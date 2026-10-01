#' # PMO1040: Class 4 · Assignment 4 (R edition)
#' ## Sepsis Early Warning in a Battlefield ICU
#'
#' **Uniformed Services University · AI and Machine Learning in Military Health Care**
#' **Course Director: Dr. Mohamed (Moe) Sallam**
#'
#' This is the R version of Assignment 4, and the same file can be run in three environments:
#'
#' | Where | How to run it |
#' |---|---|
#' | **R console** (R GUI or a terminal) | `source("PMO1040_Assignment4_Sepsis_R.R", echo = TRUE)`, or paste one STEP at a time |
#' | **RStudio** | Open the `.R` file, put the cursor in a STEP, press **Ctrl+Enter** (Mac: **Cmd+Enter**) line by line |
#' | **Google Colab** | Open the `.ipynb` version; it starts on the **R** runtime. Run the cells in order |
#'
#' Symbols: **RUN** = run as written · **EDIT** = change something first ·
#' **THINK** = write an answer (graded) · **CHECK** = compare with the manual.
#'
#' No data files are needed. All data are simulated within the script; therefore, nothing has to be downloaded.

#' ---
#' # STEP 0: Packages, helpers and settings (RUN)
#' The first run installs two packages, which takes about 1–3 minutes in Colab and less time on a local computer.

# STEP 0 - packages
for (pkg in c("randomForest", "gbm")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({ library(randomForest); library(gbm) })
cat("R            :", R.version.string, "\n")
cat("randomForest :", as.character(packageVersion("randomForest")), "\n")
cat("gbm          :", as.character(packageVersion("gbm")), "\n")
options(repr.plot.width = 11, repr.plot.height = 4.5, width = 120)   # plot size in Colab; ignored elsewhere
RANDOM_STATE <- 7
cat("\nOK. RANDOM_STATE =", RANDOM_STATE, "\n")

#' ---
#' ### Helper functions (RUN; no edits needed)
#' Plain-R versions of every metric and model used below. Run this block once per session.

# STEP 0b - helper functions
## ---- PMO1040 helper functions (identical in Assignments 4, 5 and 6) ----------
## No edits are required in this block. However, it is advisable to read it once, because every
## metric used in the assignment is written out here in plain R, so that no computation is hidden.

## ROC-AUC: the probability that a random positive is ranked above a random
## negative (ties count half). Same number as pROC::auc or sklearn roc_auc_score.
auc_roc <- function(y, p) {
  y <- as.integer(y); n1 <- sum(y == 1); n0 <- sum(y == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(p)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

## PR-AUC as average precision (step-wise area under the precision-recall curve).
auc_pr <- function(y, p) {
  y <- as.integer(y); if (sum(y) == 0) return(NA_real_)
  thr  <- sort(unique(p), decreasing = TRUE)
  tp   <- sapply(thr, function(t) sum(p >= t & y == 1))
  fp   <- sapply(thr, function(t) sum(p >= t & y == 0))
  prec <- tp / (tp + fp); rec <- tp / sum(y)
  sum(diff(c(0, rec)) * prec)
}

## Brier score: mean squared error of the predicted probabilities.
brier <- function(y, p) mean((p - as.integer(y))^2)

## One row of the standard results table.
score_row <- function(name, y, p, ...) {
  data.frame(..., model = name,
             ROC_AUC = round(auc_roc(y, p), 3),
             PR_AUC  = round(auc_pr(y, p), 3),
             Brier   = round(brier(y, p), 4), check.names = FALSE)
}

## Stratified train/test split: returns the row numbers of the TEST set.
## Keeps the positive rate the same in both halves.
split_strat <- function(y, test_size = 0.25, seed = 1) {
  set.seed(seed)
  idx <- unlist(lapply(split(seq_along(y), y), function(ix)
    ix[sample.int(length(ix), round(length(ix) * test_size))]))
  sort(unname(idx))
}

## Logistic regression with standardization learned on the TRAINING rows only
## (the R equivalent of a scikit-learn Pipeline with StandardScaler).
## balanced = TRUE up-weights the rarer class, as class_weight="balanced" does.
fit_logistic <- function(Xtr, ytr, balanced = TRUE) {
  mu <- sapply(Xtr, mean); s <- sapply(Xtr, sd); s[s == 0] <- 1
  Z  <- as.data.frame(scale(Xtr, mu, s)); Z$.y <- as.integer(ytr)
  w  <- if (balanced) ifelse(Z$.y == 1, length(ytr) / (2 * sum(ytr == 1)),
                                        length(ytr) / (2 * sum(ytr == 0)))
        else rep(1, length(ytr))
  fit <- suppressWarnings(glm(.y ~ ., data = Z, family = binomial, weights = w))
  structure(list(fit = fit, mu = mu, s = s), class = "pmo_logit")
}
predict.pmo_logit <- function(object, newdata, ...) {
  Z <- as.data.frame(scale(newdata[, names(object$mu), drop = FALSE], object$mu, object$s))
  as.numeric(predict(object$fit, newdata = Z, type = "response"))
}

## Random forest (randomForest package). Returns P(class = 1).
fit_rf <- function(Xtr, ytr, ntree = 400, nodesize = 3, seed = 1) {
  set.seed(seed)
  structure(list(fit = randomForest::randomForest(x = Xtr, y = factor(ytr, levels = c(0, 1)),
                                                  ntree = ntree, nodesize = nodesize)),
            class = "pmo_rf")
}
predict.pmo_rf <- function(object, newdata, ...)
  as.numeric(predict(object$fit, newdata = newdata, type = "prob")[, "1"])

## Gradient boosting (gbm package): 100 trees, learning rate 0.1, depth 3.
fit_gbm <- function(Xtr, ytr, seed = 1) {
  set.seed(seed)
  d <- Xtr; d$.y <- as.integer(ytr)
  structure(list(fit = gbm::gbm(.y ~ ., data = d, distribution = "bernoulli",
                                n.trees = 100, interaction.depth = 3, shrinkage = 0.1,
                                n.minobsinnode = 10, bag.fraction = 1, verbose = FALSE)),
            class = "pmo_gbm")
}
predict.pmo_gbm <- function(object, newdata, ...)
  as.numeric(predict(object$fit, newdata = newdata, n.trees = 100, type = "response"))

## Confusion-matrix counts at a threshold.
confusion <- function(y, p, thr) {
  yh <- as.integer(p >= thr); y <- as.integer(y)
  c(tn = sum(yh == 0 & y == 0), fp = sum(yh == 1 & y == 0),
    fn = sum(yh == 0 & y == 1), tp = sum(yh == 1 & y == 1))
}

## Precision / recall / F1 for each class, like a classification report.
class_report <- function(y, p, thr, names = c("negative", "positive")) {
  cm <- confusion(y, p, thr)
  pr1 <- if (cm["tp"] + cm["fp"] > 0) cm["tp"] / (cm["tp"] + cm["fp"]) else 0
  re1 <- cm["tp"] / (cm["tp"] + cm["fn"])
  pr0 <- if (cm["tn"] + cm["fn"] > 0) cm["tn"] / (cm["tn"] + cm["fn"]) else 0
  re0 <- cm["tn"] / (cm["tn"] + cm["fp"])
  f1  <- function(a, b) if (a + b > 0) 2 * a * b / (a + b) else 0
  out <- data.frame(class = names,
                    precision = round(c(pr0, pr1), 3), recall = round(c(re0, re1), 3),
                    f1 = round(c(f1(pr0, re0), f1(pr1, re1)), 3),
                    support = c(sum(y == 0), sum(y == 1)))
  print(out, row.names = FALSE)
  cat("accuracy:", round(mean(as.integer(p >= thr) == y), 3), "\n")
}

## Calibration curve with quantile bins: mean predicted vs observed rate per bin.
calib_curve <- function(y, p, n_bins = 8) {
  br <- unique(quantile(p, seq(0, 1, length.out = n_bins + 1)))
  b  <- cut(p, br, include.lowest = TRUE)
  data.frame(mean_pred = tapply(p, b, mean), frac_pos = tapply(as.integer(y), b, mean))
}

## Permutation importance: shuffle one column at a time on the TEST set and
## record how far ROC-AUC falls. Model-agnostic.
perm_importance <- function(model, X, y, n_repeats = 15, seed = 1) {
  set.seed(seed)
  base <- auc_roc(y, predict(model, X))
  res <- t(sapply(names(X), function(v) {
    d <- replicate(n_repeats, { Xp <- X; Xp[[v]] <- sample(Xp[[v]])
                                base - auc_roc(y, predict(model, Xp)) })
    c(importance = mean(d), sd = sd(d))
  }))
  out <- data.frame(feature = rownames(res), importance = res[, "importance"],
                    sd = res[, "sd"], row.names = NULL)
  out[order(-out$importance), ]
}

#' ---
#' # STEP 1: Simulate the cohort (RUN)
#' The simulated cohort includes 900 casualties × 48 hourly observations (43,200 rows). Injury severity (`inj_sev`) shifts
#' every vital sign, all variables trend over the 48 hours, and 2% of hours lose all four core vitals
#' (i.e., a monitoring blackout). This step takes a few seconds.

# STEP 1 - the simulator
simulate_cohort <- function(n = 900, hours = 48, seed = 42) {
  set.seed(seed)
  age        <- sample(18:64, n, replace = TRUE)
  inj_sev    <- runif(n)
  immunocomp <- rbinom(n, 1, 0.1)
  t     <- 0:(hours - 1)
  trend <- t / hours
  out <- vector("list", n)
  for (i in 1:n) {
    s <- inj_sev[i]
    b_hr   <- rnorm(1, 85 + 25 * s, 8);   b_map <- rnorm(1, 80 - 10 * s, 6)
    b_rr   <- rnorm(1, 18 + 6 * s, 3);    b_temp <- rnorm(1, 37.0 + 0.4 * s, 0.4)
    b_spo2 <- min(max(rnorm(1, 96 - 2 * s, 1.5), 85), 100)
    b_wbc  <- rnorm(1, 9 + 4 * s, 2.5)
    b_lac  <- abs(rnorm(1, 1.8 + 1.5 * s, 0.7)); b_cr <- abs(rnorm(1, 1.0 + 0.4 * s, 0.2))

    HR    <- rnorm(hours, b_hr * (1 + 0.15 * trend), 5)
    MAP   <- rnorm(hours, b_map * (1 - 0.10 * trend), 4)
    RR    <- rnorm(hours, b_rr * (1 + 0.10 * trend), 2)
    TEMP  <- rnorm(hours, b_temp + 0.2 * sin(2 * pi * t / 24), 0.2)
    SpO2  <- pmin(pmax(rnorm(hours, b_spo2 - 0.5 * trend, 1.2), 80), 100)
    WBC   <- abs(rnorm(hours, b_wbc * (1 + 0.05 * trend), 2.0))
    Lac   <- abs(rnorm(hours, b_lac * (1 + 0.25 * trend), 0.5))
    Creat <- abs(rnorm(hours, b_cr * (1 + 0.05 * trend), 0.1))
    Urine <- pmax(rnorm(hours, 45 - 10 * trend, 8), 0)
    Fluids  <- rbinom(hours, 1, ifelse(MAP < 65 | Lac > 2.5, 0.35, 0.15))
    Pressor <- rbinom(hours, 1, ifelse(MAP < 60, 0.25, 0.05))

    blackout <- runif(hours) < 0.02                  # field monitoring blackout
    HR[blackout] <- NA; MAP[blackout] <- NA; RR[blackout] <- NA; TEMP[blackout] <- NA

    out[[i]] <- data.frame(encounter = i - 1, hour = t, age = age[i], inj_sev = s,
                           immunocomp = immunocomp[i], HR, MAP, RR, TEMP, SpO2, WBC,
                           Lactate = Lac, Creat, Urine, Fluids, Pressor)
  }
  do.call(rbind, out)
}

ts <- simulate_cohort()
cat("rows:", format(nrow(ts), big.mark = ","), "| encounters:", length(unique(ts$encounter)),
    "| hours each:", length(unique(ts$hour)), "\n")
cat(sprintf("missing rate in the four core vitals: %.2f%%\n",
            100 * mean(is.na(ts[, c("HR", "MAP", "RR", "TEMP")]))))
head(ts)

#' ---
#' # STEP 2: How the outcome label is made (RUN)
#' Each casualty receives a risk score derived from red flags **averaged over all 48 hours**, which is then transformed
#' through a logistic function into a probability; the label is then drawn as a coin flip weighted by
#' that probability. Note that `SHARPNESS` controls how decisively the score determines the outcome.

# STEP 2 - build the label
SHARPNESS <- 1.0        # published setting. STEP 10 asks you to change this.

make_labels <- function(ts, sharpness = 1.0, seed = 1) {
  set.seed(seed)
  m  <- function(v) tapply(ts[[v]], ts$encounter, mean, na.rm = TRUE)
  f1 <- function(v) tapply(ts[[v]], ts$encounter, function(z) z[1])
  enc <- data.frame(encounter = as.numeric(names(m("Lactate"))),
                    Lactate = m("Lactate"), MAP = m("MAP"), RR = m("RR"), TEMP = m("TEMP"),
                    WBC = m("WBC"), Pressor = m("Pressor"), immunocomp = f1("immunocomp"),
                    inj_sev = f1("inj_sev"), age = f1("age"))
  risk <- (0.40 * (enc$Lactate > 2.2) + 0.30 * (enc$MAP < 65) + 0.20 * (enc$RR > 22)
         + 0.20 * (enc$TEMP > 38.3)   + 0.20 * (enc$WBC > 12) + 0.30 * (enc$Pressor > 0.15)
         + 0.25 * enc$immunocomp + 0.35 * enc$inj_sev + 0.01 * (enc$age - 40) / 10)
  prob <- 1 / (1 + exp(-sharpness * (risk - 1.2)))          # the squashing step
  prob <- pmin(pmax(prob, 0.02), 0.98)
  data.frame(encounter = enc$encounter, risk_score = risk,
             true_prob = prob,                                # ground truth we normally never see
             sepsis_next24h = rbinom(length(prob), 1, prob), row.names = NULL)
}

lab    <- make_labels(ts, sharpness = SHARPNESS)
labels <- lab[, c("encounter", "sepsis_next24h")]

cat(sprintf("prevalence of sepsis label : %.3f\n", mean(lab$sepsis_next24h)))
cat(sprintf("risk score  : min %.3f  max %.3f  sd %.3f\n",
            min(lab$risk_score), max(lab$risk_score), sd(lab$risk_score)))
cat(sprintf("true probability: min %.3f  max %.3f  mean %.3f\n",
            min(lab$true_prob), max(lab$true_prob), mean(lab$true_prob)))

par(mfrow = c(1, 2))
hist(lab$risk_score, breaks = 35, main = "Risk score across 900 casualties", xlab = "risk score")
hist(lab$true_prob, breaks = 35, main = "True probability of sepsis", xlab = "P(sepsis)")
abline(v = 0.5, lty = 2)
par(mfrow = c(1, 1))

#' **CHECK** against the manual. In the right-hand histogram, the true probability of every
#' casualty falls within a narrow band around a coin flip.
#'
#' **THINK 1.** Two features of this label are clinically implausible. Identify both of them.
#' (Specifically, one is evident in the printed prevalence, and the other concerns the time window from which the risk score is
#' computed, compared with the label's name.)
#'
#' **Your answer to THINK 1:**
#' 1.
#' 2.

#' ---
#' # STEP 3: Look at a patient (RUN)
#' It is advisable never to model a time series that has not been plotted first.

# STEP 3 - one septic and one non-septic trajectory
septic     <- labels$encounter[labels$sepsis_next24h == 1][1]
non_septic <- labels$encounter[labels$sepsis_next24h == 0][1]
options(repr.plot.width = 11, repr.plot.height = 6)
par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))
for (v in c("HR", "MAP", "Lactate", "TEMP")) {
  a <- ts[ts$encounter == septic, ]; b <- ts[ts$encounter == non_septic, ]
  plot(a$hour, a[[v]], type = "l", ylim = range(c(a[[v]], b[[v]]), na.rm = TRUE),
       xlab = "hour", ylab = v, main = v)
  lines(b$hour, b[[v]], lty = 2, col = "grey40")
}
legend("topleft", c("sepsis label = 1", "sepsis label = 0"), lty = c(1, 2), bty = "n", cex = .8)
par(mfrow = c(1, 1)); options(repr.plot.width = 11, repr.plot.height = 4.5)
cat("Note the gaps in the lines. Those are the 2% missingness bursts.\n")

#' ---
#' # STEP 4: Feature engineering on the last six hours (RUN)
#' Means, minima, maxima and **slopes** are computed over the last six hours of each casualty, in addition to
#' `miss_rate6`, which is the share of those hours with missing vitals.

# STEP 4 - last-6-hour features
slope <- function(x) {                     # gradient of a straight line through x
  if (length(x) < 2) return(0)
  t <- seq_along(x) - 1
  sum((t - mean(t)) * (x - mean(x))) / sum((t - mean(t))^2)
}
fill_mean <- function(x) { x[is.na(x)] <- mean(x, na.rm = TRUE); x }

last6h_features <- function(d) {
  d <- tail(d[order(d$hour), ], 6)
  mn <- function(x) mean(x, na.rm = TRUE)
  c(HR_mean6 = mn(d$HR), HR_min6 = min(d$HR, na.rm = TRUE), HR_slope6 = slope(fill_mean(d$HR)),
    MAP_mean6 = mn(d$MAP), MAP_min6 = min(d$MAP, na.rm = TRUE), MAP_slope6 = slope(fill_mean(d$MAP)),
    RR_mean6 = mn(d$RR), RR_slope6 = slope(fill_mean(d$RR)),
    TEMP_mean6 = mn(d$TEMP), TEMP_max6 = max(d$TEMP, na.rm = TRUE),
    SpO2_mean6 = mn(d$SpO2), SpO2_min6 = min(d$SpO2),
    WBC_mean6 = mn(d$WBC),
    Lactate_mean6 = mn(d$Lactate), Lactate_slope6 = slope(fill_mean(d$Lactate)),
    Creat_mean6 = mn(d$Creat), Urine_mean6 = mn(d$Urine),
    Fluids_rate6 = mn(d$Fluids), Pressor_rate6 = mn(d$Pressor),
    miss_rate6 = mean(is.na(d[, c("HR", "MAP", "RR", "TEMP")])))
}

by_enc <- split(ts, ts$encounter)
feat6  <- as.data.frame(do.call(rbind, lapply(by_enc, last6h_features)))
feat6$encounter <- as.numeric(names(by_enc))
static <- unique(ts[, c("encounter", "age", "inj_sev", "immunocomp")])

X_all <- merge(merge(static, feat6, by = "encounter"), labels, by = "encounter")
y <- as.integer(X_all$sepsis_next24h)
X <- X_all[, setdiff(names(X_all), c("sepsis_next24h", "encounter"))]

cat("feature matrix:", nrow(X), "x", ncol(X), "\n")
cat("features:", paste(names(X), collapse = ", "), "\n")
cat("\nany remaining missing values:", sum(is.na(X)), "\n")

#' **CHECK**: the output should read `feature matrix: 900 x 23` and `0` missing values.
#'
#' **THINK 2.** The label used means over all 48 hours, whereas the features use only the
#' last 6 hours. (1) Does that make the problem harder or easier than it appears?
#' (2) This label is a function of the same window from which the features are derived. What is the
#' difference from a real early-warning target, and why does it matter?
#'
#' **Your answer to THINK 2:**
#' 1.
#' 2.

#' ---
#' # STEP 5: Split and train three models (RUN)
#' The data are divided using a stratified 75/25 split, followed by fitting (i) logistic regression (scaled on the training rows only),
#' (ii) random forest (bagging) and (iii) gradient boosting (boosting).

# STEP 5 - split and fit
test_idx <- split_strat(y, test_size = 0.25, seed = RANDOM_STATE)
X_train <- X[-test_idx, ]; y_train <- y[-test_idx]
X_test  <- X[test_idx, ];  y_test  <- y[test_idx]
cat("train:", nrow(X_train), "rows | test:", nrow(X_test), "rows\n")
cat(sprintf("positive rate - train %.3f | test %.3f\n", mean(y_train), mean(y_test)))

models <- list(
  "Logistic Regression" = fit_logistic(X_train, y_train, balanced = TRUE),
  "Random Forest"       = fit_rf(X_train, y_train, ntree = 400, nodesize = 3, seed = RANDOM_STATE),
  "Gradient Boosting"   = fit_gbm(X_train, y_train, seed = RANDOM_STATE))
probs <- lapply(models, predict, newdata = X_test)
cat("\nTrained:", paste(names(models), collapse = ", "), "\n")

# STEP 5b - evaluate
scores <- do.call(rbind, Map(function(n, p) score_row(n, y_test, p), names(probs), probs))
print(scores, row.names = FALSE)
cat("\nReference points:\n")
cat("  ROC-AUC of pure chance          : 0.500\n")
cat(sprintf("  PR-AUC floor (= positive rate)  : %.3f\n", mean(y_test)))
cat(sprintf("  Brier of always predicting %.2f : %.4f\n", mean(y_test),
            brier(y_test, rep(mean(y_test), length(y_test)))))

#' **CHECK** against the manual. These performance values are poor, which is expected at this stage.
#' Students should therefore avoid tuning the models and complete STEP 6 first.
#'
#' **THINK 3.** *Before* running STEP 6, explain why these numbers are so poor. List the three most
#' likely explanations in the order in which you would investigate them.
#'
#' **Your answer to THINK 3:**
#' 1.
#' 2.
#' 3.

#' ---
#' # STEP 6: How good could *any* model be? (RUN)
#' Because the data are simulated, we know the **true probability** for each casualty. A predictor
#' that outputs this probability (the oracle) cannot be beaten in expectation; therefore, its score represents the ceiling.

# STEP 6 - the oracle
test_encounters <- X_all$encounter[test_idx]
oracle_prob <- lab$true_prob[match(test_encounters, lab$encounter)]
oracle_true <- lab$sepsis_next24h[match(test_encounters, lab$encounter)]
oracle <- score_row("*** ORACLE (true probability) ***", oracle_true, oracle_prob)
full <- rbind(scores, oracle)
print(full, row.names = FALSE)
cat("\nThe oracle knows the exact probability that generated every label.\n")
cat(sprintf("Best trained model ROC-AUC : %.3f\n", max(scores$ROC_AUC)))
cat(sprintf("Ceiling                    : %.3f\n", oracle$ROC_AUC))
cat(sprintf("Gap                        : %.3f\n", oracle$ROC_AUC - max(scores$ROC_AUC)))

#' Note that the oracle score, rather than the gap between models, is the relevant reference. Chance is 0.500, and the oracle
#' shows how much signal the problem contains overall. If the best model performs near the oracle, the limitation lies
#' in the information available rather than in the algorithm.
#'
#' **THINK 4.** No model can exceed the ceiling on these data. Your commander requests a
#' sepsis early-warning system. In 3–4 sentences, what would you tell them, and what would you
#' request instead of additional modeling time?
#'
#' **Your answer to THINK 4:**

#' ---
#' # STEP 7: Calibration (RUN)
#' AUC measures **ranking**, whereas calibration measures whether predicted probabilities can be taken at face value: when the
#' model predicts 40%, does the event occur 40% of the time?

# STEP 7 - calibration
cols <- c("steelblue", "darkorange", "forestgreen")
par(mfrow = c(1, 2))
plot(c(0, 1), c(0, 1), type = "l", lty = 2, xlab = "mean predicted probability",
     ylab = "observed fraction positive", main = "Calibration")
for (k in seq_along(probs)) {
  cc <- calib_curve(y_test, probs[[k]], 8); lines(cc$mean_pred, cc$frac_pos, type = "b", col = cols[k], pch = 19)
}
cc <- calib_curve(oracle_true, oracle_prob, 8); lines(cc$mean_pred, cc$frac_pos, type = "b", lty = 3, lwd = 2, pch = 15)
legend("topleft", c(names(probs), "oracle"), col = c(cols, "black"), lty = c(1, 1, 1, 3), bty = "n", cex = .75)
br <- seq(0, 1, by = 0.04)
hist(oracle_prob, breaks = br, border = "black", col = NA, lwd = 2, lty = 3, ylim = c(0, 80),
     main = "How spread out are the predictions?", xlab = "predicted probability")
for (k in seq_along(probs)) hist(probs[[k]], breaks = br, border = cols[k], col = NA, add = TRUE)
par(mfrow = c(1, 1))

cat("Brier scores (lower = better calibrated):\n")
for (n in names(probs)) cat(sprintf("  %-26s %.4f\n", n, brier(y_test, probs[[n]])))
cat(sprintf("  %-26s %.4f\n", "oracle", brier(oracle_true, oracle_prob)))
cat(sprintf("  %-26s %.4f\n", "always predict base rate", brier(y_test, rep(mean(y_train), length(y_test)))))

#' The oracle's predictions fall within a narrow band, which reflects all the information available in the data.
#' Therefore, models that spread their predictions more widely are expressing confidence that the data do not support.

#' ---
#' # STEP 8: Choosing an operating threshold (EDIT then RUN)
#' Importantly, a false positive costs cultures, antibiotics and staff attention, whereas a false negative
#' represents septic shock without warning far down an evacuation chain.

# STEP 8 - threshold sweep on the best model
best_name <- scores$model[which.max(scores$ROC_AUC)]
p_best <- probs[[best_name]]
cat("Best-ranking model:", best_name, "\n\n")
sweep <- do.call(rbind, lapply(c(0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25), function(thr) {
  cm <- confusion(y_test, p_best, thr)
  data.frame(threshold = thr, flagged = cm[["tp"]] + cm[["fp"]], caught = cm[["tp"]],
             missed = cm[["fn"]], false_alarms = cm[["fp"]],
             sensitivity = round(cm[["tp"]] / (cm[["tp"]] + cm[["fn"]]), 3),
             precision = if (cm[["tp"]] + cm[["fp"]] > 0) round(cm[["tp"]] / (cm[["tp"]] + cm[["fp"]]), 3) else NA)
}))
print(sweep, row.names = FALSE)
cat(sprintf("\n(%d casualties in the test set, %d of them septic)\n", length(y_test), sum(y_test)))

# STEP 8b - YOUR OPERATING POINT
MY_THRESHOLD <- 0.35        # <<< EDIT >>> choose from the sweep
MY_REASON <- "2-3 sentences. Who acts on the alert, what a false alarm costs a forward
surgical team, what a missed case costs. Refer to the evacuation chain and to
antimicrobial stewardship."   # <<< EDIT >>>

cm <- confusion(y_test, p_best, MY_THRESHOLD)
cat(sprintf("Threshold %.2f on %s\n", MY_THRESHOLD, best_name))
cat(sprintf("  flagged %d of %d casualties\n", cm[["tp"]] + cm[["fp"]], length(y_test)))
cat(sprintf("  caught %d of %d septic   missed %d   false alarms %d\n\n",
            cm[["tp"]], cm[["tp"]] + cm[["fn"]], cm[["fn"]], cm[["fp"]]))
class_report(y_test, p_best, MY_THRESHOLD, c("no sepsis", "sepsis"))
cat("\nA threshold cannot create information that is not there.\n")

#' ---
#' # STEP 9: What is the model using? (RUN)
#' Permutation importance is computed on the **test** set: each feature is shuffled in turn, and the decrease in ROC-AUC is recorded.
#' This step takes 20–60 seconds.

# STEP 9 - permutation importance
imp <- perm_importance(models[["Random Forest"]], X_test, y_test, n_repeats = 15, seed = RANDOM_STATE)
print(head(transform(imp, importance = round(importance, 4), sd = round(sd, 4)), 12), row.names = FALSE)
options(repr.plot.width = 8, repr.plot.height = 6)
top <- imp[15:1, ]
par(mar = c(4, 9, 2.5, 1))
barplot(top$importance, names.arg = top$feature, horiz = TRUE, las = 1, cex.names = .75,
        xlab = "drop in ROC-AUC when shuffled", main = "Permutation importance - Random Forest (test set)")
abline(v = 0)
par(mar = c(5, 4, 4, 2) + .1); options(repr.plot.width = 11, repr.plot.height = 4.5)
cat("\nFeatures at or below zero contributed nothing on unseen data.\n")

#' **THINK 5.** `miss_rate6` is random noise in this simulation. Explain what it might
#' encode in a real field hospital, and describe one way in which that could go badly wrong once the model is deployed.
#'
#' **Your answer to THINK 5:**

#' ---
#' # STEP 10: YOUR TASK, move the ceiling (EDIT)
#' `SHARPNESS` is the multiplier in `prob = 1/(1 + exp(-SHARPNESS * (risk - 1.2)))`.
#' The 900 casualties, their physiology and the features remain the same; only the link between risk and outcome changes.

# STEP 10 - how the ceiling moves with label sharpness
ceil <- do.call(rbind, lapply(c(1, 2, 4, 8, 16, 32), function(k) {
  lk <- make_labels(ts, sharpness = k, seed = 1)
  data.frame(sharpness = k, prevalence = round(mean(lk$sepsis_next24h), 3),
             ceiling_ROC_AUC = round(auc_roc(lk$sepsis_next24h, lk$true_prob), 3),
             ceiling_PR_AUC = round(auc_pr(lk$sepsis_next24h, lk$true_prob), 3))
}))
print(ceil, row.names = FALSE)

par(mar = c(5, 4, 3, 4))
plot(log2(ceil$sharpness), ceil$ceiling_ROC_AUC, type = "b", pch = 19, col = "firebrick",
     xaxt = "n", ylim = c(0.5, 1), xlab = "SHARPNESS (log scale)", ylab = "best achievable ROC-AUC",
     main = "Same patients, same features - only the label's determinism changes")
axis(1, at = log2(ceil$sharpness), labels = ceil$sharpness); abline(h = 0.5, lty = 2, col = "grey")
par(new = TRUE)
plot(log2(ceil$sharpness), ceil$prevalence, type = "b", pch = 15, col = "steelblue",
     axes = FALSE, xlab = "", ylab = "")
axis(4, col.axis = "steelblue"); mtext("sepsis prevalence", side = 4, line = 2.5, col = "steelblue")
par(mar = c(5, 4, 4, 2) + .1)

# STEP 10b - YOUR CHOICE: re-run the whole pipeline at a different sharpness
MY_SHARPNESS <- 8          # <<< EDIT >>> pick from the table above
MY_JUSTIFICATION <- "1-2 sentences: why this sharpness gives a more clinically plausible
cohort than the published setting. Cite the prevalence."   # <<< EDIT >>>

lab2 <- make_labels(ts, sharpness = MY_SHARPNESS, seed = 1)
y2 <- lab2$sepsis_next24h[match(X_all$encounter, lab2$encounter)]
te2 <- split_strat(y2, test_size = 0.25, seed = RANDOM_STATE)
X2_train <- X[-te2, ]; y2_train <- y2[-te2]; X2_test <- X[te2, ]; y2_test <- y2[te2]

m2 <- list("Logistic Regression" = fit_logistic(X2_train, y2_train, balanced = TRUE),
           "Random Forest"       = fit_rf(X2_train, y2_train, seed = RANDOM_STATE),
           "Gradient Boosting"   = fit_gbm(X2_train, y2_train, seed = RANDOM_STATE))
my_results <- do.call(rbind, lapply(names(m2), function(n) score_row(n, y2_test, predict(m2[[n]], X2_test))))
enc_te <- X_all$encounter[te2]
o_p <- lab2$true_prob[match(enc_te, lab2$encounter)]
o_y <- lab2$sepsis_next24h[match(enc_te, lab2$encounter)]
my_results <- rbind(my_results, score_row("*** ORACLE ***", o_y, o_p))
cat(sprintf("SHARPNESS = %g   prevalence = %.3f\n\n", MY_SHARPNESS, mean(y2)))
print(my_results, row.names = FALSE)
best2 <- max(my_results$ROC_AUC[1:3]); ceil2 <- my_results$ROC_AUC[4]
cat(sprintf("\nBest model %.3f vs ceiling %.3f  ->  gap %+.3f\n", best2, ceil2, ceil2 - best2))
cat("A LARGE positive gap means the models are now leaving signal on the table.\n")

#' **Your findings** (i) the gap at your sharpness; (ii) whether there is headroom now and which model uses it;
#' (iii) what prevalence does to PR-AUC; and (iv) one sentence on when tuning is worth a week of effort:

#' ---
#' # STEP 11: Streaming risk for one casualty (RUN)
#' An early-warning system re-scores each casualty every hour as new observations arrive.

# STEP 11 - hour-by-hour risk trajectory
pid <- labels$encounter[labels$sepsis_next24h == 1][4]
stream <- ts[ts$encounter == pid, ]; stream <- stream[order(stream$hour), ]
srow <- static[static$encounter == pid, ]
hours <- 6:48
risk_series <- sapply(hours, function(t) {
  f <- last6h_features(stream[stream$hour < t, ])
  row <- as.data.frame(t(c(age = srow$age, inj_sev = srow$inj_sev, immunocomp = srow$immunocomp, f)))
  predict(models[["Random Forest"]], row[, names(X)])
})
options(repr.plot.width = 10, repr.plot.height = 6)
par(mfrow = c(2, 1), mar = c(4, 4, 2.5, 1))
plot(hours, risk_series, type = "b", pch = 20, ylab = "predicted risk", xlab = "",
     main = sprintf("Casualty %d (true label = 1) - risk re-scored every hour", pid))
abline(h = MY_THRESHOLD, lty = 2, col = "firebrick")
plot(stream$hour, stream$Lactate, type = "l", xlab = "hour", ylab = "",
     ylim = range(c(stream$Lactate, stream$MAP / 20), na.rm = TRUE))
lines(stream$hour, stream$MAP / 20, col = "darkorange")
legend("topleft", c("Lactate (mmol/L)", "MAP / 20 (mmHg)"), col = c("black", "darkorange"), lty = 1, bty = "n", cex = .8)
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + .1); options(repr.plot.width = 11, repr.plot.height = 4.5)
crossings <- sum(diff(risk_series >= MY_THRESHOLD) != 0)
cat(sprintf("The risk crosses your threshold (%.2f) %d times over 42 hours.\n", MY_THRESHOLD, crossings))
cat("Crossings at other thresholds:\n")
for (thr in c(0.30, 0.35, 0.40, 0.45, 0.50, 0.55))
  cat(sprintf("  %.2f -> %2d crossings\n", thr, sum(diff(risk_series >= thr) != 0)))
cat("Each upward crossing is an alert. Would a clinician trust a system that alarms\n")
cat("and un-alarms this often?\n")

#' **THINK 6.** Name two engineering changes that would reduce the alert rate without
#' simply raising the threshold, and describe what each one costs.
#'
#' **Your answer to THINK 6:**
#' 1.
#' 2.

#' ---
#' # STEP 12: Export (EDIT then RUN)
#' Files are written to your **working directory**; see the manual for the location of that directory in
#' the R console, RStudio and Colab.

# STEP 12 - export
lastname <- "LASTNAME"          # <<< EDIT >>>
write.csv(full, sprintf("PMO1040_A4_%s_models_vs_oracle.csv", lastname), row.names = FALSE)
write.csv(ceil, sprintf("PMO1040_A4_%s_ceiling_sweep.csv", lastname), row.names = FALSE)
write.csv(transform(my_results, sharpness = MY_SHARPNESS, prevalence = round(mean(y2), 3)),
          sprintf("PMO1040_A4_%s_my_sharpness.csv", lastname), row.names = FALSE)
write.csv(transform(imp, importance = round(importance, 5), sd = round(sd, 5)),
          sprintf("PMO1040_A4_%s_importances.csv", lastname), row.names = FALSE)
cat("Working directory:", getwd(), "\n")
for (f in sort(list.files(pattern = sprintf("^PMO1040_A4_%s_.*csv$", lastname)))) cat("wrote", f, "\n")

# STEP 12b - reproducibility record
cat("Run at        :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("R             :", R.version.string, "\n")
cat("randomForest  :", as.character(packageVersion("randomForest")),
    "| gbm:", as.character(packageVersion("gbm")), "\n")
cat("RANDOM_STATE  :", RANDOM_STATE, "\n")
cat("Cohort        :", length(unique(ts$encounter)), "encounters x", length(unique(ts$hour)), "hours\n")
cat("Features      :", ncol(X), "\n")
cat(sprintf("Published SHARPNESS = 1 : best %.3f | ceiling %.3f\n", max(scores$ROC_AUC), oracle$ROC_AUC))
cat(sprintf("My SHARPNESS = %-3g      : best %.3f | ceiling %.3f\n", MY_SHARPNESS, best2, ceil2))
cat("Threshold     :", MY_THRESHOLD, "\n")

#' ---
#' # STEP 13: Write-up and submission
#' See Part 8 of the manual for the seven write-up prompts and the submission checklist.
#'
#' *PMO1040 · Class 4 · Assignment 4 (R edition) · Uniformed Services University*
