#' # PMO1040: Class 5 · Assignment 5 (R edition)
#' ## AI for Early Monitoring: Clinical Deterioration and Outbreak Risk
#'
#' **Uniformed Services University · AI and Machine Learning in Military Health Care**
#' **Course Director: Dr. Mohamed (Moe) Sallam**
#'
#' This is the R version of Assignment 5, and the same file can be run in three environments:
#'
#' | Where | How to run it |
#' |---|---|
#' | **R console** | `source("PMO1040_Assignment5_Early_Monitoring_R.R", echo = TRUE)`, or paste one STEP at a time |
#' | **RStudio** | Open the `.R` file and run each STEP with **Ctrl+Enter** (Mac: **Cmd+Enter**) |
#' | **Google Colab** | Open the `.ipynb` version (R runtime) and run the cells in order |
#'
#' **Part A: clinical early warning.** The models perform poorly; however, in this case the low
#' ceiling is caused by a bug in a single line of code, and finding it is the first graded task.
#'
#' **Part B: arthropod-borne outbreak risk.** A logistic regression fails, whereas a random forest performs
#' considerably better. As shown below, the reason lies in the biology of transmission rather than in the algorithms.
#'
#' No data files are needed; all data are simulated.

#' ---
#' # STEP 0: Packages, helpers and settings (RUN)

# STEP 0 - packages
for (pkg in c("randomForest", "gbm")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages({ library(randomForest); library(gbm) })
cat("R            :", R.version.string, "\n")
cat("randomForest :", as.character(packageVersion("randomForest")),
    "| gbm:", as.character(packageVersion("gbm")), "\n")
options(repr.plot.width = 11, repr.plot.height = 4.5, width = 120)
RANDOM_STATE <- 42
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
#' # PART A: CLINICAL EARLY WARNING
#' # STEP 1: Simulate the cohort (RUN)
#' The cohort includes 2,400 patients with EHR vitals (`hr`, `rr`, `map`, `temp`), biomarkers (`crp`, `pct`),
#' a wearable (`steps`) and `age`. Note that the `logit` line encodes the underlying biology.

# STEP 1 - simulate the clinical cohort
make_cohort <- function(n = 2400, seed = 123) {
  set.seed(seed)
  df <- data.frame(
    age   = sample(18:84, n, replace = TRUE),
    hr    = rnorm(n, 85, 12),           # heart rate, bpm
    rr    = rnorm(n, 18, 4),            # respiratory rate
    map   = rnorm(n, 80, 10),           # mean arterial pressure, mmHg
    temp  = rnorm(n, 37.0, 0.6),        # temperature, degC
    crp   = abs(rnorm(n, 12, 8)),       # C-reactive protein, mg/L
    pct   = abs(rnorm(n, 0.6, 0.4)),    # procalcitonin, ng/mL
    steps = sample(500:14999, n, replace = TRUE))   # daily steps from a wearable
  # The true relationship between physiology and risk, on the log-odds scale
  logit <- (0.012 * (df$age - 50) + 0.020 * (df$hr - 80) + 0.030 * (df$rr - 18)
          - 0.020 * (df$map - 80) + 0.800 * (df$temp - 37) + 0.050 * (df$crp - 10)
          + 0.400 * (df$pct - 0.5) - 0.00005 * (df$steps - 6000))
  list(df = df, logit = logit)
}
co <- make_cohort(); clin <- co$df; logit <- co$logit
cat("cohort:", nrow(clin), "x", ncol(clin), "\n")
cat(sprintf("\nlog-odds (logit): min %.3f  max %.3f  sd %.3f\n", min(logit), max(logit), sd(logit)))
summary(clin)

#' ---
#' # STEP 2: Turn the log-odds into a label (RUN)
#' The published Class 5 activity contains the following lines (Python):
#'
#' ```
#' p = 1/(1+np.exp(-logit))
#' df["label"] = np.random.binomial(1, 1/(1+np.exp(-(p*2 - 1))))
#' ```
#'
#' The first line applies the logistic function to the log-odds, which yields a probability `p`.
#' The second line applies the logistic function **again**, this time to `p*2 - 1`. Both versions are
#' computed side by side below.

# STEP 2 - two ways of making the label
sigmoid <- function(z) 1 / (1 + exp(-z))
p_single <- sigmoid(logit)                 # the intended probability
p_double <- sigmoid(p_single * 2 - 1)      # what the published code actually computes

comp <- data.frame(version = c("single squash (intended)", "DOUBLE squash (as published)"),
                   min = c(min(p_single), min(p_double)), max = c(max(p_single), max(p_double)),
                   mean = c(mean(p_single), mean(p_double)), sd = c(sd(p_single), sd(p_double)))
comp[, -1] <- round(comp[, -1], 4)
print(comp, row.names = FALSE)

par(mfrow = c(1, 2))
br <- seq(0, 1, by = 0.025)
hist(p_single, breaks = br, col = rgb(0.27, 0.51, 0.71, .6), border = NA, ylim = c(0, 600),
     main = "What the two versions assign", xlab = "P(deterioration)", ylab = "patients")
hist(p_double, breaks = br, col = rgb(1, 0.55, 0, .6), border = NA, add = TRUE)
legend("topleft", c("single squash", "double squash"), fill = c(rgb(0.27, 0.51, 0.71, .6), rgb(1, 0.55, 0, .6)), bty = "n", cex = .8)
o <- order(logit)
plot(logit[o], p_single[o], type = "l", col = "steelblue", lwd = 2, ylim = c(0, 1),
     xlab = "true log-odds of deterioration", ylab = "assigned probability",
     main = "Probability as a function of true risk")
lines(logit[o], p_double[o], col = "darkorange", lwd = 2); grid()
par(mfrow = c(1, 1))

#' **CHECK** against the manual. Note that the double-squash curve is almost flat.
#'
#' **THINK 1.** Explain why applying the logistic function twice destroys most of the
#' signal, with reference to the input range of the second application. (Hint: `p_single*2 - 1`
#' lies in (−1, 1). What does the sigmoid do to inputs within that narrow range?)
#'
#' **Your answer to THINK 1:**

#' ---
#' # STEP 3: Train on the published (buggy) labels (RUN)
#' Three models are fitted and evaluated against the oracle that you built in Assignment 4.

# STEP 3 - fit and score, with the oracle
fit_and_score <- function(X, y, true_prob, tag, seed = RANDOM_STATE) {
  te <- split_strat(y, test_size = 0.25, seed = seed)
  Xtr <- X[-te, , drop = FALSE]; Xte <- X[te, , drop = FALSE]; ytr <- y[-te]; yte <- y[te]
  fitted <- list("Logistic Regression" = fit_logistic(Xtr, ytr, balanced = TRUE),
                 "Random Forest"       = fit_rf(Xtr, ytr, ntree = 400, nodesize = 3, seed = seed),
                 "Gradient Boosting"   = fit_gbm(Xtr, ytr, seed = seed))
  p <- lapply(fitted, predict, newdata = Xte)
  res <- do.call(rbind, lapply(names(p), function(n) score_row(n, yte, p[[n]], dataset = tag)))
  res <- rbind(res, score_row("*** ORACLE ***", yte, true_prob[te], dataset = tag))
  list(res = res, fitted = fitted, probs = p,
       split = list(Xtr = Xtr, Xte = Xte, ytr = ytr, yte = yte, pte = true_prob[te]))
}

set.seed(7)
y_published <- rbinom(length(p_double), 1, p_double)
pub <- fit_and_score(clin, y_published, p_double, "as published")
cat(sprintf("prevalence: %.3f\n\n", mean(y_published)))
print(pub$res, row.names = FALSE)
cat(sprintf("\nPR-AUC floor (positive rate in test): %.3f\n", mean(pub$split$yte)))

#' **CHECK** against the manual: the models perform near chance, and the oracle is only marginally better.
#'
#' **THINK 2.** The Assignment 4 cohort and this one appear similar from the outside: the models perform near
#' chance, and the ceiling is barely above it. However, in one case the limit was irreducible, whereas here it is a bug.
#' Describe **two diagnostic checks** you could run, without access to the generating code, to distinguish
#' between the two situations.
#'
#' **Your answer to THINK 2:**
#' 1.
#' 2.

#' ---
#' # STEP 4: Fix the bug and re-run (RUN)
#' The fix requires one line: `p_single` instead of `p_double`. The patients, seed and models remain the same.

# STEP 4 - the same experiment with the bug fixed
set.seed(7)                                # same seed, so the only change is the fix
y_fixed <- rbinom(length(p_single), 1, p_single)
fix <- fit_and_score(clin, y_fixed, p_single, "bug fixed")
cat(sprintf("prevalence: %.3f\n\n", mean(y_fixed)))

both <- rbind(pub$res, fix$res)
piv <- reshape(both[, c("dataset", "model", "ROC_AUC")], idvar = "model",
               timevar = "dataset", direction = "wide")
names(piv) <- sub("ROC_AUC.", "", names(piv)); print(piv, row.names = FALSE)
cat("\n"); print(fix$res, row.names = FALSE)

is_or <- function(r) grepl("ORACLE", r$model)
pub_ceiling <- pub$res$ROC_AUC[is_or(pub$res)]; fix_ceiling <- fix$res$ROC_AUC[is_or(fix$res)]
pub_best <- max(pub$res$ROC_AUC[!is_or(pub$res)]); fix_best <- max(fix$res$ROC_AUC[!is_or(fix$res)])
cat(sprintf("\nceiling   : %.3f  ->  %.3f   (%+.3f)\n", pub_ceiling, fix_ceiling, fix_ceiling - pub_ceiling))
cat(sprintf("best model: %.3f  ->  %.3f   (%+.3f)\n", pub_best, fix_best, fix_best - pub_best))
cat("\nOne line of code. No new data, no new features, no tuning.\n")

#' Therefore, the ceiling in Assignment 4 was a property of the world, whereas this one was a property of the
#' code. Importantly, both situations appear identical in the metrics.

#' ---
#' # STEP 5: Evaluate the fixed model properly (RUN)
#' This step reports the ROC curve, the precision-recall curve, calibration and the operating point.

# STEP 5 - full evaluation of the fixed clinical model
yte <- fix$split$yte; pte <- fix$split$pte
best_name <- "Logistic Regression"
p_best <- fix$probs[[best_name]]

roc_points <- function(y, p) {
  thr <- c(Inf, sort(unique(p), decreasing = TRUE))
  data.frame(fpr = sapply(thr, function(t) mean(p[y == 0] >= t)),
             tpr = sapply(thr, function(t) mean(p[y == 1] >= t)))
}
pr_points <- function(y, p) {
  thr <- sort(unique(p), decreasing = TRUE)
  tp <- sapply(thr, function(t) sum(p >= t & y == 1)); fp <- sapply(thr, function(t) sum(p >= t & y == 0))
  data.frame(recall = tp / sum(y), precision = tp / (tp + fp))
}
options(repr.plot.width = 15, repr.plot.height = 4.5)
par(mfrow = c(1, 3))
r <- roc_points(yte, p_best)
plot(r$fpr, r$tpr, type = "l", lwd = 2, col = "steelblue", xlab = "false positive rate",
     ylab = "true positive rate", main = sprintf("ROC (AUC = %.3f)", auc_roc(yte, p_best)))
abline(0, 1, lty = 2); grid()
q <- pr_points(yte, p_best)
plot(q$recall, q$precision, type = "l", lwd = 2, col = "steelblue", ylim = c(0, 1),
     xlab = "recall", ylab = "precision", main = sprintf("Precision-Recall (AP = %.3f)", auc_pr(yte, p_best)))
abline(h = mean(yte), lty = 2, col = "grey40"); grid()
cc <- calib_curve(yte, p_best, 10); co2 <- calib_curve(yte, pte, 10)
plot(c(0, 1), c(0, 1), type = "l", lty = 2, xlab = "mean predicted probability",
     ylab = "observed fraction", main = "Calibration")
lines(cc$mean_pred, cc$frac_pos, type = "b", pch = 19, col = "steelblue")
lines(co2$mean_pred, co2$frac_pos, type = "b", pch = 15, lty = 3)
legend("topleft", c(best_name, "oracle"), col = c("steelblue", "black"), lty = c(1, 3), bty = "n", cex = .8)
par(mfrow = c(1, 1)); options(repr.plot.width = 11, repr.plot.height = 4.5)

cat("Threshold sweep:\n")
sweep_tab <- function(y, p, thrs) do.call(rbind, lapply(thrs, function(thr) {
  cm <- confusion(y, p, thr)
  data.frame(threshold = thr, flagged = cm[["tp"]] + cm[["fp"]], caught = cm[["tp"]],
             missed = cm[["fn"]], false_alarms = cm[["fp"]],
             sensitivity = round(cm[["tp"]] / (cm[["tp"]] + cm[["fn"]]), 3),
             precision = if (cm[["tp"]] + cm[["fp"]] > 0) round(cm[["tp"]] / (cm[["tp"]] + cm[["fp"]]), 3) else NA)
}))
print(sweep_tab(yte, p_best, c(0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35)), row.names = FALSE)

# STEP 5b - YOUR OPERATING POINT for the clinical model
MY_CLINICAL_THRESHOLD <- 0.45      # <<< EDIT >>>
MY_CLINICAL_REASON <- "2-3 sentences. A screening tool on a general inpatient population.
Who receives the alert, what happens next, and what does a false alarm cost when it
fires on hundreds of patients a week?"  # <<< EDIT >>>

cm <- confusion(yte, p_best, MY_CLINICAL_THRESHOLD)
cat(sprintf("Threshold %.2f: flagged %d of %d patients\n", MY_CLINICAL_THRESHOLD,
            cm[["tp"]] + cm[["fp"]], length(yte)))
cat(sprintf("  caught %d of %d   missed %d   false alarms %d\n\n",
            cm[["tp"]], cm[["tp"]] + cm[["fn"]], cm[["fn"]], cm[["fp"]]))
class_report(yte, p_best, MY_CLINICAL_THRESHOLD, c("stable", "deteriorating"))

#' ---
#' # STEP 6: Which signals matter? (RUN)
#' Permutation importance is computed on the test set and compared with the **true** effect per standard
#' deviation taken from the `logit` line of STEP 1. This step takes about 30 seconds.

# STEP 6 - permutation importance vs the known truth
perm <- perm_importance(fix$fitted[["Random Forest"]], fix$split$Xte, yte, n_repeats = 15, seed = RANDOM_STATE)
true_effect <- abs(c(age = 0.012 * sd(clin$age), hr = 0.020 * sd(clin$hr), rr = 0.030 * sd(clin$rr),
                     map = -0.020 * sd(clin$map), temp = 0.800 * sd(clin$temp), crp = 0.050 * sd(clin$crp),
                     pct = 0.400 * sd(clin$pct), steps = -0.00005 * sd(clin$steps)))
cmp <- data.frame(feature = names(true_effect),
                  permutation = round(perm$importance[match(names(true_effect), perm$feature)], 4),
                  true_effect_per_SD = round(true_effect, 4))
cmp <- cmp[order(-cmp$true_effect_per_SD), ]
print(cmp, row.names = FALSE)
o <- cmp[order(cmp$permutation), ]
par(mar = c(4, 6, 2.5, 1))
barplot(o$permutation, names.arg = o$feature, horiz = TRUE, las = 1,
        xlab = "drop in ROC-AUC when shuffled", main = "Permutation importance - Random Forest (test set)")
abline(v = 0); par(mar = c(5, 4, 4, 2) + .1)

#' **THINK 3.** (1) Does the permutation ranking recover the true ordering, and where does it
#' disagree? (2) `steps` has the smallest true effect here. Give one reason why a real
#' deployment might find activity data *more* valuable than this simulation suggests.
#' (Consider *when* each signal becomes available.)
#'
#' **Your answer to THINK 3:**
#' 1.
#' 2.

#' ---
#' # PART B: ARTHROPOD-BORNE OUTBREAK RISK
#' # STEP 7: A different kind of prediction problem (RUN)
#' The dataset includes 1,800 district-weeks with the predictors `temp_c`, `rain_mm`, `ndvi`, `vector_index` and `ns1_rate`.
#' Mosquito development and viral replication have a **thermal optimum** near 28 °C, and
#' rainfall has an optimum as well. Accordingly, `thermal` and `rain_response` are unimodal (hump-shaped) curves rather than straight lines.

# STEP 7 - simulate vector-borne outbreak data
make_vector_data <- function(n = 1800, seed = 2026) {
  set.seed(seed)
  temp   <- rnorm(n, 27, 4.5)
  rain   <- abs(rnorm(n, 80, 55))
  ndvi   <- pmin(pmax(rnorm(n, 0.45, 0.15), 0), 1)
  vector <- abs(rnorm(n, 6, 3))
  ns1    <- pmin(pmax(rnorm(n, 0.06, 0.04), 0), 1)
  # BIOLOGY: unimodal responses, not linear ones
  thermal       <- exp(-((temp - 28) / 4.5)^2)      # optimum near 28 degC
  rain_response <- exp(-((rain - 120) / 70)^2)      # optimum near 120 mm
  lg <- -7.0 + 3.4 * thermal + 2.0 * rain_response + 1.6 * ndvi + 0.18 * vector + 8.0 * ns1
  prob <- 1 / (1 + exp(-lg))
  y <- rbinom(n, 1, prob)
  list(X = data.frame(temp_c = temp, rain_mm = rain, ndvi = ndvi, vector_index = vector, ns1_rate = ns1),
       y = y, prob = prob)
}
vd <- make_vector_data(); Xv <- vd$X; yv <- vd$y; pv <- vd$prob
cat(sprintf("district-weeks: %d | outbreak rate: %.3f\n", nrow(Xv), mean(yv)))
cat(sprintf("temperature range: %.1f to %.1f degC\n", min(Xv$temp_c), max(Xv$temp_c)))
cat(sprintf("rainfall range   : %.0f to %.0f mm\n", min(Xv$rain_mm), max(Xv$rain_mm)))

par(mfrow = c(1, 2))
for (v in c("temp_c", "rain_mm")) {
  b <- cut(Xv[[v]], unique(quantile(Xv[[v]], seq(0, 1, length.out = 13))), include.lowest = TRUE)
  plot(tapply(Xv[[v]], b, mean), tapply(yv, b, mean), type = "b", pch = 19,
       xlab = ifelse(v == "temp_c", "temperature (degC)", "weekly rainfall (mm)"),
       ylab = "observed outbreak rate", main = paste("Outbreak rate vs", v)); grid()
}
par(mfrow = c(1, 1))
cat("\nBoth curves rise and then fall. That shape is the whole of Part B.\n")

#' **CHECK**: the plots should show two clearly humped curves, peaking near 28 °C and around 120 mm.
#'
#' **THINK 4.** A logistic regression fits a straight line on the log-odds scale. What will
#' it do when the true relationship is hump-shaped? Specifically, what single slope will it estimate for
#' temperature, and what will it predict at 15 °C compared with 40 °C?
#'
#' **Your answer to THINK 4:**

#' ---
#' # STEP 8: Linear versus trees (RUN)

# STEP 8 - three models plus the oracle on the vector data
vec <- fit_and_score(Xv, yv, pv, "vector-borne")
print(vec$res, row.names = FALSE)
cat(sprintf("\nPR-AUC floor: %.3f\n", mean(vec$split$yte)))
vec_ceiling <- vec$res$ROC_AUC[is_or(vec$res)]
cat(sprintf("\nCeiling                 : %.3f\n", vec_ceiling))
cat(sprintf("Logistic regression     : %.3f\n", vec$res$ROC_AUC[vec$res$model == "Logistic Regression"]))
cat(sprintf("Random Forest           : %.3f\n", vec$res$ROC_AUC[vec$res$model == "Random Forest"]))
cat("\nUnlike Part A, the linear model is now the WORST. Why?\n")

#' ---
#' # STEP 9: The risk surface (RUN)
#' The values of `ndvi`, `vector_index` and `ns1_rate` are fixed at their medians, while temperature and
#' rainfall are swept across their ranges; the random forest is then scored at every point.

# STEP 9 - two-dimensional risk surface
rf_vec <- vec$fitted[["Random Forest"]]
t_grid <- seq(quantile(Xv$temp_c, .01), quantile(Xv$temp_c, .99), length.out = 60)
r_grid <- seq(quantile(Xv$rain_mm, .01), quantile(Xv$rain_mm, .99), length.out = 60)
grid_df <- expand.grid(temp_c = t_grid, rain_mm = r_grid)
grid_df$ndvi <- median(Xv$ndvi); grid_df$vector_index <- median(Xv$vector_index)
grid_df$ns1_rate <- median(Xv$ns1_rate)
Z_model <- matrix(predict(rf_vec, grid_df), nrow = length(t_grid))
Z_true  <- outer(t_grid, r_grid, function(tt, rr)
  1 / (1 + exp(-(-7.0 + 3.4 * exp(-((tt - 28) / 4.5)^2) + 2.0 * exp(-((rr - 120) / 70)^2)
                 + 1.6 * median(Xv$ndvi) + 0.18 * median(Xv$vector_index) + 8.0 * median(Xv$ns1_rate)))))
options(repr.plot.width = 13, repr.plot.height = 5)
par(mfrow = c(1, 2))
for (k in 1:2) {
  Z <- if (k == 1) Z_model else Z_true
  filled.contour(t_grid, r_grid, Z, nlevels = 18, color.palette = function(n) hcl.colors(n, "YlOrRd", rev = TRUE),
                 xlab = "temperature (degC)", ylab = "weekly rainfall (mm)",
                 main = if (k == 1) "Random forest's learned risk surface" else "The true risk surface",
                 plot.axes = { axis(1); axis(2); abline(v = 28, h = 120, lty = 2) })
}
par(mfrow = c(1, 1)); options(repr.plot.width = 11, repr.plot.height = 4.5)
cat("Dashed lines mark the true optima: 28 degC and 120 mm.\n")
cat("The forest finds the ridge, but as a staircase - notice the blocky edges.\n")

#' ---
#' # STEP 10: YOUR TASK, use the biology (EDIT)
#' The linear model did not fail because it is linear; rather, it failed because temperature was supplied in
#' degrees Celsius, whereas transmission depends on the **distance from the
#' thermal optimum**. In this step, engineer features that encode the biology and re-fit the model.

# STEP 10 - YOUR FEATURE ENGINEERING
MY_FEATURES_NAME <- "Gaussian thermal and rainfall response terms"    # <<< EDIT >>>
MY_RATIONALE <- "1-2 sentences: which biological mechanism each new feature encodes,
and why a linear model can use it when it could not use raw degrees."  # <<< EDIT >>>

add_bio_features <- function(X) {
  out <- X
  # ---- EDIT FROM HERE ----
  out$thermal_response <- exp(-((X$temp_c - 28) / 4.5)^2)
  out$rain_response    <- exp(-((X$rain_mm - 120) / 70)^2)
  # ---- EDIT TO HERE ----
  out
}

# --- do not edit below --------------------------------------------------------
Xv2 <- add_bio_features(Xv)
cat("features:", paste(names(Xv2), collapse = ", "), "\n\n")
eng <- fit_and_score(Xv2, yv, pv, "with my features")
summary_tab <- rbind(vec$res, eng$res)
piv2 <- reshape(summary_tab[, c("dataset", "model", "ROC_AUC")], idvar = "model",
                timevar = "dataset", direction = "wide")
names(piv2) <- sub("ROC_AUC.", "", names(piv2)); print(piv2, row.names = FALSE)
lr_before <- vec$res$ROC_AUC[vec$res$model == "Logistic Regression"]
lr_after  <- eng$res$ROC_AUC[eng$res$model == "Logistic Regression"]
cat(sprintf("\nLogistic regression: %.3f  ->  %.3f   (%+.3f)\n", lr_before, lr_after, lr_after - lr_before))
cat(sprintf("Ceiling            : %.3f\n", vec_ceiling))
cat(sprintf("Remaining gap      : %+.3f\n", vec_ceiling - lr_after))
coefs <- coef(eng$fitted[["Logistic Regression"]]$fit)[-1]
cat("\nStandardized coefficients of your linear model:\n")
print(round(coefs[order(-abs(coefs))], 3))

#' **THINK 5 (graded; this is the core of Part B).**
#' 1. State the general principle linking feature representation and model complexity.
#' 2. The worked example uses 28 °C and 120 mm from the generating code. How would you
#'    estimate the thermal optimum from the data alone, and what published entomological
#'    evidence could you use instead?
#' 3. A colleague argues that tree models make feature engineering obsolete. Give the strongest
#'    counter-argument, and describe one situation in which the colleague would be right.
#'
#' **Your answer to THINK 5:**
#' 1.
#' 2.
#' 3.

#' ---
#' # STEP 11: Streaming risk over a transmission season (RUN)
#' A single district is observed weekly for one year: temperature and rainfall follow the seasons,
#' the vector index responds with a lag, and the model re-scores the district every week.

# STEP 11 - a season of weekly risk scores for one district
weeks <- 0:51
set.seed(11)
temp_season <- 27 + 6 * sin(2 * pi * (weeks - 14) / 52) + rnorm(52, 0, 1.0)
rain_season <- pmax(90 + 80 * sin(2 * pi * (weeks - 18) / 52) + rnorm(52, 0, 18), 0)
vec_season  <- pmax(5 + 0.05 * c(0, 0, 0, head(rain_season, -3)) + rnorm(52, 0, 1.0), 0)
ns1_season  <- pmin(pmax(0.04 + 0.004 * c(rep(0, 5), head(vec_season, -5)) + rnorm(52, 0, 0.012), 0), 1)
season <- data.frame(temp_c = temp_season, rain_mm = rain_season, ndvi = median(Xv$ndvi),
                     vector_index = vec_season, ns1_rate = ns1_season)
risk_rf <- predict(rf_vec, season)
risk_lr <- predict(eng$fitted[["Logistic Regression"]], add_bio_features(season))

options(repr.plot.width = 11, repr.plot.height = 6.5)
par(mfrow = c(2, 1), mar = c(4, 4, 2.5, 1))
plot(weeks, risk_rf, type = "b", pch = 20, col = "steelblue", ylim = c(0, 1), xlab = "",
     ylab = "P(outbreak)", main = "Weekly outbreak risk across one transmission season")
lines(weeks, risk_lr, type = "b", pch = 20, col = "darkorange"); abline(h = 0.5, lty = 2, col = "firebrick")
legend("topleft", c("Random forest (raw features)", "Logistic + biology features", "example alert threshold"),
       col = c("steelblue", "darkorange", "firebrick"), lty = c(1, 1, 2), bty = "n", cex = .75)
plot(weeks, temp_season, type = "l", ylim = range(c(0, temp_season, rain_season / 10, vec_season)),
     xlab = "week of year", ylab = "")
lines(weeks, rain_season / 10, col = "steelblue"); lines(weeks, vec_season, col = "darkgreen")
legend("topleft", c("temperature (degC)", "rainfall / 10 (mm)", "vector index"),
       col = c("black", "steelblue", "darkgreen"), lty = 1, bty = "n", cex = .75)
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + .1); options(repr.plot.width = 11, repr.plot.height = 4.5)

peak <- which.max(risk_lr) - 1
cat(sprintf("Peak predicted risk in week %d (P = %.3f)\n", peak, max(risk_lr)))
cat(sprintf("Peak temperature  in week %d\n", which.max(temp_season) - 1))
cat(sprintf("Peak rainfall     in week %d\n", which.max(rain_season) - 1))
cat("\nThe risk peak does not coincide with either driver's peak.\n")

#' **THINK 6.** (1) Why does the risk peak fall in a different week from the temperature
#' and rainfall peaks? Refer to the thermal optimum and to the lags in the vector index and
#' NS1 rate. (2) A preventive-medicine officer needs **six weeks** of lead time to requisition
#' and distribute larvicide. Does this model provide it? What would you change about the
#' model (not the data) to provide that lead time?
#'
#' **Your answer to THINK 6:**
#' 1.
#' 2.

#' ---
#' # STEP 12: Export (EDIT then RUN)

# STEP 12 - export
lastname <- "LASTNAME"          # <<< EDIT >>>
write.csv(both, sprintf("PMO1040_A5_%s_clinical_bug_comparison.csv", lastname), row.names = FALSE)
write.csv(cmp, sprintf("PMO1040_A5_%s_clinical_importances.csv", lastname), row.names = FALSE)
write.csv(summary_tab, sprintf("PMO1040_A5_%s_vector_models.csv", lastname), row.names = FALSE)
write.csv(data.frame(week = weeks, temp_c = round(temp_season, 2), rain_mm = round(rain_season, 1),
                     vector_index = round(vec_season, 2), ns1_rate = round(ns1_season, 4),
                     risk_rf = round(risk_rf, 4), risk_lr = round(risk_lr, 4)),
          sprintf("PMO1040_A5_%s_season_risk.csv", lastname), row.names = FALSE)
cat("Working directory:", getwd(), "\n")
for (f in sort(list.files(pattern = sprintf("^PMO1040_A5_%s_.*csv$", lastname)))) cat("wrote", f, "\n")

# STEP 12b - reproducibility record
cat("Run at        :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("R             :", R.version.string, "\n")
cat("RANDOM_STATE  :", RANDOM_STATE, "\n")
cat("--- Part A: clinical ---\n")
cat("  cohort      :", nrow(clin), "x", ncol(clin), "\n")
cat(sprintf("  ceiling as published %.3f -> fixed %.3f\n", pub_ceiling, fix_ceiling))
cat(sprintf("  best model  %.3f -> %.3f\n", pub_best, fix_best))
cat("  threshold   :", MY_CLINICAL_THRESHOLD, "\n")
cat("--- Part B: vector-borne ---\n")
cat(sprintf("  district-weeks: %d | outbreak rate %.3f\n", nrow(Xv), mean(yv)))
cat(sprintf("  ceiling     : %.3f\n", vec_ceiling))
cat(sprintf("  logistic %.3f -> %.3f with '%s'\n", lr_before, lr_after, MY_FEATURES_NAME))

#' ---
#' # STEP 13: Write-up and submission
#' See the manual for the seven write-up prompts and the submission checklist.
#'
#' *PMO1040 · Class 5 · Assignment 5 (R edition) · Uniformed Services University*
