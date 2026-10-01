#' # PMO1040: Class 6 · Assignment 6 (R edition)
#' ## Cardiovascular Risk Stratification, and How to Audit a Model Before You Trust It
#'
#' **Uniformed Services University · AI and Machine Learning in Military Health Care**
#' **Course Director: Dr. Mohamed (Moe) Sallam**
#'
#' This is the R version of Assignment 6, and the same file can be run in three environments:
#'
#' | Where | How to run it |
#' |---|---|
#' | **R console** | `source("PMO1040_Assignment6_CVD_Risk_R.R", echo = TRUE)`, or paste one STEP at a time |
#' | **RStudio** | Open the `.R` file and run each STEP with **Ctrl+Enter** (Mac: **Cmd+Enter**) |
#' | **Google Colab** | Open the `.ipynb` version (R runtime) and run the cells in order |
#'
#' The brief is to *build a logistic regression for cardiovascular risk on synthetic data and
#' output a ROC curve and an AUC.* In this assignment, we complete it twice: (i) on the published dataset,
#' which scores a perfect 1.00 (your task is to determine why that number is uninformative), and (ii) on a
#' properly constructed cohort of 1,200 patients.
#'
#' No data files are needed; all data are generated within the script.

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
#' # PART 1: THE AUDIT
#' # STEP 1: The dataset as published (RUN)
#' This step reproduces the ten-row dataset and logistic regression from the Class 6 activity, translated to R.

# STEP 1 - the published activity
published <- data.frame(
  age         = c(45, 54, 62, 39, 71, 50, 68, 47, 59, 64),
  bmi         = c(23, 28, 31, 26, 29, 32, 27, 25, 30, 33),
  bp          = c(120, 140, 135, 128, 150, 138, 142, 130, 145, 155),
  cholesterol = c(180, 220, 210, 195, 240, 205, 225, 200, 235, 250),
  diabetes    = c(0, 1, 1, 0, 1, 0, 1, 0, 1, 1),
  cvd_risk    = c(0, 1, 1, 0, 1, 0, 1, 0, 1, 1))
print(published, row.names = FALSE)

pub_split <- function(seed) { set.seed(seed); sort(sample.int(10, 3)) }   # 30% test, unstratified
fit_pub <- function(te) suppressWarnings(
  glm(cvd_risk ~ age + bmi + bp + cholesterol + diabetes, data = published[-te, ], family = binomial))
te_p <- pub_split(42)
model_p <- fit_pub(te_p)
probs_p <- predict(model_p, published[te_p, ], type = "response")
yp_te <- published$cvd_risk[te_p]
cat(sprintf("\nROC-AUC: %.2f\n", auc_roc(yp_te, probs_p)))

roc_points <- function(y, p) {
  thr <- c(Inf, sort(unique(p), decreasing = TRUE))
  data.frame(fpr = sapply(thr, function(t) mean(p[y == 0] >= t)),
             tpr = sapply(thr, function(t) mean(p[y == 1] >= t)))
}
options(repr.plot.width = 5, repr.plot.height = 4.5)
r <- roc_points(yp_te, probs_p)
plot(r$fpr, r$tpr, type = "b", pch = 19, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "False positive rate", ylab = "True positive rate",
     main = sprintf("CVD risk ROC curve (AUC = %.2f)", auc_roc(yp_te, probs_p)))
abline(0, 1, lty = 2); grid()
options(repr.plot.width = 11, repr.plot.height = 4.5)

#' **CHECK**: the output should show `ROC-AUC: 1.00` and a curve that rises straight up the left edge and runs along the top.
#' Importantly, a perfect score on a first attempt should raise the greatest suspicion.
#'
#' **THINK 1 (before running anything else).** Write down three specific checks you
#' would perform to decide whether that 1.00 is real. Be concrete: "count how many rows are in
#' the test set" is a check, whereas "check for overfitting" is not.
#'
#' **Your answer to THINK 1:**
#' 1.
#' 2.
#' 3.

#' ---
#' # STEP 2: The audit (RUN)
#' This step performs four checks; compare them with the checks you anticipated in THINK 1.

# STEP 2 - AUDIT CHECK 1: how much data is actually being evaluated?
cat("CHECK 1 - sample size\n")
cat("  total rows      :", nrow(published), "\n")
cat("  training rows   :", nrow(published) - length(te_p), "\n")
cat("  TEST rows       :", length(te_p), "\n")
cat("  test labels     :", yp_te, "\n")
cat("  test positives  :", sum(yp_te), " test negatives:", sum(1 - yp_te), "\n")

# STEP 2 - AUDIT CHECK 2: does any single feature give the answer away?
cat("CHECK 2 - correlation of each feature with the target\n")
corr <- sort(cor(published)[, "cvd_risk"][-6], decreasing = TRUE)
print(round(corr, 4))
perfect <- names(corr)[abs(corr) >= 0.999]
if (length(perfect)) cat("\n  *** A correlation of 1.0 with the target is the target under another name:",
                         paste(perfect, collapse = ", "), "***\n")

# STEP 2 - AUDIT CHECK 3: is the target literally a copy of a feature?
cat("CHECK 3 - direct comparison\n")
print(published[, c("diabetes", "cvd_risk")], row.names = FALSE)
cat("\n  cvd_risk == diabetes for EVERY row:", all(published$cvd_risk == published$diabetes), "\n")
cat("  The 'model' has one job: copy the diabetes column.\n")

# STEP 2 - AUDIT CHECK 4: how stable is that AUC?
cat("CHECK 4 - seed sensitivity\n")
for (seed in c(0, 1, 2, 7, 13, 42, 99)) {
  te <- pub_split(seed); yt <- published$cvd_risk[te]
  pr <- predict(fit_pub(te), published[te, ], type = "response")
  verdict <- if (length(unique(yt)) < 2) "UNDEFINED (test set has only one class)"
             else sprintf("%.3f", auc_roc(yt, pr))
  cat(sprintf("    seed=%3d  test labels [%s]  ->  AUC %s\n", seed, paste(yt, collapse = ", "), verdict))
}
cat("\n  The reported AUC is either 1.000 or mathematically undefined.\n")

#' **CHECK** against the manual.
#'
#' **THINK 2.** Defect 1 is a **leak**. You have now encountered four leaks: Assignment 2 (a rolling
#' mean that included the current day), Assignment 3 (`pixel_area`, the target × 100),
#' Assignment 4 (a label computed from the same window as the features) and Assignment 6
#' (the target copied into a feature). Rank them from hardest to easiest to detect in a real
#' project and justify the order. Then state the single check you would build into your
#' workflow to catch as many of them as possible.
#'
#' **Your answer to THINK 2:**
#' Ranking (hardest to easiest):
#' The check I would build in:

#' ---
#' # PART 2: DOING IT PROPERLY
#' # STEP 3: Build a realistic cohort (RUN)
#' The cohort includes 1,200 patients with nine risk factors. Diabetes depends on BMI and age, and antihypertensive
#' use depends on blood pressure, as in real populations.

# STEP 3 - a properly constructed synthetic cohort
make_cvd_cohort <- function(n = 1200, seed = 2026) {
  set.seed(seed)
  age    <- sample(40:79, n, replace = TRUE)
  sex_m  <- rbinom(n, 1, 0.55)
  bmi    <- pmin(pmax(rnorm(n, 28, 4.5), 16), 50)
  sbp    <- pmin(pmax(rnorm(n, 132, 18), 90), 210)      # systolic BP, mmHg
  chol   <- pmin(pmax(rnorm(n, 205, 38), 110), 340)     # total cholesterol, mg/dL
  hdl    <- pmin(pmax(rnorm(n, 52, 14), 20), 110)       # HDL cholesterol, mg/dL
  smoker <- rbinom(n, 1, 0.22)
  diabetes <- rbinom(n, 1, 1 / (1 + exp(-(-2.6 + 0.10 * (bmi - 28) + 0.02 * (age - 60)))))
  bp_meds  <- rbinom(n, 1, 1 / (1 + exp(-(-1.6 + 0.05 * (sbp - 132)))))
  lg <- (-2.40 + 0.0620 * (age - 60) + 0.5200 * sex_m + 0.0200 * (bmi - 28)
         + 0.0210 * (sbp - 132) + 0.0075 * (chol - 205) - 0.0260 * (hdl - 52)
         + 0.7800 * smoker + 0.7200 * diabetes + 0.2800 * bp_meds)
  prob <- 1 / (1 + exp(-lg)); y <- rbinom(n, 1, prob)
  list(X = data.frame(age, sex_male = sex_m, bmi = round(bmi, 1), sbp = round(sbp), chol = round(chol),
                      hdl = round(hdl), smoker, diabetes, bp_meds), y = y, prob = prob)
}
cv <- make_cvd_cohort(); X <- cv$X; y <- cv$y; true_prob <- cv$prob
cat("patients          :", nrow(X), "\n")
cat(sprintf("10-year CVD events: %d (%.1f%%)\n", sum(y), 100 * mean(y)))
cat("\nrisk factor prevalence:\n")
cat(sprintf("  smokers          : %.1f%%\n", 100 * mean(X$smoker)))
cat(sprintf("  diabetes         : %.1f%%\n", 100 * mean(X$diabetes)))
cat(sprintf("  on BP medication : %.1f%%\n", 100 * mean(X$bp_meds)))
cat("\nevent rate by age decade:\n")
dec <- cut(X$age, c(39, 49, 59, 69, 79))
print(data.frame(age = levels(dec), n = as.vector(table(dec)),
                 event_rate = round(as.vector(tapply(y, dec, mean)), 3)), row.names = FALSE)

#' ---
#' # STEP 4: Train, and check against the ceiling (RUN)

# STEP 4 - three models plus the oracle
test_idx <- split_strat(y, test_size = 0.30, seed = RANDOM_STATE)
X_train <- X[-test_idx, ]; y_train <- y[-test_idx]; X_test <- X[test_idx, ]; y_test <- y[test_idx]
p_test <- true_prob[test_idx]
cat(sprintf("train: %d patients (%d events) | test: %d patients (%d events)\n",
            nrow(X_train), sum(y_train), nrow(X_test), sum(y_test)))
models <- list("Logistic Regression" = fit_logistic(X_train, y_train, balanced = FALSE),
               "Random Forest"       = fit_rf(X_train, y_train, ntree = 400, nodesize = 5, seed = RANDOM_STATE),
               "Gradient Boosting"   = fit_gbm(X_train, y_train, seed = RANDOM_STATE))
probs <- lapply(models, predict, newdata = X_test)
results <- rbind(do.call(rbind, lapply(names(probs), function(n) score_row(n, y_test, probs[[n]]))),
                 score_row("*** ORACLE ***", y_test, p_test))
cat("\n"); print(results, row.names = FALSE)
cat(sprintf("\nPR-AUC floor (event rate in test): %.3f\n", mean(y_test)))
best_name <- "Logistic Regression"; p_best <- probs[[best_name]]
ceiling <- results$ROC_AUC[4]
cat(sprintf("\nLogistic regression %.3f vs ceiling %.3f  ->  gap %+.3f\n",
            results$ROC_AUC[1], ceiling, ceiling - results$ROC_AUC[1]))

#' For comparison, published cardiovascular risk equations (Framingham, pooled cohort equations) report
#' C-statistics of roughly 0.72–0.80 on external populations.
#'
#' **THINK 3.** A colleague sees the logistic regression's AUC and says the model is "not
#' good enough to use." Given the ceiling, explain in 2–3 sentences why that framing is
#' wrong, and what question they should ask instead.
#'
#' **Your answer to THINK 3:**

#' ---
#' # STEP 5: How many patients do you need to measure an AUC? (RUN)
#' Random subsets of the test set are drawn at increasing sizes, each with a **bootstrap 95%
#' confidence interval** for the AUC. This step takes about 30 seconds.

# STEP 5 - how test-set size controls what you can claim
set.seed(0)
bootstrap_auc_ci <- function(yt, pp, B = 600) {
  aucs <- replicate(B, { i <- sample.int(length(yt), length(yt), replace = TRUE)
                         if (length(unique(yt[i])) < 2) NA else auc_roc(yt[i], pp[i]) })
  aucs <- aucs[!is.na(aucs)]
  if (length(aucs) < 30) return(c(NA, NA))
  quantile(aucs, c(0.025, 0.975), names = FALSE)
}
prec <- do.call(rbind, lapply(c(3, 10, 25, 50, 100, 200, length(y_test)), function(n_test) {
  widths <- c(); undefined <- 0
  for (r in 1:15) {
    sub <- sample.int(length(y_test), min(n_test, length(y_test)))
    if (length(unique(y_test[sub])) < 2) { undefined <- undefined + 1; next }
    ci <- bootstrap_auc_ci(y_test[sub], p_best[sub], B = 400)
    if (!is.na(ci[1])) widths <- c(widths, ci[2] - ci[1])
  }
  data.frame(test_n = n_test, median_CI_width = if (length(widths)) round(median(widths), 3) else NA,
             subsets_with_one_class = sprintf("%d/15", undefined))
}))
print(prec, row.names = FALSE)
plot(prec$test_n, prec$median_CI_width, log = "x", type = "b", pch = 19,
     xlab = "patients in the test set (log scale)", ylab = "width of the 95% CI on AUC",
     main = "How much data you need before an AUC means anything"); grid()
ci_full <- bootstrap_auc_ci(y_test, p_best, B = 2000)
cat(sprintf("\nFull test set (n=%d): AUC %.3f, 95%% CI [%.3f, %.3f]\n",
            length(y_test), auc_roc(y_test, p_best), ci_full[1], ci_full[2]))

#' Note that the precision of an AUC is governed by the number of patients in the smaller class.
#'
#' **THINK 4.** You report the AUC and CI above, and a colleague reports an AUC 0.02 higher for
#' their model on the same test set. (1) Can you say theirs is better? (2) What would you
#' need to answer that properly?
#'
#' **Your answer to THINK 4:**
#' 1.
#' 2.

#' ---
#' # STEP 6: Coefficients and odds ratios (RUN)
#' Each logistic-regression coefficient exponentiates to an odds ratio that can be stated in a
#' sentence. Because the model was fitted on standardized features, the coefficients are per SD.

# STEP 6 - interpret the model
lr <- models[["Logistic Regression"]]
coefs_scaled <- coef(lr$fit)[names(X)]
coefs_raw <- coefs_scaled / lr$s[names(X)]           # per one natural unit
interp <- data.frame(feature = names(X), coef_per_SD = round(coefs_scaled, 3),
                     coef_raw = round(coefs_raw, 4), OR_per_SD = round(exp(coefs_scaled), 3))
interp <- interp[order(-abs(interp$coef_per_SD)), ]
print(interp, row.names = FALSE)
cat("\nRead in plain English:\n")
for (i in 1:4) with(interp[i, ], cat(sprintf("  A one-SD increase in %-9s %s the odds of a CVD event by a factor of %.2f\n",
  feature, ifelse(coef_per_SD > 0, "raises", "lowers"), exp(abs(coef_per_SD)))))
cat("\nAnd in clinical units:\n")
cat(sprintf("  every 10 years of age      : OR %.2f\n", exp(coefs_raw[["age"]] * 10)))
cat(sprintf("  every 10 mmHg systolic BP  : OR %.2f\n", exp(coefs_raw[["sbp"]] * 10)))
cat(sprintf("  current smoking            : OR %.2f\n", exp(coefs_raw[["smoker"]])))
cat(sprintf("  diabetes                   : OR %.2f\n", exp(coefs_raw[["diabetes"]])))
o <- interp[order(interp$coef_per_SD), ]
par(mar = c(4, 6, 2.5, 1))
barplot(o$coef_per_SD, names.arg = o$feature, horiz = TRUE, las = 1,
        xlab = "coefficient per standard deviation (log-odds)", main = "What drives predicted cardiovascular risk")
abline(v = 0); par(mar = c(5, 4, 4, 2) + .1)

#' Note that `bp_meds` carries a *positive* coefficient. This is not evidence that the drugs are harmful;
#' rather, being on treatment marks diagnosed hypertension. Therefore, a coefficient may appear causal
#' and be interpreted as causal without actually being causal.
#'
#' **THINK 5.** Pick one other coefficient and explain why its *causal* reading is not
#' supported even though its *predictive* value is real. What study design would be needed
#' to make the causal claim?
#'
#' **Your answer to THINK 5:**

#' ---
#' # STEP 7: Calibration and clinical risk categories (RUN)
#' Risk bands follow the ACC/AHA convention: below 5% low, 5–7.5% borderline, 7.5–20% intermediate and above 20% high.

# STEP 7 - calibration and risk bands
par(mfrow = c(1, 2))
cc <- calib_curve(y_test, p_best, 8); co <- calib_curve(y_test, p_test, 8)
plot(c(0, 0.6), c(0, 0.6), type = "l", lty = 2, xlab = "mean predicted risk",
     ylab = "observed event rate", main = "Calibration")
lines(cc$mean_pred, cc$frac_pos, type = "b", pch = 19, col = "steelblue")
lines(co$mean_pred, co$frac_pos, type = "b", pch = 15, lty = 3)
legend("topleft", c("Logistic Regression", "oracle"), col = c("steelblue", "black"), lty = c(1, 3), bty = "n", cex = .8)
bands <- cut(p_best, c(0, 0.05, 0.075, 0.20, 1), labels = c("<5% low", "5-7.5% borderline", "7.5-20% intermediate", ">20% high"))
band_tab <- data.frame(band = levels(bands), patients = as.vector(table(bands)),
                       observed_rate = round(as.vector(tapply(y_test, bands, mean)), 3))
barplot(band_tab$observed_rate, names.arg = sub(" ", "\n", band_tab$band), cex.names = .75,
        ylab = "observed event rate", main = "Observed risk by predicted risk band")
par(mfrow = c(1, 1))
print(band_tab, row.names = FALSE)
cat(sprintf("\nBrier score: %.4f (oracle %.4f, always-predict-base-rate %.4f)\n",
            brier(y_test, p_best), brier(y_test, p_test), brier(y_test, rep(mean(y_train), length(y_test)))))

#' ---
#' # STEP 8: YOUR TASK, does it work for everyone? (EDIT)
#' For each subgroup, this step reports (i) ROC-AUC (discrimination), (ii) mean predicted vs observed risk (calibration),
#' and (iii) n / events (can either number be trusted?).

# STEP 8 - subgroup performance
slice_report <- function(mask, label) {
  yt <- y_test[mask]; pr <- p_best[mask]
  if (length(yt) == 0) return(NULL)
  data.frame(subgroup = label, n = length(yt), events = sum(yt),
             ROC_AUC = round(auc_roc(yt, pr), 3), mean_predicted = round(mean(pr), 3),
             observed_rate = round(mean(yt), 3), calib_gap = round(mean(pr) - mean(yt), 3))
}
SUBGROUPS <- list(
  list(rep(TRUE, nrow(X_test)), "ALL"),
  list(X_test$sex_male == 1,    "male"),
  list(X_test$sex_male == 0,    "female"),
  list(X_test$age < 60,         "age under 60"),
  list(X_test$age >= 60,        "age 60 and over"),
  list(X_test$smoker == 1,      "smoker"),
  list(X_test$smoker == 0,      "non-smoker"),
  list(X_test$diabetes == 1,    "diabetic"),
  # <<< EDIT >>> add one subgroup of your own below, e.g. obesity or high BP
  list(X_test$bmi >= 30,        "BMI 30 and over"))
report <- do.call(rbind, lapply(SUBGROUPS, function(s) slice_report(s[[1]], s[[2]])))
print(report, row.names = FALSE)
cat("\ncalib_gap = mean predicted minus observed. Positive means OVER-prediction.\n")
overall <- report$ROC_AUC[report$subgroup == "ALL"]
others <- report[report$subgroup != "ALL", ]; worst <- others[which.min(others$ROC_AUC), ]
cat(sprintf("\nOverall AUC %.3f; worst subgroup '%s' at %.3f (n=%d, events=%d)\n",
            overall, worst$subgroup, worst$ROC_AUC, worst$n, worst$events))

#' **THINK 6 (graded; this is the core of the assignment).**
#' 1. Both age subgroups have a **lower** AUC than the whole test set. Explain why
#'    restricting the sample to one age band reduces discrimination, with reference to the coefficient table in STEP 6.
#' 2. In the smoker row, is the model over- or under-predicting for smokers, by how
#'    much, and is n / events large enough to treat it as a real finding?
#' 3. For primary prevention (counseling, statin consideration), which finding concerns you
#'    more operationally, and why? Name one specific action you would take (a modeling
#'    change, a deployment restriction or a data-collection change, but not "monitor it").
#'    Note that STEP 9 will show a third finding: how few under-60 events a single threshold catches.
#' 4. For the diabetic subgroup, what would you write in a report, given its n?
#'
#' **Your answer to THINK 6:**
#' 1.
#' 2.
#' 3.
#' 4.

#' ---
#' # STEP 9: From probability to decision (EDIT then RUN)

# STEP 9 - threshold sweep and your operating point
sweep <- do.call(rbind, lapply(c(0.30, 0.25, 0.20, 0.15, 0.125, 0.10, 0.075), function(thr) {
  cm <- confusion(y_test, p_best, thr); tr <- cm[["tp"]] + cm[["fp"]]
  data.frame(threshold = thr, treated = tr, pct_treated = round(100 * tr / length(y_test), 1),
             events_captured = cm[["tp"]], events_missed = cm[["fn"]],
             sensitivity = round(cm[["tp"]] / (cm[["tp"]] + cm[["fn"]]), 3),
             precision = if (tr) round(cm[["tp"]] / tr, 3) else NA,
             treated_per_event = if (cm[["tp"]]) round(tr / cm[["tp"]], 1) else NA)
}))
print(sweep, row.names = FALSE)
cat(sprintf("\n(%d patients in the test set, %d events)\n", length(y_test), sum(y_test)))

# STEP 9b - YOUR OPERATING POINT
MY_THRESHOLD <- 0.15        # <<< EDIT >>> choose from the sweep above
MY_REASON <- "3-4 sentences. What action does crossing this threshold trigger? What does
it cost a patient treated unnecessarily? What does it cost to miss someone? Refer to
treated_per_event, and say whether you would apply this threshold to patients
under 60 given STEP 8."   # <<< EDIT >>>

cm <- confusion(y_test, p_best, MY_THRESHOLD); tr <- cm[["tp"]] + cm[["fp"]]
cat(sprintf("Threshold %.3f:\n", MY_THRESHOLD))
cat(sprintf("  %d of %d patients (%.1f%%) recommended for treatment\n", tr, length(y_test), 100 * tr / length(y_test)))
cat(sprintf("  captures %d of %d events, misses %d\n", cm[["tp"]], cm[["tp"]] + cm[["fn"]], cm[["fn"]]))
cat(sprintf("  %.1f patients treated per event captured\n\n", tr / cm[["tp"]]))
class_report(y_test, p_best, MY_THRESHOLD, c("no event", "CVD event"))
cat("\nEffect of YOUR threshold by subgroup:\n")
print(do.call(rbind, lapply(SUBGROUPS, function(s) {
  yt <- y_test[s[[1]]]; yh <- as.integer(p_best[s[[1]]] >= MY_THRESHOLD)
  if (sum(yt) == 0) return(NULL)
  data.frame(subgroup = s[[2]], n = length(yt), pct_treated = round(100 * mean(yh), 1),
             sensitivity = round(sum(yh == 1 & yt == 1) / sum(yt), 3))
})), row.names = FALSE)

#' A single threshold does not treat every subgroup equally. This is a distributional
#' consequence of your choice, and students should be able to defend it.

#' ---
#' # STEP 10: Export (EDIT then RUN)

# STEP 10 - export
lastname <- "LASTNAME"          # <<< EDIT >>>
write.csv(results, sprintf("PMO1040_A6_%s_models_vs_oracle.csv", lastname), row.names = FALSE)
write.csv(prec, sprintf("PMO1040_A6_%s_auc_precision.csv", lastname), row.names = FALSE)
write.csv(interp, sprintf("PMO1040_A6_%s_coefficients.csv", lastname), row.names = FALSE)
write.csv(report, sprintf("PMO1040_A6_%s_subgroups.csv", lastname), row.names = FALSE)
write.csv(transform(sweep, chosen_threshold = MY_THRESHOLD),
          sprintf("PMO1040_A6_%s_threshold_sweep.csv", lastname), row.names = FALSE)
cat("Working directory:", getwd(), "\n")
for (f in sort(list.files(pattern = sprintf("^PMO1040_A6_%s_.*csv$", lastname)))) cat("wrote", f, "\n")

# STEP 10b - reproducibility record
cat("Run at        :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("R             :", R.version.string, "\n")
cat("RANDOM_STATE  :", RANDOM_STATE, "\n")
cat(sprintf("Cohort        : %d patients | event rate %.3f\n", nrow(X), mean(y)))
cat(sprintf("Test set      : %d patients | %d events\n", nrow(X_test), sum(y_test)))
cat(sprintf("Best model    : %s  ROC-AUC %.3f (ceiling %.3f)\n", best_name, auc_roc(y_test, p_best), ceiling))
cat("Threshold     :", MY_THRESHOLD, "\n")
cat(sprintf("Worst subgroup: %s at %.3f\n", worst$subgroup, worst$ROC_AUC))

#' ---
#' # STEP 11: Write-up and submission
#' See the manual for the seven write-up prompts and the submission checklist.
#'
#' *PMO1040 · Class 6 · Assignment 6 (R edition) · Uniformed Services University*
