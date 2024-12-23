<img src="pics/cover.png" width="640" alt="Data Science Diagnostic Checklist" />

## Help! My **Model** is **_worse_** on the **Test Set** than the **Train Set**

This problem is typically referred to as the "_**generalization gap**_", "_**train-test gap**_", or simply as "_**overfitting**_", where performance of a final chosen model on the hold out test set is worse than on the train set.

We can use diagnostic tests to systematically probe the data and the model in order to gather evidence about the unknown underlying cause of a performance mismatch.

**Scope**: Machine learning and deep learning predictive modeling for regression and classification tasks.

Let's walk through some categories of checklists of questions we can use to learn more.

**UPDATE**: See code examples for checks at: <https://DataScienceDiagnostics.com>

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Train/Test Split Procedure](#traintest-split-procedure)
3. [Split Size Sensitivity Analysis](#split-size-sensitivity-analysis)
4. [Data Leakage](#data-leakage)
5. [Quantify the Performance Gap](#quantify-the-performance-gap)
6. [Challenge the Performance Gap](#challenge-the-performance-gap)
7. [Data Distribution Checks](#data-distribution-checks)
8. [Performance Distribution Checks](#performance-distribution-checks)
9. [Residual Error Checks](#residual-error-checks)
10. [Residual Error Distribution Checks](#residual-error-distribution-checks)
11. [Overfitting Train Set Checks](#overfitting-train-set-checks)
12. [Overfitting Test Harness Checks](#overfitting-test-harness-checks)
13. [Overfitting Test Set Checks](#overfitting-test-set-checks)
14. [Model Robustness/Stability Checks](#model-robustnessstability-checks)
15. [So Now What? Interventions](#so-now-what-interventions)
16. [FAQ](#frequently-asked-questions)
17. [Glossary of Terms](#glossary-of-terms)
18. [Disclaimer](#disclaimer)
19. [About](#about)

## Problem Definition

Let's define our terms and the problem we are having (also see the [glossary](#glossary-of-terms)).

1. A **dataset** is collected from the domain and split into a **train set** used for model selection and a **test set** used for the evaluation for the chosen model.
2. A **test harness** is used to evaluate many candidate models on the **train set** by estimating their generalized performance (e.g. a subsequent train/test split, k-fold cross-validation, etc.).
3. A single **chosen model** is (eventually) selected using an estimate of its generalized performance on the **test harness** (here, "model" refers to the pipeline of data transforms, model architecture, model hyperparameters, calibration of predictions, etc.).
3. The chosen **model** is then fit on the entire **train set** and evaluated on the **test set** to give a single unbiased point estimate of generalized performance.
4. The difference between 1) the **test harness** model performance on the **train set** and 2) the point estimation of the model fit on the **train set** and evaluated on the **test set** _do not match_. **Why?**

Variations on this problem:

1. Performance on the **test harness** and **test set** appropriately match, but performance on a **hold out set** is worse.
2. Performance on the **test harness** and **test set** appropriately match, but performance on **data in production** is worse.

## Train/Test Split Procedure

<img src="pics/logo-procedure.svg" width="300" />

* _Is there evidence that the split of the dataset into train/test subsets followed best practices?_

### Procedure

1. Did you remove duplicates from the dataset before splitting into train and test sets?
2. Did you shuffle the dataset when you split into train and test sets?
3. Did you stratify the split by class label (only classification tasks)?
4. Did you stratify the split by domain entities? (only data with domain entities that have more than one example, e.g one customer with many transactions, one user with many recommendations, etc.)?
5. Are the train/test sets disjoint (e.g. non-overlapping, do you need to confirm this)?
6. Did you use a typical split ratio (e.g. 70/30, 80/20, 90/10)?
7. Did you use a library function to perform the split (e.g. `sklearn.model_selection.train_test_split`)?
8. Did you document the split procedure (e.g. number of duplicates removed, split percentage, random number seed, reasons, etc.)?
9. Did you save the original dataset, train set, and test set in separate files?

Related:

1. Did you receive train/test sets already split (do they have separate sources)?
2. Did a stakeholder withhold the test/validation set?
3. Did you remove unambiguously irrelevant features (e.g. database id, last updated datetime, etc.)?
4. According to domain experts, are samples from the dataset i.i.d (e.g. no known spatial/temporal/group dependence, concept/distribution shifts, sample bias, etc.)?

## Split Size Sensitivity Analysis

<img src="pics/logo-sensitivity.svg" width="300" />

Common split percentages are just heuristics, it is better to know how your data/model behave under different split scenarios.

If you have the luxury of time and compute, perform a sensitivity analysis of split sizes (e.g. between 50/50 to 99/1 with some linear interval) and optimize for one or more target properties of stability like data distribution and/or model performance.

The motivating question here is:

* _What evidence is there that one split size is better than another?_

### Split vs Distribution Sensitivity
Compare data distributions of train/test sets and find the point where they diverge (Data Distribution Tests below).

1. Compare the correlations of each input/target variable (e.g. Pearson's or Spearman's).
2. Compare the central tendencies for each input/target variable (e.g. t-Test or Mann-Whitney U Test).
3. Compare the distributions for each input/target variable (e.g. Kolmogorov-Smirnov Test or Anderson-Darling Test).
4. Compare the variance for each input/target variable (e.g. F-test, Levene's test).
5. Compare the divergence for the input/target variable distributions.

### Split vs Performance Sensitivity
Compare the distributions of standard un-tuned machine learning model performance scores on train/test sets and find the point where they diverge (Model Performance Tests below).

1. Compare the correlations of model performance scores (e.g. Pearson's or Spearman's).
2. Compare the central tendencies of model performance scores (e.g. t-Test or Mann-Whitney U Test).
3. Compare the distributions of model performance scores (e.g. Kolmogorov-Smirnov Test or Anderson-Darling Test).
4. Compare the variance of model performance scores (e.g. F-test, Levene's test).
5. Compare the divergence of model performance score distributions.

## Data Leakage

<img src="pics/logo-leakage.svg" width="300" />

Leakage of information about the test set (data or what models work well) to the train set/test harness is called "**test set leakage**" or simply **data leakage** and may introduce an optimistic bias: better results than we should expect in practice.

This bias is typically not discovered until the model is employed on entirely new data or deployed to production.

### Data Preparation Leakage

Any data cleaning (beyond removing duplicates), preparation, or analysis like tasks performed on the whole dataset (prior to splitting) may result "**data preparation leakage**".

You must prepare/analyze the train set only.

* _Is there evidence that knowledge from the test set leaked to the train set during data preparation?_

1. Are there duplicate examples in the test set and the train set (e.g. you forgot to remove duplicates or the train/test sets are not disjoint)?
2. Did you perform data scaling on the whole dataset prior to the split (e.g. standardization, normalization, etc.)?
3. Did you perform data transforms on the whole dataset prior to the split (e.g. power transform, log transform, etc.)?
4. Did you impute missing values on the whole dataset prior to the split (e.g. mean/median impute, knn impute, etc.)?
5. Did you engineer new features on the whole dataset prior to the split (e.g. one hot encode, integer encode, etc.)?
6. Did you perform exploratory data analysis (EDA) on the whole dataset prior to the split (e.g. statistical summaries, plotting, etc.)?
7. Did you engineer features that include information about the target variable?

### Test Harness Leakage

On the test harness, we may use a train/validation set split, k-fold cross-validation or similar resampling techniques to evaluate candidate models.

Performing data preparation techniques on the entire train set prior to splitting the data for candidate model evaluation may result in "**test harness leakage**".

Similarly, optimizing model hyperparameters on the same test harness as is used for model selection may result in overfitting and an optimistic bias.

1. Did you perform data preparation (e.g. scaling, transforms, imputation, feature engineering, etc.) on the entire train set before a train/val split or k-fold cross-validation?
	- Prepare data for the model on the train set/train folds only.
2. Did you tune model hyperparameters using the same train/validation split or k-fold cross-validation splits as you did for model selection?

### Model Performance Leakage

Any knowledge of what models work well or don't work well on the test set used for model selection on the training data/test harness may result in "**model performance leakage**".

* _Is there evidence that knowledge about what works well on the test set has leaked to the test harness?_

1. Did you evaluate and review/analyze the results the chosen model on the test set more than once?
2. Did you use knowledge about what models or model configurations work well on the test set to make changes to candidate models in evaluated on the train set/test harness?


## Quantify the Performance Gap

<img src="pics/logo-gap.svg" width="300" />

There is variance in the model (e.g. random seed, specific training data, etc.) and variance in its performance (e.g. test harness, specific test data, etc.).

Vary these elements and see if the distributions overlap or not.

If they do, any gap might be statistical noise (e.g. not a real effect). If they don't, it might be a real issue that requires more digging.

* _Is there evidence that the performance gap between the test harness and the test set is a warrants further investigation?_

1. Are you optimizing one (and only one) performance metric for the task?
2. What are the specific performance scores on the test harness and test set (e.g. min/max, mean/stdev, point estimate, etc.).
3. What is the generalization gap, calculated as the absolute difference in scores between the test harness and test set?
4. Does the chosen model outperform a dummy/naive model (e.g. predict mean/median/mode etc.) on the test harness and test set (e.g. does it have any skill)?
5. Are you able to evaluate a diverse suite of many (e.g. 10+) untuned standard machine learning models on the training and test sets and calculate their generalization gap (absolute difference) to establish a reference range?
	1. Is the chosen models' performance gap within the calculate a reference range for the performance gap for your test harness and test set (e.g. min/max, confidence interval, standard error, etc.)?
6. Are you able to evaluate the final chosen model many times (e.g. 30+, 100-1000) on bootstrap samples of the test set to establish a test set performance distribution?
	- Is the train set (test harness) performance within the test set distribution (e.g. min/max, confidence interval, etc.)
7. Are you able to develop multiple estimates of model performance on the training dataset (e.g. k-fold cross-validation, repeated train/test splits, bootstrap, checkpoint models, etc.)?
	- Is the test set performance within the train set (test harness) performance distribution (e.g. min/max, confidence interval, etc.)?
8. Are you able to fit and evaluate multiple different versions of the chosen model on the train set (e.g. with different random seeds)?
	- Is the test set performance within the train set (test harness) performance distribution (e.g. min/max, confidence interval, etc.)?

## Challenge the Performance Gap

<img src="pics/logo-challenge.svg" width="300" />

Often the train set score on the test harness is a mean (e.g. via k-fold cross-validation) and the test set score is a point estimate. Perhaps comparing these different types of quantities is the source of the problem.

We change the model evaluation scheme in one or both cases so that we can attempt an apples-to-apples comparison of model performance on the train and test sets.

* _Is there statistical evidence that the chosen model has the same apples-to-apples performance on the train and test sets?_

### Prerequisite
We require a single test harness that we can use to evaluate the single chosen model on the train set and the test set that results in multiple (>3) estimates of model performance on hold out data.

* Did you use a train/test split for model selection on the test harness?
	* Perform multiple (e.g. 5 or 10) train/test splits on the train set and the test set and gather the two samples of performance scores.
* Did you use k-fold cross-validation for model selection (e.g. kfold, stratified kfold, repeated kfold, etc.) in your test harness?
	* Apply the same cross-validation evaluation procedure to the train set and the test set and gather the two samples of performance scores.
* Is performing multiple fits of the chosen model too expensive and you used one train/val split on the train set?
	* Evaluate the same chosen model on many (e.g. 30, 100, 1000) bootstrap samples of the validation set and the test set and gather the two samples of performance scores.

### Checks
We now have a sample of performance scores on the train set and another for the test set.

Next, compare the distributions of these performance scores.

**Warning**: To avoid leakage, do not review performance scores directly, only the output and interpretation of the statistical tests.

1. Are performance scores highly correlated (e.g. Pearson's or Spearman's)?
2. Do the performance scores on train and test sets have the same central tendencies (e.g. t-Test or Mann-Whitney U Test)?
3. Do the performance scores on train and test sets have the same distributions (e.g. Kolmogorov-Smirnov Test or Anderson-Darling Test)?
4. Do the performance scores on train and test sets have the same variance (e.g. F-test, Levene's test)?
5. Is the effect size of the difference in performance between the train and test scores small (e.g. Cohen's d)?
6. Is divergence in the performance distributions between the train and test scores is small (e.g. KL-divergence or JS-divergence)?

## Data Distribution Checks

<img src="pics/logo-distribution.svg" width="300" />

We assume that the train set and the test set are a representative statistical sample from the domain. As such, they should have the same data distributions.

* _Is there statistical evidence that the train and test sets have the same data distributions?_

**Warning**: To avoid leakage, do not review summary statistics of the data directly, only the output and interpretation of the statistical tests.

These checks are most useful if train and test sets had separate sources or we suspect a sampling bias of some kind.

1. Do numerical input/target variables have the same central tendencies (e.g. t-Test or Mann-Whitney U Test)?
2. Do numerical input/target variables have the same distributions (e.g. Kolmogorov-Smirnov Test or Anderson-Darling Test)?
3. Do categorical input/target variables have the same distributions (e.g. Chi-Square Test)?
4. Do numerical input/target variables have the same distribution variance (e.g. F-test, Levene’s test)?
5. Are pair-wise correlations between numerical input variable distributions consistent (e.g. Pearson’s or Spearman’s, threshold difference, Fisher’s z-test, etc.)?
6. Are pair-wise correlations between numerical input and target variable distributions consistent (e.g. Pearson’s or Spearman’s, threshold difference, Fisher’s z-test, etc.)?
7. Do numerical variables have a low divergence in distributions (e.g. KL-divergence or JS-divergence)?
8. Does each variable in the train vs test set have a small effect size (e.g. Cohen’s d or Cliff’s delta)?
9. Do input/target variables in the train and test sets have a similar distributions of univariate outliers (e.g. 1.5xIQR, 3xstdev, etc.)?
10. Do the train and test sets have a similar pattern of missing values (e.g. statistical tests applied to number of missing values)?

Extensions:

* Compare the train set and the test set to the whole dataset (train+test).
* Compare the train set and the test set to a new data sample (if available).

## Performance Distribution Checks

<img src="pics/logo-performance.svg" width="300" />

We assume model performance on the train set and test set generalizes to new data. As such, model performance on the train and test sets have the same distribution.

* _Is there statistical evidence that general model performance on the train and test sets have the same distributions?_

### Prerequisite
We require a single test harness that we can use to evaluate a suite of standard models on the train set and the test set that results in multiple (>3) estimates of model performance on hold out data.

1. Choose a diverse suite (10+) of standard machine learning algorithms with good default hyperparameters that appropriate for your predictive modeling task (e.g. DummyModel, SVM, KNN, DecisionTree, Linear, NeuralNet, etc.)
2. Choose a standard test harness:
	* Did you use a train/test split for model selection on the test harness?
		* Perform multiple (e.g. 5 or 10) train/test splits on the train set and the test set and gather the two samples of performance scores.
	* Did you use k-fold cross-validation for model selection (e.g. kfold, stratified kfold, repeated kfold, etc.) in your test harness?
		* Apply the same cross-validation evaluation procedure to the train set and the test set and gather the two samples of performance scores.
	* Is performing multiple fits of the chosen model too expensive and you used one train/val split on the train set?
		* Evaluate the same chosen model on many (e.g. 30, 100, 1000) bootstrap samples of the validation set and the test set and gather the two samples of performance scores.
3. Evaluate the suite of algorithms on the train set and test set using the same test harness.
4. Gather the sample of hold out performance scores on the train set and test set for each algorithm.

### Checks
We now have a sample of performance scores on the train set and another for the test set for each algorithm.

Next, compare the distributions of these performance scores (all together or per algorithm or both).

**Warning**: To avoid leakage, do not review performance scores directly, only the output and interpretation of the statistical tests.

1. Are performance scores highly correlated (e.g. Pearson's or Spearman's)?
2. Do the performance scores on train and test sets have the same central tendencies (e.g. t-Test or Mann-Whitney U Test)?
3. Do the performance scores on train and test sets have the same distributions (e.g. Kolmogorov-Smirnov Test or Anderson-Darling Test)?
4. Do the performance scores on train and test sets have the same variance (e.g. F-test, Levene's test)?
5. Is the effect size of the difference in performance between the train and test scores small (e.g. Cohen's d)?
6. Do performance score points on a scatter plot fall on the expected diagonal line?

## Residual Error Checks

<img src="pics/logo-residuals.svg" width="300" />

We assume the train and test sets have a balanced sample of difficult-to-predict-examples.

* _Is there evidence that examples in the train set and test set are equally challenging to predict?_

1. Are there examples in the train set or the test set that are always predicted incorrectly?
2. Are there domain segments of the train set or the test set that the are harder to predict than other segments?
3. Are there some classes that are more challenging to predict than others on the train or test sets (classification only)?
4. Are there deciles of the target variable that are more challenging to predict than others on the train or test sets (regression only)?

## Residual Error Distribution Checks

We assume model prediction errors on the train set and test set are statistically representative of errors we will encounter on new data. As such, residual errors on the train set have the same distribution as the test set.

* _Is there evidence that model prediction errors on the train set and test set have the same distributions?_

### Prerequisite
We require a single test harness that we can use to evaluate a suite of standard models on the train set and the test set that results in a modest sample of prediction errors.

1. Choose a diverse suite (10+) of standard machine learning algorithms with good default hyperparameters that appropriate for your predictive modeling task (e.g. DummyModel, SVM, KNN, DecisionTree, Linear, NeuralNet, etc.)
2. Choose a standard test harness:
	* Did you use a train/test split for model selection on the test harness?
		* Perform multiple (e.g. 5 or 10) train/test splits on the train set and the test set and gather the two samples of performance scores.
	* Did you use k-fold cross-validation for model selection (e.g. kfold, stratified kfold, repeated kfold, etc.) in your test harness?
		* Apply the same cross-validation evaluation procedure to the train set and the test set and gather the two samples of performance scores.
	* Is performing multiple fits of the chosen model too expensive and you used one train/val split on the train set?
		* Evaluate the same chosen model on many (e.g. 30, 100, 1000) bootstrap samples of the validation set and the test set and gather the two samples of performance scores.
3. Evaluate the suite of algorithms on the train set and test set using the same test harness.
4. Gather the sample of prediction errors on the hold out data for the train set and test set for each algorithm.

### Checks
We now have a sample of prediction errors on the train set and another for the test set for each algorithm.

Next, compare the distributions of these errors (all together, or per algorithm, or both).

For regression tasks and classification tasks that require probabilities:

1. Are residual errors highly correlated (e.g. Pearson's or Spearman's)?
2. Do the residual errors on train and test sets have the same central tendencies (e.g. t-Test or Mann-Whitney U Test)?
3. Do the residual errors on train and test sets have the same distributions (e.g. Kolmogorov-Smirnov Test or Anderson-Darling Test)?
4. Do the residual errors on train and test sets have the same variance (e.g. F-test, Levene's test)?
5. Is the effect size of the difference in residual errors between the train and test scores small (e.g. Cohen's d)?

For classification tasks that require a nominal class label:

1. Do the confusion matrices of predictions from the train and test sets have the same distributions (e.g. Chi-Squared Test)?


## Overfitting Train Set Checks

<img src="pics/logo-overfitting.svg" width="300" />

We assume that model performance on train data generalizes to new data. As such, model performance on the train and hold out data in the test harness should have the same distribution.

Significantly better model performance on train data compared to performance on hold out data (e.g. validation data or test folds) may indicate overfitting.

* _Is there evidence that the chosen model has overfit the train data compared to hold out data in the test harness?_

### Performance Distribution Tests

* Are you able to evaluate the chosen model using resampling (e.g. k-fold cross-validation)?

Evaluate the model and gather performance scores on train and test folds.

1. Are performance scores on train and test folds highly correlated (e.g. Pearson's or Spearman's)?
2. Do performance scores on train and test folds have the same central tendencies (e.g. paired t-Test or Wilcoxon signed-rank test)?
3. Do performance scores on train and test folds have the same distributions (e.g. Kolmogorov-Smirnov Test or Anderson-Darling Test)?
4. Do performance score points on a scatter plot fall on the expected diagonal line?

### Loss Curve Tests

* Are you able to evaluate the model after each training update (aka training epoch, model update, boosting round, etc.)?

Split the training dataset into train/validation sets and evaluate the model loss on each set after each model update.

1. Does a line plot of model loss show signs of overfitting (improving/level performance on train, improving then worsening performance on validation i.e., the so-called "U" shape of a loss plot)?

### Validation Curve Tests

* Are you able to vary model capacity?

Vary model capacity and evaluate model performance with each configuration value and record train and hold out performance.

Note where the currently chosen configuration (if any) should be included.

1. Does a line plot of capacity vs performance scores show signs of overfitting for the chosen configuration relative to other configurations?

### Learning Curve Tests

Vary the size of the training data and evaluate model performance.

Note whether the model performance scales, e.g. performance improves with more data.

1. Does the line plot of train set size vs train and validation performance scores show signs of overfitting (or does performance scale proportionally to dataset size)?

### Visualize The Fit

In some cases we can visualize the domain and the model fit in the domain (possible in only the simplest domains).

Visualizing the fit relative to the data can often reveal the condition of the fit, e.g. good fit, underfit, overfit.

1. Visualize the decision surface for the model (classification only).
2. Visualize a scatter plot for data and line plot for the fit (regression only).


## Overfitting Test Harness Checks

A model has overfit the test harness if the performance of the model on the hold out set in the test harness (test folds in cross-validation or validation set) is better than performance when a new model is fit on the entire train set and evaluated on the test set.

* _Is there evidence that the chosen model has overfit the train set (whole test harness) compared to the test set?_

This is the premise for the entire checklist, go back to the top and work your way down.

Specifically focus on:

1. Train/Test Split Procedure
2. Data Distribution Checks
3. Performance Distribution Checks

## Overfitting Test Set Checks

A model has overfit the test set if a model is trained on the entire train set and evaluated on the test set and its performance is better than performance of a new model fit on the whole dataset (train + test sets) and evaluated on new data (e.g. in production).

* _Is there evidence that the chosen model has overfit the test set?_

This can happen if the test set is evaluated and knowledge is used in the test harness in some way, such as: choosing a model, model configuration, data preparation, etc.

This is called test set leakage and results in an optimistic estimate of model performance.

See the section on Data Leakage above.


## Model Robustness/Stability Checks

<img src="pics/logo-robustness.svg" width="300" />

Perhaps we can describe the robustness or stability of a model as how much its performance varies under noise.

A model that is sensitive to noise or small perturbations might be overly optimized to its training dataset and in turn fragile to a change in data, i.e., overfit.

We assume that small changes to the model and data result in proportionally small changes to model performance on the test harness.

* _Is there evidence that the model is robust to perturbations?_

1. Is the variance in model performance small when the chosen model is evaluated with differing random seeds for the learning algorithm?
2. Is the variance in model performance modest when Gaussian noise is added to input features?
3. Is the variance in model performance modest when Gaussian noise is added to target variable (regression only)?
4. Is the variance in model performance modest when class labels are flipped in target variable (classification only)?
5. Is the variance in model performance modest when Gaussian noise is added to key model hyperparameters?

### Sensitivity Analysis

1. Perform a sensitivity analysis of noise perturbation tests and confirm that model performance degrades gracefully (perhaps proportionality) to the amount of noise added to the dataset.


## So Now What? Interventions

<img src="pics/logo-fixes.svg" width="300" />

So, there may be an issue with your test set, train set, or your model. Now what do we do?

Below are some plans of attack:

### 1. Do Nothing

Take the estimate of model performance on the test set as the expected behavior of the model on new data.

If test set performance is to be presented to stakeholders, don't present a point estimate:

* Estimate performance using many (30+) bootstrap samples of the test set and report the median and 95% confidence interval.
* Fit multiple different versions of the final model (e.g. with different random seeds) and ensemble their predictions as the “chosen model” to reduce the variance of predictions made by the model.

### 2. Fix the Test Harness

In most cases, the fix involves making the test harness results less biased (avoid overfitting) and more stable/robust (more performance sampling).

1. Use k-fold cross-validation, if you're not already.
	- See [`KFold`](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.KFold.html) and [`StratifiedKFold`](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.StratifiedKFold.html).
2. Use 10 folds instead of 5 or 3, if you can afford the computational cost.
3. Use repeated 10-fold cross-validation, if you can afford the computational cost.
	- See [`RepeatedKFold`](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.RepeatedKFold.html) and [`RepeatedStratifiedKFold`](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)
4. Use a pipeline of data cleaning, transforms, feature engineering, etc. steps that are applied automatically.
	- The pipeline steps must be fit on train data (same data used to fit a candidate/final model) and applied to all data before it touches the model.
	- See [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html)
5. Use nested cross-validation or nested train/validation split to tune model hyperparameters for a chosen model.

**Warning**: After fixing the test harness, if you want to choose a different model/configuration and evaluate it on the test set, you should probably develop a new test set to avoid any test set leakage.

### 3. Fix Overfit Models

Even with best practices, high-capacity models can overfit.

This is a big topic, but to get started, consider:

1. **Simplification**: Reducing model complexity by using fewer layers/parameters, removing features, or choosing a simpler model architecture, etc. This limits the model's capacity to memorize training data and forces it to learn more generalizable patterns instead.
2. **Regularization**: Adding penalty terms to the loss function that discourage complex models by penalizing large weights. Common approaches include L1 (Lasso) and L2 (Ridge) regularization, which help prevent the model from relying too heavily on any single feature.
3. **Early Stopping**: Monitoring model performance on a validation set during training and stopping when performance begins to degrade. This prevents the model from continuing to fit noise in the training data after it has learned the true underlying patterns.
4. **Data Augmentation**: Artificially increasing the size and diversity of the training dataset by applying transformations or adding noise to existing samples. This exposes the model to more variation and helps it learn more robust features rather than overfitting to specific training examples.

Beyond these simple measures: dive into the literature for your domain and for your model and sniff out common/helpful regularization techniques you can test.

**Warning**: After fixing your model, if you want to evaluate it on the test set, you should probably develop a new test set to avoid any test set leakage.

### 4. Fix the Test Set (_danger!_)

Perhaps the test set is large enough and data/performance distributions match well enough, but there are specific examples that are causing problems.

* Perhaps some examples are outliers (cannot be predicted effectively) and should be removed from the train and test sets?
* Perhaps a specific subset of the test set can be used (careful of selection bias)?
* Perhaps the distribution of challenging examples is not balanced across train/test sets and domain-specific stratification of the dataset into train/test sets is required?
* Perhaps a weighting can be applied to samples and a sample-weighting aware learning algorithm can be used?
* Perhaps you can augment the test set with artificial/updated/contrived examples (danger)?

**Warning**: There is a slippery slope of removing all the hard-to-predict examples or too many examples. You're intentionally biasing the data and manipulating the results. Your decisions need to be objectively defensible in front of your harshest critic.

### 5. Get a New Test Set (_do this_)

Perhaps the test set is too small, or the data/performance distributions are significantly different, or some other key failing.

* Discard the test set, gather a new test set (using best practices) and develop a new unbiased estimate of the model's generalized performance.

Consider using some distribution and/or performance checks to have prior confidence in the test set before adopting it.

### 6. Reconstitute the Test Set

Sometimes acquiring a new test set  is not possible, in which case:

* Combine train and test set into the whole dataset again and develop a new split using best practices above
	* Ideally use a sensitivity analysis to ensure to choose an optimal split/avoid the same problem.

Although risky, this is common because of real world constraints on time, data, domain experts, etc.

**Warning**: There is a risk of an optimistic bias with this approach as you already know something about what does/doesn't work on examples in the original test set.




## Frequently Asked Questions

**Q. Isn't this just undergrad stats?**

Yes, applied systematically.

**Q. Do I need to perform all of these checks?**

No. Use the checks that are appropriate for your project. For example, checks that use k-fold cross-validation may not be appropriate for a deep learning model that takes days/weeks to fit. Use your judgement.

**Q. Are these checks overkill?**

Yes, probably.

**Q. What's the 80/20?**

Follow best practices in the split procedure. Using distribution checks will catch most problems on a pre-defined test set. Avoid test set leakage with test harness best practices like pipelines and nested cross-validation. Avoid a final point performance estimates with bootstrapping.

**Q. What's the most common cause of the problem?**

The most common cause is poor/fragile test harness (use best practices!). After that, the cause is statistical noise, which is ironed out with more robust performance estimates (e.g. bootstrap the test set and compare distributions/confidence intervals). After that, its some kind of test set leakage or a test set provided by a third party/stakeholder with differing distributions.

**Q. Don't I risk test set leakage and overfitting if I evaluate a model on the test set more than once?**

Yes. Strictly, the test set must be discarded after being used once. You can choose to bend this rule at your own risk. If you use these checks at the beginning of a project and carefully avoid and omit any summary stats/plots of data and model performance on the test set (e.g. only report the outcomes of statistical tests), then I suspect the risk is minimal.

**Q. Do all of these checks help?**

No. There is a lot of redundancy across the categories and the checks. This is a good thing as it gives many different perspectives on evidence gathering and more opportunities that we will catch the cause of the fault. Also, the specific aspects of some projects make some avenues of checking available and some not.

**Q. Don't we always expect a generalization gap?**

Perhaps. But how do you know the gap you see is reasonable for your specific predictive modeling task and test harness?

**Q. Doesn't R/scikit-learn/LLMs/XGBoost/AGI/etc. solve this for me?**

No.

**Q. Does this really matter?**

Yes, perhaps. It comes down to a judgement call, like most things.

**Q. Do you do this on your own projects?**

Yeah, a lot of it. I want to be able to (metaphorically) stand tall, put my hand on my heart, and testify that the test set results are as correct and indicative of real world performance as I know how to make them.

**Q. Why do we sometimes use a paired t-Test and sometimes not (and Mann-Whitney U Test vs Wilcoxon signed-rank test)**

When comparing distribution central tendencies on train vs test folds under k-fold cross-validation, we use a paired statistical hypothesis test like the paired Student's t-Test or the Wilcoxon signed-rank test because the train/test folds are naturally dependent upon each other. If we do not account for this dependence, we would miss-estimate the significance. When the two data distributions are independent, we can use an independent test such as the (unpaired) Student's t-Test or the Mann-Whitney U Test.

**Q. What about measure of model complexity to help diagnose overfitting (e.g. AIC/BIC/MDL/etc.)?**

I've not had much luck with these measures in practice. Also, classical ideas of overfitting were focused on "overparameterization". I'm not convinced this is entirely relevant with modern machine learning and deep learning models. We often overparameterize and still achieve SOTA performance. Put another way: Overfitting != Overparameterization. That being said, regularizing away from large magnitude coefficients/weights remains a good practice when overfit.

**Q. The test result is ambiguous, what do I do?**

This is the case more often than not. Don't panic, here are some ideas:
- Can you increase the sensitivity of the test (e.g. smaller p value, larger sample, etc.)?
- Can you perform a sensitivity analysis for the test and see if there is a point of transition?
- Can you try a different but related test to see if it provides less ambiguous results?
- Can you try a completely different test type to see if it gives a stronger signal?

**Q. All checks suggest my test set is no good, what do I do?**

Good, we learned something about your project! See the interventions.

**Q. Can you recommend further reading to learn more about this type of analysis?**

Yes, read these books:

* [Statistics in Plain English](https://amzn.to/3Vii1Mi) (2016). Get up to speed on parametric statistics.
* [Nonparametric Statistics for Non-Statisticians](https://amzn.to/3CWo82L) (2009). Get up to speed on non-parametric statistics.
* [Introduction to the New Statistics](https://amzn.to/49f9ako) (2024). Get up to speed on estimation statistics.
* [Empirical Methods for Artificial Intelligence](https://amzn.to/4gfMJ0X) (1995). This is an old favorite that will give some practical grounding.

## Glossary of Terms

Let's ensure we're talking about the same things:

* **Chosen Model**: A Chosen Model is the specific machine learning algorithm or architecture selected as the final candidate for solving a particular problem after comparing different options during the model selection phase.
* **Dataset**: A Dataset is a structured collection of data samples, where each sample contains features (input variables) and typically a target variable (for supervised learning), used to train and evaluate machine learning models.
* **Generalization Gap**: The Generalization Gap is the mathematical difference between a model's performance metrics on the training data versus previously unseen test data, indicating how well the model generalizes to new examples.
* **Leakage**: Leakage occurs when information from outside the training set inappropriately influences the model training process, leading to unrealistically optimistic performance estimates that won't hold in production.
* **Overfit**: Overfit happens when a model learns the training data too precisely, including its noise and peculiarities, resulting in poor generalization to new, unseen data.
* **Performance**: Performance is a quantitative measure of how well a model accomplishes its intended task, typically expressed through metrics like accuracy, precision, recall, or mean squared error.
* **Resampling**: Resampling is a family of statistical methods that involve repeatedly drawing different samples from a dataset to estimate model properties or population parameters, with common approaches including bootstrapping and cross-validation.
* **Test Harness**: A Test Harness is the complete experimental framework that includes data preprocessing, model training, validation procedures, and evaluation metrics used to assess and compare different models systematically.
* **Test Set**: The Test Set is a portion of the dataset that is completely held out during training and only used once at the very end to provide an unbiased evaluation of the final model's performance.
* **Train Set**: The Train Set is the portion of the dataset used to train the model by adjusting its parameters through the optimization of a loss function.
* **Validation Set**: The Validation Set is a portion of the dataset, separate from both training and test sets, used to tune hyperparameters and assess model performance during the development process before final testing.

## Disclaimer

Use these checks, interpret their results, and take resulting action all at your own risk.

I'm trying to help, but I'm not a statistician, merely a humble engineer.

Use your best judgement.

Chat with an LLM, with your data science friend, with your statistician friend and double check your findings, interpretations and resulting actions.

## About

Hi there, I'm Jason Brownlee ([twitter](https://x.com/jason2brownlee), [linkedin](https://www.linkedin.com/in/jasonbrownlee/)), the author of this checklist.

I've worked as an engineer on modeling and high-performance computing projects in industry, government, startups, and a long time ago, I had more academic aspirations and completed a Masters and PhD in stochastic optimization.

I've been helping data scientists for more than a decade via coaching and consulting and answering tens of thousands of emailed questions. I've also authored 1,000+ tutorials and 20+ books on machine learning and deep learning over at my former company [Machine Learning Mastery](https://machinelearningmastery.com), which I sold in 2021.

I've worked with hundreds (probably thousands) of budding and professional data scientists one-on-one over the years. The **most common problem** that we work on together is the **Generalization Gap**. This is where performance on new data in the test set or in production is worse than the expected performance from the test harness.

The checklist above is (an updated version of) a diagnostic tool I used to work through the problem with my clients.

I don't do so much consulting anymore, but I know from experience how valuable this checklist can be for those struggling with the generalization gap problem, so I've decided to share it here publicly.

I hope that you find this checklist useful too!

If it helps you make progress (or if you have ideas for more/better questions), please email me any time: Jason.Brownlee05@gmail.com

