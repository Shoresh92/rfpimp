<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

[Back to the Main Page](https://shoresh92.github.io)

# On Feature Importance in Random Forest
As a part of the Lead Scoring project at Spreedly, I used feature importance of Random Forest (RF) to determine the important features that lead trial sign-ups to conversion. However, due to the lack of robustness and the fact that the results did not match the intuition from the domain knowledge, I decided to dig deeper into the problem where I learned about the extensive research on the RF' feature importance. This note, that I try to keep short, is a summary of the problem and a couple of solutions discussed in literature.

Please note this post is a summary of the feature importance topic discussed in the three references below for future quick references.
* [Beware Default Random Forest Importances](http://parrt.cs.usfca.edu/doc/rf-importance/index.html)
* [Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
* [Bias in random forest variable importance measures: Illustrations, sources and a solution](https://link.springer.com/article/10.1186%2F1471-2105-8-25)

### Problem
#### Random Forest is popular
* Built from ensemble of decision trees, they are interpretable!
* They are high-performance algorithms.
* They require minimum data preparation compared to algorithms such as logistic regression.
* Compared to baggig models, they are more immune to overfitting.
* They can be used for evaluating feature importance.
* They do well with datasets with small number of observations, $n$, and large feature space, $p$.

#### Feature Importance in Random Forest
* Three ways to measure feature importance via RF:
  * A naive variable importance measure: to merely count the number of times each variable is selected by all individual trees in the ensemble.
  * Gini Index: the improvement in the "Gini gain" splitting criterion.
  * Permutation Significance: the permutation accuracy importance measure.

#### Feature Importance in Random Forest Is Biased
* RF variable importance measures are not reliable in situations where
  * predictor variables vary in their scale of measurement and/or
  * predictor variables vary in their number of categories

* If predictors are categorical, feature importance measures are biased in favor of variables taking more categories.  

* Severe effect of selection bias on permutation importance: When permuting the variables to compute their permutation importance measure, the variables that appear in more trees and are situated closer to the root node can affect the prediction accuracy of a larger set of observations, while variables that appear in fewer trees and are situated closer to the bottom nodes affect only small subsets of observations. Thus, the range of possible changes in prediction accuracy in the random forest, i.e. the deviation of the variable importance measure, is higher for variables that are preferred by the individual trees due to variable selection bias.

#### The Root of the Bias
* The bias is due to the use of
  * **bootstrap sampling**
  * **Gini split criterion**
* The bias can also be due to the
  * **collinearity**

**1. Gini Split Criterion**
* In traditional classification tree algorithms, for each variable a split criterion like the "Gini index" is computed for all possible cutpoints within the range of that variable. The variable selected for the next split is the one that produced the highest criterion value overall, i.e. in its best cutpoint. Obviously variables with more potential cutpoints are more likely to produce a good criterion value by chance.
* Therefore, if we compare the highest criterion value of a variable with two categories, say, that provides only one cutpoint from which the criterion was computed, with a variable with four categories, that provides seven cutpoints from which the best criterion value is used, the latter is often preferred.
* Because the number of cutpoints grows exponentially with the number of categories of unordered categorical predictors we find a preference for variables with more categories in CART-like classification trees.
* Since the Gini importance measure in randomForest is directly derived from the Gini index split criterion used in the underlying individual classification trees, it carries forward the same bias.

**2. Bootstrapping**
* The bootstrap sampling artificially induces an association between the variables. This effect is always present when statistical inference, such as an association test, is carried out on bootstrap samples: Bootstrap hypothesis testing fails whenever the distribution of any statistic in the bootstrap sample, rather than the distribution of the statistic under the null hypothesis, is used for statistical inference. This issue directly affects variable selection in random forest, because the deviation from the null hypothesis is more pronounced for variables that have more categories. However, if subsamples are drawn without replacement the effect disappears.

* The apparent association between the variables that is induced by bootstrap sampling, affects both feature and permutation importance measures: The selection frequency is again directly affected, and the permutation importance is affected because variables with many categories are selected more often and gain positions closer to the root node in the individual trees.

**3. Collinearity**
* Why collinearity is important? Because the importance is shared between the two collinear features: Permutation Importance (and mean-decrease-in-impurity importance) spreads importance across collinear variables. The amount of sharing appears to be a function of how much noise there is in between the two.

### Solution
#### 1. cForest
* cForest which is based on Conditional Inference Trees is, to the best of my knowledge, only available in $R$.

* The variable importance measure available in cForest, when used together with sampling without replacement, reliably reflects the true importance of potential predictor variables in a scenario where the potential predictor variables vary in their scale of measurement or number of categories.

* Why feature selection based on conditional trees is not biased? Here, the variable selection is conducted by minimizing the $p$ value of a conditional inference independence test that incorporates the number of categories of each variable in the degrees of freedom.

#### 2. Permutation Importance
* Permutation is like randomizing a column and random column should have no significance in predicting the output. Comparing the prediction accuracy of the actual data with the one from the dataset where a column values are shuffled can indicate the the significance of that particular column. If the difference is small, we conclude that the feature has no significance. If the significance is significant, we conclude otherwise.

* Pros:
  * It is broadly-applicable because it doesn't rely on internal model parameters, such as linear regression coefficients
  * It it is recommended for any model including regression models since interpreting regression coefficients requires great care and expertise; landmines include not normalizing input data, properly interpreting coefficients when using Lasso or Ridge regularization, and avoiding highly-correlated variables.
  * Permutation importance does not require the retraining of the underlying model in order to measure the effect of shuffling variables on overall model accuracy.

* Cons:
  * Permutation Importance over-estimates the importance of correlated predictor variables.

* Generalization: Model-neutral permutation importance
  * Use a generic scoring function instead of Out-Of-Bag approach that works for RF and a couple of other ensemble methods.

#### 3. Drop-column Importance
* If we ignore the computation cost of retraining the model, we can get the most accurate feature importance using a brute force drop-column importance mechanism.
* The idea is to get a baseline performance score as with permutation importance but then drop a column entirely, retrain the model, and recompute the performance score.
* This strategy answers the question of how important a feature is to overall model performance even more directly than the permutation importance strategy.
* Drop-column approach reminds me of **SelectKBest()** methods in scikit-learn package. The latter, however, is more general since the goal is to find the $k$ best features for prediction rather than evaluating one feature's significance.

#### 4. Repeated Permutation
* The method is based on repeated permutations of the outcome vector for estimating the distribution of measured importance for each variable in a non-informative setting. The $p$-value of the observed importance provides a corrected measure of feature importance.
* Non-informative predictors do not receive significant $p$-values so informative variables can successfully be recovered among non-informative variables.

* The major drawback of this method is the requirement of time-consuming permutations of the response vector and subsequent computation of feature importance. Simulations showed that around 10 permutations provides improvements over a biased base method. For stability of the results any number from 50 to 100 permutations is recommended.

### In Practice
* This is the order of steps I take
  * Drop-out importance is my first choice if the feature space is small and the approach is computationally affordable.
  * Calculating feature importances:
    * Use [rfpimp](http://parrt.cs.usfca.edu/doc/rf-importance/index.html) library: A library of functions that can be used for drop-column and permutation importances.
    * Make sure to identify collinear features. If using Permutation Importance, make sure to permute them altogether.
* Notable functions in `rfpimp`:
  * Permutation Feature Importance using cross-validation:
  `cv_importances(model, X_train, y_train, k=3)`
  * Permutation Feature Importance using a predefined metric:
  `permutation_importances(rf, X_train, y_train, metric)`
  * Drop-column Feature Importance:
  `def dropcol_importances(rf, X_train, y_train)`
  
* Final Note: currently dealing with a small feature space, I used drop-column importance and observed a significant imrovements in results. More on Lead Scoring project later.


[Back to the Main Page](https://shoresh92.github.io)
