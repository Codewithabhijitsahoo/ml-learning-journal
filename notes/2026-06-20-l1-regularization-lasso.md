# L1 Regularization (Lasso)

## Overview
L1 Regularization, also known as Lasso (Least Absolute Shrinkage and Selection Operator), is a powerful technique used in machine learning, particularly with linear models, to prevent overfitting and perform feature selection. Imagine you have a model that's trying to learn from data, and it's becoming too complex, memorizing the training examples rather than understanding the underlying patterns. This is overfitting. Lasso helps by simplifying the model. It does this by adding a penalty to the model's cost function that is proportional to the absolute value of the magnitude of the coefficients. This penalty encourages the model to make some of the less important feature coefficients exactly zero, effectively removing those features from the model. This not only makes the model simpler and less prone to overfitting but also makes it more interpretable by highlighting the most relevant features.

## What Problem It Solves
L1 Regularization (Lasso) primarily addresses the following critical problems in machine learning:

1.  **Overfitting**: When a model learns the training data too well, including noise and random fluctuations, it performs poorly on unseen data. This is overfitting. Lasso combats this by penalizing large coefficients, forcing the model to be simpler and generalize better.
2.  **High Dimensionality / Too Many Features**: In many real-world datasets, we have a vast number of features (e.g., thousands of genes in genomics, hundreds of financial indicators). Many of these features might be irrelevant or redundant. Including all of them can lead to a more complex model, increased computational cost, and higher risk of overfitting. Lasso automatically performs **feature selection** by driving the coefficients of irrelevant features to exactly zero, effectively removing them from the model.
3.  **Multicollinearity**: This occurs when independent variables in a regression model are highly correlated with each other. Multicollinearity can make it difficult to determine the individual impact of each feature on the target variable, leading to unstable and high-variance coefficient estimates. While Lasso doesn't explicitly solve multicollinearity in the same way Ridge does, its feature selection property can help by picking one feature from a group of highly correlated ones and shrinking others to zero, thus simplifying the model.
4.  **Model Interpretability**: When a model uses too many features, it becomes difficult to understand which features are truly driving the predictions. By shrinking less important feature coefficients to zero, Lasso results in a sparser model (fewer active features), making it easier to interpret and explain the relationships between features and the target variable.

## How It Works
Lasso works by modifying the standard cost function (e.g., Mean Squared Error for linear regression) by adding a penalty term. This penalty term is the sum of the absolute values of the model's coefficients, multiplied by a regularization parameter, $\lambda$ (lambda).

Here's a breakdown of the mechanism:

1.  **Standard Linear Regression Goal**: In standard linear regression, the goal is to find the coefficients ($\beta$) that minimize the sum of squared residuals (the difference between predicted and actual values). This is often called the Residual Sum of Squares (RSS).
    $$ \text{Minimize: } \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
    where $\hat{y}_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}$.

2.  **Adding the L1 Penalty**: Lasso adds an L1 penalty term to this objective function. The L1 penalty is the sum of the absolute values of the coefficients.
    $$ \text{L1 Penalty: } \lambda \sum_{j=1}^{p} |\beta_j| $$
    Here, $\lambda$ (lambda) is a non-negative tuning parameter that controls the strength of the regularization.
    *   If $\lambda = 0$, the L1 penalty term has no effect, and Lasso becomes equivalent to standard linear regression.
    *   As $\lambda$ increases, the penalty for large coefficients becomes stronger, forcing more coefficients towards zero.

3.  **The Lasso Objective Function**: The model now tries to minimize the sum of the squared residuals *plus* the L1 penalty term:
    $$ \text{Minimize: } \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| $$
    The model has to find a balance: fit the data well (minimize RSS) *and* keep the coefficients small (minimize the L1 penalty).

4.  **Coefficient Shrinkage and Sparsity**:
    *   The key characteristic of the L1 penalty is that it encourages coefficients to become *exactly zero*. This is because the absolute value function has a "sharp corner" at zero. When the optimization algorithm tries to minimize the total cost, it finds it more efficient to push some coefficients all the way to zero rather than just shrinking them slightly.
    *   This property leads to **sparsity**, meaning that many coefficients become zero, effectively performing automatic feature selection. Features with non-zero coefficients are considered important by the model, while those with zero coefficients are discarded.

5.  **Training Process**: During training, an optimization algorithm (like coordinate descent) iteratively adjusts the coefficients to minimize this combined objective function. It evaluates the trade-off between fitting the data and keeping the coefficients small/sparse. The optimal $\lambda$ is typically chosen through cross-validation, where different values of $\lambda$ are tested, and the one that yields the best performance on unseen data is selected.

## Mathematical Intuition
Let's dive into the mathematical underpinnings of L1 Regularization.

The standard objective function for Ordinary Least Squares (OLS) linear regression aims to minimize the Residual Sum of Squares (RSS):
$$ \text{RSS}(\beta) = \sum_{i=1}^{n} (y_i - (\beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}))^2 $$
where:
*   $y_i$ is the actual target value for the $i$-th observation.
*   $\beta_0$ is the intercept term.
*   $\beta_j$ are the coefficients for the $p$ features.
*   $x_{ij}$ is the value of the $j$-th feature for the $i$-th observation.

Lasso modifies this objective function by adding an L1 penalty term. The new objective function to minimize is:
$$ \text{Minimize: } \sum_{i=1}^{n} (y_i - (\beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}))^2 + \lambda \sum_{j=1}^{p} |\beta_j| $$
Let's break down the components:

*   **First term: $\sum_{i=1}^{n} (y_i - (\beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}))^2$**
    This is the familiar Residual Sum of Squares (RSS). It measures how well the model fits the training data. A smaller RSS means a better fit. The model wants to make this term as small as possible.

*   **Second term: $\lambda \sum_{j=1}^{p} |\beta_j|$**
    This is the L1 penalty term.
    *   $\sum_{j=1}^{p} |\beta_j|$ represents the sum of the absolute values of the coefficients (excluding the intercept $\beta_0$, which is typically not regularized). This is also known as the L1 norm of the coefficient vector.
    *   $\lambda$ (lambda) is the regularization parameter. It's a non-negative hyperparameter that you choose before training.
        *   When $\lambda = 0$, the penalty term vanishes, and Lasso becomes equivalent to OLS.
        *   As $\lambda$ increases, the penalty for large coefficients becomes stronger. The model is forced to shrink coefficients more aggressively.

**Why does the L1 penalty lead to sparsity (coefficients becoming exactly zero)?**

Consider the optimization problem geometrically. We are trying to minimize the RSS (a parabolic bowl shape) subject to a constraint on the size of the coefficients.
*   For standard OLS, there's no constraint on the size of coefficients.
*   For Lasso, the constraint is $\sum_{j=1}^{p} |\beta_j| \le t$ for some constant $t$. This constraint region forms a **diamond shape** (or an octahedron in higher dimensions) centered at the origin.

The solution for the coefficients $\beta$ occurs at the first point where the "level sets" (contours) of the RSS function touch the constraint region.
*   With the diamond-shaped L1 constraint, the corners of the diamond are often where the RSS contours first touch. These corners lie on the axes, meaning that some coefficients are exactly zero.
*   In contrast, L2 Regularization (Ridge) uses a constraint $\sum_{j=1}^{p} \beta_j^2 \le t$, which forms a **circle** (or sphere). The tangent point between a circular constraint and the elliptical RSS contours is rarely exactly on an axis, so Ridge shrinks coefficients towards zero but rarely makes them exactly zero.

This geometric property of the L1 norm's "sharp corners" at the axes is what drives coefficients to exactly zero, enabling feature selection.

## Advantages
*   **Feature Selection**: The most significant advantage of Lasso is its ability to perform automatic feature selection. It drives the coefficients of less important features to exactly zero, effectively removing them from the model. This results in a simpler, more interpretable model.
*   **Improved Interpretability**: By reducing the number of active features, Lasso makes the model easier to understand and explain. You can clearly see which features are considered most important for making predictions.
*   **Handles High Dimensionality**: When dealing with datasets that have many features (high dimensionality), Lasso can effectively identify and select the most relevant ones, preventing the model from being overwhelmed by noise or irrelevant information.
*   **Reduces Overfitting**: By penalizing large coefficients and promoting sparsity, Lasso helps prevent the model from memorizing the training data, leading to better generalization performance on unseen data.
*   **Can be more robust to irrelevant features**: Compared to OLS, Lasso is less sensitive to the inclusion of many irrelevant features because it can simply zero them out.

## Disadvantages
*   **Instability with Highly Correlated Features**: If there is a group of highly correlated features, Lasso tends to arbitrarily pick one feature from the group and shrink the others to zero. It doesn't perform "group selection" where all correlated features are kept or dropped together. This can lead to instability in feature selection if the data changes slightly.
*   **Doesn't Perform Group Selection**: As mentioned above, if you have a group of features that are all relevant and highly correlated, Lasso will typically select only one of them and zero out the rest. This might not be ideal if you believe all features in the group contribute meaningfully.
*   **Non-differentiable at Zero**: The absolute value function $|x|$ is not differentiable at $x=0$. This means that standard gradient descent algorithms cannot be directly applied to optimize the Lasso objective function. Specialized optimization algorithms like coordinate descent or subgradient methods are required.
*   **Can struggle with $N < P$ (more features than samples)**: While Lasso is good for high dimensionality, if the number of features ($P$) is much larger than the number of samples ($N$), Lasso can select at most $N$ features. If more than $N$ features are truly relevant, Lasso might miss some.
*   **Bias in Coefficient Estimates**: While Lasso reduces variance and performs feature selection, the coefficients it estimates for the selected features can be biased (shrunk towards zero more than their true values).

## Real World Applications
L1 Regularization (Lasso) is widely used across various domains due to its feature selection capabilities:

1.  **Genomics and Bioinformatics**: In gene expression analysis, researchers often deal with thousands of genes (features) but relatively few patient samples. Lasso can identify a small subset of genes that are most predictive of a disease outcome or drug response, making the biological interpretation much easier and guiding further research.
2.  **Finance and Economics**: Predicting stock prices, credit risk, or economic indicators often involves a large number of financial ratios, market data, and macroeconomic variables. Lasso can help select the most influential factors, building more robust and interpretable predictive models for risk assessment, portfolio management, or economic forecasting.
3.  **Marketing and Customer Analytics**: Companies collect vast amounts of data on customer demographics, purchase history, website interactions, and marketing campaign responses. Lasso can be used to identify key customer attributes or marketing touchpoints that drive customer churn, product adoption, or conversion rates, allowing for more targeted and effective marketing strategies.
4.  **Image and Signal Processing**: In tasks like image reconstruction or denoising, signals can be represented by a large number of coefficients. Lasso can promote sparse representations, meaning only a few coefficients are non-zero, which is useful for compression and identifying key components in the signal.
5.  **Drug Discovery**: Identifying molecular features or chemical properties that contribute to a drug's efficacy or toxicity. With a large number of molecular descriptors, Lasso can pinpoint the most relevant ones, accelerating the drug discovery process.

## Python Example

This example demonstrates how to use `Lasso` from `scikit-learn` to perform linear regression with L1 regularization, highlighting its feature selection capability by setting some coefficients to zero.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate a dummy dataset
# Let's create a dataset where only a few features are truly relevant.
np.random.seed(42)
n_samples = 100
n_features = 10

# Create some truly relevant features
X_relevant = np.random.randn(n_samples, 3) * 5 # Features 0, 1, 2
# Create some irrelevant features (noise)
X_irrelevant = np.random.randn(n_samples, n_features - 3) * 2 # Features 3 to 9

# Combine them
X = np.hstack((X_relevant, X_irrelevant))

# Define the true coefficients for the relevant features
# The true model is y = 2*x0 - 1.5*x1 + 3*x2 + noise
true_coefficients = np.array([2.0, -1.5, 3.0] + [0.0] * (n_features - 3))

# Generate the target variable y
y = X @ true_coefficients + np.random.randn(n_samples) * 5 # Add some noise

# Convert to DataFrame for better visualization of feature names
feature_names = [f'feature_{i}' for i in range(n_features)]
df_X = pd.DataFrame(X, columns=feature_names)
df_y = pd.Series(y, name='target')

print("--- Dataset Information ---")
print(f"Shape of X: {df_X.shape}")
print(f"Shape of y: {df_y.shape}")
print("\nFirst 5 rows of X:")
print(df_X.head())
print("\nFirst 5 rows of y:")
print(df_y.head())
print(f"\nTrue coefficients: {true_coefficients}")

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=42)

print("\n--- Model Training ---")

# 3. Fit a standard Linear Regression model (for comparison)
print("\nFitting Standard Linear Regression (OLS)...")
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

print("\nOLS Coefficients:")
for i, coef in enumerate(ols_model.coef_):
    print(f"{feature_names[i]}: {coef:.4f}")
print(f"OLS Intercept: {ols_model.intercept_:.4f}")

# 4. Fit a Lasso Regression model
# We need to choose an 'alpha' (lambda) value.
# A larger alpha means stronger regularization (more coefficients driven to zero).
# Let's try a moderate alpha first.
alpha_lasso = 0.5 # This is our lambda
print(f"\nFitting Lasso Regression with alpha (lambda) = {alpha_lasso}...")
lasso_model = Lasso(alpha=alpha_lasso, random_state=42)
lasso_model.fit(X_train, y_train)

print(f"\nLasso Coefficients (alpha={alpha_lasso}):")
for i, coef in enumerate(lasso_model.coef_):
    print(f"{feature_names[i]}: {coef:.4f}")
print(f"Lasso Intercept: {lasso_model.intercept_:.4f}")

# 5. Make predictions and evaluate
print("\n--- Model Evaluation ---")

# OLS predictions and evaluation
y_pred_ols = ols_model.predict(X_test)
mse_ols = mean_squared_error(y_test, y_pred_ols)
print(f"\nOLS Mean Squared Error on Test Set: {mse_ols:.4f}")

# Lasso predictions and evaluation
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Lasso Mean Squared Error on Test Set (alpha={alpha_lasso}): {mse_lasso:.4f}")

# 6. Visualize coefficients to highlight feature selection
plt.figure(figsize=(12, 6))
plt.bar(feature_names, ols_model.coef_, width=0.4, label='OLS Coefficients', align='center')
plt.bar(feature_names, lasso_model.coef_, width=0.4, label=f'Lasso Coefficients (alpha={alpha_lasso})', align='edge')
plt.axhline(0, color='grey', linewidth=0.8)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Comparison of OLS and Lasso Coefficients')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Let's try a larger alpha to see more aggressive feature selection
alpha_lasso_strong = 5.0
print(f"\nFitting Lasso Regression with a stronger alpha (lambda) = {alpha_lasso_strong}...")
lasso_strong_model = Lasso(alpha=alpha_lasso_strong, random_state=42)
lasso_strong_model.fit(X_train, y_train)

print(f"\nLasso Coefficients (alpha={alpha_lasso_strong}):")
for i, coef in enumerate(lasso_strong_model.coef_):
    print(f"{feature_names[i]}: {coef:.4f}")
print(f"Lasso Intercept: {lasso_strong_model.intercept_:.4f}")

y_pred_lasso_strong = lasso_strong_model.predict(X_test)
mse_lasso_strong = mean_squared_error(y_test, y_pred_lasso_strong)
print(f"Lasso Mean Squared Error on Test Set (alpha={alpha_lasso_strong}): {mse_lasso_strong:.4f}")

# Visualize coefficients again with stronger regularization
plt.figure(figsize=(12, 6))
plt.bar(feature_names, ols_model.coef_, width=0.3, label='OLS Coefficients', align='center')
plt.bar(feature_names, lasso_model.coef_, width=0.3, label=f'Lasso (alpha={alpha_lasso})', align='center', alpha=0.7, hatch='//')
plt.bar(feature_names, lasso_strong_model.coef_, width=0.3, label=f'Lasso (alpha={alpha_lasso_strong})', align='edge', alpha=0.7, hatch='--')
plt.axhline(0, color='grey', linewidth=0.8)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Comparison of OLS and Lasso Coefficients with Different Alphas')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Observation:
# Notice how OLS assigns small non-zero coefficients to irrelevant features (feature_3 to feature_9).
# Lasso with alpha=0.5 shrinks these coefficients closer to zero, and some might become exactly zero.
# Lasso with alpha=5.0 (stronger regularization) drives even more coefficients to exactly zero,
# clearly demonstrating its feature selection capability.
# The relevant features (feature_0, feature_1, feature_2) retain non-zero coefficients,
# though they might be slightly shrunk compared to their true values or OLS estimates.
```

**Explanation of the Output:**

1.  **Dataset Generation**: We create a synthetic dataset with 10 features. Crucially, only `feature_0`, `feature_1`, and `feature_2` are truly relevant to the target variable `y`. The remaining features (`feature_3` to `feature_9`) are just random noise, meaning their true coefficients are 0.
2.  **OLS Coefficients**: The standard `LinearRegression` (OLS) model will try to find coefficients for *all* features. You'll observe that even the irrelevant features (`feature_3` to `feature_9`) will have small, non-zero coefficients due to noise in the data.
3.  **Lasso Coefficients (alpha=0.5)**: With a moderate `alpha` (which is $\lambda$), Lasso starts to shrink coefficients. You'll see that the coefficients for the irrelevant features are much closer to zero, and some might even become exactly zero. The coefficients for the relevant features are also slightly shrunk but remain significant.
4.  **Lasso Coefficients (alpha=5.0)**: With a stronger `alpha`, the regularization effect is more pronounced. You'll likely see that most, if not all, of the irrelevant features (`feature_3` to `feature_9`) have their coefficients driven to *exactly zero*. This clearly demonstrates Lasso's feature selection capability. The coefficients for the relevant features will be shrunk even more compared to the `alpha=0.5` case.
5.  **Mean Squared Error (MSE)**: You might observe that Lasso, especially with an appropriate `alpha`, can achieve a similar or even better MSE on the test set compared to OLS, particularly if the dataset has many irrelevant features, because it builds a simpler, less overfit model.
6.  **Visualizations**: The bar plots visually confirm the shrinkage and zeroing out of coefficients by Lasso, especially for the irrelevant features, as `alpha` increases.

## Interview Questions

Here are 10 relevant technical interview questions about L1 Regularization (Lasso), along with detailed answers:

1.  **What is L1 Regularization (Lasso) and what is its primary purpose?**
    *   **Answer:** L1 Regularization, or Lasso (Least Absolute Shrinkage and Selection Operator), is a technique used in linear models to prevent overfitting and perform feature selection. Its primary purpose is to add a penalty term to the standard loss function (e.g., Mean Squared Error) that is proportional to the sum of the absolute values of the model's coefficients. This penalty encourages some coefficients to become exactly zero, effectively removing the corresponding features from the model.

2.  **How does Lasso differ from Ridge Regression (L2 Regularization) mathematically?**
    *   **Answer:** Both Lasso and Ridge add a penalty term to the loss function.
        *   **Lasso (L1):** Adds a penalty proportional to the sum of the *absolute values* of the coefficients: $\lambda \sum_{j=1}^{p} |\beta_j|$.
        *   **Ridge (L2):** Adds a penalty proportional to the sum of the *squares* of the coefficients: $\lambda \sum_{j=1}^{p} \beta_j^2$.
    *   The key mathematical difference lies in the L1 norm versus the L2 norm, which leads to their distinct behaviors.

3.  **What is the main advantage of Lasso over Ridge Regression?**
    *   **Answer:** The main advantage of Lasso is its ability to perform **automatic feature selection**. Because of the nature of the L1 penalty, it can drive the coefficients of less important features to *exactly zero*, effectively removing them from the model. Ridge, on the other hand, shrinks coefficients towards zero but rarely makes them exactly zero. This makes Lasso particularly useful for building sparse, interpretable models and handling high-dimensional data.

4.  **Explain the role of the regularization parameter $\lambda$ (alpha in scikit-learn) in Lasso.**
    *   **Answer:** The regularization parameter $\lambda$ (often called `alpha` in scikit-learn) controls the strength of the L1 penalty.
        *   **$\lambda = 0$**: No regularization is applied, and Lasso behaves like standard Ordinary Least Squares (OLS) regression.
        *   **Small $\lambda$**: Weak regularization. Coefficients are slightly shrunk, and only the least important features might have their coefficients driven to zero.
        *   **Large $\lambda$**: Strong regularization. More coefficients are aggressively shrunk towards zero, and many will become exactly zero, leading to a sparser model.
    *   The optimal $\lambda$ is typically chosen through techniques like cross-validation to balance model fit and complexity.

5.  **Why does Lasso lead to sparse models (coefficients becoming zero), while Ridge only shrinks them towards zero?**
    *   **Answer:** This can be understood geometrically. The optimization problem involves minimizing the RSS (a parabolic bowl) subject to a constraint on the coefficients.
        *   **Lasso (L1 constraint):** The constraint region is a **diamond shape** (or octahedron in higher dimensions). The corners of this diamond lie on the axes. When the elliptical contours of the RSS function touch these corners, the corresponding coefficients are forced to be exactly zero.
        *   **Ridge (L2 constraint):** The constraint region is a **circle** (or sphere). The tangent point between the elliptical RSS contours and the circular constraint is almost never exactly on an axis, so coefficients are shrunk but rarely become exactly zero. The "sharp corners" of the L1 norm are crucial for sparsity.

6.  **When would you prefer to use Lasso over Ridge Regression?**
    *   **Answer:** You would prefer Lasso when:
        *   You suspect that many features in your dataset are irrelevant or redundant, and you want to perform automatic feature selection.
        *   You need a simpler, more interpretable model with fewer active features.
        *   You are dealing with high-dimensional data where $P > N$ (more features than samples) and want to reduce dimensionality.

7.  **What are some limitations or disadvantages of using Lasso?**
    *   **Answer:**
        *   **Instability with Correlated Features:** If there's a group of highly correlated features, Lasso tends to arbitrarily pick one and zero out the others. This selection can be unstable with slight changes in data.
        *   **Doesn't Perform Group Selection:** It doesn't select or drop entire groups of correlated features together.
        *   **Non-differentiable at Zero:** The absolute value function is not differentiable at zero, requiring specialized optimization algorithms (like coordinate descent).
        *   **Bias in Coefficients:** While it reduces variance, the coefficients of the selected features can be biased (shrunk more than their true values).
        *   **Limited Feature Selection when $P > N$**: Lasso can select at most $N$ features (where $N$ is the number of samples). If more than $N$ features are truly relevant, Lasso might miss some.

8.  **Can Lasso be used for non-linear models?**
    *   **Answer:** Lasso regularization is primarily applied to linear models (e.g., linear regression, logistic regression). However, the *concept* of adding an L1 penalty can be extended to other models. For instance, if you transform your features using non-linear basis functions (e.g., polynomial features) and then apply Lasso, you are effectively building a non-linear model with L1 regularization. There are also L1 regularized versions of SVMs, neural networks, etc., but the direct "Lasso" term usually refers to its application in linear regression.

9.  **How do you typically choose the optimal $\lambda$ for a Lasso model?**
    *   **Answer:** The optimal $\lambda$ (alpha) is typically chosen using **cross-validation**.
        *   The dataset is split into multiple folds.
        *   For each fold, the model is trained on the remaining folds with a range of different $\lambda$ values.
        *   The performance (e.g., Mean Squared Error for regression, accuracy for classification) is evaluated on the held-out fold for each $\lambda$.
        *   The $\lambda$ value that yields the best average performance across all folds is selected as the optimal hyperparameter. `scikit-learn` provides `LassoCV` and `GridSearchCV` for this purpose.

10. **What happens to the bias and variance of a model when you increase $\lambda$ in Lasso?**
    *   **Answer:**
        *   **Bias:** As $\lambda$ increases, the regularization strength increases, forcing more coefficients towards zero. This makes the model simpler and less flexible, which generally leads to an **increase in bias**. The model might underfit if $\lambda$ is too large.
        *   **Variance:** By shrinking coefficients and performing feature selection, Lasso reduces the model's complexity and its sensitivity to small fluctuations in the training data. This generally leads to a **decrease in variance**.
    *   Lasso aims to find a sweet spot where the reduction in variance outweighs the increase in bias, leading to a lower overall prediction error on unseen data.

## Quiz

1.  What is the primary goal of L1 Regularization (Lasso)?
    A) To increase model complexity
    B) To prevent underfitting
    C) To perform feature selection and prevent overfitting
    D) To make all coefficients equal

2.  Which of the following mathematical terms represents the L1 penalty in Lasso Regression?
    A) $\lambda \sum_{j=1}^{p} \beta_j^2$
    B) $\lambda \sum_{j=1}^{p} |\beta_j|$
    C) $\frac{1}{2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
    D) $\sum_{j=1}^{p} \beta_j$

3.  What happens to the coefficients of irrelevant features when using Lasso with a sufficiently large $\lambda$?
    A) They increase significantly.
    B) They are shrunk towards zero but remain non-zero.
    C) They become exactly zero.
    D) They become equal to the coefficients of relevant features.

4.  Compared to Ridge Regression, a key advantage of Lasso is:
    A) It always yields lower Mean Squared Error.
    B) It handles multicollinearity better by keeping all correlated features.
    C) It performs automatic feature selection.
    D) It is computationally faster for very large datasets.

5.  If the regularization parameter $\lambda$ in Lasso is set to 0, what does the Lasso model become equivalent to?
    A) Ridge Regression
    B) Logistic Regression
    C) Standard Ordinary Least Squares (OLS) Regression
    D) A model with no features

---

### Answer Key

1.  **C) To perform feature selection and prevent overfitting**
    *   **Explanation:** Lasso's unique property of driving coefficients to zero enables automatic feature selection, which in turn simplifies the model and helps prevent it from overfitting to the training data.

2.  **B) $\lambda \sum_{j=1}^{p} |\beta_j|$**
    *   **Explanation:** This term represents the sum of the absolute values of the coefficients, multiplied by the regularization parameter $\lambda$. Option A is the L2 penalty used in Ridge Regression.

3.  **C) They become exactly zero.**
    *   **Explanation:** This is the defining characteristic of Lasso. The L1 penalty's geometric shape (diamond) encourages the optimization process to push less important coefficients all the way to zero, effectively removing those features.

4.  **C) It performs automatic feature selection.**
    *   **Explanation:** While both Lasso and Ridge help with overfitting, Lasso's distinct advantage is its ability to make coefficients exactly zero, thus performing feature selection. Ridge only shrinks coefficients towards zero but rarely makes them exactly zero.

5.  **C) Standard Ordinary Least Squares (OLS) Regression**
    *   **Explanation:** When $\lambda = 0$, the L1 penalty term becomes zero, and the Lasso objective function reduces to the standard OLS objective function (minimizing the sum of squared residuals).

## Further Reading

1.  **Scikit-learn Documentation on Lasso**: The official documentation is an excellent resource for understanding the implementation and practical usage of Lasso in Python.
    *   [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

2.  **"An Introduction to Statistical Learning" (ISLR) - Chapter 6**: This widely acclaimed textbook provides a clear and intuitive explanation of Lasso, Ridge, and other regularization techniques, including their mathematical foundations and practical applications. It's freely available online.
    *   [https://www.statlearning.com/](https://www.statlearning.com/) (Look for Chapter 6: Linear Model Selection and Regularization)

3.  **"The Elements of Statistical Learning" (ESL) - Chapter 3**: For a more advanced and in-depth mathematical treatment of Lasso and related topics, ESL is a classic reference.
    *   [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/) (Look for Chapter 3: Linear Methods for Regression)