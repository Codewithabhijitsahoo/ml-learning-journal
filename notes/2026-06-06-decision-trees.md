# Decision Trees

## Overview
Decision Trees are a powerful and intuitive supervised machine learning algorithm used for both classification and regression tasks. Imagine a flowchart where each internal node represents a "test" on an attribute (e.g., "Is the customer's age > 30?"), each branch represents the outcome of the test, and each leaf node represents a class label (for classification) or a numerical value (for regression). The goal of a Decision Tree is to learn simple decision rules inferred from the data features to predict the value of a target variable. They are called "trees" because the model structure resembles an inverted tree, with the root at the top and branches extending downwards.

Decision Trees are highly interpretable, meaning it's easy to understand how they arrive at a particular decision. This "white-box" nature makes them popular in fields where transparency and explainability are crucial, such as finance or medicine.

## What Problem It Solves
Decision Trees primarily solve the problem of making predictions based on a set of input features by learning a hierarchical set of rules.

1.  **Classification**: Given a set of features, predict which category or class an observation belongs to. For example, classifying an email as "spam" or "not spam" based on its content, sender, and subject.
2.  **Regression**: Given a set of features, predict a continuous numerical value. For example, predicting the price of a house based on its size, number of bedrooms, and location.
3.  **Interpretability**: Many machine learning models, like neural networks, are "black boxes" – they make predictions, but it's hard to understand *why*. Decision Trees provide a clear, step-by-step decision path, making them excellent for understanding the underlying logic and feature importance. This is crucial in applications where understanding the decision-making process is as important as the prediction itself.
4.  **Handling Mixed Data Types**: Decision Trees can naturally handle both numerical and categorical features without requiring extensive preprocessing like one-hot encoding for categorical variables or feature scaling for numerical ones.
5.  **Non-linear Relationships**: They can capture complex non-linear relationships between features and the target variable, unlike linear models that assume a linear relationship.

In essence, Decision Trees are needed when you want a model that can make accurate predictions, is easy to understand, and can handle various types of data without extensive preprocessing.

## How It Works
The core idea behind a Decision Tree is to recursively split the dataset into smaller, more homogeneous subsets based on the values of the input features. This process continues until a stopping criterion is met.

Here's a step-by-step breakdown:

1.  **Start with the Root Node**: The entire dataset is considered the root node. The algorithm then looks for the best feature to split this dataset.

2.  **Feature Selection and Splitting**:
    *   The algorithm evaluates all possible features and all possible split points (for numerical features) or categories (for categorical features) to divide the data.
    *   The goal is to find a split that best separates the data into distinct groups, where each group is as "pure" as possible with respect to the target variable. "Purity" means that most (ideally all) instances in a group belong to the same class (for classification) or have similar target values (for regression).
    *   Common metrics used to measure impurity and determine the "best" split include Gini Impurity and Entropy (for classification) or Mean Squared Error (MSE) and Mean Absolute Error (MAE) (for regression). The split that maximizes "Information Gain" (reduction in impurity) is chosen.

3.  **Recursive Splitting**:
    *   Once the best split is found, the root node is divided into two or more child nodes.
    *   The process then repeats for each child node: the algorithm again looks for the best feature to split *that* subset of data. This is a recursive process.

4.  **Stopping Criteria**: The tree building process stops when one or more of the following conditions are met:
    *   **Maximum Depth**: The tree reaches a predefined maximum depth.
    *   **Minimum Samples per Leaf**: A node contains fewer than a predefined minimum number of samples required to split.
    *   **Minimum Samples per Split**: A split would result in a child node having fewer than a predefined minimum number of samples.
    *   **Purity**: All samples in a node belong to the same class (for classification) or have a very small variance (for regression), meaning the node is "pure" enough.
    *   **No Information Gain**: No further split can improve the purity of the nodes significantly.

5.  **Leaf Nodes**: When the splitting stops, the final nodes are called leaf nodes. Each leaf node represents a prediction:
    *   For **classification**: The class label that is most frequent among the samples in that leaf node is assigned.
    *   For **regression**: The average (or median) of the target values of the samples in that leaf node is assigned.

6.  **Prediction**: To make a prediction for a new, unseen data point, you simply traverse the tree from the root node down to a leaf node by following the decision rules based on the data point's features. The value or class associated with that leaf node is the prediction.

This greedy, top-down approach ensures that at each step, the best possible split is chosen, aiming to create the most informative and pure partitions.

## Mathematical Intuition
The core mathematical concept behind building a Decision Tree, especially for classification, revolves around measuring the "impurity" or "disorder" of a set of data points and then finding splits that reduce this impurity. The goal is to make leaf nodes as "pure" as possible, meaning they contain data points predominantly belonging to a single class.

Let's focus on classification trees and the two most common impurity measures: Gini Impurity and Entropy.

### 1. Impurity Measures

#### a) Gini Impurity
Gini Impurity measures the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the distribution of labels in the subset. A Gini Impurity of 0 means the node is perfectly pure (all samples belong to the same class). A Gini Impurity of 0.5 (for a binary classification) means the node is perfectly impure (classes are equally distributed).

For a node $t$ and $C$ classes, the Gini Impurity is calculated as:
$$G(t) = 1 - \sum_{i=1}^{C} (p_i)^2$$
Where:
*   $p_i$ is the proportion of samples belonging to class $i$ in node $t$.
*   $\sum_{i=1}^{C} (p_i)^2$ is the sum of the squared probabilities of each class.

**Example**:
Suppose a node has 10 samples: 7 belong to Class A and 3 belong to Class B.
*   $p_A = \frac{7}{10} = 0.7$
*   $p_B = \frac{3}{10} = 0.3$
*   $G(t) = 1 - ((0.7)^2 + (0.3)^2)$
*   $G(t) = 1 - (0.49 + 0.09)$
*   $G(t) = 1 - 0.58 = 0.42$

If a node has 10 samples, all belonging to Class A:
*   $p_A = \frac{10}{10} = 1.0$
*   $p_B = \frac{0}{10} = 0.0$
*   $G(t) = 1 - ((1.0)^2 + (0.0)^2)$
*   $G(t) = 1 - (1 + 0) = 0$ (Perfectly pure)

#### b) Entropy
Entropy, originating from information theory, measures the average amount of "information" or "surprise" in a random variable. In the context of Decision Trees, it quantifies the disorder or uncertainty in a node. Higher entropy means more disorder, and lower entropy means more purity. An entropy of 0 means the node is perfectly pure.

For a node $t$ and $C$ classes, the Entropy is calculated as:
$$E(t) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$
Where:
*   $p_i$ is the proportion of samples belonging to class $i$ in node $t$.
*   We use $\log_2$ because information is often measured in bits.
*   If $p_i = 0$, then $p_i \log_2(p_i)$ is taken as 0 (as $0 \times \log_2(0)$ is undefined, but $\lim_{p \to 0} p \log_2(p) = 0$).

**Example**:
Suppose a node has 10 samples: 7 belong to Class A and 3 belong to Class B.
*   $p_A = 0.7$
*   $p_B = 0.3$
*   $E(t) = - (0.7 \log_2(0.7) + 0.3 \log_2(0.3))$
*   $E(t) = - (0.7 \times (-0.515) + 0.3 \times (-1.737))$
*   $E(t) = - (-0.3605 - 0.5211)$
*   $E(t) = - (-0.8816) \approx 0.8816$

If a node has 10 samples, all belonging to Class A:
*   $p_A = 1.0$
*   $p_B = 0.0$
*   $E(t) = - (1.0 \log_2(1.0) + 0.0 \log_2(0.0))$
*   $E(t) = - (1.0 \times 0 + 0) = 0$ (Perfectly pure)

### 2. Information Gain
Information Gain (IG) is the primary criterion used to decide which feature to split on at each step in building a Decision Tree. It measures the reduction in entropy (or Gini Impurity) after a dataset is split on an attribute. The attribute with the highest Information Gain is chosen as the splitting criterion.

For a split on attribute $A$, Information Gain is calculated as:
$$IG(S, A) = E(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} E(S_v)$$
Where:
*   $S$ is the current dataset (parent node).
*   $A$ is the attribute being considered for the split.
*   $E(S)$ is the Entropy (or Gini Impurity) of the parent node $S$.
*   $Values(A)$ is the set of all possible values for attribute $A$.
*   $S_v$ is the subset of $S$ for which attribute $A$ has value $v$.
*   $|S_v|$ is the number of samples in subset $S_v$.
*   $|S|$ is the total number of samples in the parent node $S$.
*   $\frac{|S_v|}{|S|}$ is the proportion of samples in $S_v$ relative to $S$, acting as a weighting factor.

**Explanation**:
The formula essentially says: "The Information Gain from splitting on attribute $A$ is the impurity of the parent node minus the weighted average of the impurities of the child nodes created by the split." The algorithm tries all possible splits for all features and picks the one that yields the maximum Information Gain, meaning it results in the greatest reduction in impurity.

For **regression trees**, instead of Gini Impurity or Entropy, the algorithm typically uses metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE) to measure the impurity (variance) of a node. The goal is to find splits that minimize the variance within the child nodes.
$$MSE(t) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2$$
Where:
*   $N$ is the number of samples in node $t$.
*   $y_i$ is the actual target value for sample $i$.
*   $\bar{y}$ is the mean of the target values in node $t$.
The splitting criterion would then be to choose the split that results in the largest reduction in MSE.

## Advantages
*   **Interpretability and Explainability**: Decision Trees are easy to understand and visualize. Their "if-then-else" logic mirrors human decision-making, making them excellent for explaining predictions to non-technical stakeholders.
*   **No Feature Scaling Required**: They are not sensitive to feature scaling (like standardization or normalization) because their splitting criteria are based on individual feature values, not distances.
*   **Handles Both Numerical and Categorical Data**: They can naturally handle both types of features without special preprocessing.
*   **Handles Non-linear Relationships**: Can capture complex non-linear relationships between features and the target variable.
*   **Robust to Outliers (to some extent)**: The splitting process focuses on relative ordering of features rather than absolute values, making them somewhat robust to outliers.
*   **Feature Selection**: The tree structure implicitly performs feature selection, as more important features appear closer to the root.
*   **Relatively Fast for Prediction**: Once trained, prediction is very fast as it only involves traversing the tree.

## Disadvantages
*   **Prone to Overfitting**: Decision Trees can easily overfit the training data, especially if they are allowed to grow too deep. This means they might perform very well on the training set but poorly on unseen data.
*   **Instability**: Small changes in the training data can lead to a completely different tree structure. This makes them quite unstable.
*   **Bias with Imbalanced Data**: If the dataset is imbalanced (one class dominates), the tree might be biased towards the majority class, leading to poor performance on the minority class.
*   **Greedy Algorithm**: The tree building algorithm is greedy; it makes the locally optimal split at each node without considering the global optimal tree. This means it might not find the globally optimal tree.
*   **High Variance**: Due to their instability and tendency to overfit, individual Decision Trees often have high variance. This is why they are often used as base estimators in ensemble methods like Random Forests or Gradient Boosting.
*   **Difficulty with Continuous Features**: For continuous features, the tree might create many splits, leading to a complex tree and potentially overfitting.
*   **Limited to Axis-Parallel Splits**: Decision Trees make splits that are parallel to the feature axes. This can be inefficient for datasets where the true decision boundary is diagonal or curved.

## Real World Applications
1.  **Medical Diagnosis**: Decision Trees are used to help diagnose diseases based on patient symptoms, medical history, and test results. For example, predicting the likelihood of a patient having a certain condition (e.g., heart disease, diabetes) based on their age, blood pressure, cholesterol levels, etc. Their interpretability is highly valued here, as doctors need to understand the reasoning behind a diagnosis.

2.  **Customer Churn Prediction**: Businesses use Decision Trees to identify customers who are likely to churn (cancel their service or subscription). By analyzing customer demographics, usage patterns, billing history, and interactions, companies can predict churn and proactively offer incentives or interventions to retain at-risk customers.

3.  **Credit Risk Assessment**: Financial institutions employ Decision Trees to assess the creditworthiness of loan applicants. They analyze factors like income, debt-to-income ratio, credit score, employment history, and past payment behavior to decide whether to approve a loan and at what interest rate. The transparent nature of Decision Trees helps comply with regulatory requirements for explaining credit decisions.

4.  **Fraud Detection**: In banking and e-commerce, Decision Trees can be used to detect fraudulent transactions. By examining transaction details such as amount, location, time, frequency, and merchant type, the model can flag suspicious activities that deviate from typical user behavior.

5.  **Marketing and Targeted Advertising**: Marketers use Decision Trees to segment customers and identify target audiences for specific products or campaigns. By analyzing purchasing history, browsing behavior, demographics, and responses to previous campaigns, they can predict which customers are most likely to respond positively to a new offer, optimizing marketing spend.

## Python Example
This example demonstrates how to train a Decision Tree Classifier using `scikit-learn` on a synthetic dataset, make predictions, and visualize the tree.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Generate a dummy dataset
# We'll create a synthetic dataset with 1000 samples, 4 features, and 2 classes.
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Convert to a pandas DataFrame for better readability and feature naming
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.Series(y, name='target')

print("Dataset Head:")
print(X_df.head())
print("\nTarget Distribution:")
print(y_df.value_counts())

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 3. Initialize and train the Decision Tree Classifier
# We'll limit the max_depth to prevent overfitting and make the tree more interpretable.
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

print("\nDecision Tree Classifier trained successfully!")

# 4. Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier,
          feature_names=feature_names,
          class_names=['Class 0', 'Class 1'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Classifier Visualization (Max Depth = 3)")
plt.show()

# 7. Demonstrate a single prediction
sample_data = np.array([[0.5, -1.2, 0.8, 0.1]]) # Example new data point
sample_prediction = dt_classifier.predict(sample_data)
sample_proba = dt_classifier.predict_proba(sample_data)

print(f"\nPrediction for sample {sample_data[0]}: Class {sample_prediction[0]}")
print(f"Prediction probabilities: Class 0: {sample_proba[0][0]:.4f}, Class 1: {sample_proba[0][1]:.4f}")

# You can also inspect feature importances
print("\nFeature Importances:")
for feature, importance in zip(feature_names, dt_classifier.feature_importances_):
    print(f"{feature}: {importance:.4f}")
```

**Explanation of the Code:**
1.  **Generate Data**: `make_classification` creates a synthetic dataset suitable for binary classification. We convert it to a Pandas DataFrame for easier handling and naming features.
2.  **Split Data**: The dataset is divided into training (70%) and testing (30%) sets using `train_test_split`. This ensures we evaluate the model on unseen data.
3.  **Train Model**: `DecisionTreeClassifier` is initialized. `max_depth=3` is set to limit the tree's complexity and prevent overfitting, making the visualization clearer. The `fit` method trains the model on the training data.
4.  **Predict**: The trained model makes predictions on the `X_test` data.
5.  **Evaluate**: `accuracy_score` and `classification_report` are used to assess the model's performance.
6.  **Visualize Tree**: `plot_tree` from `sklearn.tree` is used to visualize the trained decision tree. This is incredibly helpful for understanding the decision rules learned by the model. `filled=True` colors the nodes by class, and `rounded=True` gives them rounded corners.
7.  **Single Prediction & Feature Importance**: Shows how to predict for a new data point and how to get the importance of each feature in the decision-making process.

## Interview Questions

1.  **What is a Decision Tree and how does it work?**
    *   **Answer:** A Decision Tree is a supervised machine learning algorithm that uses a tree-like model of decisions and their possible consequences. It works by recursively splitting the dataset into smaller, more homogeneous subsets based on feature values. Each internal node represents a test on an attribute, each branch represents an outcome of the test, and each leaf node represents a class label (for classification) or a numerical value (for regression). The algorithm selects the best split at each step to maximize information gain or minimize impurity.

2.  **Explain the concepts of Gini Impurity and Entropy in Decision Trees.**
    *   **Answer:** Both Gini Impurity and Entropy are measures used to quantify the "impurity" or "disorder" of a node in a Decision Tree, primarily for classification tasks.
        *   **Gini Impurity:** Measures the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the distribution of labels in the subset. A Gini of 0 means perfect purity (all samples belong to the same class).
        *   **Entropy:** Originating from information theory, it measures the average amount of "information" or "surprise" in a node. Higher entropy means more disorder, and lower entropy means more purity. An entropy of 0 means perfect purity.
    *   The algorithm aims to find splits that reduce these impurity measures, leading to more homogeneous child nodes.

3.  **What is Information Gain and how is it used in Decision Trees?**
    *   **Answer:** Information Gain is the primary criterion used to decide which feature to split on at each step. It measures the reduction in entropy (or Gini Impurity) after a dataset is split on an attribute. The attribute that yields the highest Information Gain is chosen as the splitting criterion because it results in the greatest reduction in impurity, leading to the most informative split.

4.  **How does a Decision Tree handle continuous numerical features?**
    *   **Answer:** For continuous numerical features, a Decision Tree algorithm considers all possible split points between adjacent unique values of that feature. For each potential split point, it calculates the impurity (e.g., Gini or Entropy) of the resulting child nodes and determines the Information Gain. The split point that maximizes Information Gain is chosen. Essentially, it converts a continuous feature into a binary categorical feature at each split (e.g., "feature X <= threshold" vs. "feature X > threshold").

5.  **What are the main advantages of using Decision Trees?**
    *   **Answer:**
        *   **Interpretability:** Easy to understand and visualize.
        *   **No Feature Scaling:** Not sensitive to feature scaling.
        *   **Handles Mixed Data:** Can handle both numerical and categorical data.
        *   **Non-linear Relationships:** Can capture complex non-linear patterns.
        *   **Implicit Feature Selection:** More important features appear closer to the root.

6.  **What are the main disadvantages of using Decision Trees?**
    *   **Answer:**
        *   **Overfitting:** Prone to overfitting, especially deep trees.
        *   **Instability:** Small changes in data can lead to large changes in tree structure.
        *   **Bias with Imbalanced Data:** Can be biased towards dominant classes.
        *   **Greedy Approach:** Uses a greedy algorithm, which might not find the globally optimal tree.
        *   **High Variance:** Individual trees often have high variance.

7.  **How do you prevent overfitting in Decision Trees?**
    *   **Answer:** Overfitting can be prevented through techniques collectively known as "pruning" or by setting hyperparameter constraints:
        *   **Max Depth (`max_depth`):** Limit the maximum depth of the tree.
        *   **Min Samples per Leaf (`min_samples_leaf`):** Require a minimum number of samples in a leaf node to allow a split.
        *   **Min Samples per Split (`min_samples_split`):** Require a minimum number of samples in a node to consider splitting it.
        *   **Max Leaf Nodes (`max_leaf_nodes`):** Limit the total number of leaf nodes.
        *   **Cost-Complexity Pruning (Post-pruning):** Build a full tree and then prune back branches that provide little additional value.

8.  **Explain the difference between a Decision Tree for classification and one for regression.**
    *   **Answer:** The fundamental structure is the same, but the splitting criteria and leaf node predictions differ:
        *   **Classification Tree:** Uses impurity measures like Gini Impurity or Entropy to find splits that maximize Information Gain. Leaf nodes predict the majority class of the samples within them.
        *   **Regression Tree:** Uses measures like Mean Squared Error (MSE) or Mean Absolute Error (MAE) to find splits that minimize the variance of target values within child nodes. Leaf nodes predict the average (or median) target value of the samples within them.

9.  **Why are Decision Trees often used as base estimators in ensemble methods like Random Forests or Gradient Boosting?**
    *   **Answer:** Individual Decision Trees are prone to high variance and overfitting. Ensemble methods leverage the "wisdom of crowds" by combining multiple Decision Trees to reduce these issues.
        *   **Random Forests:** Build multiple Decision Trees independently using bootstrapped samples and random subsets of features, then average their predictions (or take a majority vote). This reduces variance.
        *   **Gradient Boosting (e.g., XGBoost, LightGBM):** Builds trees sequentially, where each new tree tries to correct the errors of the previous ones. This reduces bias.
    *   Decision Trees are good base estimators because they are flexible, can capture complex patterns, and are relatively fast to train.

10. **What is a "greedy" algorithm in the context of Decision Trees?**
    *   **Answer:** The Decision Tree algorithm is considered greedy because at each step (i.e., at each node), it makes the locally optimal decision by choosing the split that provides the maximum immediate Information Gain (or impurity reduction) without considering the potential impact of this split on future splits or the overall global optimality of the tree. It doesn't backtrack or explore alternative paths that might lead to a better overall tree structure. This greedy approach makes the algorithm computationally efficient but doesn't guarantee the globally optimal tree.

## Quiz

1.  Which of the following is NOT a common impurity measure used in Decision Trees for classification?
    A) Gini Impurity
    B) Entropy
    C) Mean Squared Error (MSE)
    D) Information Gain

2.  What is the primary goal of splitting a node in a Decision Tree?
    A) To increase the number of features.
    B) To maximize the depth of the tree.
    C) To create child nodes that are more homogeneous (pure).
    D) To ensure all leaf nodes have an equal number of samples.

3.  Which of the following is a major disadvantage of a single, unpruned Decision Tree?
    A) It requires extensive feature scaling.
    B) It is highly resistant to overfitting.
    C) It is prone to overfitting the training data.
    D) It cannot handle categorical features.

4.  If a leaf node in a classification Decision Tree contains 10 samples, with 8 belonging to Class A and 2 belonging to Class B, what would be the predicted class for a new sample reaching this leaf?
    A) Class B
    B) Class A
    C) Undetermined, as it's not perfectly pure.
    D) A probability distribution of 80% Class A and 20% Class B.

5.  What technique is commonly used to prevent overfitting in Decision Trees?
    A) One-hot encoding
    B) Feature scaling
    C) Pruning
    D) Increasing the number of features

---

### Answer Key

1.  **C) Mean Squared Error (MSE)**
    *   **Explanation:** Gini Impurity and Entropy are used for classification trees to measure impurity. Information Gain is the criterion used to select the best split based on these impurity measures. MSE is typically used in regression trees to measure the variance of target values.

2.  **C) To create child nodes that are more homogeneous (pure).**
    *   **Explanation:** The entire process of building a Decision Tree revolves around finding splits that reduce the impurity of the nodes, making them as pure as possible with respect to the target variable.

3.  **C) It is prone to overfitting the training data.**
    *   **Explanation:** Unpruned Decision Trees can grow very deep and learn noise in the training data, leading to excellent performance on the training set but poor generalization to unseen data.

4.  **B) Class A**
    *   **Explanation:** For classification trees, a leaf node predicts the majority class among the samples it contains. In this case, Class A has 8 samples, which is more than Class B's 2 samples.

5.  **C) Pruning**
    *   **Explanation:** Pruning techniques (like setting `max_depth`, `min_samples_leaf`, etc.) are specifically designed to limit the complexity of a Decision Tree and prevent it from overfitting the training data. One-hot encoding and feature scaling are preprocessing steps, and increasing features can sometimes worsen overfitting.

## Further Reading

1.  **Scikit-learn Documentation - Decision Trees**: The official documentation provides a great overview, mathematical details, and practical usage examples for `DecisionTreeClassifier` and `DecisionTreeRegressor`.
    *   [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)

2.  **"An Introduction to Statistical Learning" (ISLR) - Chapter 8: Tree-Based Methods**: This textbook provides a comprehensive and accessible explanation of Decision Trees, Bagging, Random Forests, and Boosting. It's a highly recommended resource for understanding the theoretical underpinnings.
    *   [http://www-bcf.usc.edu/~gareth/ISL/](http://www-bcf.usc.edu/~gareth/ISL/) (Look for Chapter 8 PDF)

3.  **StatQuest with Josh Starmer - Decision Trees (video series)**: Josh Starmer provides incredibly intuitive and visually engaging explanations of complex ML topics, including Decision Trees, Gini Impurity, and Entropy.
    *   [https://www.youtube.com/watch?v=7VeUPuFGJHk](https://www.youtube.com/watch?v=7VeUPuFGJHk) (Start with "Decision Trees Part 1")