# Non-negative Matrix Factorization (NMF)

## Overview
Non-negative Matrix Factorization (NMF) is a powerful unsupervised learning technique used for dimensionality reduction and feature extraction. At its core, NMF decomposes a non-negative matrix (a matrix where all elements are greater than or equal to zero) into two smaller non-negative matrices. Think of it like this: if you have a complex object, NMF tries to break it down into its fundamental, non-negative "parts." For example, if you have a collection of images of faces, NMF might decompose them into a set of "basis faces" (like eyes, noses, mouths), and then represent each original face as a unique combination of these basis parts. The "non-negative" constraint is crucial because it often leads to more interpretable and meaningful components, as real-world quantities (like pixel intensities, word counts, or sound frequencies) are typically non-negative.

## What Problem It Solves
NMF addresses several key problems and challenges in machine learning and data analysis:

1.  **Dimensionality Reduction:** Like Principal Component Analysis (PCA), NMF can reduce the number of features in a dataset while retaining important information. This is crucial for handling high-dimensional data, speeding up subsequent machine learning algorithms, and mitigating the "curse of dimensionality."
2.  **Feature Extraction and Part-Based Representation:** Unlike PCA, which often produces components that are linear combinations of original features and can have negative values (making them harder to interpret), NMF's non-negativity constraint forces it to learn "parts" or "building blocks" of the data. For instance, in text analysis, NMF can identify topics where each topic is a collection of words, and each document is a combination of these topics. In image analysis, it can identify facial features (eyes, nose, mouth) rather than abstract principal components.
3.  **Interpretability:** The non-negative nature of the decomposed matrices often leads to more interpretable results. If your input data represents counts (e.g., word counts in documents) or intensities (e.g., pixel values in images), then the "parts" discovered by NMF will also represent counts or intensities, which are easier for humans to understand than abstract negative values.
4.  **Source Separation:** In scenarios where a signal is a mixture of several underlying sources (e.g., a musical track with vocals and instruments), NMF can sometimes separate these sources, assuming they combine additively and have non-negative characteristics.
5.  **Handling Sparse Data:** NMF can be particularly effective with sparse data (matrices with many zero entries), which is common in areas like text mining and recommender systems.

In essence, NMF is needed when you want to discover underlying, interpretable, additive components or features from non-negative data, providing a more intuitive understanding of the data's structure than some other factorization methods.

## How It Works
The core idea behind NMF is to approximate a given non-negative data matrix $V$ as the product of two other non-negative matrices, $W$ and $H$.

Here's a step-by-step breakdown of how NMF typically works:

1.  **Input Data:** You start with a data matrix $V$ of dimensions $m \times n$, where $m$ is the number of features (e.g., words, pixels) and $n$ is the number of samples (e.g., documents, images). All elements in $V$ must be non-negative ($V_{ij} \ge 0$).

2.  **Choose the Number of Components (k):** You need to decide on the desired number of components, $k$. This $k$ represents the number of "parts" or "latent features" you want to extract. It's a hyperparameter that needs to be chosen carefully, often through experimentation or domain knowledge. The resulting matrices will have dimensions $W_{m \times k}$ and $H_{k \times n}$.

3.  **Initialization:** Randomly initialize the two factor matrices, $W$ and $H$, with non-negative values. This initialization can significantly impact the final result, as NMF is a non-convex optimization problem. Common initialization strategies include random initialization or NNDSVD (Non-negative Double Singular Value Decomposition), which often leads to faster convergence and better results.

4.  **Iterative Update Process:** The goal is to find $W$ and $H$ such that their product $WH$ is a good approximation of $V$, while ensuring all elements remain non-negative. This is typically achieved through an iterative optimization process that minimizes a cost function (often the Frobenius norm of the difference $V - WH$).
    *   **Update $W$:** While holding $H$ constant, update the values in $W$ to reduce the error between $V$ and $WH$.
    *   **Update $H$:** While holding $W$ constant, update the values in $H$ to further reduce the error.
    *   These updates are usually performed using multiplicative update rules, which are guaranteed to maintain non-negativity and converge to a local minimum.

5.  **Convergence:** The iterative process continues until a stopping criterion is met. This could be:
    *   The change in the objective function value between iterations falls below a certain threshold.
    *   The maximum number of iterations is reached.
    *   The change in $W$ and $H$ matrices becomes negligible.

6.  **Output:**
    *   **$W$ (Basis Matrix or Feature Matrix):** This $m \times k$ matrix contains the "basis vectors" or "components." Each column of $W$ represents a learned "part" or "feature" (e.g., a topic in text, a facial feature in images).
    *   **$H$ (Coefficient Matrix or Encoding Matrix):** This $k \times n$ matrix contains the "weights" or "coefficients." Each column of $H$ represents how much of each "part" (from $W$) is present in the corresponding original sample from $V$. In other words, it encodes the original data in terms of the learned components.

By decomposing $V$ into $W$ and $H$, NMF effectively extracts latent features ($W$) and represents the original data in terms of these features ($H$), all while maintaining the interpretability offered by non-negativity.

## Mathematical Intuition

Let's dive into the mathematical underpinnings of NMF.

Given a non-negative data matrix $V \in \mathbb{R}^{m \times n}$, where all $V_{ij} \ge 0$, the goal of NMF is to find two non-negative matrices, $W \in \mathbb{R}^{m \times k}$ and $H \in \mathbb{R}^{k \times n}$, such that their product $WH$ approximates $V$.
$$V \approx WH$$
Here, $k$ is the chosen number of latent components, where typically $k \ll \min(m, n)$.

The non-negativity constraints are crucial:
$$W_{ij} \ge 0 \quad \text{for all } i, j$$
$$H_{ij} \ge 0 \quad \text{for all } i, j$$

The approximation quality is measured by an objective function, often the Frobenius norm of the difference between $V$ and $WH$:
$$J(W, H) = ||V - WH||_F^2 = \sum_{i=1}^{m} \sum_{j=1}^{n} (V_{ij} - (WH)_{ij})^2$$
The problem then becomes an optimization task:
$$\min_{W, H} ||V - WH||_F^2 \quad \text{subject to } W_{ij} \ge 0, H_{ij} \ge 0$$

Let's break down the matrix multiplication $WH$:
The element $(WH)_{ij}$ is the dot product of the $i$-th row of $W$ and the $j$-th column of $H$:
$$(WH)_{ij} = \sum_{l=1}^{k} W_{il} H_{lj}$$
This means that each column of $V$ (which represents a data sample) is approximated as a linear combination of the columns of $W$, with the coefficients given by the corresponding column of $H$.
$$V_j \approx \sum_{l=1}^{k} W_l H_{lj}$$
where $V_j$ is the $j$-th column of $V$, and $W_l$ is the $l$-th column of $W$.
Since all $W_{il}$ and $H_{lj}$ are non-negative, this implies that the original data samples are represented as additive combinations of the "parts" defined by $W$. This additive, non-negative combination is what gives NMF its "part-based" interpretation.

To minimize the objective function, iterative update rules are commonly used. One popular set of multiplicative update rules, proposed by Lee and Seung, ensures that $W$ and $H$ remain non-negative and the objective function is non-increasing.

The update rules for $W$ and $H$ are derived using gradient descent-like methods, but with a multiplicative factor to enforce non-negativity.
The gradient of the objective function with respect to $W$ and $H$ can be calculated. For example, for $W$:
$$\frac{\partial J}{\partial W} = -2(V - WH)H^T$$
And for $H$:
$$\frac{\partial J}{\partial H} = -2W^T(V - WH)$$

The multiplicative update rules are then:
$$H_{lj} \leftarrow H_{lj} \frac{(W^T V)_{lj}}{(W^T W H)_{lj}}$$
$$W_{il} \leftarrow W_{il} \frac{(V H^T)_{il}}{(W H H^T)_{il}}$$

Let's understand these rules intuitively:
*   The numerator $(W^T V)_{lj}$ represents how well the $l$-th basis vector (column of $W$) correlates with the $j$-th data sample (column of $V$).
*   The denominator $(W^T W H)_{lj}$ represents how well the $l$-th basis vector correlates with the current approximation of the $j$-th data sample ($WH_j$).
*   If the numerator is larger than the denominator, it means the $l$-th basis vector is "under-represented" in the current approximation of the $j$-th sample, so $H_{lj}$ is increased.
*   If the numerator is smaller, it means it's "over-represented," so $H_{lj}$ is decreased.
*   The same logic applies to the update rule for $W$.

These multiplicative updates ensure that $W$ and $H$ remain non-negative throughout the optimization process, as they are always multiplied by a non-negative ratio. The process iterates, refining $W$ and $H$ until convergence, yielding a locally optimal factorization.

## Advantages
*   **Interpretability:** The non-negativity constraint often leads to more interpretable components (e.g., topics in text, parts of faces in images) because real-world quantities are often non-negative.
*   **Part-Based Representation:** NMF excels at finding "parts" or "building blocks" of the data, where each data point is represented as an additive combination of these parts. This is a key differentiator from methods like PCA, which find holistic components.
*   **Dimensionality Reduction:** Effectively reduces the dimensionality of data, making it easier to analyze and process, and potentially improving the performance of subsequent machine learning models.
*   **Handles Sparse Data Well:** NMF can be particularly effective for sparse datasets (matrices with many zero entries), which are common in text analysis and recommender systems.
*   **Flexibility in Objective Functions:** While the Frobenius norm is common, NMF can be formulated with other divergence measures (e.g., Kullback-Leibler divergence) which might be more suitable for certain types of data (e.g., count data).

## Disadvantages
*   **Non-Uniqueness of Solutions:** NMF is a non-convex optimization problem, meaning there isn't a single unique global minimum. Different initializations can lead to different local minima and thus different factorizations.
*   **Sensitivity to Initialization:** The quality of the factorization can depend heavily on the initial random values chosen for $W$ and $H$. Running NMF multiple times with different initializations and choosing the best result is a common practice.
*   **Choosing the Number of Components (k):** Determining the optimal number of components $k$ is often challenging and requires domain knowledge, cross-validation, or heuristic methods (e.g., elbow method, reconstruction error analysis).
*   **Computational Cost:** For very large matrices, the iterative optimization process can be computationally intensive and time-consuming.
*   **Scalability:** While there are efforts to make NMF more scalable, it can still be challenging for extremely large datasets compared to some other dimensionality reduction techniques.
*   **Requires Non-Negative Data:** NMF is strictly applicable only to data where all values are non-negative. If your data contains negative values, you must preprocess it (e.g., by shifting the data) before applying NMF, which might alter its interpretability.

## Real World Applications
1.  **Document Clustering and Topic Modeling:** NMF is widely used to discover latent topics within a collection of documents. Each document can be represented as a mixture of these topics, and each topic is characterized by a set of highly weighted words. This helps in organizing, summarizing, and searching large text corpora. For example, identifying trending topics in news articles or categorizing customer reviews.
2.  **Image Processing and Computer Vision:** NMF can decompose images into their constituent parts. For instance, in facial recognition, it can learn basis components (like eyes, nose, mouth) from a dataset of faces, and then represent new faces as combinations of these parts. It's also used for object recognition, image compression, and denoising.
3.  **Recommender Systems:** In collaborative filtering, NMF can be used to factorize user-item interaction matrices (e.g., ratings, purchase history). The resulting matrices can represent latent features of users and items, which can then be used to predict user preferences for unrated items, leading to personalized recommendations (e.g., suggesting movies, products, or music).
4.  **Audio Signal Processing:** NMF can be applied to audio spectrograms (which are non-negative representations of sound frequencies over time) for tasks like source separation (e.g., separating vocals from instrumental music), music transcription, and identifying distinct sound events in an audio stream.
5.  **Bioinformatics and Genomics:** NMF is used to analyze gene expression data, where it can identify underlying biological pathways or cell types from gene expression profiles. It helps in discovering patterns in large genomic datasets, classifying diseases, and understanding cellular processes.

## Python Example

This example demonstrates NMF using `scikit-learn` to decompose a simple synthetic dataset. We'll create a matrix representing "documents" and "words" and then use NMF to find "topics."

```python
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate a dummy non-negative dataset
# Let's imagine a simple document-term matrix.
# Rows are documents, columns are words.
# Values represent word counts.
# We'll create a matrix that inherently has some "topics".

# Example: 3 documents, 5 words
# Document 1: "apple banana apple" -> [2, 1, 0, 0, 0] (fruit topic)
# Document 2: "car truck car" -> [0, 0, 2, 1, 0] (vehicle topic)
# Document 3: "apple car banana" -> [1, 1, 1, 0, 0] (mixed topic)

# Let's make a slightly larger, more structured example.
# We'll simulate 10 documents and 8 words.
# Two underlying "topics": 'fruit' and 'vehicle'.

# Words: apple, banana, orange, car, truck, bus, red, green
word_names = ["apple", "banana", "orange", "car", "truck", "bus", "red", "green"]

# Create a data matrix V (documents x words)
# Document 1-5 are mostly 'fruit' related
# Document 6-10 are mostly 'vehicle' related
# Some noise/overlap to make it realistic

data = np.array([
    [5, 2, 1, 0, 0, 0, 1, 0],  # Doc 1: mostly fruit
    [3, 4, 0, 0, 0, 0, 0, 1],  # Doc 2: mostly fruit
    [1, 5, 2, 0, 0, 0, 1, 1],  # Doc 3: mostly fruit
    [4, 1, 3, 0, 0, 0, 0, 0],  # Doc 4: mostly fruit
    [2, 3, 1, 1, 0, 0, 1, 0],  # Doc 5: fruit with a little vehicle
    [0, 0, 0, 5, 2, 1, 0, 1],  # Doc 6: mostly vehicle
    [0, 0, 0, 3, 4, 0, 1, 0],  # Doc 7: mostly vehicle
    [1, 0, 0, 1, 5, 2, 0, 1],  # Doc 8: vehicle with a little fruit
    [0, 0, 0, 4, 1, 3, 1, 0],  # Doc 9: mostly vehicle
    [0, 1, 0, 2, 3, 1, 0, 1]   # Doc 10: mixed, more vehicle
])

print("Original Data Matrix (V):")
print(data)
print(f"Shape of V: {data.shape}\n")

# 2. Initialize and fit the NMF model
# We expect 2 underlying topics (k=2)
n_components = 2
model = NMF(n_components=n_components, init='random', random_state=42, max_iter=1000)

# Fit the model to the data and transform it
# W will be (n_documents x n_components)
# H will be (n_components x n_words)
W = model.fit_transform(data)
H = model.components_

print(f"Factorized Matrix W (Document-Topic Matrix, shape: {W.shape}):")
print(W)
print("\nInterpretation of W:")
print("Each row represents a document, and each column represents the weight of a topic in that document.")
print("For example, W[0,0] is the weight of Topic 0 in Document 0.")

print(f"\nFactorized Matrix H (Topic-Word Matrix, shape: {H.shape}):")
print(H)
print("\nInterpretation of H:")
print("Each row represents a topic, and each column represents the weight of a word in that topic.")
print("For example, H[0,0] is the weight of 'apple' in Topic 0.")

# 3. Reconstruct the original matrix (approximation)
V_reconstructed = np.dot(W, H)
print(f"\nReconstructed Data Matrix (V_reconstructed, shape: {V_reconstructed.shape}):")
print(V_reconstructed)

# 4. Evaluate the reconstruction error (Frobenius norm)
reconstruction_error = np.linalg.norm(data - V_reconstructed, 'fro')
print(f"\nReconstruction Error (Frobenius Norm): {reconstruction_error:.4f}")

# 5. Interpret the results
print("\n--- Interpreting the NMF Results ---")

# Print top words for each topic
print("\nTop words for each topic:")
for topic_idx, topic in enumerate(H):
    # Get indices of words sorted by their weights in this topic
    top_word_indices = topic.argsort()[-5:][::-1] # Top 5 words
    top_words = [word_names[i] for i in top_word_indices]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")

# Visualize the topic-word matrix (H)
plt.figure(figsize=(10, 4))
sns.heatmap(H, cmap="viridis", annot=True, fmt=".2f",
            xticklabels=word_names, yticklabels=[f"Topic {i}" for i in range(n_components)])
plt.title("Topic-Word Matrix (H)")
plt.xlabel("Words")
plt.ylabel("Topics")
plt.show()

# Visualize the document-topic matrix (W)
plt.figure(figsize=(10, 6))
sns.heatmap(W, cmap="plasma", annot=True, fmt=".2f",
            xticklabels=[f"Topic {i}" for i in range(n_components)],
            yticklabels=[f"Doc {i}" for i in range(data.shape[0])])
plt.title("Document-Topic Matrix (W)")
plt.xlabel("Topics")
plt.ylabel("Documents")
plt.show()

# Example: Which topic is Document 0 mostly about?
doc_0_topic_weights = W[0, :]
dominant_topic_idx = np.argmax(doc_0_topic_weights)
print(f"\nDocument 0 is mostly about Topic {dominant_topic_idx} (weights: {doc_0_topic_weights})")

# Example: Which documents are mostly about Topic 1?
docs_for_topic_1 = np.where(W[:, 1] > W[:, 0])[0] # Documents where Topic 1 weight > Topic 0 weight
print(f"Documents mostly about Topic 1: {docs_for_topic_1}")
```

**Explanation of the Python Example:**

1.  **Dummy Dataset Creation:** We create a `data` matrix representing word counts in documents. The `word_names` list helps us interpret the columns. We intentionally structure the data so that the first few documents are "fruit-related" and the latter ones are "vehicle-related," with some overlap.
2.  **NMF Model Initialization:** We instantiate `NMF` from `sklearn.decomposition`.
    *   `n_components=2`: We tell NMF to find 2 underlying components (topics), matching our expectation.
    *   `init='random'`: Specifies the initialization method for W and H. 'random' is simple, 'nndsvd' is often better.
    *   `random_state=42`: Ensures reproducibility of the random initialization.
    *   `max_iter=1000`: Sets the maximum number of iterations for the optimization.
3.  **Fitting and Transformation:** `model.fit_transform(data)` performs the NMF decomposition.
    *   It returns `W`, the document-topic matrix. Each row is a document, and its values indicate the contribution of each topic to that document.
    *   `model.components_` gives `H`, the topic-word matrix. Each row is a topic, and its values indicate the importance of each word to that topic.
4.  **Reconstruction:** We multiply `W` and `H` back together (`np.dot(W, H)`) to get `V_reconstructed`, which should be an approximation of the original `data` matrix.
5.  **Reconstruction Error:** We calculate the Frobenius norm of the difference between the original and reconstructed matrices to quantify how good the approximation is. A smaller error means a better approximation.
6.  **Interpretation and Visualization:**
    *   We print the top words for each discovered topic by looking at the highest values in each row of `H`. This helps us label and understand what each topic represents (e.g., "Topic 0: apple, banana, orange" -> Fruit Topic).
    *   Heatmaps are used to visually inspect `H` and `W`. The `H` heatmap shows which words are strong indicators for which topics. The `W` heatmap shows which documents are strongly associated with which topics.
    *   We demonstrate how to use `W` to determine the dominant topic for a specific document and how to find documents associated with a particular topic.

This example clearly shows how NMF decomposes the original data into interpretable parts (topics) and how each original data point (document) is composed of these parts.

## Interview Questions

1.  **What is Non-negative Matrix Factorization (NMF) and what is its primary goal?**
    *   **Answer:** NMF is an unsupervised learning technique that decomposes a non-negative data matrix $V$ into two non-negative matrices, $W$ (basis matrix) and $H$ (coefficient matrix), such that $V \approx WH$. Its primary goal is dimensionality reduction and feature extraction, specifically to find a "part-based" representation of the data, where the parts are interpretable due to the non-negativity constraint.

2.  **Why is the "non-negative" constraint so important in NMF? What benefits does it offer?**
    *   **Answer:** The non-negative constraint ($W_{ij} \ge 0, H_{ij} \ge 0$) is crucial because it forces the learned components (in $W$) and their combinations (in $H$) to be additive. This often leads to more interpretable results, as many real-world quantities (like pixel intensities, word counts, sound frequencies) are inherently non-negative. It helps in discovering "parts" of objects (e.g., facial features, topics in text) rather than holistic or abstract components that might include negative values.

3.  **How does NMF differ from Principal Component Analysis (PCA)?**
    *   **Answer:**
        *   **Non-negativity:** NMF requires and maintains non-negative data and components, while PCA does not.
        *   **Interpretability:** NMF components are often more interpretable as "parts" due to non-negativity and additive combinations. PCA components are orthogonal and can have negative values, making them less intuitive for part-based understanding.
        *   **Basis Vectors:** PCA finds orthogonal basis vectors that capture maximum variance. NMF finds basis vectors that, when additively combined, reconstruct the original data.
        *   **Data Type:** NMF is suitable for non-negative data (e.g., counts, intensities). PCA can handle any real-valued data.

4.  **Explain the objective function typically minimized in NMF.**
    *   **Answer:** The most common objective function for NMF is the Frobenius norm of the difference between the original matrix $V$ and its approximation $WH$. It's defined as $||V - WH||_F^2 = \sum_{i,j} (V_{ij} - (WH)_{ij})^2$. The goal is to find $W$ and $H$ that minimize this squared error, subject to the non-negativity constraints. Other divergence measures like Kullback-Leibler divergence can also be used, especially for count data.

5.  **What are the matrices $W$ and $H$ called, and what do their rows/columns represent in a typical application like topic modeling?**
    *   **Answer:**
        *   **$W$ (Basis Matrix or Feature Matrix):** In topic modeling, if $V$ is a document-term matrix, $W$ would be a document-topic matrix. Each row represents a document, and each column represents a topic. The values $W_{ij}$ indicate the contribution of topic $j$ to document $i$.
        *   **$H$ (Coefficient Matrix or Encoding Matrix):** In topic modeling, $H$ would be a topic-term matrix. Each row represents a topic, and each column represents a word (term). The values $H_{ij}$ indicate the importance or weight of word $j$ in topic $i$.

6.  **What are the main challenges or disadvantages of using NMF?**
    *   **Answer:** Key challenges include:
        *   **Non-uniqueness of solutions:** NMF is a non-convex problem, so different initializations can lead to different local minima.
        *   **Sensitivity to initialization:** The choice of initial $W$ and $H$ can significantly impact the final factorization.
        *   **Choosing the number of components ($k$):** There's no definitive method to determine the optimal $k$, often requiring heuristics or domain knowledge.
        *   **Computational cost:** Can be high for very large matrices.
        *   **Requires non-negative data:** Cannot directly handle negative values.

7.  **How do you typically choose the number of components ($k$) in NMF?**
    *   **Answer:** Choosing $k$ is often heuristic. Methods include:
        *   **Domain knowledge:** If you know how many underlying "parts" or "topics" exist.
        *   **Reconstruction error:** Plotting the reconstruction error against different $k$ values and looking for an "elbow point" where the error reduction diminishes.
        *   **Coherence measures:** For topic modeling, metrics like topic coherence can help evaluate the quality of topics for different $k$.
        *   **Cross-validation:** Evaluating the performance of a downstream task (e.g., classification) using features derived from NMF with different $k$.

8.  **Describe the iterative update process in NMF. Why are multiplicative update rules often preferred?**
    *   **Answer:** NMF typically uses an iterative optimization process. It starts with random non-negative $W$ and $H$, then alternately updates $W$ (holding $H$ fixed) and $H$ (holding $W$ fixed) to minimize the objective function. Multiplicative update rules (like those by Lee and Seung) are preferred because they are simple to implement, are guaranteed to maintain the non-negativity constraints, and ensure that the objective function is non-increasing, leading to convergence to a local minimum.

9.  **Can NMF be used for data with negative values? If not, what preprocessing steps might be necessary?**
    *   **Answer:** No, NMF strictly requires non-negative input data. If the data contains negative values, it must be preprocessed. A common approach is to shift the data by adding a constant value to all elements such that the minimum value becomes zero. However, this transformation might alter the interpretability of the factorization. Another approach is to split the matrix into positive and negative parts, but this is less common.

10. **Provide 2-3 real-world applications where NMF is effectively used.**
    *   **Answer:**
        *   **Topic Modeling:** Extracting latent topics from large collections of text documents (e.g., news articles, scientific papers).
        *   **Recommender Systems:** Decomposing user-item interaction matrices to discover latent features of users and items, used for personalized recommendations.
        *   **Image Processing:** Decomposing images into basis components (e.g., facial features, object parts) for tasks like facial recognition, object recognition, or image compression.

## Quiz

1.  What is the fundamental constraint that defines Non-negative Matrix Factorization (NMF)?
    A) The factorized matrices $W$ and $H$ must be orthogonal.
    B) The original data matrix $V$ must be square.
    C) All elements in the original matrix $V$ and the factorized matrices $W$ and $H$ must be non-negative.
    D) The number of components $k$ must be greater than the number of features $m$.

2.  Which of the following is a key advantage of NMF over Principal Component Analysis (PCA)?
    A) NMF guarantees a unique global optimum.
    B) NMF can directly handle data with negative values.
    C) NMF often produces more interpretable, part-based representations.
    D) NMF is computationally less expensive for large datasets.

3.  In the context of topic modeling, if $V$ is a document-term matrix, what does the matrix $H$ (from $V \approx WH$) typically represent?
    A) Document-document similarity.
    B) Term-term similarity.
    C) Document-topic weights.
    D) Topic-term weights.

4.  What is the most common objective function minimized in NMF?
    A) Euclidean distance.
    B) Kullback-Leibler divergence.
    C) Frobenius norm of the reconstruction error.
    D) Cosine similarity.

5.  Which of the following is a common challenge when applying NMF?
    A) It always converges to the same unique solution regardless of initialization.
    B) It requires the input data to be normalized to a unit vector.
    C) Determining the optimal number of components ($k$) can be difficult.
    D) It is primarily used for supervised learning tasks.

### Answer Key

1.  **C) All elements in the original matrix $V$ and the factorized matrices $W$ and $H$ must be non-negative.**
    *   **Explanation:** The "non-negative" constraint is the defining characteristic of NMF, ensuring that all values in the input data and the resulting components are greater than or equal to zero.

2.  **C) NMF often produces more interpretable, part-based representations.**
    *   **Explanation:** Due to the non-negativity constraint, NMF tends to decompose data into additive "parts" (e.g., facial features, topics), which are often more intuitive and interpretable than the abstract, potentially negative components found by PCA.

3.  **D) Topic-term weights.**
    *   **Explanation:** If $V$ is documents x terms, and $W$ is documents x topics, then $H$ must be topics x terms to make the matrix multiplication $WH$ valid and approximate $V$. Thus, $H$ represents the weights of terms within each topic.

4.  **C) Frobenius norm of the reconstruction error.**
    *   **Explanation:** The Frobenius norm of the difference $||V - WH||_F^2$ is the most widely used objective function for NMF, measuring the squared error between the original matrix and its approximation.

5.  **C) Determining the optimal number of components ($k$) can be difficult.**
    *   **Explanation:** Choosing the number of latent components $k$ is a hyperparameter that often requires experimentation, domain knowledge, or heuristic methods, as there's no single definitive way to determine it. NMF is also a non-convex problem, meaning it does not always converge to the same unique solution.

## Further Reading

1.  **Scikit-learn NMF Documentation:** The official documentation for NMF in scikit-learn provides a great overview, parameters, and examples.
    *   [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

2.  **"Learning the parts of objects by non-negative matrix factorization" by Lee and Seung (1999):** This is one of the foundational papers that popularized NMF. It provides the mathematical basis and demonstrates its application to image data.
    *   [https://www.nature.com/articles/44565](https://www.nature.com/articles/44565) (You might need institutional access or search for a free PDF version online.)

3.  **"Algorithms for Non-negative Matrix Factorization" by Lee and Seung (2001):** A follow-up paper detailing the multiplicative update rules that are widely used in NMF implementations.
    *   [https://papers.nips.cc/paper/2001/file/f9d1152a7774724cd0f1683f2898a499-Paper.pdf](https://papers.nips.cc/paper/2001/file/f9d1152a7774724cd0f1683f2898a499-Paper.pdf)

4.  **"Non-negative Matrix Factorization (NMF) for Dummies" by Benyamin Ghojogh:** A blog post that offers a simplified, intuitive explanation of NMF.
    *   [https://towardsdatascience.com/non-negative-matrix-factorization-nmf-for-dummies-a745806f7228](https://towardsdatascience.com/non-negative-matrix-factorization-nmf-for-dummies-a745806f7228)