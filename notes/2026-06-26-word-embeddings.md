# Word Embeddings

## Overview
Word Embeddings are a fundamental concept in Natural Language Processing (NLP) that transforms words into numerical representations called vectors. Instead of treating words as discrete, atomic units, word embeddings represent them as dense, low-dimensional vectors in a continuous vector space. The core idea is that words with similar meanings or that appear in similar contexts will have similar vector representations, meaning their vectors will be "close" to each other in this space. This allows machine learning models to understand the semantic and syntactic relationships between words, which is crucial for many NLP tasks.

## What Problem It Solves
Before word embeddings, words were often represented using methods like One-Hot Encoding.
*   **One-Hot Encoding Limitations:**
    *   **Sparsity:** Each word is represented by a vector with a single '1' and many '0's. For a vocabulary of 10,000 words, each vector is 10,000 dimensions long, making it very sparse and inefficient.
    *   **No Semantic Relationship:** One-hot vectors are orthogonal to each other, meaning the dot product between any two distinct word vectors is zero. This implies that "king" and "queen" are as different as "king" and "banana," which is semantically incorrect. There's no way to capture the idea that "king" is related to "queen" or "man" is related to "woman."
    *   **High Dimensionality:** As vocabulary grows, the dimensionality of one-hot vectors increases linearly, leading to the "curse of dimensionality."

Word embeddings solve these problems by providing:
*   **Dense Representations:** Vectors are typically much smaller (e.g., 50-300 dimensions) and contain real numbers, making them dense and computationally efficient.
*   **Semantic Relationships:** Words with similar meanings are mapped to similar vectors, allowing models to infer relationships like "king - man + woman = queen."
*   **Dimensionality Reduction:** They reduce the high dimensionality of sparse representations to a much lower, more manageable size.

## How It Works
The general principle behind word embeddings is to learn these dense vector representations by analyzing large amounts of text data. The most common methods learn embeddings by predicting the context of words or by leveraging co-occurrence statistics.

Here's a simplified explanation of the core idea:
1.  **Contextual Learning:** Algorithms like Word2Vec (Skip-gram and CBOW) and FastText learn embeddings by training a shallow neural network to perform a "proxy" task.
    *   **Skip-gram:** Given a target word, predict its surrounding context words.
    *   **CBOW (Continuous Bag-of-Words):** Given a set of context words, predict the target word in the middle.
    *   During this training, the hidden layer weights of the neural network become the word embeddings. Words that frequently appear in similar contexts will have their embeddings adjusted to be closer to each other.
2.  **Co-occurrence Statistics:** Algorithms like GloVe (Global Vectors for Word Representation) learn embeddings by analyzing the global word-word co-occurrence matrix from a corpus. They aim to learn vectors such that their dot product is proportional to the logarithm of their co-occurrence probability.
3.  **Resulting Vectors:** Regardless of the method, the output is a lookup table where each word in the vocabulary is associated with a unique, fixed-size vector of real numbers. These vectors capture various linguistic regularities, such as semantic similarity, gender, tense, and plurality.

## Mathematical Intuition
Word embeddings represent words as points in a high-dimensional vector space. The "meaning" of a word is encoded by its position relative to other words in this space.

The key mathematical concept is **vector similarity**, often measured using **cosine similarity**. Cosine similarity measures the cosine of the angle between two non-zero vectors. If two vectors are pointing in roughly the same direction, their cosine similarity will be close to 1, indicating high similarity. If they are orthogonal, it's 0, and if they point in opposite directions, it's -1.

For two word vectors, $\mathbf{A}$ and $\mathbf{B}$, the cosine similarity is calculated as:

$$ \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}} $$

Where:
*   $\mathbf{A} \cdot \mathbf{B}$ is the dot product of vectors $\mathbf{A}$ and $\mathbf{B}$.
*   $\|\mathbf{A}\|$ and $\|\mathbf{B}\|$ are the Euclidean norms (magnitudes) of vectors $\mathbf{A}$ and $\mathbf{B}$, respectively.
*   $n$ is the dimensionality of the word vectors.

This formula allows us to quantify how semantically similar two words are based on the "angle" between their vector representations.

## Advantages
1.  **Capture Semantic Relationships:** Embeddings capture nuanced meanings and relationships between words (e.g., "king" is to "man" as "queen" is to "woman").
2.  **Dimensionality Reduction:** They reduce sparse, high-dimensional one-hot vectors to dense, low-dimensional representations, saving memory and computation.
3.  **Improved Performance:** Machine learning models perform better on various NLP tasks when fed with rich, semantically meaningful word embeddings compared to sparse representations.
4.  **Transfer Learning:** Pre-trained word embeddings (trained on massive text corpora) can be used as features in new NLP tasks, even with limited task-specific data, significantly boosting performance.
5.  **Generalization:** Models can generalize better to unseen words or phrases if their components have meaningful embeddings.

## Disadvantages
1.  **Context-Independent (for static embeddings):** Traditional word embeddings (like Word2Vec, GloVe) assign a single, fixed vector to each word, regardless of its context. This means "bank" has the same embedding whether it refers to a financial institution or a river bank.
2.  **Out-of-Vocabulary (OOV) Words:** They cannot generate embeddings for words not seen during training. This is a significant challenge for rare words or new words. (Some models like FastText address this by using subword information).
3.  **Computationally Intensive:** Training word embeddings from scratch on very large corpora can be computationally expensive and time-consuming.
4.  **Requires Large Corpora:** Effective word embeddings typically require vast amounts of text data for training to capture robust semantic relationships.
5.  **Bias:** Embeddings can inherit and amplify biases present in the training data (e.g., gender stereotypes, racial biases), which can lead to unfair or discriminatory outcomes in downstream applications.

## Real World Applications
1.  **Sentiment Analysis:** By representing words like "amazing," "terrible," "good," and "bad" with vectors that reflect their sentiment, models can more accurately classify the overall sentiment of reviews or social media posts.
2.  **Machine Translation:** Word embeddings help translation models understand the semantic equivalence between words in different languages, improving the quality and fluency of translations.
3.  **Recommendation Systems:** If users like items associated with certain words (e.g., "adventure," "sci-fi"), word embeddings can help recommend other items whose descriptions contain semantically similar words.
4.  **Search Engines:** Embeddings allow search engines to understand the intent behind a query, even if the exact keywords aren't present in the documents. For example, a query for "car repair" might also return results for "auto mechanic" because the embeddings for "car" and "auto," and "repair" and "mechanic" are close.

## Python Example
This example uses the `gensim` library to train a simple Word2Vec model on a small corpus and then demonstrates finding similar words.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Sample corpus
corpus = [
    "I love natural language processing.",
    "Word embeddings are very useful for NLP tasks.",
    "Machine learning is a powerful tool.",
    "Deep learning is a subset of machine learning.",
    "The quick brown fox jumps over the lazy dog.",
    "NLP is an exciting field."
]

# Tokenize the corpus into sentences of words
# Word2Vec expects a list of lists of words
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train a Word2Vec model
# vector_size: dimensionality of the word vectors
# window: maximum distance between the current and predicted word within a sentence
# min_count: ignores all words with total frequency lower than this
# workers: use these many worker threads to train the model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Access the vector for a specific word
word_vector = model.wv['nlp']
print(f"Vector for 'nlp' (first 5 dimensions): {word_vector[:5]}\n")

# Find the most similar words to a given word
similar_words = model.wv.most_similar('nlp', topn=3)
print(f"Words similar to 'nlp': {similar_words}\n")

similar_words_ml = model.wv.most_similar('learning', topn=2)
print(f"Words similar to 'learning': {similar_words_ml}\n")

# You can also perform vector arithmetic (though with a small corpus, results might not be perfect)
# Example: "king - man + woman = queen" analogy
# Here, we'll try a simpler one: "fox" - "brown" + "quick" (might not make sense with this small data)
# Let's try to find a word that is to 'machine' what 'deep' is to 'learning' (not perfect analogy)
# result = model.wv.most_similar(positive=['deep', 'machine'], negative=['learning'], topn=1)
# print(f"Analogy 'deep' is to 'learning' as X is to 'machine': {result}")
```

## Interview Questions
1.  **What are Word Embeddings, and why are they preferred over One-Hot Encoding in NLP?**
    *   **Answer:** Word Embeddings are dense, low-dimensional vector representations of words that capture semantic and syntactic relationships. They are preferred over One-Hot Encoding because one-hot vectors are sparse, high-dimensional, and fail to capture any semantic similarity between words (all distinct one-hot vectors are orthogonal). Embeddings, conversely, represent similar words with similar vectors, allowing models to understand context and meaning.
2.  **Explain the core idea behind how Word Embeddings are learned (e.g., Word2Vec).**
    *   **Answer:** Word embeddings are typically learned by training a shallow neural network on a large text corpus. For Word2Vec, there are two main architectures: Skip-gram and CBOW. Skip-gram predicts context words given a target word, while CBOW predicts a target word given its context words. During this training process, the weights of the hidden layer of the neural network become the word embeddings. The model learns to adjust these vectors such that words appearing in similar contexts have similar vector representations.
3.  **What is cosine similarity, and how is it used with word embeddings?**
    *   **Answer:** Cosine similarity is a metric used to measure the similarity between two non-zero vectors by calculating the cosine of the angle between them. In the context of word embeddings, it quantifies the semantic similarity between two words. A cosine similarity close to 1 indicates high similarity (small angle), while a value close to 0 indicates low similarity (large angle, orthogonal vectors). It's widely used to find the most similar words to a given word or to compare the relatedness of different concepts.

## Quiz
1.  Which of the following is a primary advantage of Word Embeddings over One-Hot Encoding?
    a) They result in higher-dimensional representations.
    b) They explicitly capture semantic relationships between words.
    c) They are easier to compute for very large vocabularies.
    d) They are immune to biases present in training data.
    *   **Answer:** b) They explicitly capture semantic relationships between words.

2.  What is a significant limitation of static word embeddings like Word2Vec?
    a) They are too computationally cheap to train.
    b) They can easily handle out-of-vocabulary (OOV) words.
    c) They assign a single, fixed vector to a word regardless of its context.
    d) They cannot be used for transfer learning.
    *   **Answer:** c) They assign a single, fixed vector to a word regardless of its context.

## Further Reading
1.  **Word2Vec Paper:** [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) by Tomas Mikolov et al.
2.  **GloVe Paper:** [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) by Jeffrey Pennington et al.
3.  **Gensim Word2Vec Documentation:** [Gensim Word2Vec Model](https://radimrehurek.com/gensim/models/word2vec.html)