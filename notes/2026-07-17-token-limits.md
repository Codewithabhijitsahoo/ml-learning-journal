# Token Limits

## Overview
In the exciting world of Large Language Models (LLMs) and Natural Language Processing (NLP), "Token Limits" refer to the maximum number of "tokens" that a model can process or generate in a single interaction. Think of tokens as the fundamental building blocks of text that an LLM understands. They can be whole words, parts of words (like "un-" or "-ing"), punctuation marks, or even individual characters. For example, the sentence "Hello, world!" might be broken down into tokens like ["Hello", ",", " world", "!"].

Every LLM has a specific "context window" or "token window" size, which dictates how much information it can "see" or "remember" at any given time. This limit applies to both the input you provide to the model (your prompt) and the output it generates. If your input prompt exceeds this limit, it will often be truncated (cut off). Similarly, if the model tries to generate a response longer than its output limit, it will stop generating text prematurely. Understanding and managing token limits is crucial for effectively using and designing applications with LLMs.

## What Problem It Solves
Token limits address several fundamental challenges in the design and operation of large language models:

1.  **Computational Cost and Efficiency**: Processing text, especially with complex neural network architectures like Transformers, is computationally intensive. The computational cost of the attention mechanism, a core component of Transformers, scales quadratically with the sequence length ($O(N^2)$, where $N$ is the number of tokens). Without limits, processing extremely long texts would become prohibitively expensive and slow, requiring immense processing power and time. Token limits keep this cost manageable.

2.  **Memory Constraints**: Storing the intermediate representations (embeddings) of tokens during processing requires significant memory (RAM or GPU VRAM). As the number of tokens increases, the memory footprint grows. Token limits prevent models from running out of memory, especially on consumer-grade hardware or in large-scale deployments where many requests are processed concurrently.

3.  **Latency and Response Time**: Longer inputs and outputs directly translate to longer processing times. For real-time applications like chatbots or interactive assistants, low latency is critical. Token limits help ensure that responses are generated within an acceptable timeframe, improving the user experience.

4.  **Context Management and "Hallucination"**: While LLMs are powerful, they don't have infinite memory. The "context window" defines what information the model can actively consider when generating its next token. If the input is too long, the model might struggle to focus on the most relevant parts or even "forget" earlier details, leading to less coherent or accurate responses (sometimes referred to as "context stuffing" issues). Token limits force developers to be concise and provide the most pertinent information, helping the model stay focused.

5.  **Preventing "Runaway" Generation**: Without an output token limit, a model could potentially generate an infinitely long response, consuming resources indefinitely. Output limits provide a safeguard, ensuring that responses are concise and terminate appropriately.

6.  **API Cost Control**: Many LLM providers charge based on the number of tokens processed (both input and output). Token limits allow users and developers to control their spending by setting a maximum amount of text that can be processed per request.

## How It Works
The mechanism of token limits involves several steps, primarily handled by the tokenization process and the model's architecture:

1.  **Tokenization**:
    *   When you provide text to an LLM, the first step is to convert it into a sequence of tokens. This process is called tokenization.
    *   Different models use different tokenization strategies (e.g., WordPiece, Byte-Pair Encoding (BPE), SentencePiece). These strategies break down text into subword units, which helps handle out-of-vocabulary words and reduces the overall vocabulary size.
    *   For example, "unbelievable" might be tokenized as ["un", "believe", "able"]. Each of these subword units is a token.
    *   Each unique token is then mapped to a numerical ID from the model's vocabulary. This sequence of numerical IDs is what the model actually processes.

2.  **Applying the Limit (Input)**:
    *   Before the tokenized input sequence is fed into the model, its length (number of tokens) is checked against the model's predefined maximum input token limit (e.g., 4096, 8192, 128k tokens).
    *   If the input sequence exceeds this limit, one of two things typically happens:
        *   **Truncation**: The most common approach is to truncate the input. This means cutting off tokens from either the beginning, the end, or sometimes the middle of the sequence until it fits within the limit. The specific truncation strategy (e.g., `truncation_strategy='longest_first'`, `truncation_strategy='only_second'`) depends on the tokenizer and the task. For conversational models, often the oldest parts of the conversation are truncated.
        *   **Error/Warning**: Some APIs or local implementations might raise an error or warning, indicating that the input is too long and needs to be shortened by the user.

3.  **Model Processing**:
    *   The truncated (or appropriately sized) sequence of token IDs is then passed through the LLM's neural network architecture (typically a Transformer).
    *   The model processes these tokens, attending to relationships between them, and generates a probability distribution over the next possible token in the sequence.

4.  **Applying the Limit (Output)**:
    *   When the model starts generating its response, it predicts one token at a time.
    *   An output token limit (often specified by the user or API as `max_new_tokens` or `max_tokens`) is continuously monitored.
    *   Once the number of generated tokens reaches this limit, the model stops generating, even if it hasn't completed a sentence or thought.
    *   The generated token IDs are then converted back into human-readable text (detokenization).

In essence, token limits act as a gatekeeper, ensuring that the data flowing into and out of the LLM adheres to the computational and memory constraints of the underlying architecture.

## Mathematical Intuition
While "Token Limits" itself isn't a mathematical algorithm, its existence is deeply rooted in the mathematical and computational properties of the Transformer architecture, especially the self-attention mechanism.

Let's consider a sequence of $N$ tokens. Each token $t_i$ in the sequence is first converted into a numerical representation called an embedding, typically a vector of dimension $d_{model}$. So, our input sequence becomes a matrix of embeddings $X \in \mathbb{R}^{N \times d_{model}}$.

1.  **Self-Attention Complexity**:
    The core of the Transformer is the self-attention mechanism, which allows each token in a sequence to "attend" to every other token to understand context. This involves calculating attention scores between all pairs of tokens.
    For a sequence of length $N$, the attention mechanism computes three matrices: Query ($Q$), Key ($K$), and Value ($V$). These are derived from the input embeddings $X$ by multiplying them with learned weight matrices $W_Q, W_K, W_V$:
    $$Q = X W_Q$$
    $$K = X W_K$$
    $$V = X W_V$$
    The attention scores are then calculated as:
    $$Attention(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
    Where $d_k$ is the dimension of the key vectors.
    The crucial part here is the matrix multiplication $Q K^T$. If $Q$ is $N \times d_k$ and $K^T$ is $d_k \times N$, their product $Q K^T$ results in an $N \times N$ matrix. This operation has a computational complexity of $O(N^2 \cdot d_k)$ or simply $O(N^2)$ if we consider $d_k$ constant.
    This quadratic scaling means that doubling the sequence length $N$ quadruples the computational effort for the attention mechanism. This rapid increase in computation is the primary mathematical reason for token limits.

2.  **Memory Usage**:
    The intermediate matrices ($Q, K, V$, and the attention scores) also scale with $N$. For example, the attention score matrix $\frac{Q K^T}{\sqrt{d_k}}$ is $N \times N$. Storing this matrix requires $O(N^2)$ memory.
    Additionally, the embeddings for each token $X \in \mathbb{R}^{N \times d_{model}}$ require $O(N \cdot d_{model})$ memory. While this is linear, the $N^2$ memory for attention scores quickly dominates for large $N$.
    $$Memory \propto N^2$$
    This memory constraint further necessitates token limits, especially when running models on hardware with finite GPU memory.

3.  **Probability Distribution and Generation**:
    During text generation, the model predicts the next token $t_{i+1}$ based on the preceding sequence $t_1, ..., t_i$. This is a conditional probability:
    $$P(t_{i+1} | t_1, ..., t_i)$$
    The token limit $N_{max}$ defines the maximum length of this sequence $i$. The model is trained to learn these conditional probabilities up to a certain sequence length. Beyond this length, its ability to maintain coherence and context might degrade, even if computational resources were infinite. The limit ensures the model operates within its trained and effective context window.

In summary, the $O(N^2)$ computational complexity and memory requirements of the self-attention mechanism are the fundamental mathematical reasons why token limits are imposed on large language models. They are a practical necessity to make these powerful models feasible to train and deploy.

## Advantages
*   **Resource Management**: Prevents excessive consumption of computational resources (CPU/GPU, memory) during inference and training.
*   **Cost Control**: For API-based LLMs, token limits directly help manage and reduce API costs, as billing is often token-based.
*   **Faster Inference**: Shorter input and output sequences lead to quicker processing times, improving latency for real-time applications.
*   **Stability and Reliability**: Reduces the likelihood of out-of-memory errors or crashes when processing very long texts.
*   **Focused Context**: Encourages users to provide concise and relevant information, which can sometimes lead to better model performance by reducing irrelevant noise.
*   **Prevents Runaway Generation**: Output token limits ensure that the model stops generating text after a reasonable length, preventing infinite loops or excessively long, unhelpful responses.

## Disadvantages
*   **Information Loss (Truncation)**: The most significant drawback is that important information might be lost if the input text exceeds the token limit and is truncated. This can lead to incomplete understanding or incorrect responses from the model.
*   **Difficulty with Long Documents**: LLMs struggle to process and understand very long documents (e.g., entire books, lengthy research papers) in a single pass due to these limits.
*   **Context Window Limitations**: Even with large token limits, the model might still struggle to maintain coherence and recall information from the very beginning of a long context, a phenomenon sometimes called "lost in the middle."
*   **Incomplete Responses**: Output token limits can cut off a model's response mid-sentence or mid-thought, leading to fragmented or unsatisfying answers.
*   **Requires Pre-processing/Chunking**: To work with long texts, users often need to implement complex pre-processing strategies like chunking (breaking text into smaller pieces) and retrieval-augmented generation (RAG), adding complexity to application development.
*   **Creative Constraints**: For creative writing tasks, a strict output token limit can hinder the model's ability to develop elaborate narratives or detailed descriptions.

## Real World Applications
1.  **Chatbots and Conversational AI**:
    *   **Application**: Customer service chatbots, virtual assistants, and interactive AI companions.
    *   **How Token Limits Apply**: In a conversation, the entire chat history (or a significant portion of it) is often sent as input to the LLM to maintain context. Token limits dictate how much of this history can be included. If the conversation gets too long, older messages are truncated to fit within the limit, ensuring the bot remains responsive and cost-effective. Output limits prevent the bot from generating overly verbose responses.

2.  **Document Summarization**:
    *   **Application**: Generating concise summaries of articles, reports, legal documents, or research papers.
    *   **How Token Limits Apply**: A long document often exceeds the LLM's input token limit. Developers must employ strategies like "chunking" the document into smaller, overlapping segments, summarizing each chunk, and then recursively summarizing the summaries. The final summary itself must also adhere to an output token limit.

3.  **Code Generation and Completion**:
    *   **Application**: AI assistants that write code, suggest completions, or debug existing code (e.g., GitHub Copilot).
    *   **How Token Limits Apply**: The input typically includes the current code file, relevant surrounding files, and the user's prompt. This combined context can quickly hit token limits. The model must intelligently select the most relevant code snippets to include in the prompt. The generated code snippet also adheres to an output token limit to prevent generating excessively long or irrelevant code.

4.  **Content Creation and Marketing**:
    *   **Application**: Generating blog posts, marketing copy, social media updates, or product descriptions.
    *   **How Token Limits Apply**: While generating a full-length article might require multiple calls, individual paragraphs or sections are generated within token limits. For shorter content like social media posts, the entire output is often constrained by a single token limit, ensuring brevity and adherence to platform requirements.

5.  **Data Extraction and Information Retrieval**:
    *   **Application**: Extracting specific entities (names, dates, addresses), answering questions from documents, or structuring unstructured text.
    *   **How Token Limits Apply**: When querying a large document for specific information, the relevant sections (chunks) of the document are retrieved and passed to the LLM along with the question. Token limits ensure that only the most pertinent context is provided to the model, improving accuracy and efficiency. The extracted answer itself is also subject to an output token limit.

## Python Example
This example demonstrates how tokenization works with the `transformers` library and how `max_length` and `truncation` parameters are used to manage token limits. We'll use a pre-trained tokenizer from a common model like BERT.

```python
import torch
from transformers import AutoTokenizer

# 1. Load a pre-trained tokenizer
# We'll use a BERT-base-uncased tokenizer as an example.
# This tokenizer has a default max_length of 512 tokens.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print(f"Tokenizer loaded: {tokenizer.name_or_path}")
print(f"Default max_length for this tokenizer: {tokenizer.model_max_length} tokens\n")

# 2. Define some example texts
short_text = "Hello, this is a short sentence."
long_text = (
    "In the vast expanse of the digital cosmos, where algorithms dance and data flows like rivers of information, "
    "large language models stand as colossal monuments of artificial intelligence. They process, understand, "
    "and generate human-like text with an astonishing degree of fluency and coherence. However, their power "
    "is not without bounds, as they are inherently constrained by what we call 'token limits'. These limits "
    "are crucial for managing computational resources, controlling costs, and ensuring the practical "
    "deployability of these complex systems in real-world applications. Understanding how these limits "
    "impact the processing of extensive documents or lengthy conversations is paramount for developers "
    "and users alike. Without careful consideration, vital information might be truncated, leading to "
    "incomplete or inaccurate responses, thereby undermining the utility of these advanced AI tools."
    * 5 # Make it even longer to ensure it exceeds typical limits
)

# 3. Tokenize a short text (within limits)
print("--- Tokenizing a short text ---")
encoded_short = tokenizer(short_text, return_tensors="pt")
print(f"Original short text: '{short_text}'")
print(f"Token IDs: {encoded_short['input_ids'][0].tolist()}")
print(f"Number of tokens: {len(encoded_short['input_ids'][0])}\n")
# Decode to see the tokens
decoded_short = tokenizer.decode(encoded_short['input_ids'][0])
print(f"Decoded short text: '{decoded_short}'\n")


# 4. Tokenize a long text without truncation (will exceed limit if not handled)
print("--- Tokenizing a long text (without explicit truncation) ---")
# By default, if max_length is not set and truncation is False, it will just tokenize everything.
# However, if you were to pass this to a model, it would likely error out if it exceeds the model's capacity.
encoded_long_no_trunc = tokenizer(long_text, return_tensors="pt")
print(f"Original long text length: {len(long_text)} characters")
print(f"Number of tokens (no truncation): {len(encoded_long_no_trunc['input_ids'][0])}")
print(f"Exceeds model_max_length ({tokenizer.model_max_length})? "
      f"{len(encoded_long_no_trunc['input_ids'][0]) > tokenizer.model_max_length}\n")

# 5. Tokenize a long text with truncation
print("--- Tokenizing a long text WITH truncation ---")
# We'll set a max_length explicitly, which is often done when preparing input for a model.
# The `truncation=True` argument tells the tokenizer to cut off tokens if they exceed `max_length`.
# `max_length` here overrides the tokenizer's default `model_max_length` for this specific call.
custom_max_length = 128 # A smaller limit for demonstration
encoded_long_truncated = tokenizer(long_text,
                                   max_length=custom_max_length,
                                   truncation=True,
                                   return_tensors="pt")

print(f"Number of tokens (truncated to {custom_max_length}): {len(encoded_long_truncated['input_ids'][0])}")
print(f"Does it fit within {custom_max_length} tokens? "
      f"{len(encoded_long_truncated['input_ids'][0]) <= custom_max_length}")

# Decode the truncated text to see what was kept
decoded_truncated = tokenizer.decode(encoded_long_truncated['input_ids'][0], skip_special_tokens=True)
print(f"Decoded truncated text (first {custom_max_length} tokens):\n'{decoded_truncated}'\n")
print(f"Notice how the text ends abruptly, demonstrating truncation.\n")

# 6. Simulating an output token limit (conceptual)
print("--- Simulating an output token limit ---")
# In a real LLM, you'd pass the input_ids to the model's generate method.
# For demonstration, let's imagine a model generates tokens one by one.
# We'll use the short text as a starting point and simulate generating 5 new tokens.

# Let's assume the model generates these token IDs after the short_text:
# (These are arbitrary token IDs for demonstration)
generated_token_ids = [2054, 2003, 1037, 3014, 1012, 2054, 2003, 1037, 3014, 1012] # " this is a test . this is a test ."

output_max_tokens = 5 # We want to limit the *new* tokens generated to 5

# The model would typically take the input_ids and generate from there.
# For simplicity, let's just take the first `output_max_tokens` from our simulated generation.
limited_generated_ids = generated_token_ids[:output_max_tokens]

# Combine input tokens with limited generated tokens (excluding special tokens from input for clarity)
full_output_ids = encoded_short['input_ids'][0].tolist()[:-1] + limited_generated_ids # remove [SEP] from input

decoded_full_output = tokenizer.decode(full_output_ids, skip_special_tokens=True)

print(f"Input text: '{short_text}'")
print(f"Simulated generated tokens (limited to {output_max_tokens}): {limited_generated_ids}")
print(f"Full output with limited generation:\n'{decoded_full_output}'")
print(f"Notice how the generation stops after 5 new tokens, even if more were 'intended'.")

```

**Explanation of the Python Example:**

1.  **Loading Tokenizer**: We start by loading a `bert-base-uncased` tokenizer. This tokenizer knows how to convert raw text into numerical tokens that BERT understands and vice-versa. It also has a `model_max_length` attribute, which is its inherent token limit (512 for BERT-base).
2.  **Short Text Tokenization**: We tokenize a short sentence. The output shows the numerical `input_ids` and the total number of tokens, including special tokens like `[CLS]` (start of sentence) and `[SEP]` (separator).
3.  **Long Text (No Truncation)**: We tokenize a very long text without explicitly setting `max_length` or `truncation=True`. This demonstrates that the tokenizer *can* process long texts, but the resulting token count will exceed the model's typical `model_max_length`. If you tried to feed this directly into a BERT model, it would likely raise an error.
4.  **Long Text (With Truncation)**: This is the core demonstration of token limits.
    *   We set `max_length=128` (a custom, smaller limit for clarity).
    *   We set `truncation=True`. This tells the tokenizer: "If the text is longer than `max_length`, cut off the excess tokens."
    *   The output clearly shows that the number of tokens is now exactly `128`.
    *   When decoded, you can see that the text abruptly ends, illustrating the loss of information due to truncation.
5.  **Simulating Output Limit**: This section conceptually shows how an output token limit works. In a real LLM, you'd use a `model.generate()` method with a `max_new_tokens` parameter. Here, we simulate a model generating a sequence of tokens and then manually apply a `max_new_tokens` limit to show how the generation would stop prematurely.

This example highlights how `max_length` and `truncation` are practical tools for managing token limits when preparing input for LLMs.

## Interview Questions

1.  **What are Token Limits in the context of LLMs?**
    *   **Answer**: Token limits refer to the maximum number of tokens (subword units, words, punctuation) that a Large Language Model can process in its input (prompt) or generate in its output (response) during a single interaction. Each LLM has a predefined "context window" size, which is essentially its token limit.

2.  **Why are Token Limits necessary for LLMs?**
    *   **Answer**: They are crucial for several reasons:
        *   **Computational Cost**: The attention mechanism in Transformers scales quadratically with sequence length ($O(N^2)$), making long sequences very expensive to process.
        *   **Memory Constraints**: Storing intermediate representations and attention matrices for long sequences requires significant memory.
        *   **Latency**: Shorter sequences lead to faster inference times, which is critical for real-time applications.
        *   **Resource Management**: Prevents models from consuming excessive resources or running indefinitely.
        *   **API Cost Control**: Many LLM APIs charge per token, so limits help manage expenses.

3.  **How do LLMs handle input text that exceeds the token limit?**
    *   **Answer**: The most common method is **truncation**. The input text is cut off (either from the beginning, end, or sometimes middle) until it fits within the `max_length`. Some systems might also raise an error or warning, requiring the user to shorten the input manually.

4.  **What is the primary disadvantage of token limits, especially for long documents?**
    *   **Answer**: The primary disadvantage is **information loss due to truncation**. If important context or details are cut off from the input, the model may not have all the necessary information to generate an accurate or complete response. This makes processing very long documents in a single pass challenging.

5.  **Explain the concept of "context window" in relation to token limits.**
    *   **Answer**: The "context window" is synonymous with the token limit. It defines the maximum number of tokens an LLM can "see" or "remember" at any given moment to generate its next token. It's the window of information the model uses to understand the current state and predict the future.

6.  **How does the quadratic scaling of attention mechanisms relate to token limits?**
    *   **Answer**: The self-attention mechanism in Transformer models has a computational complexity of $O(N^2)$, where $N$ is the sequence length (number of tokens). This means that if you double the sequence length, the computation required for attention quadruples. This rapid increase in computational demand and memory usage is the fundamental mathematical reason why token limits are imposed to keep models feasible.

7.  **What strategies can be employed to work with documents longer than an LLM's token limit?**
    *   **Answer**: Common strategies include:
        *   **Chunking**: Breaking the document into smaller, overlapping segments (chunks) that fit within the token limit.
        *   **Summarization**: Summarizing each chunk and then recursively summarizing the summaries.
        *   **Retrieval-Augmented Generation (RAG)**: Using a retrieval system to find the most relevant chunks of a document based on a query, and then feeding only those relevant chunks to the LLM.
        *   **Sliding Window**: Processing a document by moving a "window" of tokens across it, potentially with some overlap.

8.  **Can an LLM generate an infinitely long response if there's no output token limit?**
    *   **Answer**: Theoretically, yes. Without an explicit output token limit, an LLM could enter a repetitive loop or continue generating text indefinitely, consuming resources and never terminating. Output token limits are a safeguard to ensure responses are concise and terminate appropriately.

9.  **How do token limits impact the cost of using commercial LLM APIs?**
    *   **Answer**: Most commercial LLM APIs (like OpenAI's GPT models) charge based on the number of tokens processed, both for input and output. Higher token limits or longer prompts/responses directly translate to higher costs. Managing token usage is essential for cost-effective API integration.

10. **What is the difference between `max_length` and `max_new_tokens` when interacting with an LLM?**
    *   **Answer**:
        *   `max_length` (or `max_input_tokens`): Refers to the maximum total length of the *input sequence* that the model can accept. If the input exceeds this, it's typically truncated.
        *   `max_new_tokens`: Refers to the maximum number of *new tokens* that the model is allowed to generate as its output, *after* processing the input. This parameter controls the length of the model's response.

## Quiz

1.  What is the primary reason for token limits in Large Language Models?
    A) To make models forget past information quickly.
    B) To reduce computational cost and memory usage.
    C) To force users to write shorter prompts.
    D) To prevent models from learning too much.

2.  If an input prompt exceeds an LLM's token limit, what is the most common way it's handled?
    A) The model requests a shorter prompt from the user.
    B) The model processes only the middle part of the prompt.
    C) The prompt is truncated (cut off) to fit the limit.
    D) The model automatically expands its token limit for that request.

3.  Which of the following best describes a "token" in the context of LLMs?
    A) A fixed-length string of 10 characters.
    B) The smallest unit of text (word, subword, punctuation) that an LLM processes.
    C) A numerical representation of an entire sentence.
    D) A measure of the model's processing speed.

4.  The computational complexity of the self-attention mechanism in Transformers scales quadratically with the sequence length ($O(N^2)$). What does this imply for token limits?
    A) Longer sequences are easier to process.
    B) Token limits are necessary to keep computation and memory manageable for longer sequences.
    C) Token limits are irrelevant to computational complexity.
    D) The model becomes more accurate with longer sequences regardless of limits.

5.  Which strategy is commonly used to process a document that is significantly longer than an LLM's input token limit?
    A) Sending the entire document in multiple, sequential API calls without modification.
    B) Increasing the LLM's token limit dynamically for that specific document.
    C) Truncating the document to the first few sentences and hoping for the best.
    D) Chunking the document into smaller, manageable segments and potentially using retrieval techniques.

---

**Answer Key:**

1.  **B) To reduce computational cost and memory usage.**
    *   **Explanation**: The quadratic scaling of the attention mechanism makes processing long sequences very expensive in terms of computation and memory. Token limits are a practical necessity to manage these resources.

2.  **C) The prompt is truncated (cut off) to fit the limit.**
    *   **Explanation**: Truncation is the most common method. Parts of the input (often the oldest or least relevant) are removed until the prompt fits within the `max_length`.

3.  **B) The smallest unit of text (word, subword, punctuation) that an LLM processes.**
    *   **Explanation**: Tokens are the fundamental building blocks of text for LLMs, which can be whole words, parts of words (subwords), or punctuation.

4.  **B) Token limits are necessary to keep computation and memory manageable for longer sequences.**
    *   **Explanation**: The $O(N^2)$ complexity means that computational resources grow very rapidly with sequence length, making token limits essential for practical deployment.

5.  **D) Chunking the document into smaller, manageable segments and potentially using retrieval techniques.**
    *   **Explanation**: Since an LLM cannot process an entire long document at once, breaking it into smaller chunks and feeding those chunks (possibly after retrieving the most relevant ones) is the standard approach.

## Further Reading

1.  **Hugging Face Transformers Documentation - Tokenizers**: A comprehensive guide to how tokenizers work, including different tokenization strategies and handling `max_length` and `truncation`.
    *   [https://huggingface.co/docs/transformers/main_classes/tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)
2.  **OpenAI API Documentation - Understanding Tokens**: Official explanation of tokens, how they are counted, and their implications for API usage and costs.
    *   [https://platform.openai.com/docs/introduction/tokens](https://platform.openai.com/docs/introduction/tokens)
3.  **"Attention Is All You Need" (The Transformer Paper)**: While highly technical, understanding the original Transformer paper provides the foundational knowledge for why the attention mechanism has $O(N^2)$ complexity, which directly leads to the necessity of token limits. Focus on the "Scaled Dot-Product Attention" section.
    *   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)