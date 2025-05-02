**DATASETS USED**

**VAST (Versatile Argument-Based Stance Detection)**
16,000+ labeled tweets.
Targets with FAVOR, AGAINST, NONE labels.
Includes train/val/test splits and "Seen?" indicator.
**SemEval-2016 Task 6**
Benchmark dataset with 5 controversial topics (e.g., abortion, climate change).
Widely used for general stance classification.
**COVID-19 Stance Dataset**
Tweets about vaccination, mask mandates, and policies.
Focuses on public health stance analysis.

**EZ-Stance**
Built for zero-shot stance detection.
Tests model generalization on unseen topics.
Balanced label distribution.
**P-Stance (Political Stance Dataset)**
Focuses on stances toward political figures (e.g., Trump, Biden).
Captures political opinion and bias in social media content.

**MODELS USED TO FIND STANCE DETECTION:**
Model-1.1 (Mini-GRU + GloVe + Basic Attention)
Model-1.2 (Mini-GRU + GloVe + Topic-Aware Attention)
Model-2 (GRU + Attention)
Model-3 (BERT Fine-Tuned)
Model-4(BART)


**1. Model-1.1 (Mini-GRU + GloVe + Basic Attention):**
Mini-GRU: This is a compact version of the Gated Recurrent Unit (GRU), a type of recurrent neural network (RNN) designed for sequence modeling. GRUs are particularly useful for capturing dependencies in sequences like text. The "Mini" version suggests a more lightweight and computationally efficient architecture compared to full-scale GRUs.

GloVe: GloVe (Global Vectors for Word Representation) is a pre-trained word embedding model that captures semantic relationships between words. It represents each word as a dense vector in a high-dimensional space, helping the model to understand the context and meaning of words in relation to one another.

Basic Attention: Attention mechanisms allow the model to focus on specific parts of the input sequence when making predictions, which helps in handling long-range dependencies better. The Basic Attention mechanism is likely a simpler attention mechanism that allows the model to weigh different words in the input sentence based on their importance for predicting the stance.

Working: The model processes input text through the Mini-GRU to capture the sequential dependencies and uses GloVe word embeddings to enrich the representation of each word. The Basic Attention mechanism is applied to focus on relevant parts of the text, improving the model's ability to determine the stance of a given text (i.e., whether it is supportive, against, or neutral).

**2. Model-1.2 (Mini-GRU + GloVe + Topic-Aware Attention):**
Mini-GRU: Same as in Model-1.1, a lightweight GRU model for sequence processing.

GloVe: Used here as well for word embedding, providing semantic context to the model.

Topic-Aware Attention: This is an enhancement over basic attention. It likely incorporates knowledge about the topics of the text or the task at hand. Topic-aware attention helps the model focus on parts of the text that are not only contextually important but also relevant to the specific topic or domain.

Working: This model works similarly to Model-1.1 but incorporates topic-aware attention. Instead of focusing only on the words’ importance in context, it also takes into account the relevance to a specific topic, improving stance detection by focusing on topic-related information in the text.

**3. Model-2 (GRU + Attention):**
GRU: A standard GRU network is used in this model. It's designed to capture sequential patterns in the text.

Attention: This model employs an attention mechanism to focus on the most relevant parts of the input text, which allows it to better handle long-range dependencies and context that are important for stance detection.

Working: The GRU processes the input text to capture its sequential patterns. The attention mechanism is applied afterward to allow the model to focus on the most important words or phrases for stance prediction. The model uses this attention to determine whether the stance towards the given text is supportive, against, or neutral.

**4. Model-3 (BERT Fine-Tuned):**
BERT (Bidirectional Encoder Representations from Transformers): BERT is a pre-trained transformer-based model that excels at understanding the context of words in a sentence bidirectionally (i.e., both from the left and the right). It is pre-trained on a large corpus and can be fine-tuned on specific tasks, such as stance detection.

Fine-Tuning: Fine-tuning involves training the pre-trained BERT model on your specific dataset (stance detection) to adapt its knowledge to your task.

Working: This model uses the powerful pre-trained BERT architecture, which understands the relationship between words and their context in a sentence. It is fine-tuned on a stance detection dataset to predict the stance of a given text, leveraging its understanding of language to determine whether the stance is supportive, against, or neutral.

**5. Model-4 (BART):**
BART (Bidirectional and Auto-Regressive Transformers): BART is a transformer-based model that combines the benefits of both BERT (bidirectional encoder) and GPT (autoregressive decoder). It is effective for various natural language processing tasks, including text generation and understanding. In your case, BART is likely being used for sequence-to-sequence learning, which is particularly useful for tasks like stance detection, where you may need to understand both the context and the generative nature of the input text.

Working: BART works by encoding the input text into a sequence of embeddings and then generating a representation for stance detection. Its architecture allows it to capture both local and global dependencies in the text, which is crucial for accurately detecting stances.
