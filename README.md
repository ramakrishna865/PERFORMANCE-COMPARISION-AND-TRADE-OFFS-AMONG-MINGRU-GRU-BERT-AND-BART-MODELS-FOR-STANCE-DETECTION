**Datasets Used**
**1. VAST (Versatile Argument-Based Stance Detection)**
Size: 16,000+ labeled tweets

Targets: FAVOR, AGAINST, NONE labels

Splits: Includes train/val/test splits and "Seen?" indicator

**2. SemEval-2016 Task 6**
Content: Benchmark dataset with 5 controversial topics (e.g., abortion, climate change)

Use: Widely used for general stance classification
**
**3. COVID-19 Stance Dataset****
Content: Tweets about vaccination, mask mandates, and policies

Use: Focuses on public health stance analysis
**
**4. EZ-Stance****
Purpose: Built for zero-shot stance detection

Focus: Tests model generalization on unseen topics

Labels: Balanced label distribution

**5. P-Stance (Political Stance Dataset)**
Focus: Stances toward political figures (e.g., Trump, Biden)

Use: Captures political opinion and bias in social media content

**Models Used for Stance Detection**
**Model-1.1: Mini-GRU + GloVe + Basic Attention**
Mini-GRU:
A compact version of the Gated Recurrent Unit (GRU), designed for efficient sequence modeling. GRUs capture dependencies in sequences like text, and the "Mini" version is computationally lighter.

GloVe:
Global Vectors for Word Representation (GloVe) is a pre-trained model that captures semantic relationships between words. It represents each word as a dense vector in a high-dimensional space.

Basic Attention:
An attention mechanism that focuses on the most relevant parts of the input text when making predictions, helping the model handle long-range dependencies.

Working:
The Mini-GRU processes input text, capturing sequential dependencies, and GloVe enriches the word representations. Basic Attention allows the model to focus on important parts of the text for stance prediction (supportive, against, or neutral).

**Model-1.2: Mini-GRU + GloVe + Topic-Aware Attention**
Mini-GRU:
Same as in Model-1.1, a lightweight GRU model for sequence processing.

GloVe:
Pre-trained word embeddings that provide semantic context to the model.

Topic-Aware Attention:
An enhanced attention mechanism that incorporates knowledge of the text's topic, allowing the model to focus on domain-relevant parts of the input.

Working:
This model is similar to Model-1.1 but includes topic-aware attention, which focuses not only on contextual importance but also on the relevance to the specific topic, improving stance detection.

Model-2: GRU + Attention
GRU:
A standard GRU network for capturing sequential patterns in text.

Attention:
An attention mechanism that focuses on the most important words or phrases for stance prediction.

Working:
The GRU processes the input text, capturing sequential patterns. The attention mechanism then highlights the most relevant parts of the text for stance classification (supportive, against, or neutral).

**Model-3: BERT Fine-Tuned**
BERT:
A transformer-based model that understands the context of words bidirectionally, both from the left and the right. It is pre-trained on a large corpus and fine-tuned on specific tasks like stance detection.

Fine-Tuning:
Involves training the pre-trained BERT model on the stance detection dataset to adapt its knowledge to the specific task.

Working:
This model uses the pre-trained BERT architecture to understand word relationships in context. It is fine-tuned to predict the stance of a given text, identifying whether the stance is supportive, against, or neutral.
**
Model-4: BART**
BART:
A transformer-based model that combines the benefits of both BERT (bidirectional encoder) and GPT (autoregressive decoder). BART is effective for sequence-to-sequence learning, making it useful for stance detection, where context and generative capabilities are key.

Working:
BART encodes input text into embeddings and generates a stance prediction by capturing both local and global dependencies in the text, making it effective for stance detection tasks.
