# Sentiment Analysis with Sarcasm Detection on News Headlines
 
A deep learning project that detects sarcasm in news headlines and combines the result with sentiment analysis to produce a more accurate overall sentiment label. Four neural architectures (LSTM, Bi-LSTM, Attention-based LSTM, and a Transformer encoder) are trained and compared, and the best model is paired with VADER sentiment scoring.
 
## Overview
 
Sarcasm flips the apparent sentiment of a sentence — a positive-sounding headline can carry a negative intent. This project first trains a classifier to flag sarcastic headlines, then uses that signal alongside VADER sentiment polarity to correct the final sentiment label (e.g. a "positive" headline that is sarcastic is reclassified as negative).
 
## Dataset
 
The data comes from the **News Headlines Dataset for Sarcasm Detection**, sourced from The Onion (sarcastic) and HuffPost (non-sarcastic). Two files (`Sarcasm_Headlines_Dataset.json` and `Sarcasm_Headlines_Dataset_v2.json`) are combined.
 
- **Instances:** 55,328
- **Variables:** 3
| Variable | Description |
| :--- | :--- |
| `article_link` | Link to the news article (dropped during cleaning) |
| `headline` | Headline of the news article |
| `is_sarcastic` | 1 if the headline is sarcastic, 0 otherwise |
 
The data is loaded directly from GitHub raw URLs in the notebook.
 
## Pipeline
 
1. **Data understanding** — load and combine both JSON files, inspect the class distribution (used in full, without balancing).
2. **Cleaning & quality check** — drop `article_link`, confirm no missing values, verify `is_sarcastic` contains only 0/1.
3. **Label encoding** — encode the target with `LabelEncoder`.
4. **Text preprocessing** — regex cleaning, lowercasing, stopword removal (keeping "not"), lemmatization with WordNet; headlines longer than 40 tokens are dropped.
5. **EDA** — Bag-of-Words frequency analysis and WordClouds for sarcastic vs. non-sarcastic headlines.
6. **Partitioning** — split into train / validation / test (80/10/10, stratified).
7. **Tokenisation & padding** — Keras `Tokenizer`, sequences padded/truncated to `max_len = 40`.
8. **Modeling** — train four architectures (below).
9. **Comparison** — ROC curves, accuracy, precision, recall, F1, and a confusion matrix for the best model.
10. **Sentiment analysis** — VADER scoring combined with the sarcasm label to derive a final Positive / Negative / Neutral output.
## Models
 
| Model | Description |
| :--- | :--- |
| **LSTM** | Embedding → SpatialDropout → LSTM → Dropout → Dense (sigmoid) |
| **Bi-LSTM** | Embedding → Bidirectional LSTM → Dropout → Dense (sigmoid) |
| **Attention-based LSTM** | Embedding → LSTM (sequences) → custom attention layer → Dropout → Dense |
| **Transformer** | Token + positional embeddings → multi-head self-attention encoder block → global average pooling → Dense |
 
All models use 100-dim embeddings, binary cross-entropy loss, Adam optimizer, and early stopping.
 
## Results
 
The **Bi-LSTM** was the best-performing model on the test set:
 
- **Accuracy:** 0.9671
- **F1 score:** 0.9639
Models are compared via ROC/AUC, accuracy, precision, recall, and F1, with a confusion matrix plotted for the Bi-LSTM.
 
## Tech Stack
 
Python · TensorFlow / Keras · scikit-learn · NLTK · Gensim · VADER (vaderSentiment) · pandas · NumPy · Matplotlib · Seaborn · WordCloud
 
## Getting Started
 
```bash
pip install tensorflow scikit-learn nltk gensim wordcloud vaderSentiment pandas numpy matplotlib seaborn
```
 
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
 
Then open `Sarcasm_Detection.ipynb` and run the cells top to bottom. The datasets load automatically from their GitHub URLs, so no manual download is required.
 
## Project Structure
 
```
.
├── Sarcasm_Detection.ipynb    # Main notebook: preprocessing, modeling, evaluation
└── README.md
```
