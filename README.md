# CS918 Natural Language Processing - Sentiment Analysis Project

This project explores machine learning and deep learning approaches for Twitter sentiment analysis, classifying tweets as **positive**, **negative**, or **neutral**.

## 📋 Overview

We implement and compare traditional ML models (Naive Bayes, Logistic Regression, SVM) and deep learning models (Bi-LSTM, Bi-LSTM with Attention, BERT). The workflow covers data preprocessing, EDA, model training, and evaluation.

## 🗂️ Structure

```
CS918-natural-language-processing/
├── README.md
├── requirements.txt
├── solution.ipynb
├── data/
│   ├── glove.6B.100d.txt
│   ├── twitter-training-data.txt
│   ├── twitter-dev-data.txt
│   ├── twitter-test1.txt
│   ├── twitter-test2.txt
│   └── twitter-test3.txt
├── models_weights/
│   ├── naive_bayes_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── svm_model.joblib
│   ├── lstm_model.pt
│   ├── lstm_with_attention_model.pt
│   ├── bert_raw_tweets.pt
│   └── bert_cleaned_tweets.pt
└── scripts/
    ├── __init__.py
    ├── data_loading_utils.py
    ├── model_training_utils.py
    ├── models.py
    ├── plotting_utilities.py
    ├── text_preprocessing_utils.py
    └── tweet_data_set.py
```

## 📊 Dataset

- **Training**: 45,101 tweets (46% neutral, 35% positive, 18% negative)
- **Development**: 2,000 tweets (similar distribution)

## 🔧 Preprocessing

Pipeline includes normalization, mention/URL removal, hashtag and emoji handling, slang/contraction expansion, tokenization (NLTK TweetTokenizer), and cleaning.

## 🤖 Models

- **Naive Bayes**: TF-IDF, best CV score 0.615
- **Logistic Regression**: TF-IDF/Count n-grams, accuracy 0.63
- **SVM**: TF-IDF, accuracy 0.63
- **Bi-LSTM**: GloVe embeddings, 2 layers, bidirectional
- **Bi-LSTM + Attention**: Adds self-attention
- **BERT**: bert-base-uncased, tested on raw and cleaned tweets

## 📈 Results (Validation)

| Model                | Precision | Recall | F1   | Acc  |
|----------------------|-----------|--------|------|------|
| Naive Bayes          | 0.62      | 0.56   | 0.58 | 0.62 |
| Logistic Regression  | 0.63      | 0.63   | 0.62 | 0.63 |
| SVM                  | 0.64      | 0.63   | 0.61 | 0.63 |

- **Class imbalance**: Negative class recall is lowest.
- **Neutral class**: Highest performance due to size.
- **Traditional ML and DL**: Comparable results.

## 📊 EDA

- Tweet length by sentiment
- N-gram and word cloud analysis
- UMAP visualization

## 🚀 Getting Started

1. Clone repo & install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download NLTK data:
   ```python
   import nltk; nltk.download('stopwords')
   ```
3. Run `solution.ipynb` in Jupyter.

## 📦 Key Dependencies

- pandas, numpy, nltk, ekphrasis, emoji, contractions
- scikit-learn, joblib
- PyTorch, torchtext, transformers
- matplotlib, seaborn, wordcloud, umap-learn

See [requirements.txt](requirements.txt) for details.

## 📝 Modules

- `models.py`: LSTM, LSTM+Attention, BERT architectures
- `text_preprocessing_utils.py`: Cleaning, normalization
- `data_loading_utils.py`: Data and embedding loaders
- `model_training_utils.py`: Training, metrics, plots
- `tweet_data_set.py`: PyTorch datasets

---

**Note**: This project is part of an academic assignment for CS918 at the University of Warwick.
