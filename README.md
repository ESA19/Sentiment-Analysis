# Sentiment Analysis with BERT

This project focuses on using BERT (Bidirectional Encoder Representations from Transformers) to perform sentiment analysis on the IMDb movie reviews dataset. The BERT model is a state-of-the-art deep learning model that utilizes the transformer architecture to understand the context of words in a sentence, enabling it to determine sentiment with high accuracy.

## Overview

BERT is a transformer-based model where each output is connected to every input element, and the weights between them are dynamically computed based on their relationships. This model uses bidirectional reading, which sets it apart from previous language models. It allows BERT to understand the meaning of words based on the context of surrounding words in the sentence.

BERT has been trained in an unsupervised manner on the English Wikipedia and Brown Corpus using unlabeled text, allowing it to develop a deep understanding of the language's structure.

## Dataset

The dataset used for training and evaluating the model is the [IMDb movie reviews dataset](https://huggingface.co/datasets/stanfordnlp/imdb). It contains movie reviews labeled with sentiment (positive/negative), making it suitable for supervised training.

## Implementation

The implementation is based on the code found in [PyTorch Sentiment Analysis by ben trevett](https://github.com/bentrevett/pytorch-sentiment-analysis?tab=readme-ov-file). You can access the Colab notebook for a detailed walk-through of the implementation [here](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/4%20-%20Transformers.ipynb).

## Model Details

- **Algorithm**: BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture**: Transformer-based, leveraging attention mechanisms for bidirectional context reading.
- **Training Data**: IMDb movie reviews (accessed via [Hugging Face Datasets](https://huggingface.co/datasets/stanfordnlp/imdb))

BERT is capable of understanding the sentiment of each word in a sentence by using the surrounding context. This contextual understanding enables BERT to determine whether a review is positive or negative with higher accuracy than traditional NLP models.


