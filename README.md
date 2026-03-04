# Fake News Detection

A machine learning project to detect fake news using deep learning and natural language processing (NLP) techniques.

## Overview

This project implements a classification model to distinguish between fake and genuine news articles. It uses TensorFlow/Keras for building neural networks with LSTM layers and text preprocessing techniques including tokenization, lemmatization, and removal of stopwords.

## Features

- **Data Processing**: Loads and preprocesses news datasets from Kaggle
- **Text Preprocessing**: 
  - Tokenization
  - Stopword removal
  - Lemmatization and stemming
  - Text cleaning and normalization
- **Feature Engineering**: Word embeddings and sequence padding
- **Deep Learning Models**:
  - LSTM (Long Short-Term Memory) networks
  - Bidirectional LSTM
  - Convolutional Neural Networks (CNN) for text
  - Dense neural networks with dropout for regularization

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- NLTK
- Gensim
- Matplotlib
- Seaborn
- WordCloud
- kagglehub

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Goutam16-Withcode/Fake-News-Detection.git
cd Fake-News-Detection
```

2. Install required packages:
```bash
pip install tensorflow keras pandas numpy nltk gensim matplotlib seaborn wordcloud kagglehub
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
```

## Usage

Open and run the Jupyter notebook:
```bash
jupyter notebook FAKE__NEWS_Detection.ipynb
```

The notebook includes:
1. Data loading from Kaggle datasets
2. Exploratory data analysis
3. Text preprocessing and cleaning
4. Model training and evaluation
5. Visualization of results

## Dataset

The project uses the Fake News Detection Dataset from Kaggle, which contains labeled news articles classified as either fake or true news.

## Model Architecture

The project explores multiple neural network architectures:
- LSTM with embeddings
- Bidirectional LSTM for better context understanding
- CNN with pooling for feature extraction
- Hybrid models combining LSTM and CNN

## Author

**Goutam16-Withcode**

## Contact

Email: goutamshaarma001212@gmail.com
