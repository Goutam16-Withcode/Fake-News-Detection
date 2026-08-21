# 🛡️ VeriFact AI - Fake News & Credibility Detection Platform

A comprehensive Machine Learning and Natural Language Processing (NLP) system built to detect fake news, clickbait propaganda, and unverified rumors in real time. The platform features an interactive, modern **Streamlit Web Application** alongside a complete **Jupyter Notebook pipeline** (`FAKE__NEWS_Detection.ipynb`).

---

## 🌟 Key Features

### 1. 🔍 Real-Time News & Headline Classifier
- **Authenticity Rating (0–100%)**: Instant classification of news articles into **🟢 Authentic**, **🚨 Fabricated**, or **⚠️ Sensationalist**.
- **Interactive Plotly Authenticity Gauge**: Color-coded veracity meter with real-time confidence scores.
- **Sensationalism Index & Linguistic Markers**: Analyzes capitalization ratio, exclamation mark density, and emotional trigger words.
- **1-Click Preset Library**: Pre-loaded test cases for genuine international trade reports, space science discoveries, political conspiracies, and miracle health cures.

### 2. 🔍 Linguistic Trigger Word Heatmap
- **Red Highlights**: Highlights clickbait, sensational, and panic-inducing trigger terms (`shocking`, `leaked`, `secret`, `outlaw`, `conspiracy`).
- **Green Highlights**: Highlights factual reporting indicators (`reuters`, `according to`, `spokesperson`, `published`, `officials`).

### 3. 📊 Interactive Analytics & Plotly Graphs
- **Dataset Class Balance**: Donut chart comparing Fake News (23,481) vs True News (21,417).
- **Subject Distribution**: Bar chart breakdown by category (Politics, World News, Government, etc.).
- **Training & Validation Convergence**: Line chart tracking accuracy across training epochs.
- **Confusion Matrix Heatmap**: Interactive evaluation metric chart.

### 4. 📁 Batch CSV News Analyzer
- Drag-and-drop CSV file uploader for bulk article classification.
- Interactive table view of veracity verdicts and authenticity percentages.
- 1-click CSV report export button.

### 5. ⚖️ Side-by-Side News Comparison Tool
- Compare two articles or rumors simultaneously with dual progress bars and stance indicators.

### 6. 🎨 Ultra-Modern Light Mode UI
- Built with a clean slate background (`#f8fafc`), elevated white cards, sapphire gradient headers (`#0284c7`), and custom CSS.

---

## 🛠️ Technology Stack

- **Machine Learning & NLP**: Scikit-Learn (`TfidfVectorizer`, `PassiveAggressiveClassifier`), NLTK, Gensim, WordCloud, TensorFlow / Keras (`BiLSTM` + `Conv1D`).
- **Frontend & Visualization**: Streamlit, Plotly (`plotly.express`, `plotly.graph_objects`), Pandas, NumPy, Custom HTML/CSS.
- **Dataset**: Kaggle Fake News Detection Dataset (`emineyetm/fake-news-detection-datasets`).

---

## 📁 Project Directory Structure

```text
e:\Fake-News-Detection\
│
├── FAKE__NEWS_Detection.ipynb   # Jupyter Notebook (EDA, Tokenization, BiLSTM & ML Model Training)
├── app.py                      # Production Streamlit Web Application (In-Memory ML Engine & Light UI)
├── train_model.py              # ML Model Training & Export Script
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
└── .streamlit/
    └── config.toml             # Streamlit Light Mode Design Theme Configuration
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Goutam16-Withcode/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```
Open your browser and navigate to **[http://localhost:8501](http://localhost:8501)**.

### 4. Run the Jupyter Notebook
```bash
jupyter notebook FAKE__NEWS_Detection.ipynb
```

---

## 📊 Dataset Overview

The project is trained on the Kaggle Fake News Detection Dataset containing **44,898 news articles**:
- **Fake News**: 23,481 articles
- **True News**: 21,417 articles
- **Subjects**: Politics, World News, Government News, US News, Middle East News.

---

## 👨‍💻 Author

**Goutam16-Withcode**
- **GitHub**: [Goutam16-Withcode](https://github.com/Goutam16-Withcode)
- **Email**: goutam.sharrma@gmail.com

---
*Built with ❤️ for Machine Learning and NLP Research.*
