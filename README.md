# 📰 Fake News Detector with Sentiment Analysis

This project aims to detect fake news headlines and analyze their sentiment using machine learning and natural language processing (NLP) techniques.

---

## 🔍 Project Overview

Fake news is a growing concern in the digital age. The goal of this project is to create a machine learning model that can classify news headlines based on their truthfulness. It also includes sentiment analysis to detect the tone of the headline (positive, negative, or neutral).

---

##  Key Features

- ✅ **Text Preprocessing**: Cleaned and vectorized news headlines using TF-IDF.
- 📊 **Model Training**: Trained multiple models (Naive Bayes, Decision Tree, Logistic Regression, etc.) and compared their performance.
- 🏆 **Best Model Selected**: Logistic Regression with 24% accuracy was chosen as the final model.
- 💬 **Sentiment Analysis**: Added sentiment polarity using `TextBlob`.
- 🌐 **Web App (Streamlit)**: Simple UI to input a news headline and predict:
  - Truthfulness label (e.g., *True*, *False*, *Pants on Fire!*, etc.)
  - Sentiment score (positive/negative/neutral)
  - Color-coded prediction and score

---

## 📁 Project Structure
fake-news-detector/
│
├── app.py                   # Streamlit web application
├── lr_model.pkl             # Trained logistic regression model
├── vectorizer.pkl           # TF-IDF vectorizer
├── model-evaluation.ipynb   # Jupyter notebook with EDA & model comparison
├── requirements.txt         # Python dependencies
├── Report-Findings.docx              # Final IEEE format report
└── README.md                # This file

---

## 📌 Labels Used

The labels used for classification are:

| Label Code | Label Name        |
|------------|-------------------|
| 0          | Barely True       |
| 1          | False             |
| 2          | Half True         |
| 3          | Mostly True       |
| 4          | Pants on Fire!    |
| 5          | True              |

---

## 🌐 Web App Features

- Enter a news headline and get a prediction from one of six categories:
  - Barely True
  - False
  - Half True
  - Mostly True
  - Pants on Fire!
  - True
- Sentiment score and tone (Positive, Negative, Neutral)
- Clean and minimal UI with color-coded predictions

---

## 🙌 Credits

Made with 💻, 🧠, and a bit of ☕ by Taqwa Rasheed  
Semester 6
AI [AI2002] Course

---

## ⚠️ Disclaimer

> This project is a **prototype created for academic purposes**. The current version may produce **inaccurate predictions** due to limited preprocessing and model tuning.  
> 
> 🛠️ In the future, I plan to:
> - Improve the data preprocessing pipeline (e.g., handling stopwords, lemmatization, named entity recognition).
> - Revisit feature engineering techniques.
> - Experiment with deep learning or transformer-based models for better accuracy.
>
> For now, please do not rely on this app for real-world use.

---

## 📎 License

This project is for academic purposes only.
