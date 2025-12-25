# Phishing Website Detection using Machine Learning

## 🔐 Project Overview
This project detects phishing websites using Machine Learning techniques.
It analyzes URLs and classifies them as **Phishing** or **Legitimate**, and also identifies the **type of phishing attack**.

## 🚀 Features
- URL-based phishing detection
- Attack type classification (Credential Phishing, Brand Impersonation, etc.)
- Machine Learning models (Random Forest, XGBoost)
- GUI-based interface for real-time analysis
- Offline batch URL prediction support

## 🧠 Machine Learning Models Used
- Logistic Regression
- Random Forest
- XGBoost

## 🖥️ GUI
A desktop GUI built using **Tkinter** allows users to:
- Enter a URL
- Analyze phishing probability
- View attack type and final verdict

## 📂 Project Structure
phish-detect/
│── data/
│── models/
│── results/
│── gui_detector.py
│── phishing_detection.py
│── step1_environment_and_abstract.py
│── step2_feature_engineering.py
│── step3_merge_datasets.py
│── step4_train_with_attacktypes.py
│── step5_integrate_predict.py
│── requirements.txt
│── README.md
│── .gitignore
