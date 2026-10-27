# Smart Complaint Management System

An AI/ML-based Smart Complaint Management System that uses Natural Language Processing (NLP) and Machine Learning to automatically analyze and classify complaints.

## 📌 Project Description

The system accepts user complaints and automatically predicts:

- Complaint Category
- Priority
- Appropriate Department

It also generates a unique Complaint ID and allows administrators to track and update complaint status through an Admin Dashboard.

## 🚀 Features

- Submit complaints online
- AI/ML-based complaint classification
- Category prediction
- Priority prediction
- Department prediction
- Unique Complaint ID generation
- Complaint status tracking
- Admin Login
- Admin Dashboard
- Complaint statistics and charts
- SQLite database for storing complaints

## 🛠️ Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- SQLite
- Pandas
- Matplotlib

## 🤖 Machine Learning Models

The project uses TF-IDF and Logistic Regression for classification.

### Model Accuracy

- Category Prediction: **84.62%**
- Priority Prediction: **69.23%**
- Department Prediction: **65.38%**

## 📂 Project Structure

```text
Smart Complaint System/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── complaints.csv
│
├── models/
│   ├── category_model.py
│   ├── priority_model.py
│   └── department_model.py
│
└── database/
    └── database.py
