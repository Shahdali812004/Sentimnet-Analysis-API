# 🧠 Sentiment-Analysis-API

A lightweight and production-ready RESTful API for training and serving a sentiment analysis model. Built using FastAPI and Scikit-learn, this project allows users to train a text classification model, monitor its status, and perform both single and batch predictions.

---

## 🚀 Project Overview

This project provides an end-to-end pipeline for:

- Training a sentiment analysis model dynamically via API
- Serving predictions through REST endpoints
- Monitoring model status and evaluation metrics
- Securing endpoints using API key authentication

The system is designed to be simple, extensible, and suitable for real-world NLP applications.

---

## 🛠 Tech Stack

- **Backend Framework:** FastAPI  
- **Machine Learning:** Scikit-learn  
- **Model Pipeline:** TF-IDF + Logistic Regression  
- **Data Validation:** Pydantic  
- **Model Persistence:** Joblib  
- **Environment Management:** python-dotenv  
- **Concurrency:** Python Threading  

---

### Key Components:

- **API Layer (main.py)**  
  Handles routing, validation, and authentication.

- **Controller Layer (NLPTrainer)**  
  Manages training, prediction, and model lifecycle.

- **Schema Layer (Pydantic Models)**  
  Defines request and response structures.

- **Storage Layer**  
  Saves trained model and status as files.

---

## ✨ Features

- 🔐 API Key Authentication  
- 🧠 Train model dynamically via API  
- ⚡ Asynchronous training using background threads  
- 📊 Model evaluation with classification report  
- 🔍 Single text prediction  
- 📦 Batch prediction support  
- 📈 Model status tracking (Training / Ready / Not Trained)  
- 💾 Persistent model storage  

---

## 🧪 Testing

You can test the API using:

- **Swagger UI** (automatically available at `/docs`)
- **Postman**
- **cURL**

### Example Request (Train Model)

```json
POST /model/train
{
  "texts": ["I love this!", "This is bad"],
  "labels": ["positive", "negative"]
}
Sentiment-Analysis-API/
│
├── main.py                     # FastAPI entry point
│
├── src/
│   ├── controllers/
│   │   └── NLPController.py   # Core ML logic (training + prediction)
│   │
│   ├── models/
│   │   └── schemas.py         # Pydantic schemas
│   │
│   ├── helper/
│   │   └── config.py          # Environment & configuration
│   │
│   └── assets/
│       └── storage/           # Saved model & status files
│
└── .env                       # Environment variables

