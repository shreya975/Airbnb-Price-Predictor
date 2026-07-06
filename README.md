# 🏡 AI Airbnb Price Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange?style=for-the-badge&logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Dashboard-3F4F75?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An AI-powered web application that predicts the optimal nightly price of Airbnb listings using Machine Learning. The platform analyzes property details, location, room type, availability, and host activity to provide accurate pricing recommendations through a premium Streamlit dashboard.

---

# 🌐 Live Demo

## 🚀 Try the Application

👉 https://airbnb-price-predictor-cxfvk8zny5lyudxcxje25g.streamlit.app/

---

# 📌 Project Overview

Setting the right Airbnb price is essential for maximizing occupancy and revenue. Pricing too high may reduce bookings, while pricing too low results in lost income.

This project uses Machine Learning to estimate the ideal nightly rental price for Airbnb properties based on real New York City Airbnb listings. The application combines predictive analytics with an elegant Streamlit interface to help hosts make data-driven pricing decisions.

---

# ✨ Features

✅ AI-powered Airbnb Price Prediction

✅ Premium Interactive Dashboard

✅ Real-time Price Estimation

✅ Property Value Analysis

✅ Market Demand Score

✅ Booking Potential Analysis

✅ AI Confidence Score

✅ Location-based Pricing

✅ Interactive Analytics

✅ Prediction History

✅ Download Prediction Report

---

# 📊 Dataset

**Dataset Used**

New York City Airbnb Open Dataset (2019)

The dataset contains thousands of Airbnb listings across New York City with property, host, and booking information.

### Features Include

- Neighbourhood Group
- Neighbourhood
- Latitude
- Longitude
- Room Type
- Minimum Nights
- Number of Reviews
- Reviews Per Month
- Host Listings Count
- Availability (365 Days)

### Target Variable

- Nightly Price (USD)

---

# 🧹 Data Preprocessing

The dataset is cleaned and prepared before training the Machine Learning model.

### Steps

- Removed unnecessary columns
- Missing value handling
- Median imputation
- Outlier removal using IQR
- One-Hot Encoding
- Feature Engineering
- Train-Test Split

---

# 🤖 Machine Learning Model

The project uses

## Random Forest Regressor

The model learns complex relationships between property characteristics and market prices to estimate accurate nightly rental values. :contentReference[oaicite:0]{index=0}

---

# 📈 Project Workflow

```
Airbnb Listing Details
          │
          ▼
Data Cleaning
          │
          ▼
Feature Engineering
          │
          ▼
One-Hot Encoding
          │
          ▼
Random Forest Regressor
          │
          ▼
Price Prediction
          │
          ▼
Market Analysis
          │
          ▼
Interactive Dashboard
```

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Machine Learning

- Scikit-Learn
- Random Forest Regressor

## Data Processing

- Pandas
- NumPy

## Visualization

- Plotly

## Deployment

- Streamlit

---

# 📂 Project Structure

```
Airbnb-Price-Predictor/

│── app.py
│── AB_NYC_2019.csv
│── Airbnb_price_prediction.ipynb
│── airbnb_price_model.pkl
│── feature_names.pkl
│── requirements.txt
│── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/shreya975/Airbnb-Price-Predictor.git
```

## Navigate into Project

```bash
cd Airbnb-Price-Predictor
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

---

# 📊 Input Parameters

The prediction model considers multiple property-related features including:

- Neighbourhood Group
- Neighbourhood
- Latitude
- Longitude
- Room Type
- Minimum Nights
- Number of Reviews
- Reviews Per Month
- Host Listing Count
- Availability (365 Days)

---

# 📈 Dashboard Features

The application provides:

- 🏡 Airbnb Price Prediction
- 📊 AI Confidence Score
- 💰 Estimated Nightly Price
- 📈 Market Demand Analysis
- ⭐ Booking Potential Score
- 🏘️ Property Value Insights
- 📉 Price Distribution Analysis
- 📊 Interactive Visualizations
- 📋 AI Recommendations
- 📄 Prediction History Download

---

# 💻 Application Screens

## Home Dashboard

(Add Screenshot Here)

---

## Prediction Dashboard

(Add Screenshot Here)

---

## Analytics Dashboard

(Add Screenshot Here)

---

# 🎯 Future Improvements

- XGBoost Regressor
- LightGBM Model
- Explainable AI (SHAP)
- Seasonal Price Forecasting
- Dynamic Pricing Engine
- Google Maps Integration
- REST API Deployment
- Multi-city Support
- Revenue Optimization Dashboard

---

# 🚀 Deployment

The application is deployed using **Streamlit Cloud**.

### Live Application

👉 https://airbnb-price-predictor-cxfvk8zny5lyudxcxje25g.streamlit.app/

---

# 📈 Business Impact

This application helps Airbnb hosts to:

- Set competitive nightly prices
- Improve occupancy rates
- Maximize rental revenue
- Analyze market trends
- Make data-driven pricing decisions

---

# 👩‍💻 Author

## Shreya Mahajan

### GitHub

https://github.com/shreya975

### LinkedIn

https://www.linkedin.com/in/shreya-mahajan-b38b28385/

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork this repository and submit a Pull Request.

---
