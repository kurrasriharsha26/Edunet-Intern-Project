# 🌍 AI-Driven Climate Risk Prediction & Mitigation Framework


This repository contains the implementation of Edunet foundation AI GreenSkills Internship project, AI-Driven Climate Risk Prediction and Mitigation Framework, which predicts weather conditions and classifies climate risks using AI-powered LSTM models.


This project is an **AI-powered climate risk prediction system** that predicts potential climate risks such as temperature extremes, rainfall, and humidity fluctuations, and provides suggested mitigation strategies. It includes a **Streamlit web application** for interactive usage.

---

## 🛠 Tools & Technologies

- **Programming Language:** Python 3.x  
- **Frameworks & Libraries:** 
  - Streamlit (Web App)  
  - Pandas, NumPy (Data manipulation)  
  - Scikit-learn (Machine Learning)  
  - Plotly (Data visualization)  
  - Joblib (Model saving/loading)  
- **IDE:** Jupyter Notebook / VS Code  

---

## ⚡ Features

- Generates a **synthetic climate dataset** (or you can use your own dataset).  
- Trains a **RandomForestRegressor** to predict climate risk index.  
- **Predict climate risk** based on input parameters (Temperature, Rainfall, Humidity).  
- Displays **recommended mitigation strategies**.  
- **Interactive data visualization** with Plotly.  
- Animated gradient **background for better UI**.

- streamlit run app.py

🧠 Usage

Open the Streamlit app in your browser.

Preview the synthetic climate dataset.

Select the target variable for prediction (default is ClimateRiskIndex).

Click Train Model to train the RandomForestRegressor.

Input your own Temperature, Rainfall, and Humidity values to predict climate risk.

View predicted climate risk and mitigation strategies.

Explore interactive plots to visualize climate trends.

📝 Notes

The dataset is synthetic, but you can replace it with real climate data.

The ML model can be replaced or retrained as needed.

The visualization uses Plotly with a light/dark theme for better clarity.
