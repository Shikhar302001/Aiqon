# 🚢 Port Congestion & Shipment Delay Prediction System

This repository contains a machine learning application designed to predict shipment delays and assess congestion risk at various ports. The project uses a Random Forest Regressor model trained on synthetic operational data. An interactive web interface, built with Streamlit, allows users to input different scenarios and receive real-time predictions.

This project was developed as a take-home assignment for the Aiqon AI Internship.

## ✨ Features

-   **Real-time Delay Prediction**: Predicts the potential arrival delay of vessels in hours based on operational parameters.
-   **Interactive Dashboard**: A user-friendly Streamlit interface to adjust inputs like vessel traffic, port capacity, weather, and more.
-   **Confidence Scoring**: Provides a confidence score for each prediction, derived from the variance of individual decision trees in the Random Forest model.
-   **Explainable AI (XAI)**: Lists the key factors contributing to the predicted delay, offering insights into why a prediction was made.

## ⚙️ How It Works

The system is built around a `RandomForestRegressor` model trained to predict the `arrival_delay_hrs`.

### Dataset

The model is trained on the `port_congestion_synthetic_data.csv` dataset, which includes the following features:
-   `vessel_arrival_count`: Number of vessels arriving.
-   `port_capacity`: The maximum number of vessels the port can handle.
-   `queue_length`: Number of vessels currently waiting.
-   `weather_condition`: Environmental conditions (e.g., Good, Moderate, Bad, Severe).
-   `historical_delay_hrs`: The average historical delay for similar conditions.
-   `port_name`: The specific port location.

### Model Performance

The model was trained and evaluated in the `congestion.ipynb` notebook, achieving the following performance metrics on the test set:
-   **Root Mean Squared Error (RMSE)**: 2.24 hours
-   **R² Score**: 0.75

## 🚀 Getting Started

To run the prediction application locally, follow these steps.

### Prerequisites

Ensure you have Python 3.8+ installed. You will also need to install the required libraries.

```bash
pip install streamlit pandas scikit-learn numpy
```

### Running the Application

1.  Clone the repository:
    ```bash
    git clone https://github.com/shikhar302001/Aiqon.git
    cd Aiqon
    ```
2.  Run the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## 📁 Repository Structure

```
.
├── app.py                      # Main Streamlit application file for the interactive UI
├── congestion.ipynb            # Jupyter Notebook for data exploration, model training, and evaluation
└── port_congestion_synthetic_data.csv # The synthetic dataset used for model training