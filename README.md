# Norwich House Price Forecasting Web Application using MLOps

This project is a web application that forecasts house prices in Norwich using an ensemble of machine learning and deep learning models. The ensemble combines Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), Convolutional Neural Networks (CNN), and XGBoost to leverage complementary modelling strengths. Models are trained on historical data from January 2016 to October 2022, and out-of-sample forecasts run from November 2022 through December 2025. The trained ensemble is served via a Flask web interface that accepts feature inputs and returns predicted prices.

- Live demo: https://norwich-house-prices-prediction.onrender.com/
- Online article: https://medium.com/@muadeyus/end-to-end-mlops-deployment-of-the-norwich-house-price-predictor-71d6bd57c16f


## Project Structure

```
norwich-flask-app
├── app
│   ├── __init__.py          # Initializes the Flask application
│   ├── main.py              # Entry point for the application
│   ├── routes.py            # Defines the routes for the web application
│   ├── utils.py             # Utility functions for model loading and predictions
│   ├── model                # Contains the trained model and related files
│   │   ├── model.pkl  # Saved Keras model
│   │   ├── scaler.pkl       # Saved StandardScaler for input features
│   │   └── feature_columns.pkl  # List of feature columns used in the model
│   ├── templates            # HTML templates for the web interface
│   │   └── index.html       # Main HTML file for user input
│   └── static               # Static files (CSS, JS)
│       ├── css
│       │   └── styles.css   # CSS styles for the web interface
│       └── js
│           └── app.js       # JavaScript for handling user interactions
├── notebooks                 # Jupyter notebooks for model training
│   ├── norwich.ipynb        # Notebook containing the model training code
│   └── house_prices.csv     # Data
├── tests                     # Unit tests for the application
│   └── test_app.py          # Test cases for the Flask application
├── Dockerfile                # Instructions for building the Docker image
├── docker-compose.yml        # Defines services for Docker deployment
├── requirements.txt          # Python dependencies for the application
├── .dockerignore             # Files to ignore when building the Docker image
├── run.sh                   # Shell script to run the application
└── README.md                 # Documentation for the project
```

---

## Objectives

- Build robust, production-ready models for house-price prediction using time-series and tabular approaches.
- Combine sequence models (RNN, LSTM, CNN) with a gradient-boosted tree (XGBoost) in an ensemble to improve accuracy and resilience.
- Produce out-of-sample forecasts for November 2022 — December 2025.
- Deploy the ensemble as a Flask application with a user-friendly front end for feature input and prediction retrieval.

---

## Data

- **Training timeframe:** January 2016 — October 2022.
- **Forecast horizon:** November 2022 — December 2025.
- **Typical features:** temporal variables; property attributes (size, rooms, age); location encodings; engineered features from price history and external indicators.
- **Preprocessing steps:** missing-value handling; scaling/normalisation of numeric features; encoding categorical variables; constructing sequence windows for neural models.

---

## Modeling Approach

### RNN and LSTM
- Capture temporal dependencies in price series and lagged indicators.
- Inputs prepared as sliding windows to model short- and medium-term dynamics.

### CNN (1D)
- Extract local temporal patterns from sliding windows using 1D convolutions.
- Detect short-lived patterns and time-local feature interactions.

### XGBoost
- Model non-linear relationships in tabular features.
- Provide a robust baseline and strong performance on engineered features.

### Ensemble Strategy
- Combine model outputs via weighted averaging or a stacking meta-learner.
- Select weights or stacking model using time-aware cross-validation to avoid look-ahead bias.

---

## Evaluation

- Use time-series aware holdout and cross-validation schemes.
- **Primary metrics:** Root Mean Squared Error (RMSE), Goodness-of-fit (R-squared).
- Backtest across multiple historical windows to assess stability and generalisation for the November 2022 — December 2025 horizon.

---

## Deployment

- Export trained models and integrate them into a Flask application.
- Flask provides an API endpoint that accepts feature inputs (JSON or form data) and returns ensemble predictions.
- The web interface allows users to enter required features and view predicted prices.
- **Production considerations:** model artifact serialisation, input validation, logging, performance monitoring, scaling, and containerisation.

---

## Usage

1. Start the Flask application or visit the hosted demo at https://norwich-house-prices-prediction.onrender.com/.
2. Open the web interface in a browser.
3. Provide property features and any required temporal context.
4. Submit to receive a point forecast.

---