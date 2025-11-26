🔮 Universal AI Price Predictor

A Time-Series Forecasting project that uses a Long Short-Term Memory (LSTM) neural network to predict Daily Log-Returns of financial assets (Crypto, Stocks, Commodities).

Unlike traditional price predictors, this model is trained on volatility and percentage moves, making it "Universal"—it can be applied to Bitcoin, Apple, or Gold without retraining.

🌟 Features

Universal Inference: Trained on BTC, but adapts to any asset using dynamic scaling.

Log-Return Forecasting: Predicts the change in price rather than the raw price for better stability.

Explainable AI (XAI): Uses Integrated Gradients (Captum) to visualize which market factors (RSI, Volume, etc.) influenced the prediction.

Interactive Dashboard: Built with Streamlit for real-time analysis and backtesting.

🛠️ Tech Stack

Core: PyTorch, NumPy, Pandas

Data: yfinance, pandas-ta

XAI: Captum

UI: Streamlit, Plotly

🚀 Getting Started

1. Install Dependencies

pip install -r requirements.txt


2. Train the Model (Optional)

The project comes with a pre-trained model, but you can retrain it:

Open notebooks/01_train_model.ipynb

Run all cells to generate models/lstm_v1.pth

3. Run the Dashboard

streamlit run app.py


📂 Project Structure

├── data/               # Data storage
├── models/             # Saved PyTorch models (.pth)
├── notebooks/          # Training & Research notebooks
├── src/                # Shared source code
│   ├── data_utils.py   # Data fetching & feature engineering
│   └── model_defs.py   # LSTM Architecture
├── app.py              # Streamlit Dashboard Entry Point
└── requirements.txt    # Python dependencies
