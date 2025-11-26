import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients

# Import our modules
from src.model_defs import CryptoLSTM
from src.data_utils import fetch_crypto_data, add_technical_indicators

# --- PAGE CONFIG ---
st.set_page_config(page_title="Universal XAI Trader (Batch Model)", layout="wide")

st.title("🔮 Universal AI Price Predictor (Batch Trained)")
st.markdown("""
This dashboard uses the **Batch-Trained LSTM** to predict **Daily Log-Returns**. 
It dynamically adapts to any asset by scaling live data to match the model's training distribution.
""")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker Symbol", value="BTC-USD", help="Try AAPL, NVDA, ETH-USD")
    lookback = st.slider("Lookback Window (Days)", 30, 90, 30) # Default to 30 to match training
    
    if st.button("Run Analysis"):
        st.session_state['run'] = True

# --- MAIN LOGIC ---
if st.session_state.get('run', False):
    
    # 1. LOAD MODEL
    try:
        # Initialize model architecture (MUST MATCH TRAINING NOTEBOOK)
        # UPDATED: Changed num_layers to 2 to match your saved checkpoint
        model = CryptoLSTM(input_dim=7, hidden_dim=128, num_layers=2, output_dim=1)
        
        # Load weights
        model.load_state_dict(torch.load('models/lstm_v1.pth'))
        model.eval()
        
    except FileNotFoundError:
        st.error("❌ Model file 'models/lstm_v1.pth' not found. Please run the training notebook first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    # 2. FETCH DATA
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # Fetch enough data for lookback + backtesting
            raw_df = fetch_crypto_data(ticker, period="2y", interval="1d")
            
            # Process Features (Log Returns, RSI, MACD...)
            # df_features has the 7 columns we trained on
            # df_close has the raw prices we need for conversion
            df_features, df_close = add_technical_indicators(raw_df)
            
            feature_cols = df_features.columns.tolist()
            data_values = df_features.values
            
            # --- DYNAMIC SCALING ---
            # Use StandardScaler because that is what we trained with!
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_values)
            
        except Exception as e:
            st.error(f"Error fetching/processing data: {e}")
            st.stop()

    # Tabs
    tab1, tab2 = st.tabs(["🔮 Forecast & Explanation", "📉 Accuracy Check"])

    # ==========================================
    # TAB 1: LIVE PREDICTION
    # ==========================================
    with tab1:
        if len(scaled_data) < lookback:
            st.error("Not enough data for the requested lookback.")
        else:
            # Prepare latest sequence
            last_sequence = scaled_data[-lookback:]
            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
            input_tensor.requires_grad = True

            # Predict (Output is Scaled Log Return)
            with torch.no_grad():
                pred_scaled = model(input_tensor)

            # Inverse Scale
            # Create dummy row with 7 columns (0th is Log_Ret)
            dummy_row = np.zeros((1, 7))
            dummy_row[0, 0] = pred_scaled.item()
            
            # Inverse transform using the scaler fitted on THIS asset
            pred_log_ret = scaler.inverse_transform(dummy_row)[0, 0]

            # CONVERT LOG-RETURN TO PRICE
            # Price_Next = Price_Today * exp(Log_Return)
            current_price = df_close.iloc[-1]
            pred_price = current_price * np.exp(pred_log_ret)
            
            delta = pred_price - current_price
            delta_pct = (np.exp(pred_log_ret) - 1) * 100
            
            # Display Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:,.2f}")
            col2.metric("AI Predicted Price", f"${pred_price:,.2f}", f"{delta:,.2f} ({delta_pct:.2f}%)")
            col3.metric("AI Predicted Return", f"{pred_log_ret:.5f}")

            # Explainability (Captum)
            st.divider()
            st.subheader("🧠 Why this prediction?")
            st.markdown("Using **Integrated Gradients** to analyze the last 30 days.")
            
            ig = IntegratedGradients(model)
            # Baseline: Zero tensor (Neutral return assumption)
            baseline = torch.zeros_like(input_tensor) 
            attributions, _ = ig.attribute(input_tensor, baseline, return_convergence_delta=True)
            
            attr_matrix = attributions.squeeze().detach().numpy()
            attr_df = pd.DataFrame(attr_matrix, columns=feature_cols)
            attr_df.index = df_features.index[-lookback:]

            # Heatmap
            fig_heat = go.Figure(data=go.Heatmap(
                z=attr_df.T.values, 
                x=attr_df.index, 
                y=attr_df.columns,
                colorscale='RdBu', 
                zmid=0
            ))
            fig_heat.update_layout(title="Factor Influence (Red=Bearish, Blue=Bullish)", height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

    # ==========================================
    # TAB 2: BACKTEST (REALITY CHECK)
    # ==========================================
    with tab2:
        st.subheader("📊 Model vs Reality (Last 60 Days)")
        
        backtest_days = 60
        required_len = backtest_days + lookback
        
        if len(scaled_data) >= required_len:
            # Slice data for backtest
            subset = scaled_data[-required_len:]
            X_back = []
            
            # Create batches
            for i in range(backtest_days):
                seq = subset[i : i+lookback]
                X_back.append(seq)
            
            X_back = torch.FloatTensor(np.array(X_back))
            
            # Batch Prediction
            with torch.no_grad():
                y_pred_scaled = model(X_back).squeeze().numpy()
            
            # Inverse Transform
            dummy_preds = np.zeros((backtest_days, 7))
            dummy_preds[:, 0] = y_pred_scaled
            pred_log_rets = scaler.inverse_transform(dummy_preds)[:, 0]
            
            # RECONSTRUCT PRICE PATH
            # Get Close prices aligned with the prediction days
            # If we predict t=1 using t=0 data, we apply return to Price[0]
            
            # Actual Prices for the backtest period
            dates_for_plot = df_features.index[-backtest_days:]
            actual_prices = df_close.values[-backtest_days:]
            
            # Previous Prices (to apply returns to)
            prev_prices = df_close.values[-(backtest_days+1):-1]
            
            # Calculate Predicted Prices
            ai_prices = prev_prices * np.exp(pred_log_rets)
            
            # Plot
            fig_back = go.Figure()
            fig_back.add_trace(go.Scatter(x=dates_for_plot, y=actual_prices, name="Actual Price", line=dict(color='gray', width=2)))
            fig_back.add_trace(go.Scatter(x=dates_for_plot, y=ai_prices, name="AI Prediction", line=dict(color='blue', width=2, dash='dot')))
            
            fig_back.update_layout(title=f"Backtest Accuracy: {ticker}", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_back, use_container_width=True)
            
            # Directional Accuracy
            # Did the AI predict the right SIGN? (Up vs Down)
            actual_returns = df_features['Log_Ret'].values[-backtest_days:]
            correct_direction = np.sign(pred_log_rets) == np.sign(actual_returns)
            accuracy = np.mean(correct_direction) * 100
            
            st.metric("Directional Accuracy (Last 60 Days)", f"{accuracy:.1f}%")
            
        else:
            st.warning("Not enough data history to run backtest.")

else:
    st.info("👈 Enter ticker settings and click **Run Analysis**.")