import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Analyzer & Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .prediction-card {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00a6c4;
    }
    .metric-good {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .metric-moderate {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .metric-poor {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Advanced Stock Price Analyzer & Predictor")
st.markdown("**Powered by Random Forest Machine Learning** - Advanced ensemble learning for accurate price predictions")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Stock symbols dictionary
stock_options = {
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Berkshire Hathaway (BRK-B)": "BRK-B",
    "JPMorgan Chase (JPM)": "JPM",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Visa (V)": "V",
    "Walmart (WMT)": "WMT"
}

# Stock selection
selected_stock = st.sidebar.selectbox(
    "Select Stock",
    options=list(stock_options.keys()),
    index=0
)
stock_symbol = stock_options[selected_stock]

# Date range selection
st.sidebar.markdown("### Time Frame Analysis")
end_date = datetime.now()
start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime('2015-01-01'),
    max_value=end_date
)

# Model settings
st.sidebar.markdown("### Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees", min_value=50, max_value=300, value=100, step=10)
max_depth = st.sidebar.slider("Max Tree Depth", min_value=5, max_value=30, value=10, step=1)
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=30, value=20, step=5) / 100

# Prediction horizon settings
st.sidebar.markdown("### Prediction Horizon")
prediction_unit = st.sidebar.selectbox(
    "Prediction Unit",
    options=["Weeks", "Months"],
    index=1
)

if prediction_unit == "Weeks":
    prediction_value = st.sidebar.slider(
        "Number of Weeks to Predict",
        min_value=1,
        max_value=52,
        value=4,
        step=1
    )
    trading_days = prediction_value * 5
    horizon_text = f"{prediction_value} Week{'s' if prediction_value > 1 else ''}"
else:
    prediction_value = st.sidebar.slider(
        "Number of Months to Predict",
        min_value=1,
        max_value=24,
        value=3,
        step=1
    )
    trading_days = prediction_value * 21
    horizon_text = f"{prediction_value} Month{'s' if prediction_value > 1 else ''}"

# Load data with caching
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start, end):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        if df.empty:
            st.error(f"No data found for {symbol}")
            return None
        df = df[['Close', 'Open', 'High', 'Low', 'Volume']]
        df.columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        return df
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return None

# Create features for machine learning
def create_features(df, lookback_days=30):
    """Create features for time series prediction"""
    data = df.copy()
    
    # Price-based features
    for i in range(1, lookback_days + 1):
        data[f'Close_lag_{i}'] = data['Close'].shift(i)
        data[f'Volume_lag_{i}'] = data['Volume'].shift(i)
    
    # Moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Price returns
    data['returns_1d'] = data['Close'].pct_change()
    data['returns_5d'] = data['Close'].pct_change(5)
    data['returns_10d'] = data['Close'].pct_change(10)
    
    # Volatility
    data['volatility'] = data['returns_1d'].rolling(window=20).std()
    
    # Price range
    data['price_range'] = (data['High'] - data['Low']) / data['Close']
    
    # Volume features
    data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
    
    # Drop NaN values
    data = data.dropna()
    
    return data

# Train Random Forest model
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10):
    """Train Random Forest Regressor"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# Predict future values
def predict_future(model, last_data, scaler, days_ahead, feature_columns):
    """Generate multi-step future predictions"""
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(days_ahead):
        # Prepare features for prediction
        features = pd.DataFrame([current_data], columns=feature_columns)
        next_pred = model.predict(features)[0]
        predictions.append(next_pred)
        
        # Update the data for next iteration
        # Shift all lag features
        for i in range(29, 0, -1):
            current_data[f'Close_lag_{i+1}'] = current_data[f'Close_lag_{i}']
        current_data['Close_lag_1'] = next_pred
        
        # Update moving averages (simplified)
        current_data['MA5'] = (current_data['Close_lag_1'] + current_data['Close_lag_2'] + 
                               current_data['Close_lag_3'] + current_data['Close_lag_4'] + 
                               current_data['Close_lag_5']) / 5
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    dummy_array = np.zeros((len(predictions), 1))
    dummy_array[:, 0] = predictions.flatten()
    predictions_scaled = scaler.inverse_transform(dummy_array)[:, 0]
    
    return predictions_scaled

# Main execution
# Load data
with st.spinner(f"Loading {selected_stock} data..."):
    df = load_stock_data(stock_symbol, start_date, end_date)

if df is not None and not df.empty:
    # Display basic information
    st.header(f"📊 {selected_stock} Stock Analysis")
    st.markdown(f"**Period:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    st.markdown(f"**Data Points:** {len(df)} trading days")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    with col2:
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252] * 100) if len(df) > 252 else 0
        st.metric("1-Year Change", f"{price_change:.1f}%")
    with col3:
        st.metric("Average Daily Volume", f"{df['Volume'].mean():,.0f}")
    with col4:
        st.metric("52-Week Range", f"${df['Low'].tail(252).min():.2f} - ${df['High'].tail(252).max():.2f}")
    
    # Interactive candlestick chart
    st.subheader("📈 Interactive Price Chart")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', showlegend=False
    ), row=1, col=1)
    
    # Volume chart
    colors = ['red' if df['Close'].iloc[i] < df['Close'].iloc[i-1] else 'green' for i in range(1, len(df))]
    colors.insert(0, 'grey')
    
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume', showlegend=False), row=2, col=1)
    
    fig.update_layout(title=f'{selected_stock} - Candlestick Chart with Volume', yaxis_title='Price ($)', 
                     yaxis2_title='Volume', xaxis_title='Date', template='plotly_white', height=600, hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving averages toggle
    show_ma = st.checkbox("Show Moving Averages (20-day & 50-day)", value=True)
    if show_ma:
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue', width=1)))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), mode='lines', name='20-day MA', line=dict(color='orange', width=2)))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), mode='lines', name='50-day MA', line=dict(color='red', width=2)))
        fig_ma.update_layout(title='Price with Moving Averages', xaxis_title='Date', yaxis_title='Price ($)', template='plotly_white', height=500)
        st.plotly_chart(fig_ma, use_container_width=True)
    
    # Model Training Section
    st.header("🤖 Random Forest Price Prediction")
    st.markdown(f"Train machine learning model to predict stock prices for the next **{horizon_text}**")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        train_model = st.button("🚀 Train Model & Generate Predictions", type="primary")
    with col2:
        lookback_days = st.selectbox("Lookback Days", options=[10, 20, 30, 45, 60], index=2)
    
    if train_model:
        with st.spinner("Training Random Forest Model... This may take a minute..."):
            try:
                # Prepare data with features
                df_features = create_features(df, lookback_days=lookback_days)
                
                # Define feature columns (exclude target and date index)
                exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
                feature_columns = [col for col in df_features.columns if col not in exclude_cols]
                target_column = 'Close'
                
                # Prepare X and y
                X = df_features[feature_columns]
                y = df_features[target_column]
                
                # Scale the target variable
                scaler = MinMaxScaler()
                y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
                
                # Split data chronologically
                split_idx = int(len(X) * (1 - test_size))
                X_train = X[:split_idx]
                X_test = X[split_idx:]
                y_train = y_scaled[:split_idx]
                y_test = y_scaled[split_idx:]
                
                # Train model
                model = train_random_forest(X_train, y_train, n_estimators, max_depth)
                
                # Make predictions
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
                mae = mean_absolute_error(y_actual, y_pred)
                r2 = r2_score(y_actual, y_pred)
                mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
                
                # Display metrics
                st.subheader("📊 Model Performance Metrics")
                
                # R² score indicator
                if r2 > 0.85:
                    r2_class = "metric-good"
                    r2_message = "🏆 EXCELLENT! Model captures price patterns with high accuracy"
                elif r2 > 0.7:
                    r2_class = "metric-good"
                    r2_message = "✅ Good - Model shows strong predictive capability"
                elif r2 > 0.5:
                    r2_class = "metric-moderate"
                    r2_message = "⚠️ Moderate - Some predictive capability"
                else:
                    r2_class = "metric-poor"
                    r2_message = "❌ Poor - Market may be highly volatile"
                
                st.markdown(f"""
                <div class="{r2_class}" style="margin-bottom: 20px;">
                    <h4>Model Accuracy Assessment</h4>
                    <p><b>R² Score:</b> {r2:.4f}</p>
                    <p>{r2_message}</p>
                    <p style="font-size: 12px; margin-top: 10px;">💡 R² > 0.85 indicates excellent predictive accuracy.</p>
                </div>
                """, unsafe_allow_html=True)
                
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with met_col2:
                    st.metric("MAE", f"${mae:.2f}")
                with met_col3:
                    st.metric("R² Score", f"{r2:.4f}")
                with met_col4:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # Plot actual vs predicted
                test_dates = df_features.index[split_idx:]
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=test_dates, y=y_actual, mode='lines', name='Actual', line=dict(color='blue', width=2)))
                fig_pred.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', name='Predicted', line=dict(color='red', width=2, dash='dash')))
                fig_pred.update_layout(title='Model Predictions vs Actual Prices', xaxis_title='Date', yaxis_title='Price ($)', template='plotly_white', height=500)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                        title='Top 15 Most Important Features',
                                        template='plotly_white', height=500)
                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Generate future predictions
                st.subheader(f"🔮 {horizon_text} Price Prediction")
                st.markdown(f"Predicting stock prices for the next {horizon_text} (approximately {trading_days} trading days)")
                
                with st.spinner("Generating future predictions..."):
                    # Get last data point
                    last_data_dict = {}
                    for col in feature_columns:
                        if col in df_features.columns:
                            last_data_dict[col] = df_features[col].iloc[-1]
                    
                    # Generate predictions
                    future_prices = predict_future(model, last_data_dict, scaler, trading_days, feature_columns)
                    
                    # Create future dates
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=trading_days, freq='B')
                    
                    # Calculate confidence intervals (using prediction error)
                    std_dev = np.std(y_actual - y_pred)
                    z_score = 1.96  # 95% confidence interval
                    
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_prices,
                        'Lower Bound': future_prices - (z_score * std_dev),
                        'Upper Bound': future_prices + (z_score * std_dev)
                    })
                    
                    # Plot predictions
                    fig_future = go.Figure()
                    
                    # Historical data
                    historical_days = min(252, len(df))
                    fig_future.add_trace(go.Scatter(x=df.index[-historical_days:], y=df['Close'].tail(historical_days),
                                                    mode='lines', name='Historical', line=dict(color='blue', width=2)))
                    
                    # Predictions
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Price'],
                                                    mode='lines', name='ML Prediction', line=dict(color='red', width=2)))
                    
                    # Confidence interval
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Upper Bound'],
                                                    mode='lines', name='95% Upper Bound',
                                                    line=dict(color='rgba(255,0,0,0)', width=0), showlegend=True))
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Lower Bound'],
                                                    mode='lines', name='95% Lower Bound',
                                                    fill='tonexty', line=dict(color='rgba(255,0,0,0)', width=0), showlegend=True))
                    
                    fig_future.update_layout(title=f'{selected_stock} - {horizon_text} Price Prediction',
                                            xaxis_title='Date', yaxis_title='Price ($)', template='plotly_white',
                                            height=600, hovermode='x unified')
                    st.plotly_chart(fig_future, use_container_width=True)
                    
                    # Prediction summary
                    st.markdown("### 📈 Prediction Summary")
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        final_price = future_df['Predicted Price'].iloc[-1]
                        current_price = df['Close'].iloc[-1]
                        total_return = ((final_price - current_price) / current_price) * 100
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>Target Price</h4>
                            <h2>${final_price:.2f}</h2>
                            <p>Expected Return: <b style="color: {'green' if total_return > 0 else 'red'}">{total_return:+.1f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col2:
                        max_price = future_df['Predicted Price'].max()
                        max_idx = future_df['Predicted Price'].idxmax()
                        max_date = future_df.loc[max_idx, 'Date'].strftime('%Y-%m-%d')
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>Peak Price</h4>
                            <h2>${max_price:.2f}</h2>
                            <p>Expected on: {max_date}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col3:
                        min_price = future_df['Predicted Price'].min()
                        min_idx = future_df['Predicted Price'].idxmin()
                        min_date = future_df.loc[min_idx, 'Date'].strftime('%Y-%m-%d')
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>Minimum Price</h4>
                            <h2>${min_price:.2f}</h2>
                            <p>Expected on: {min_date}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download predictions
                    csv_pred = future_df.to_csv(index=False)
                    st.download_button(label="📥 Download Predictions (CSV)", data=csv_pred,
                                      file_name=f"{stock_symbol}_predictions.csv", mime="text/csv")
                
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.info("Try reducing the lookback days or prediction horizon.")
    
    # Download historical data
    st.header("💾 Download Historical Data")
    csv_hist = df.to_csv()
    st.download_button(label="Download Historical Data (CSV)", data=csv_hist,
                      file_name=f"{stock_symbol}_historical_data.csv", mime="text/csv")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Why Random Forest for Stock Prediction?**
    - **Ensemble Learning**: Combines multiple decision trees for better accuracy
    - **Handles Non-linearity**: Captures complex market patterns without overfitting
    - **Feature Importance**: Identifies which indicators most influence prices
    - **Robust**: Less sensitive to outliers and noisy financial data
    
    **Model Features:**
    - Multiple lag features for time series patterns
    - Technical indicators (moving averages, volatility)
    - Volume analysis and price ranges
    - Feature importance ranking
    
    **Performance Expectation:**
    - R² > 0.85: Excellent predictive accuracy
    - R² 0.70-0.85: Good predictive capability  
    - R² < 0.70: High market volatility or insufficient data
    
    **Disclaimer:** This is for educational purposes only. Stock market predictions are inherently uncertain.
    """)
    
else:
    st.error("Failed to load data. Please check your internet connection and try again.")
