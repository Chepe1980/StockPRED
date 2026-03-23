import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf  # Replace pandas_datareader with yfinance
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
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Analyzer & Future Predictor")
st.markdown("Analyze historical stock data and get AI-powered 2-year price predictions")

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
    "Walmart (WMT)": "WMT",
    "Procter & Gamble (PG)": "PG",
    "Coca-Cola (KO)": "KO"
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
    value=pd.to_datetime('2020-01-01'),
    max_value=end_date
)

# Advanced settings
st.sidebar.markdown("### Advanced Settings")
show_model_details = st.sidebar.checkbox("Show Model Details", value=False)
confidence_interval = st.sidebar.slider(
    "Confidence Interval (%)",
    min_value=80,
    max_value=99,
    value=95,
    step=1
)

# Load data with caching - using yfinance instead of pandas_datareader
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

# Prepare features for XGBoost
def prepare_features(data, lags=30):
    """Prepare features with technical indicators"""
    df = data.copy()
    
    # Price-based features
    for i in range(1, min(lags + 1, len(df) - 1)):
        df[f'lag_{i}'] = df['Close'].shift(i)
    
    # Rolling statistics
    for window in [5, 10, 20, 50]:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
    
    # Price changes
    df['daily_return'] = df['Close'].pct_change()
    df['daily_volatility'] = df['daily_return'].rolling(window=20).std()
    
    # Technical indicators
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volume features
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Train XGBoost model
def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Split training data for validation
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

# Generate future predictions
def predict_future(model, last_data, days_ahead=730):
    """Predict future prices for specified number of days (2 years ≈ 730 trading days)"""
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(days_ahead):
        # Prepare features for prediction
        features = current_data[-1:].drop('Close', axis=1)
        next_pred = model.predict(features)[0]
        predictions.append(next_pred)
        
        # Update current_data for next prediction
        new_row = current_data.iloc[-1:].copy()
        new_row['Close'] = next_pred
        
        # Update lag features
        for i in range(1, 31):
            if f'lag_{i}' in new_row.columns:
                if i == 1:
                    new_row[f'lag_{i}'] = current_data['Close'].iloc[-1]
                else:
                    new_row[f'lag_{i}'] = current_data[f'lag_{i-1}'].iloc[-1] if i-1 <= len(current_data) else current_data['Close'].iloc[-1]
        
        current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    return np.array(predictions)

# Load data
with st.spinner(f"Loading {selected_stock} data..."):
    df = load_stock_data(stock_symbol, start_date, end_date)

if df is not None and not df.empty:
    # Display basic information
    st.header(f"📊 {selected_stock} Stock Analysis")
    st.markdown(f"**Period:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    with col2:
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252] * 100) if len(df) > 252 else 0
        st.metric("1-Year Change", f"{price_change:.1f}%", 
                 delta=f"{price_change:.1f}%")
    with col3:
        st.metric("Average Daily Volume", f"{df['Volume'].mean():,.0f}")
    with col4:
        st.metric("52-Week Range", f"${df['Low'].tail(252).min():.2f} - ${df['High'].tail(252).max():.2f}")
    
    # Interactive candlestick chart with Plotly
    st.subheader("📈 Interactive Price Chart")
    
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05,
                        row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        showlegend=False
    ), row=1, col=1)
    
    # Volume chart
    colors = ['red' if df['Close'].iloc[i] < df['Close'].iloc[i-1] else 'green' 
              for i in range(1, len(df))]
    colors.insert(0, 'grey')
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color=colors,
        name='Volume',
        showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        title=f'{selected_stock} - Candlestick Chart with Volume',
        yaxis_title='Price ($)',
        yaxis2_title='Volume',
        xaxis_title='Date',
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving averages toggle
    show_ma = st.checkbox("Show Moving Averages (20-day & 50-day)", value=True)
    if show_ma:
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'], 
                                    mode='lines', name='Close Price',
                                    line=dict(color='blue', width=1)))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(),
                                    mode='lines', name='20-day MA',
                                    line=dict(color='orange', width=2)))
        fig_ma.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(),
                                    mode='lines', name='50-day MA',
                                    line=dict(color='red', width=2)))
        fig_ma.update_layout(title='Price with Moving Averages',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            template='plotly_white',
                            height=500)
        st.plotly_chart(fig_ma, use_container_width=True)
    
    # Model Training Section
    st.header("🤖 AI Price Prediction Model")
    st.markdown("Train XGBoost model to predict future stock prices for the next 2 years")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        train_model = st.button("🚀 Train Model & Predict Next 2 Years", type="primary")
    with col2:
        use_advanced_features = st.checkbox("Use Advanced Technical Indicators", value=True)
    
    if train_model:
        with st.spinner("Training XGBoost model... This may take a few moments..."):
            try:
                # Prepare data
                if use_advanced_features:
                    df_features = prepare_features(df)
                else:
                    # Simple features only
                    df_features = df.copy()
                    for i in range(1, 31):
                        df_features[f'lag_{i}'] = df_features['Close'].shift(i)
                    df_features = df_features.dropna()
                
                # Split data
                split_date = df_features.index[int(len(df_features) * 0.8)]
                train_data = df_features[df_features.index < split_date]
                test_data = df_features[df_features.index >= split_date]
                
                # Prepare features
                feature_cols = [col for col in train_data.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
                X_train = train_data[feature_cols]
                y_train = train_data['Close']
                X_test = test_data[feature_cols]
                y_test = test_data['Close']
                
                # Train model
                model = train_xgboost_model(X_train.values, y_train.values)
                
                # Make predictions on test set
                y_pred = model.predict(X_test.values)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Display metrics
                st.subheader("📊 Model Performance Metrics")
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with met_col2:
                    st.metric("MAE", f"${mae:.2f}")
                with met_col3:
                    st.metric("R² Score", f"{r2:.3f}")
                with met_col4:
                    st.metric("MAPE", f"{np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.2f}%")
                
                # Plot actual vs predicted
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=test_data.index, y=y_test,
                                              mode='lines', name='Actual',
                                              line=dict(color='blue', width=2)))
                fig_pred.add_trace(go.Scatter(x=test_data.index, y=y_pred,
                                              mode='lines', name='Predicted',
                                              line=dict(color='red', width=2, dash='dash')))
                fig_pred.update_layout(title='Model Predictions vs Actual Prices',
                                       xaxis_title='Date',
                                       yaxis_title='Price ($)',
                                       template='plotly_white',
                                       height=500)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                if show_model_details:
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'feature': feature_cols[:30],  # Limit to top 30
                        'importance': model.feature_importances_[:30]
                    }).sort_values('importance', ascending=True)
                    
                    fig_imp = go.Figure(go.Bar(
                        x=importance_df['importance'],
                        y=importance_df['feature'],
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    fig_imp.update_layout(title='Top 30 Feature Importance',
                                         xaxis_title='Importance',
                                         yaxis_title='Features',
                                         template='plotly_white',
                                         height=600)
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Generate 2-year prediction
                st.subheader("🔮 2-Year Price Prediction")
                st.markdown(f"Predicting stock prices for the next 2 years (approximately 730 trading days)")
                
                with st.spinner("Generating future predictions..."):
                    # Prepare last data point for prediction
                    last_data = df_features.tail(1).copy()
                    future_prices = predict_future(model, last_data, days_ahead=730)
                    
                    # Create future dates (trading days only approximation)
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                 periods=730, 
                                                 freq='B')  # Business days
                    
                    # Create prediction dataframe
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_prices
                    })
                    
                    # Calculate confidence intervals
                    std_dev = np.std(y_test - y_pred)
                    z_score = 1.96 if confidence_interval == 95 else (2.576 if confidence_interval == 99 else 1.282)
                    future_df['Upper Bound'] = future_df['Predicted Price'] + (z_score * std_dev)
                    future_df['Lower Bound'] = future_df['Predicted Price'] - (z_score * std_dev)
                    
                    # Plot predictions
                    fig_future = go.Figure()
                    
                    # Historical data
                    fig_future.add_trace(go.Scatter(x=df.index[-252:], 
                                                    y=df['Close'].tail(252),
                                                    mode='lines', 
                                                    name='Historical (Last Year)',
                                                    line=dict(color='blue', width=2)))
                    
                    # Predictions with confidence interval
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], 
                                                    y=future_df['Predicted Price'],
                                                    mode='lines', 
                                                    name='Predicted Price',
                                                    line=dict(color='red', width=2, dash='dash')))
                    
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], 
                                                    y=future_df['Upper Bound'],
                                                    mode='lines', 
                                                    name=f'{confidence_interval}% Upper Bound',
                                                    line=dict(color='rgba(255,0,0,0.3)', width=0),
                                                    showlegend=True))
                    
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], 
                                                    y=future_df['Lower Bound'],
                                                    mode='lines', 
                                                    name=f'{confidence_interval}% Lower Bound',
                                                    fill='tonexty',
                                                    line=dict(color='rgba(255,0,0,0.3)', width=0),
                                                    showlegend=True))
                    
                    fig_future.update_layout(title=f'{selected_stock} - 2-Year Price Prediction',
                                            xaxis_title='Date',
                                            yaxis_title='Price ($)',
                                            template='plotly_white',
                                            height=600,
                                            hovermode='x unified')
                    
                    st.plotly_chart(fig_future, use_container_width=True)
                    
                    # Key prediction metrics
                    st.markdown("### 📈 Prediction Summary")
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    
                    with pred_col1:
                        final_price = future_df['Predicted Price'].iloc[-1]
                        current_price = df['Close'].iloc[-1]
                        total_return = ((final_price - current_price) / current_price) * 100
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>2-Year Target</h4>
                            <h2>${final_price:.2f}</h2>
                            <p>Expected Return: <b style="color: {'green' if total_return > 0 else 'red'}">{total_return:+.1f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col2:
                        max_price = future_df['Predicted Price'].max()
                        max_date = future_df.loc[future_df['Predicted Price'].idxmax(), 'Date'].strftime('%Y-%m-%d')
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>Peak Price</h4>
                            <h2>${max_price:.2f}</h2>
                            <p>Expected on: {max_date}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_col3:
                        min_price = future_df['Predicted Price'].min()
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>Minimum Price</h4>
                            <h2>${min_price:.2f}</h2>
                            <p>During prediction period</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download predictions
                    csv_pred = future_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv_pred,
                        file_name=f"{stock_symbol}_2year_predictions.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.info("Try using fewer features or increasing the amount of historical data.")
    
    # Download historical data
    st.header("💾 Download Historical Data")
    csv_hist = df.to_csv()
    st.download_button(
        label="Download Historical Data (CSV)",
        data=csv_hist,
        file_name=f"{stock_symbol}_historical_data.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This app uses machine learning for educational purposes only. 
    Stock market predictions are inherently uncertain. Always do your own research before making investment decisions.
    """)
    
else:
    st.error("Failed to load data. Please check your internet connection and try again.")
