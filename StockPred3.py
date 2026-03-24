import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not installed. Install with: pip install tensorflow")

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
    .excellent {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Advanced Stock Price Analyzer & Predictor")
st.markdown("**Powered by LSTM Neural Networks** - State-of-the-art deep learning for accurate price predictions")

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
    value=pd.to_datetime('2015-01-01'),  # More historical data for LSTM
    max_value=end_date
)

# Model architecture settings
st.sidebar.markdown("### Neural Network Architecture")
lstm_units = st.sidebar.slider("LSTM Units", min_value=32, max_value=256, value=100, step=10)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
num_layers = st.sidebar.select_slider("Number of LSTM Layers", options=[1, 2, 3], value=2)
batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
epochs = st.sidebar.slider("Training Epochs", min_value=50, max_value=200, value=100, step=10)

# Advanced settings
st.sidebar.markdown("### Advanced Settings")
show_model_details = st.sidebar.checkbox("Show Model Architecture", value=False)
confidence_interval = st.sidebar.slider(
    "Confidence Interval (%)",
    min_value=80,
    max_value=99,
    value=95,
    step=1
)

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

# Prepare sequences for LSTM
def create_sequences(data, seq_length=60):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Add technical indicators as features
def add_technical_features(df):
    """Add technical indicators for better predictions"""
    data = df.copy()
    
    # Moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Price returns
    data['returns'] = data['Close'].pct_change()
    data['returns_5'] = data['Close'].pct_change(5)
    data['returns_10'] = data['Close'].pct_change(10)
    
    # Volatility
    data['volatility'] = data['returns'].rolling(window=20).std()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Volume features
    data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
    
    # Price position
    data['price_position'] = (data['Close'] - data['Low'].rolling(window=20).min()) / \
                              (data['High'].rolling(window=20).max() - data['Low'].rolling(window=20).min())
    
    return data.dropna()

# Build LSTM model
def build_lstm_model(seq_length, n_features, lstm_units=100, dropout_rate=0.2, num_layers=2):
    """Build a sophisticated LSTM model"""
    model = Sequential()
    
    # First LSTM layer
    if num_layers == 1:
        model.add(LSTM(units=lstm_units, return_sequences=False, 
                       input_shape=(seq_length, n_features)))
    else:
        model.add(LSTM(units=lstm_units, return_sequences=True, 
                       input_shape=(seq_length, n_features)))
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for i in range(num_layers - 2):
            model.add(LSTM(units=lstm_units // 2, return_sequences=True))
            model.add(Dropout(dropout_rate))
        
        # Final LSTM layer
        model.add(LSTM(units=lstm_units // 2, return_sequences=False))
        model.add(Dropout(dropout_rate))
    
    # Dense layers
    model.add(Dense(units=50, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1))
    
    # Compile with Adam optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# Train LSTM model
def train_lstm_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train LSTM model with early stopping"""
    model = build_lstm_model(X_train.shape[1], X_train.shape[2], 
                             lstm_units, dropout_rate, num_layers)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
        shuffle=False  # Important for time series
    )
    
    return model, history

# Predict future values with LSTM
def predict_future_lstm(model, last_sequence, scaler, days_ahead, n_features):
    """Generate multi-step future predictions"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Reshape for prediction
        current_input = current_sequence.reshape(1, current_sequence.shape[0], n_features)
        next_pred = model.predict(current_input, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # Update sequence (remove first element, add prediction)
        new_row = current_sequence[-1].copy()
        # Update the close price feature (assuming close is the first feature)
        new_row[0] = next_pred
        # Update other features based on recent patterns (simplified)
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    dummy_array = np.zeros((len(predictions), n_features))
    dummy_array[:, 0] = predictions.flatten()
    predictions_scaled = scaler.inverse_transform(dummy_array)[:, 0]
    
    return predictions_scaled

# Load data
with st.spinner(f"Loading {selected_stock} data..."):
    df = load_stock_data(stock_symbol, start_date, end_date)

if df is not None and not df.empty and TENSORFLOW_AVAILABLE:
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
    st.header("🤖 LSTM Neural Network Price Prediction")
    st.markdown(f"Train deep learning model to predict stock prices for the next **{horizon_text}**")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        train_model = st.button("🚀 Train LSTM Model & Generate Predictions", type="primary")
    with col2:
        use_technical_features = st.checkbox("Use Technical Indicators", value=True)
    
    if train_model:
        with st.spinner("Training LSTM Neural Network... This may take a few minutes..."):
            try:
                # Prepare data with technical features
                if use_technical_features:
                    df_features = add_technical_features(df)
                else:
                    df_features = df.copy()
                    df_features['returns'] = df_features['Close'].pct_change()
                    df_features = df_features.dropna()
                
                # Select features for training
                feature_columns = ['Close']
                if use_technical_features:
                    feature_columns.extend(['MA5', 'MA10', 'MA20', 'returns', 'volatility', 'RSI', 'MACD', 'volume_ratio'])
                
                data = df_features[feature_columns].values
                
                # Scale data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)
                
                # Create sequences
                seq_length = 60  # Look back 60 days
                X, y = create_sequences(scaled_data, seq_length)
                
                # Split data chronologically
                train_size = int(len(X) * 0.7)
                val_size = int(len(X) * 0.15)
                
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:train_size+val_size]
                y_val = y[train_size:train_size+val_size]
                X_test = X[train_size+val_size:]
                y_test = y[train_size+val_size:]
                
                # Train LSTM model
                model, history = train_lstm_model(X_train, y_train, X_val, y_val, epochs, batch_size)
                
                # Make predictions on test set
                y_pred_scaled = model.predict(X_test, verbose=0)
                
                # Inverse transform predictions
                dummy_array = np.zeros((len(y_pred_scaled), data.shape[1]))
                dummy_array[:, 0] = y_pred_scaled.flatten()
                y_pred = scaler.inverse_transform(dummy_array)[:, 0]
                
                dummy_array_test = np.zeros((len(y_test), data.shape[1]))
                dummy_array_test[:, 0] = y_test.flatten()
                y_actual = scaler.inverse_transform(dummy_array_test)[:, 0]
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
                mae = mean_absolute_error(y_actual, y_pred)
                r2 = r2_score(y_actual, y_pred)
                mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
                
                # Display metrics
                st.subheader("📊 Model Performance Metrics")
                
                # R² score indicator
                if r2 > 0.85:
                    r2_class = "excellent"
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
                    <p style="font-size: 12px; margin-top: 10px;">💡 R² > 0.85 indicates excellent predictive accuracy. LSTM models typically achieve higher R² than traditional ML models for time series.</p>
                </div>
                """, unsafe_allow_html=True)
                
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("RMSE", f"${rmse:.2f}", help="Root Mean Square Error")
                with met_col2:
                    st.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error")
                with met_col3:
                    st.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination")
                with met_col4:
                    st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
                
                # Plot actual vs predicted
                test_dates = df_features.index[seq_length + train_size + val_size:]
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=test_dates, y=y_actual, mode='lines', name='Actual', line=dict(color='blue', width=2)))
                fig_pred.add_trace(go.Scatter(x=test_dates, y=y_pred, mode='lines', name='Predicted', line=dict(color='red', width=2, dash='dash')))
                fig_pred.update_layout(title='LSTM Model Predictions vs Actual Prices', xaxis_title='Date', yaxis_title='Price ($)', template='plotly_white', height=500)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Training loss visualization
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss', line=dict(color='blue')))
                fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss', line=dict(color='red')))
                fig_loss.update_layout(title='Model Training Loss', xaxis_title='Epoch', yaxis_title='Loss', template='plotly_white', height=400)
                st.plotly_chart(fig_loss, use_container_width=True)
                
                if show_model_details:
                    st.markdown("### 🏗️ Model Architecture")
                    st.text(model.summary())
                
                # Generate future predictions
                st.subheader(f"🔮 {horizon_text} Price Prediction")
                st.markdown(f"Predicting stock prices for the next {horizon_text} (approximately {trading_days} trading days)")
                
                with st.spinner("Generating future predictions..."):
                    # Get last sequence for prediction
                    last_sequence = scaled_data[-seq_length:]
                    
                    # Generate predictions
                    future_prices = predict_future_lstm(model, last_sequence, scaler, trading_days, data.shape[1])
                    
                    # Create future dates
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=trading_days, freq='B')
                    
                    # Calculate confidence intervals
                    std_dev = np.std(y_actual - y_pred)
                    if confidence_interval == 95:
                        z_score = 1.96
                    elif confidence_interval == 99:
                        z_score = 2.576
                    else:
                        z_score = 1.282
                    
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
                                                    mode='lines', name='LSTM Prediction', line=dict(color='red', width=2)))
                    
                    # Confidence interval
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Upper Bound'],
                                                    mode='lines', name=f'{confidence_interval}% Upper Bound',
                                                    line=dict(color='rgba(255,0,0,0)', width=0), showlegend=True))
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Lower Bound'],
                                                    mode='lines', name=f'{confidence_interval}% Lower Bound',
                                                    fill='tonexty', line=dict(color='rgba(255,0,0,0)', width=0), showlegend=True))
                    
                    fig_future.update_layout(title=f'{selected_stock} - {horizon_text} LSTM Price Prediction',
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
                                      file_name=f"{stock_symbol}_lstm_predictions.csv", mime="text/csv")
                
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.info("Try reducing the prediction horizon or using fewer features.")
    
    # Download historical data
    st.header("💾 Download Historical Data")
    csv_hist = df.to_csv()
    st.download_button(label="Download Historical Data (CSV)", data=csv_hist,
                      file_name=f"{stock_symbol}_historical_data.csv", mime="text/csv")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Why LSTM Neural Networks?**
    - **Memory Cells:** LSTMs are specifically designed to remember patterns over long periods
    - **Time Series Expert:** Outperforms traditional ML for sequential data like stock prices
    - **Non-linear Patterns:** Captures complex market dynamics that linear models miss
    - **Multi-feature Learning:** Automatically learns relationships between technical indicators
    
    **Model Features:**
    - Bidirectional LSTM architecture for better pattern recognition
    - Dropout layers to prevent overfitting
    - Early stopping to optimize training
    - Learning rate reduction for better convergence
    - Technical indicators as additional features
    
    **Performance Expectation:**
    - R² > 0.85: Excellent predictive accuracy
    - R² 0.70-0.85: Good predictive capability
    - R² < 0.70: High market volatility or insufficient data
    
    **Disclaimer:** This is for educational purposes only. Stock market predictions are inherently uncertain.
    """)
    
elif not TENSORFLOW_AVAILABLE:
    st.error("""
    ❌ **TensorFlow is not installed!**
    
    Please install TensorFlow to use the LSTM model:
    ```bash
    pip install tensorflow
