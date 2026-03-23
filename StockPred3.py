import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
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
    .metric-poor {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Analyzer & Predictor")
st.markdown("Analyze historical stock data and get AI-powered price predictions with adjustable time horizons")

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
st.sidebar.markdown("### Model Settings")
show_model_details = st.sidebar.checkbox("Show Model Details", value=False)
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
        max_value=52,  # 1 year max in weeks
        value=12,
        step=1
    )
    trading_days = prediction_value * 5  # Approximate trading days per week
    horizon_text = f"{prediction_value} Week{'s' if prediction_value > 1 else ''}"
else:  # Months
    prediction_value = st.sidebar.slider(
        "Number of Months to Predict",
        min_value=1,
        max_value=24,  # 2 years max
        value=6,
        step=1
    )
    trading_days = prediction_value * 21  # Approximate trading days per month
    horizon_text = f"{prediction_value} Month{'s' if prediction_value > 1 else ''}"

# Advanced model tuning
st.sidebar.markdown("### Advanced Model Tuning")
enable_grid_search = st.sidebar.checkbox("Enable Grid Search (Slower but Better)", value=False)
model_complexity = st.sidebar.select_slider(
    "Model Complexity",
    options=["Low", "Medium", "High"],
    value="Medium"
)

# Set model parameters based on complexity
if model_complexity == "Low":
    n_estimators = 100
    max_depth = 3
    learning_rate = 0.1
elif model_complexity == "Medium":
    n_estimators = 200
    max_depth = 5
    learning_rate = 0.05
else:  # High
    n_estimators = 300
    max_depth = 7
    learning_rate = 0.03

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

# Enhanced feature engineering
def prepare_features(data, lags=20, use_all_indicators=True):
    """Prepare features with comprehensive technical indicators"""
    df = data.copy()
    
    # Price-based features
    for i in range(1, min(lags + 1, len(df) - 1)):
        df[f'lag_{i}'] = df['Close'].shift(i)
    
    # Rolling statistics
    for window in [5, 10, 20, 50]:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'rolling_max_{window}'] = df['Close'].rolling(window=window).max()
        df[f'rolling_min_{window}'] = df['Close'].rolling(window=window).min()
    
    # Price changes and returns
    df['daily_return'] = df['Close'].pct_change()
    df['weekly_return'] = df['Close'].pct_change(5)
    df['monthly_return'] = df['Close'].pct_change(21)
    df['daily_volatility'] = df['daily_return'].rolling(window=20).std()
    
    # Price ratios
    df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
    df['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open']
    
    if use_all_indicators:
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
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume features
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        df['volume_price_trend'] = df['Volume'] * df['daily_return']
        
        # Price momentum
        df['momentum_5'] = df['Close'].pct_change(5)
        df['momentum_10'] = df['Close'].pct_change(10)
        df['momentum_20'] = df['Close'].pct_change(20)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Optimized XGBoost model training with time series cross-validation
def train_xgboost_model(X_train, y_train, enable_grid_search=False):
    base_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    if enable_grid_search and len(X_train) > 100:
        # Simplified grid search for better performance
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        st.info(f"Best parameters: {grid_search.best_params_}")
    else:
        # Simple train/validation split
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        
        model = base_model
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    return model

# Generate future predictions with improved dynamics
def predict_future(model, last_data, feature_cols, days_ahead, scaler=None):
    """Predict future prices with rolling updates"""
    predictions = []
    current_data = last_data.copy()
    
    for i in range(days_ahead):
        # Prepare features for prediction
        features = current_data[feature_cols].iloc[-1:].values
        if scaler:
            features = scaler.transform(features)
        next_pred = model.predict(features)[0]
        predictions.append(next_pred)
        
        # Create new row with updated features
        new_row = current_data.iloc[-1:].copy()
        new_row['Close'] = next_pred
        
        # Update lag features
        for lag in range(1, 21):
            if f'lag_{lag}' in new_row.columns:
                if lag == 1:
                    new_row[f'lag_{lag}'] = current_data['Close'].iloc[-1]
                else:
                    new_row[f'lag_{lag}'] = current_data[f'lag_{lag-1}'].iloc[-1] if lag-1 < len(current_data) else current_data['Close'].iloc[-1]
        
        # Update rolling statistics with improved logic
        for window in [5, 10, 20, 50]:
            if f'rolling_mean_{window}' in new_row.columns:
                recent_values = current_data['Close'].tail(window)
                new_row[f'rolling_mean_{window}'] = recent_values.mean()
                new_row[f'rolling_std_{window}'] = recent_values.std()
                new_row[f'rolling_max_{window}'] = recent_values.max()
                new_row[f'rolling_min_{window}'] = recent_values.min()
        
        # Update returns
        if len(current_data) > 1:
            prev_close = current_data['Close'].iloc[-2]
            new_row['daily_return'] = (next_pred - prev_close) / prev_close
            if len(current_data) >= 5:
                prev_close_5 = current_data['Close'].iloc[-5] if len(current_data) >= 5 else current_data['Close'].iloc[0]
                new_row['weekly_return'] = (next_pred - prev_close_5) / prev_close_5
            if len(current_data) >= 21:
                prev_close_21 = current_data['Close'].iloc[-21] if len(current_data) >= 21 else current_data['Close'].iloc[0]
                new_row['monthly_return'] = (next_pred - prev_close_21) / prev_close_21
        
        # Update other features
        if 'high_low_ratio' in new_row.columns:
            new_row['high_low_ratio'] = 0.02  # Assume moderate volatility
        if 'open_close_ratio' in new_row.columns:
            new_row['open_close_ratio'] = 0.01
        
        # Update technical indicators with simplified logic for speed
        if 'RSI' in new_row.columns:
            recent_closes = current_data['Close'].tail(14)
            gains = []
            losses = []
            for j in range(1, len(recent_closes)):
                change = recent_closes.iloc[j] - recent_closes.iloc[j-1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            if avg_loss == 0:
                new_row['RSI'] = 100
            else:
                rs = avg_gain / avg_loss
                new_row['RSI'] = 100 - (100 / (1 + rs))
        
        if 'MACD' in new_row.columns:
            recent_closes = current_data['Close'].tail(26)
            exp12 = recent_closes.ewm(span=12, adjust=False).mean().iloc[-1]
            exp26 = recent_closes.ewm(span=26, adjust=False).mean().iloc[-1]
            new_row['MACD'] = exp12 - exp26
            macd_values = current_data['MACD'].tail(9)
            new_row['MACD_signal'] = macd_values.mean() if len(macd_values) > 0 else new_row['MACD']
            new_row['MACD_histogram'] = new_row['MACD'] - new_row['MACD_signal']
        
        if 'BB_middle' in new_row.columns:
            recent_closes = current_data['Close'].tail(20)
            new_row['BB_middle'] = recent_closes.mean()
            bb_std = recent_closes.std()
            new_row['BB_upper'] = new_row['BB_middle'] + (bb_std * 2)
            new_row['BB_lower'] = new_row['BB_middle'] - (bb_std * 2)
            new_row['BB_width'] = (new_row['BB_upper'] - new_row['BB_lower']) / new_row['BB_middle']
        
        if 'ATR' in new_row.columns:
            new_row['ATR'] = current_data['ATR'].tail(14).mean() if len(current_data) >= 14 else 0.02 * next_pred
        
        # Update momentum
        if 'momentum_5' in new_row.columns and len(current_data) >= 5:
            prev_close_5 = current_data['Close'].iloc[-5]
            new_row['momentum_5'] = (next_pred - prev_close_5) / prev_close_5
        if 'momentum_10' in new_row.columns and len(current_data) >= 10:
            prev_close_10 = current_data['Close'].iloc[-10]
            new_row['momentum_10'] = (next_pred - prev_close_10) / prev_close_10
        if 'momentum_20' in new_row.columns and len(current_data) >= 20:
            prev_close_20 = current_data['Close'].iloc[-20]
            new_row['momentum_20'] = (next_pred - prev_close_20) / prev_close_20
        
        # Append new row
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
    
    # Interactive candlestick chart
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
    st.markdown(f"Train XGBoost model to predict stock prices for the next **{horizon_text}**")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        train_model = st.button("🚀 Train Model & Generate Predictions", type="primary")
    with col2:
        use_advanced_features = st.checkbox("Use Advanced Technical Indicators", value=True)
    
    if train_model:
        with st.spinner("Training XGBoost model... This may take a few moments..."):
            try:
                # Prepare data
                df_features = prepare_features(df, use_all_indicators=use_advanced_features)
                
                if len(df_features) < 100:
                    st.warning("Not enough historical data for reliable predictions. Consider selecting an earlier start date.")
                
                # Define feature columns
                exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
                feature_cols = [col for col in df_features.columns if col not in exclude_cols]
                
                # Split data using time series split
                split_idx = int(len(df_features) * 0.8)
                train_data = df_features.iloc[:split_idx]
                test_data = df_features.iloc[split_idx:]
                
                # Prepare features with optional scaling
                X_train = train_data[feature_cols].values
                y_train = train_data['Close'].values
                X_test = test_data[feature_cols].values
                y_test = test_data['Close'].values
                
                # Optional scaling for better performance
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model with grid search option
                model = train_xgboost_model(X_train_scaled, y_train, enable_grid_search)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Display metrics with visual indicators
                st.subheader("📊 Model Performance Metrics")
                
                # R² score indicator
                r2_color = "metric-good" if r2 > 0.7 else ("metric-poor" if r2 < 0.5 else "prediction-card")
                st.markdown(f"""
                <div class="{r2_color}" style="margin-bottom: 20px;">
                    <h4>Model Accuracy Assessment</h4>
                    <p>R² Score: <b>{r2:.3f}</b> - {'Good fit' if r2 > 0.7 else 'Moderate fit' if r2 > 0.5 else 'Poor fit'}</p>
                    <p>Higher R² values indicate better predictive accuracy. Values above 0.7 are considered good for stock prediction.</p>
                </div>
                """, unsafe_allow_html=True)
                
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with met_col2:
                    st.metric("MAE", f"${mae:.2f}")
                with met_col3:
                    st.metric("R² Score", f"{r2:.3f}")
                with met_col4:
                    st.metric("MAPE", f"{mape:.2f}%")
                
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
                
                # Residual plot
                residuals = y_test - y_pred
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(x=test_data.index, y=residuals,
                                              mode='markers', name='Residuals',
                                              marker=dict(color='purple', size=5)))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                fig_resid.update_layout(title='Prediction Residuals',
                                        xaxis_title='Date',
                                        yaxis_title='Residual ($)',
                                        template='plotly_white',
                                        height=400)
                st.plotly_chart(fig_resid, use_container_width=True)
                
                if show_model_details:
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'feature': feature_cols[:30],
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
                
                # Generate predictions based on selected horizon
                st.subheader(f"🔮 {horizon_text} Price Prediction")
                st.markdown(f"Predicting stock prices for the next {horizon_text} (approximately {trading_days} trading days)")
                
                with st.spinner("Generating future predictions..."):
                    # Prepare last data point for prediction
                    last_data = df_features.tail(1).copy()
                    
                    # Generate future predictions
                    future_prices = predict_future(model, last_data, feature_cols, trading_days, scaler)
                    
                    # Create future dates
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                 periods=trading_days, 
                                                 freq='B')
                    
                    # Create prediction dataframe
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_prices
                    })
                    
                    # Calculate confidence intervals
                    std_dev = np.std(residuals)
                    if confidence_interval == 95:
                        z_score = 1.96
                    elif confidence_interval == 99:
                        z_score = 2.576
                    else:
                        z_score = 1.282
                    
                    future_df['Upper Bound'] = future_df['Predicted Price'] + (z_score * std_dev)
                    future_df['Lower Bound'] = future_df['Predicted Price'] - (z_score * std_dev)
                    
                    # Plot predictions
                    fig_future = go.Figure()
                    
                    # Historical data (last 6 months for context)
                    historical_days = min(126, len(df))  # 6 months of trading days
                    fig_future.add_trace(go.Scatter(x=df.index[-historical_days:], 
                                                    y=df['Close'].tail(historical_days),
                                                    mode='lines', 
                                                    name='Historical',
                                                    line=dict(color='blue', width=2)))
                    
                    # Predictions with confidence interval
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], 
                                                    y=future_df['Predicted Price'],
                                                    mode='lines', 
                                                    name='Predicted Price',
                                                    line=dict(color='red', width=2)))
                    
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], 
                                                    y=future_df['Upper Bound'],
                                                    mode='lines', 
                                                    name=f'{confidence_interval}% Upper Bound',
                                                    line=dict(color='rgba(255,0,0,0)', width=0),
                                                    showlegend=True))
                    
                    fig_future.add_trace(go.Scatter(x=future_df['Date'], 
                                                    y=future_df['Lower Bound'],
                                                    mode='lines', 
                                                    name=f'{confidence_interval}% Lower Bound',
                                                    fill='tonexty',
                                                    line=dict(color='rgba(255,0,0,0)', width=0),
                                                    showlegend=True))
                    
                    fig_future.update_layout(title=f'{selected_stock} - {horizon_text} Price Prediction',
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
                    
                    # Additional metrics for longer horizons
                    if trading_days > 60:
                        st.markdown("### 📊 Key Milestones")
                        milestone_col1, milestone_col2 = st.columns(2)
                        
                        with milestone_col1:
                            # 3-month prediction
                            quarter_days = min(63, len(future_df))
                            quarter_price = future_df['Predicted Price'].iloc[quarter_days-1] if quarter_days > 0 else final_price
                            quarter_return = ((quarter_price - current_price) / current_price) * 100
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>3-Month Outlook</h4>
                                <p>Target: <b>${quarter_price:.2f}</b></p>
                                <p>Expected Return: <b style="color: {'green' if quarter_return > 0 else 'red'}">{quarter_return:+.1f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with milestone_col2:
                            # 6-month prediction
                            half_year_days = min(126, len(future_df))
                            half_year_price = future_df['Predicted Price'].iloc[half_year_days-1] if half_year_days > 0 else final_price
                            half_year_return = ((half_year_price - current_price) / current_price) * 100
                            st.markdown(f"""
                            <div class="prediction-card">
                                <h4>6-Month Outlook</h4>
                                <p>Target: <b>${half_year_price:.2f}</b></p>
                                <p>Expected Return: <b style="color: {'green' if half_year_return > 0 else 'red'}">{half_year_return:+.1f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Download predictions
                    csv_pred = future_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv_pred,
                        file_name=f"{stock_symbol}_{horizon_text.lower().replace(' ', '_')}_predictions.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.info("Try reducing the prediction horizon or using fewer features.")
    
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
    
    **Model Information:** The XGBoost model uses technical indicators and price patterns to generate predictions. 
    R² scores above 0.7 indicate good predictive accuracy, while lower scores suggest high market volatility.
    """)
    
else:
    st.error("Failed to load data. Please check your internet connection and try again.")
