import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
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
st.title("📈 Stock Price Analyzer & Predictor")
st.markdown("Analyze historical stock data and get AI-powered price predictions with improved time series forecasting")

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
    value=pd.to_datetime('2018-01-01'),
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

# Cross-validation settings
st.sidebar.markdown("### Cross-Validation")
cv_folds = st.sidebar.slider(
    "Time Series CV Folds",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    help="More folds = more robust but slower training"
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
        value=1,
        step=1
    )
    trading_days = prediction_value * 21
    horizon_text = f"{prediction_value} Month{'s' if prediction_value > 1 else ''}"

# Model regularization settings
st.sidebar.markdown("### Regularization (Prevent Overfitting)")
reg_alpha = st.sidebar.slider(
    "L1 Regularization (reg_alpha)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.5,
    help="Higher values = more regularization, reduces overfitting"
)
reg_lambda = st.sidebar.slider(
    "L2 Regularization (reg_lambda)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.5,
    help="Higher values = more regularization, reduces overfitting"
)

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

# Improved feature engineering
def prepare_features(data, use_all_indicators=True):
    """Prepare features with technical indicators"""
    df = data.copy()
    
    # Price-based features (reduced lags to prevent overfitting)
    for i in range(1, 16):
        df[f'lag_{i}'] = df['Close'].shift(i)
    
    # Rolling statistics with shorter windows
    for window in [5, 10, 20]:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
    
    # Price changes (returns)
    df['daily_return'] = df['Close'].pct_change()
    df['return_volatility'] = df['daily_return'].rolling(window=10).std()
    
    # Key technical indicators
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
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Trend indicators
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['price_vs_sma50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        df['price_vs_sma200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Time series cross-validation
def time_series_cv_score(X, y, n_splits=5):
    """Calculate cross-validation scores with time series split"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42
        )
        model.fit(X_train, y_train, verbose=False)
        
        # Predict and score
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        cv_scores.append(score)
    
    return np.mean(cv_scores), np.std(cv_scores)

# Train XGBoost model with regularization (simplified - no early stopping)
def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train model with regularization - simplified version without early stopping"""
    # Create model
    model = xgb.XGBRegressor(
        n_estimators=200,  # Fixed number of trees
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1
    )
    
    # Simple fit without eval_set to avoid compatibility issues
    model.fit(X_train, y_train, verbose=False)
    
    return model

# Generate future predictions with uncertainty estimation
def predict_future(model, last_data, feature_cols, days_ahead, scaler, model_uncertainty):
    """Predict future prices with uncertainty propagation"""
    predictions = []
    uncertainties = []
    current_data = last_data.copy()
    
    for i in range(days_ahead):
        # Prepare features for prediction
        features = current_data[feature_cols].iloc[-1:].values
        features_scaled = scaler.transform(features)
        next_pred = model.predict(features_scaled)[0]
        predictions.append(next_pred)
        
        # Add prediction uncertainty (grows with prediction horizon)
        uncertainties.append(model_uncertainty * (1 + i / days_ahead))
        
        # Create new row with updated features
        new_row = current_data.iloc[-1:].copy()
        new_row['Close'] = next_pred
        
        # Update lag features
        for lag in range(1, 16):
            if f'lag_{lag}' in new_row.columns:
                if lag == 1:
                    new_row[f'lag_{lag}'] = current_data['Close'].iloc[-1]
                else:
                    new_row[f'lag_{lag}'] = current_data[f'lag_{lag-1}'].iloc[-1] if lag-1 < len(current_data) else current_data['Close'].iloc[-1]
        
        # Update rolling statistics
        for window in [5, 10, 20]:
            if f'rolling_mean_{window}' in new_row.columns:
                recent_values = current_data['Close'].tail(window)
                new_row[f'rolling_mean_{window}'] = recent_values.mean()
                new_row[f'rolling_std_{window}'] = recent_values.std()
        
        # Update returns
        if len(current_data) > 1:
            prev_close = current_data['Close'].iloc[-2]
            new_row['daily_return'] = (next_pred - prev_close) / prev_close
            new_row['return_volatility'] = current_data['daily_return'].tail(10).std() if len(current_data) >= 10 else 0.02
        
        # Update technical indicators (simplified for predictions)
        if 'RSI' in new_row.columns:
            new_row['RSI'] = 50
        
        if 'MACD' in new_row.columns:
            new_row['MACD'] = 0
            new_row['MACD_signal'] = 0
        
        if 'BB_middle' in new_row.columns:
            recent_closes = current_data['Close'].tail(20)
            new_row['BB_middle'] = recent_closes.mean()
            bb_std = recent_closes.std()
            new_row['BB_upper'] = new_row['BB_middle'] + (bb_std * 2)
            new_row['BB_lower'] = new_row['BB_middle'] - (bb_std * 2)
            new_row['BB_position'] = 0.5
        
        if 'volume_ratio' in new_row.columns:
            new_row['volume_ratio'] = 1.0
        
        if 'price_vs_sma50' in new_row.columns:
            new_row['price_vs_sma50'] = 0
        if 'price_vs_sma200' in new_row.columns:
            new_row['price_vs_sma200'] = 0
        
        current_data = pd.concat([current_data, new_row], ignore_index=True)
    
    return np.array(predictions), np.array(uncertainties)

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
                # Prepare data with selected feature set
                df_features = prepare_features(df, use_all_indicators=use_advanced_features)
                
                if len(df_features) < 100:
                    st.warning(f"Limited historical data available ({len(df_features)} data points). Predictions may be less reliable.")
                    st.info("Consider selecting an earlier start date for more training data.")
                
                # Define feature columns (exclude target and original columns)
                exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
                if not use_advanced_features:
                    # If not using advanced features, exclude them from feature list
                    advanced_cols = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal', 
                                    'BB_middle', 'BB_upper', 'BB_lower', 'BB_position', 
                                    'volume_ratio', 'price_vs_sma50', 'price_vs_sma200']
                    exclude_cols.extend(advanced_cols)
                
                feature_cols = [col for col in df_features.columns if col not in exclude_cols]
                
                st.info(f"Using {len(feature_cols)} features for model training")
                
                # Split data using time series split (chronological order)
                train_size = int(len(df_features) * 0.7)
                val_size = int(len(df_features) * 0.15)
                
                train_data = df_features.iloc[:train_size]
                val_data = df_features.iloc[train_size:train_size+val_size]
                test_data = df_features.iloc[train_size+val_size:]
                
                # Prepare features and targets
                X_train = train_data[feature_cols].values
                y_train = train_data['Close'].values
                X_val = val_data[feature_cols].values
                y_val = val_data['Close'].values
                X_test = test_data[feature_cols].values
                y_test = test_data['Close'].values
                
                # Scale features
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
                
                # Evaluate on test set
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
                mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
                
                # Cross-validation score
                cv_mean, cv_std = time_series_cv_score(X_train_scaled, y_train, n_splits=cv_folds)
                
                # Display metrics with visual indicators
                st.subheader("📊 Model Performance Metrics")
                
                # R² score indicator
                if r2 > 0.7:
                    r2_class = "metric-good"
                    r2_message = "Excellent fit - Model captures price patterns well"
                elif r2 > 0.5:
                    r2_class = "metric-moderate"
                    r2_message = "Moderate fit - Some predictive capability"
                else:
                    r2_class = "metric-poor"
                    r2_message = "Poor fit - Market may be too volatile or data insufficient"
                
                st.markdown(f"""
                <div class="{r2_class}" style="margin-bottom: 20px;">
                    <h4>Model Accuracy Assessment</h4>
                    <p><b>Test Set R² Score:</b> {r2:.3f}</p>
                    <p><b>Cross-Validation R²:</b> {cv_mean:.3f} (±{cv_std:.3f})</p>
                    <p>{r2_message}</p>
                    <p style="font-size: 12px; margin-top: 10px;">💡 R² > 0.7 indicates good predictive accuracy. Values between 0.5-0.7 show moderate accuracy. Lower values suggest high market volatility or insufficient data.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics in columns
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("RMSE", f"${rmse:.2f}", help="Root Mean Square Error")
                with met_col2:
                    st.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error")
                with met_col3:
                    st.metric("R² Score", f"{r2:.3f}", help=f"Test Set Performance")
                with met_col4:
                    st.metric("CV R² Score", f"{cv_mean:.3f}", help=f"Cross-validation average ({cv_folds} folds)")
                
                # Plot actual vs predicted
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=test_data.index, y=y_test,
                                              mode='lines', name='Actual',
                                              line=dict(color='blue', width=2)))
                fig_pred.add_trace(go.Scatter(x=test_data.index, y=y_pred_test,
                                              mode='lines', name='Predicted',
                                              line=dict(color='red', width=2, dash='dash')))
                fig_pred.update_layout(title='Model Predictions vs Actual Prices (Test Set)',
                                       xaxis_title='Date',
                                       yaxis_title='Price ($)',
                                       template='plotly_white',
                                       height=500)
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Residual plot
                residuals = y_test - y_pred_test
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(x=test_data.index, y=residuals,
                                              mode='markers', name='Residuals',
                                              marker=dict(color='purple', size=5,
                                                         opacity=0.6)))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                fig_resid.add_hline(y=residuals.std() * 2, line_dash="dot", line_color="gray", 
                                   annotation_text="+2σ")
                fig_resid.add_hline(y=-residuals.std() * 2, line_dash="dot", line_color="gray",
                                   annotation_text="-2σ")
                fig_resid.update_layout(title='Prediction Residuals (Error Analysis)',
                                        xaxis_title='Date',
                                        yaxis_title='Residual ($)',
                                        template='plotly_white',
                                        height=400)
                st.plotly_chart(fig_resid, use_container_width=True)
                
                # Show residual statistics
                st.markdown("### 📉 Residual Analysis")
                resid_col1, resid_col2, resid_col3 = st.columns(3)
                with resid_col1:
                    st.metric("Mean Residual", f"${residuals.mean():.2f}")
                with resid_col2:
                    st.metric("Std Deviation", f"${residuals.std():.2f}")
                with resid_col3:
                    st.metric("95% Range", f"${residuals.std() * 1.96:.2f}")
                
                if show_model_details:
                    # Feature importance (top 20)
                    importance_df = pd.DataFrame({
                        'feature': feature_cols[:20],
                        'importance': model.feature_importances_[:20]
                    }).sort_values('importance', ascending=True)
                    
                    fig_imp = go.Figure(go.Bar(
                        x=importance_df['importance'],
                        y=importance_df['feature'],
                        orientation='h',
                        marker_color='lightblue',
                        text=importance_df['importance'].round(3),
                        textposition='outside'
                    ))
                    fig_imp.update_layout(title='Top 20 Feature Importance',
                                         xaxis_title='Importance Score',
                                         yaxis_title='Features',
                                         template='plotly_white',
                                         height=500)
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Generate predictions based on selected horizon
                st.subheader(f"🔮 {horizon_text} Price Prediction")
                st.markdown(f"Predicting stock prices for the next {horizon_text} (approximately {trading_days} trading days)")
                
                with st.spinner("Generating future predictions with uncertainty estimation..."):
                    # Prepare last data point for prediction
                    last_data = df_features.tail(1).copy()
                    
                    # Calculate model uncertainty from residuals
                    model_uncertainty = residuals.std()
                    
                    # Generate future predictions
                    future_prices, uncertainties = predict_future(model, last_data, feature_cols, 
                                                                  trading_days, scaler, model_uncertainty)
                    
                    # Create future dates
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                                 periods=trading_days, 
                                                 freq='B')
                    
                    # Create prediction dataframe
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_prices,
                        'Uncertainty': uncertainties
                    })
                    
                    # Calculate confidence intervals
                    if confidence_interval == 95:
                        z_score = 1.96
                    elif confidence_interval == 99:
                        z_score = 2.576
                    else:
                        z_score = 1.282
                    
                    future_df['Upper Bound'] = future_df['Predicted Price'] + (z_score * future_df['Uncertainty'])
                    future_df['Lower Bound'] = future_df['Predicted Price'] - (z_score * future_df['Uncertainty'])
                    
                    # Plot predictions
                    fig_future = go.Figure()
                    
                    # Historical data (last year for context)
                    historical_days = min(252, len(df))
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
                    
                    fig_future.update_layout(title=f'{selected_stock} - {horizon_text} Price Prediction with Confidence Intervals',
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
                            <p style="font-size: 12px;">±${z_score * future_df['Uncertainty'].iloc[-1]:.2f}</p>
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
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv_pred,
                        file_name=f"{stock_symbol}_{horizon_text.lower().replace(' ', '_')}_predictions.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.info("Try reducing the prediction horizon, using fewer features, or selecting a stock with more historical data.")
    
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
    
    **Model Features:**
    - **XGBoost with Regularization:** L1 and L2 regularization to reduce overfitting
    - **Time Series Cross-Validation:** Prevents look-ahead bias
    - **Robust Scaling:** Handles outliers better than standard scaling
    - **Uncertainty Estimation:** Provides confidence intervals based on historical errors
    - **Adjustable Prediction Horizon:** Predict from 1 week to 2 years ahead
    
    **Interpretation:** R² scores above 0.7 indicate good predictive capability. Lower scores suggest the stock is highly volatile or unpredictable.
    """)
    
else:
    st.error("Failed to load data. Please check your internet connection and try again.")
