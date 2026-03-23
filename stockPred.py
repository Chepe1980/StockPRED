import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas_datareader.data as web
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Google Stock Price Analyzer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Google Stock Price Analysis")
st.markdown("This app analyzes Google stock prices and splits the data into training and test sets.")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.markdown("Configure the analysis parameters below:")

# Date selection for train/test split
split_date = st.sidebar.date_input(
    "Select Train/Test Split Date",
    value=pd.to_datetime('2025-01-01'),
    min_value=pd.to_datetime('2004-01-01'),
    max_value=datetime.now()
)

# Option to show/hide model training
show_model = st.sidebar.checkbox("Show XGBoost Model Training", value=False)

# Load data with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        df = web.DataReader('GOOGL', 'stooq')
        df = df[['Close']]
        df.columns = ['Close Price']
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
with st.spinner("Loading stock data..."):
    df = load_data()

if df is not None:
    # Display raw data
    st.subheader("📊 Raw Data")
    st.write(f"Data from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    st.dataframe(df.tail(10))
    
    # Display basic statistics
    st.subheader("📈 Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${df['Close Price'].iloc[-1]:.2f}")
    with col2:
        st.metric("Mean Price", f"${df['Close Price'].mean():.2f}")
    with col3:
        st.metric("Max Price", f"${df['Close Price'].max():.2f}")
    with col4:
        st.metric("Min Price", f"${df['Close Price'].min():.2f}")
    
    # Plot full time series
    st.subheader("📉 Full Time Series")
    fig1, ax1 = plt.subplots(figsize=(15, 5))
    df.plot(ax=ax1, style='-', color='#1f77b4', title='Stock Price of GOOGLE')
    ax1.set_ylabel('Price ($)')
    ax1.set_xlabel('Date')
    st.pyplot(fig1)
    plt.close()
    
    # Train/Test split
    split_date_str = split_date.strftime('%Y-%m-%d')
    train = df.loc[df.index < split_date_str]
    test = df.loc[df.index >= split_date_str]
    
    # Display split information
    st.subheader("🔄 Train/Test Split")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Set Size", f"{len(train)} days")
        st.metric("Training Date Range", 
                 f"{train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')}")
    with col2:
        st.metric("Test Set Size", f"{len(test)} days")
        st.metric("Test Date Range", 
                 f"{test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')}")
    
    # Plot train/test split
    st.subheader("📊 Train/Test Split Visualization")
    fig2, ax2 = plt.subplots(figsize=(15, 5))
    train.plot(ax=ax2, label='Training Set', color='blue')
    test.plot(ax=ax2, label='Test Set', color='orange')
    ax2.axvline(split_date_str, color='black', ls='--', label='Split Date')
    ax2.set_title('Data Train/Test Split')
    ax2.set_ylabel('Price ($)')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()
    
    # Optional: XGBoost Model
    if show_model and len(train) > 0 and len(test) > 0:
        st.subheader("🤖 XGBoost Model Training")
        
        # Prepare data for XGBoost
        with st.spinner("Preparing data for XGBoost..."):
            # Create features (using lagged values)
            def prepare_features(data, lags=7):
                df_features = data.copy()
                for i in range(1, lags + 1):
                    df_features[f'lag_{i}'] = df_features['Close Price'].shift(i)
                df_features = df_features.dropna()
                return df_features
            
            train_features = prepare_features(train)
            test_features = prepare_features(test)
            
            if len(train_features) > 0 and len(test_features) > 0:
                X_train = train_features.drop('Close Price', axis=1)
                y_train = train_features['Close Price']
                X_test = test_features.drop('Close Price', axis=1)
                y_test = test_features['Close Price']
                
                # Train XGBoost model
                with st.spinner("Training XGBoost model..."):
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with col2:
                    st.metric("MSE", f"${mse:.2f}")
                with col3:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # Plot predictions vs actual
                st.subheader("📊 Model Predictions vs Actual")
                fig3, ax3 = plt.subplots(figsize=(15, 5))
                ax3.plot(test_features.index, y_test, label='Actual', color='blue')
                ax3.plot(test_features.index, y_pred, label='Predicted', color='red', alpha=0.7)
                ax3.set_title('XGBoost Model Predictions vs Actual Prices')
                ax3.set_ylabel('Price ($)')
                ax3.set_xlabel('Date')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
                plt.close()
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.barh(importance_df['feature'], importance_df['importance'])
                ax4.set_xlabel('Importance')
                ax4.set_title('XGBoost Feature Importance')
                st.pyplot(fig4)
                plt.close()
            else:
                st.warning("Not enough data for model training after creating lagged features.")
    
    # Download button for data
    st.subheader("💾 Download Data")
    csv = df.to_csv()
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"google_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("Data source: STOOQ (Yahoo Finance via pandas_datareader)")

else:
    st.error("Failed to load data. Please check your internet connection and try again.")
