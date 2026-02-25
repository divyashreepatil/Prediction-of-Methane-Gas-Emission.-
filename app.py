"""
Streamlit App for Methane Gas Emission Prediction
Deployable on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Methane Emission Prediction",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def extract_dataset(zip_path, extract_to):
    """Extract dataset from zip file"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except:
            return False
    return True


def load_data():
    """Load the dataset"""
    # Try to find the dataset
    possible_paths = [
        "./extracted_dataset/FAOSTAT_data_en_11-14-2023.csv",
        "FAOSTAT_data_en_11-14-2023.csv",
        "extracted_dataset/FAOSTAT_data_en_11-14-2023.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    # Check for zip file
    zip_paths = [
        "./FAOSTAT_data_en_11-14-2023.csv.zip",
        "FAOSTAT_data_en_11-14-2023.csv.zip"
    ]
    
    for zip_path in zip_paths:
        if os.path.exists(zip_path):
            extract_dataset(zip_path, "./extracted_dataset")
            if os.path.exists("./extracted_dataset/FAOSTAT_data_en_11-14-2023.csv"):
                return pd.read_csv("./extracted_dataset/FAOSTAT_data_en_11-14-2023.csv")
    
    return None


def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Ensure Value column exists
    if "Value" not in df.columns:
        st.error("'Value' column is missing from the dataset!")
        return None, None, None, None, None
    
    # Drop rows with missing values in target column
    df = df.dropna(subset=["Value"])
    
    # Remove unnecessary categorical columns
    columns_to_exclude = ["Area", "Element", "Item", "Source", "Note"]
    df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns], errors="ignore")
    
    # Select only numerical columns
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ensure "Value" exists and remove it from features
    if "Value" in numeric_features:
        numeric_features.remove("Value")
    
    X = df[numeric_features]
    y = df["Value"]
    
    return X, y, numeric_features, df


def build_knn_model(X_train, y_train, n_neighbors=5):
    """Build KNN Model"""
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def build_gbr_model(X_train, y_train, n_estimators=500, learning_rate=0.1, max_depth=5):
    """Build Gradient Boosting Model"""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def plot_predictions(y_test, y_pred):
    """Plot actual vs predicted"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_test.values,
        mode='lines',
        name='Actual',
        line=dict(color='#4CAF50', width=2)
    ))
    fig.add_trace(go.Scatter(
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='#F44336', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Methane Emission: Actual vs Predicted',
        xaxis_title='Sample Index',
        yaxis_title='Emission Value',
        template='plotly_white',
        height=450
    )
    return fig


def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        fig = px.bar(
            x=importance, 
            y=feature_names, 
            orientation='h',
            title='Feature Importance',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        return fig
    return None


# Main App
def main():
    # Header
    st.title("🌍 Methane Gas Emission Prediction")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model Selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["Gradient Boosting", "K-Nearest Neighbors"],
        help="Choose between Gradient Boosting and KNN architectures"
    )
    
    # Parameters
    st.sidebar.subheader("Model Parameters")
    
    if model_type == "K-Nearest Neighbors":
        n_neighbors = st.sidebar.slider(
            "Number of Neighbors (K)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of neighbors to use"
        )
    else:
        n_estimators = st.sidebar.slider(
            "Number of Estimators",
            min_value=100,
            max_value=1000,
            value=500,
            step=100
        )
        learning_rate = st.sidebar.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01
        )
        max_depth = st.sidebar.slider(
            "Max Depth",
            min_value=2,
            max_value=10,
            value=5
        )
    
    test_size = st.sidebar.slider(
        "Test Size (%)",
        min_value=10,
        max_value=40,
        value=20
    ) / 100
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your CSV or ZIP file",
            type=['csv', 'zip'],
            help="Upload a file with methane emission data"
        )
    
    # Load data
    data = None
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            # Save uploaded file
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.endswith('.zip'):
                extract_dataset(uploaded_file.name, "./extracted_dataset")
                data_path = "./extracted_dataset/FAOSTAT_data_en_11-14-2023.csv"
                if os.path.exists(data_path):
                    data = pd.read_csv(data_path)
            else:
                data = pd.read_csv(uploaded_file.name)
        
        st.success("Data loaded successfully!")
    
    # Try to load default data if no upload
    if data is None:
        with st.spinner("Loading default dataset..."):
            data = load_data()
    
    if data is None:
        st.info("Please upload a data file. The file should contain methane emission data.")
        st.markdown("""
        ### Expected Data Format
        - Column 'Value' for emission values
        - Numerical features for prediction
        - Optional: Area, Element, Item, Year columns
        """)
        return
    
    with col2:
        st.subheader("📋 Data Preview")
        st.dataframe(data.head(), height=150)
    
    # Show data info
    st.subheader("📈 Dataset Information")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Rows", data.shape[0])
    with col_info2:
        st.metric("Columns", data.shape[1])
    with col_info3:
        st.metric("Features", len(data.columns))
    
    # Show columns
    st.write("Available columns:", ", ".join(data.columns.tolist()))
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        X, y, feature_names, processed_data = preprocess_data(data)
    
    if X is None:
        return
    
    st.success(f"Data preprocessed! Shape: X={X.shape}, y={y.shape}")
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.markdown("---")
    
    # Training section
    st.subheader("🧠 Model Training")
    
    if st.button("🚀 Train Model", type="primary"):
        # Build model
        with st.spinner(f"Building {model_type} model..."):
            if model_type == "K-Nearest Neighbors":
                model = build_knn_model(X_train_scaled, y_train, n_neighbors)
            else:
                model = build_gbr_model(X_train_scaled, y_train, n_estimators, learning_rate, max_depth)
        
        st.success(f"{model_type} Model built successfully!")
        
        # Predictions
        with st.spinner("Making predictions..."):
            y_pred = model.predict(X_test_scaled)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        st.subheader("📊 Model Performance")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("R² Score", f"{r2:.4f}")
        with col_m2:
            st.metric("MAE", f"{mae:.2f}")
        with col_m3:
            st.metric("RMSE", f"{rmse:.2f}")
        
        # Plot predictions
        st.subheader("📉 Predictions vs Actual")
        fig_pred = plot_predictions(y_test, y_pred)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature importance (for GBR)
        if model_type == "Gradient Boosting":
            st.subheader("📊 Feature Importance")
            fig_importance = plot_feature_importance(model, feature_names)
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Methane Gas Emission Prediction using Machine Learning</p>
        <p>Powered by Streamlit, Scikit-learn & Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
