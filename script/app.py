import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model # type: ignore
import pickle
import joblib 

# --- Configuration and Model Loading ---
st.set_page_config(page_title="Indian YouTube Virality Predictor", layout="wide")

# Use a cache to load models only once, speeding up the app
@st.cache_resource
def load_all_models():
    """
    Loads all the trained models and scalers from disk.
    This function is cached so it only runs once.
    """
    # Load the XGBoost Classifier
    classifier = joblib.load('../models/xgb_classifier.joblib')

    # Defining model and scaler paths
    model_paths = {
        0: "../models/lstm_model_cluster_0.h5",
        1: "../models/lstm_model_cluster_1.h5",
        3: "../models/lstm_model_cluster_3.h5" 
    }
    scaler_paths = {
        0: "../models/scaler_cluster_0.pkl",
        1: "../models/scaler_cluster_1.pkl",
        3: "../models/scaler_cluster_3.pkl"
    }

    # Loading LSTM models and scalers
    lstm_models = {key: load_model(path) for key, path in model_paths.items()}
    scalers = {}
    for key, path in scaler_paths.items():
        with open(path, 'rb') as f:
            scalers[key] = pickle.load(f)
            
    return classifier, lstm_models, scalers

# Archetype mapping
archetype_names = {
    0: "The Explosive Hit",
    1: "The Steady Climb",
    2: "Niche Anomaly", 
    3: "The Standard Takeoff"
}

# Loading the models
classifier, lstm_models, scalers = load_all_models()

# --- User Interface ---
st.title("🚀 Indian YouTube Virality Prediction Engine")
st.markdown("This tool predicts the virality archetype of a new video based on its first-day stats and forecasts its view trajectory for the next 7 days.")

st.sidebar.header("Enter Video's First-Day Stats")

# Create input fields in the sidebar
views = st.sidebar.number_input("Views", min_value=0, value=50000)
likes = st.sidebar.number_input("Likes", min_value=0, value=2500)
dislikes = st.sidebar.number_input("Dislikes", min_value=0, value=100)
comment_count = st.sidebar.number_input("Comment Count", min_value=0, value=500)
title_length = st.sidebar.slider("Title Length", 5, 100, 40)


if st.sidebar.button("Predict Virality"):
    with st.spinner('Running the ML pipeline... This may take a moment.'):

        # --- 1. Archetype Classification ---
        
        # Create a DataFrame for the single prediction.
        # IMPORTANT: The columns MUST match the order and names used for training the classifier.
        
        # A simplified input for demonstration.
        input_data = pd.DataFrame({
            'views': [views],
            'likes': [likes],
            'dislikes': [dislikes],
            'comment_count': [comment_count],
            'comments_disabled': [0], # Assuming not disabled
            'ratings_disabled': [0], # Assuming not disabled
            'title_length': [title_length],
            'has_question_mark': [0],
            'has_cricket_keyword': [0],
            'has_bollywood_keyword': [1], # Assume it's a Bollywood video
            'has_comedy_keyword': [0],
            'has_news_keyword': [0],
            'is_from_top_channel': [1]
        
        })
        

        expected_cols = classifier.get_booster().feature_names
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
                
        input_data = input_data[expected_cols] # Ensure column order is correct

        predicted_cluster = classifier.predict(input_data)[0]
        predicted_archetype_name = archetype_names.get(predicted_cluster, "Unknown")

    st.header("Step 1: Predicted Virality Archetype")
    st.metric(label="Predicted Archetype", value=predicted_archetype_name)
    with st.expander("What does 'The Standard Takeoff' mean?"):
        st.write("""
            This archetype represents the most common successful video trajectory. 
            It typically features a 24-hour delay after publishing, followed by 
            2-3 days of strong, sustained growth before reaching a plateau. 
            This is the bread-and-butter of standard, well-performing content on YouTube.
        """)

    # --- 2. View Trajectory Forecasting ---
    st.header("Step 2: 7-Day View Forecast")

    if predicted_cluster in lstm_models:
        # Select the correct LSTM model and scaler
        forecast_model = lstm_models[predicted_cluster]
        scaler = scalers[predicted_cluster]

        # Start the forecast using the initial view count
        last_views = np.array([[views]])
        scaled_last_views = scaler.transform(last_views)

        # We need 3 initial points to start the LSTM. We will just repeat the first day's views.
        input_sequence = np.array([scaled_last_views[0], scaled_last_views[0], scaled_last_views[0]])
        # Ensure the starting sequence has the correct 3D shape: (1, 3, 1)
        input_sequence = input_sequence.reshape(1, 3, 1) 

        forecast_list = []
        for _ in range(7): # Forecast for 7 days
            # Predict the next step. Prediction will have shape (1, 1)
            next_pred_scaled = forecast_model.predict(input_sequence, verbose=0)
            
            # Inverse transform the 2D prediction to get the actual view count
            next_pred = scaler.inverse_transform(next_pred_scaled)
            forecast_list.append(int(next_pred[0][0]))
            
            
            # Reshape the 2D prediction (shape (1,1)) into a 3D array (shape (1,1,1))
            # so it can be concatenated with the 3D input_sequence.
            new_pred_3d = next_pred_scaled.reshape(1, 1, 1)

            # Get the last two time steps from the input sequence
            last_two_steps = input_sequence[:, 1:, :]
            
            # Create the new sequence by concatenating the old part with the new prediction
            input_sequence = np.concatenate([last_two_steps, new_pred_3d], axis=1)
            

        # Display the forecast
        df_forecast = pd.DataFrame({
            'Day': range(1, 8),
            'Predicted Views': forecast_list
        })
        
        st.line_chart(df_forecast.set_index('Day'))
        df_forecast_display = df_forecast.set_index('Day')
        st.dataframe(df_forecast_display.style.format({'Predicted Views': '{:,.0f}'}))

    else:
        st.warning(f"No forecasting model available for the '{predicted_archetype_name}' archetype.")