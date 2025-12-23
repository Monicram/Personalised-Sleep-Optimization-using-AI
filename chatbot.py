import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained models
sleep_disorder_model = load_model("sleep_disorder_model.h5")
recommendation_model = joblib.load("sleep_recommendation_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to preprocess eye image
def preprocess_eye_image(image):
    image = cv2.resize(image, (64, 64))  # Resize to model input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Reshape for model input

# Function to predict sleep disorder
def predict_sleep_disorder(image):
    processed_image = preprocess_eye_image(image)
    prediction = sleep_disorder_model.predict(processed_image)
    return "High Risk of Sleep Disorder" if prediction[0] > 0.5 else "Low Risk of Sleep Disorder"

# Function to recommend sleep optimization
def recommend_sleep(age, sleeping_hours, occupation):
    input_data = np.array([[age, sleeping_hours]])
    input_data = scaler.transform(input_data)  # Apply scaling
    recommendation = recommendation_model.predict(input_data)
    return recommendation[0]

# Function to integrate everything
def chatbot_interface(image, age, sleeping_hours, occupation):
    if image is not None:
        disorder_result = predict_sleep_disorder(image)
        recommendation = recommend_sleep(age, sleeping_hours, occupation)
        return disorder_result, recommendation
    return "No Image Provided", "Please provide an image"

# Gradio UI
inputs = [
    gr.Image(source="webcam", label="Capture Eye Image"),
    gr.Number(label="Age"),
    gr.Number(label="Sleeping Hours"),
    gr.Textbox(label="Occupation"),
]

outputs = [
    gr.Textbox(label="Sleep Disorder Prediction"),
    gr.Textbox(label="Personalized Recommendation"),
]

# Launch the chatbot
gr.Interface(fn=chatbot_interface, inputs=inputs, outputs=outputs, title="AI-Powered Sleep Optimization Chatbot").launch()
