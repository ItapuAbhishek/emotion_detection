import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image


# Load the saved model
model = keras.models.load_model('emotion_detection_model.h5')

# Define the emotions list
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# Function to predict emotion from the uploaded image
def predict_emotion(image):
    # Preprocess the image
    img = image.convert('RGB').resize((48, 48))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)

    # Get the index of the predicted emotion
    emotion_index = np.argmax(prediction)

    return emotions[emotion_index]


# Define the Streamlit app
def app():
    st.set_page_config(page_title="Emotion Detection App")

    # Add a title
    st.title("Emotion Detection App")

    # Add a file uploader to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # If the user has uploaded an image
    if uploaded_file is not None:
        # Open and display the image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_column_width=True)

        # Add a button to predict the emotion
        if st.button('Predict Emotion'):
            emotion = predict_emotion(image_display)
            st.write('The predicted emotion is:', emotion)


# Run the app
if __name__ == '__main__':
    app()
