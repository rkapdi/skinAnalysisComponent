import streamlit as st
from fastai.vision.all import load_learner, PILImage
import time
from PIL import Image

# Load your trained model
learn = load_learner('./export.pkl')

# Function to predict and display results
def predict(image):
    img = PILImage.create(image)
    pred, pred_idx, probs = learn.predict(img)
    return pred, probs

# Function to resize image
def resize_image(image, max_width, max_height):
    if isinstance(image, str):  # If image is a file path
        img = Image.open(image)
    else:  # If image is already an opened image file
        img = image

    img.thumbnail((max_width, max_height))
    return img

# Sample images (add paths to your sample images here)
sample_images = {
    "Sample 1": "./sample1.jpeg",
    "Sample 2": "./sample2.jpeg",
    "Sample 3": "./sample3.jpeg"
}

# Use the full page width
st.set_page_config(layout="wide")

# Streamlit UI


# Create two columns for the layout
col1, col2 = st.columns(2)

# Column 1: Image Upload, Sample Selection, and Display Results
with col1:
    st.title("Face Condition Analyzer")
    st.write("Upload an image of the face to analyze the skin condition, or use a sample image.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    sample_image_selection = st.selectbox("Or choose a sample image:", list(sample_images.keys()))
    analyze_button = st.button("ANALYZE")

    if uploaded_file is not None:
        display_image = PILImage.create(uploaded_file)
    elif sample_image_selection in sample_images:
        display_image = Image.open(sample_images[sample_image_selection])
    else:
        display_image = None

    if display_image is not None:
        resized_image = resize_image(display_image, max_width=250, max_height=250)
        st.image(resized_image, caption='Uploaded Image.')

    if analyze_button and display_image is not None:
        with st.spinner(text='Analyzing...'):
            time.sleep(2)  # Simulate processing time
            label, probabilities = predict(display_image)

        st.success('Analysis Done')
        st.write(f"Prediction: {label}")

# Column 2: Probability Sliders
with col2:
    if analyze_button and display_image is not None:
        st.write("Probabilities:")
        for i, prob in enumerate(probabilities):
            st.slider(label=learn.dls.vocab[i], min_value=0.0, max_value=100.0, value=float(prob*100), format='%.2f%%', disabled=True)



