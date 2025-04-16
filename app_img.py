import numpy as np
import pandas as pd
import easyocr
import streamlit as st
from PIL import Image
import cv2
import base64
from textblob import TextBlob
from transformers import pipeline
import torch

# Function to add app background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(f"""<style>.stApp {{background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover}}</style>""", unsafe_allow_html=True)


def display_ocr_image(img, results):
    img_np = np.array(img)

    for detection in results:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.rectangle(img_np, top_left, bottom_right, (0, 255, 0), 5)
        cv2.putText(img_np, text, top_left, font, 1, (125, 29, 241), 2, cv2.LINE_AA)

    st.image(img_np, channels="BGR", use_column_width=True)


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity


# Load the image captioning model
@st.cache(allow_output_mutation=True)
def load_image_captioning_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


# Generate image description
def generate_image_description(model, image):
    description = model(image)[0]['generated_text']
    return description


# Streamlit app
st.markdown("""
    <svg width="600" height="100">
        <text x="50%" y="50%" font-family="monospace" font-size="42px" fill="Turquoise" text-anchor="middle" stroke="white"
         stroke-width="0.3" stroke-linejoin="round">📃 Image Analysis by Techylla  📃
        </text>
    </svg>
""", unsafe_allow_html=True)

add_bg_from_local('background.jpg')

file = st.file_uploader(label="Upload Image Here (png/jpg/jpeg):", type=['png', 'jpg', 'jpeg'])

if file is not None:
    image = Image.open(file)
    st.image(image)

    # Load models
    caption_model = load_image_captioning_model()
    reader = easyocr.Reader(['en'], gpu=False)

    # Step 1: OCR for text extraction
    results = reader.readtext(np.array(image))
    img_df = pd.DataFrame(results, columns=['bbox', 'Predicted Text', 'Prediction Confidence'])
    text_combined = " ".join(img_df['Predicted Text'])
    st.write("Extracted Text from Image: ", text_combined)

    # Step 2: Image description generation
    image_description = generate_image_description(caption_model, image)
    st.write("Generated Image Description: ", image_description)

    # Step 3: Sentiment analysis
    text_polarity, text_subjectivity = analyze_sentiment(text_combined)
    desc_polarity, desc_subjectivity = analyze_sentiment(image_description)

    st.write("Sentiment Analysis:")
    st.write(f"- **Extracted Text Polarity**: {text_polarity}, **Subjectivity**: {text_subjectivity}")
    st.write(f"- **Image Description Polarity**: {desc_polarity}, **Subjectivity**: {desc_subjectivity}")

    # Step 4: Visualize OCR with bounding boxes
    display_ocr_image(image, results)

    # Step 5: Generate report
    st.subheader("Detailed Report")
    st.write("Extracted Text:")
    st.write(text_combined)
    st.write("Generated Image Description:")
    st.write(image_description)
    st.write("Sentiment Analysis Summary:")
    st.write(f"- Extracted Text Polarity: {text_polarity}, Subjectivity: {text_subjectivity}")
    st.write(f"- Image Description Polarity: {desc_polarity}, Subjectivity: {desc_subjectivity}")

else:
    st.warning("!! Please Upload an Image !!")
