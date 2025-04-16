import numpy as np
import pandas as pd
import easyocr
import streamlit as st
from PIL import Image
import cv2
import base64
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

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


def extracted_text(col):
    return " ".join(img_df[col])


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity


def generate_topics(text, num_topics=2, num_words=5):
    vectorizer = CountVectorizer(stop_words="english")
    data_vectorized = vectorizer.fit_transform([text])
    lda = LDA(n_components=num_topics, random_state=42)
    lda.fit(data_vectorized)
    words = vectorizer.get_feature_names_out()
    topics = [
        ", ".join([words[i] for i in topic.argsort()[-num_words:]])
        for topic in lda.components_
    ]
    return topics


def summarize_text(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


# Streamlit app
st.markdown("""
    <svg width="600" height="100">
        <text x="50%" y="50%" font-family="monospace" font-size="42px" fill="Turquoise" text-anchor="middle" stroke="white"
         stroke-width="0.3" stroke-linejoin="round">📃 ScanMaster OCR 📃
        </text>
    </svg>
""", unsafe_allow_html=True)

add_bg_from_local('background.jpg')

file = st.file_uploader(label="Upload Image Here (png/jpg/jpeg) : ", type=['png', 'jpg', 'jpeg'])

if file is not None:
    image = Image.open(file)
    st.image(image)

    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(np.array(image))

    img_df = pd.DataFrame(results, columns=['bbox', 'Predicted Text', 'Prediction Confidence'])

    text_combined = extracted_text(col='Predicted Text')
    st.write("Text Generated :- ", text_combined)

    # Sentiment analysis
    polarity, subjectivity = analyze_sentiment(text_combined)
    st.write(f"Sentiment Analysis:\n- Polarity: {polarity}\n- Subjectivity: {subjectivity}")

    # Topic modeling
    topics = generate_topics(text_combined)
    st.write("Extracted Topics:")
    for i, topic in enumerate(topics, start=1):
        st.write(f"Topic {i}: {topic}")

    # Text summarization
    if len(text_combined.split()) > 20:
        summary = summarize_text(text_combined)
        st.write("Summary of Extracted Text:")
        st.write(summary)
    else:
        st.write("Text too short for summarization.")

    # Printing results in tabular form
    st.write("Table Showing Predicted Text and Prediction Confidence : ")
    st.table(img_df.iloc[:, 1:])

    # Getting final image with drawing annotations
    display_ocr_image(image, results)

else:
    st.warning("!! Please Upload your image !!")
