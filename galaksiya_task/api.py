from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import pickle

app = FastAPI()

# Pydantic model
class Review(BaseModel):
    review_text: str
    summary: str
    helpful_ratio: float

# Model and tokenizer
with open('rating_predictor.pkl', 'rb') as file:
    model = pickle.load(file)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldırma
    text = text.lower()  # Küçük harfe çevirme
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])  # Stopwords kaldırma
    return text

def encode_text(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    return tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy().squeeze()

@app.post("/predict")
def predict(review: Review):
    try:
        print("Request received")
        cleaned_text = clean_text(review.review_text + ' ' + review.summary)
        print(f"Cleaned text: {cleaned_text}")
        bert_vector = encode_text(cleaned_text)
        print(f"BERT vector shape: {bert_vector.shape}")
        bert_vector = bert_vector.reshape(1, -1)  # Tek boyutlu BERT vektörünü iki boyutluya dönüştür
        print(f"Reshaped BERT vector shape: {bert_vector.shape}")
        features = np.hstack((bert_vector, np.array([[review.helpful_ratio, len(cleaned_text.split())]])))
        print(f"Features shape: {features.shape}")
        prediction = model.predict(features)[0]
        print(f"Prediction: {prediction}")
        return {"predicted_rating": float(prediction)}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



#check http://localhost:8000/predict on postman
import nest_asyncio
nest_asyncio.apply()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
