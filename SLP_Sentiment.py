import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional, Attention, Input
from tensorflow.keras.preprocessing.text import Tokenizer

# ✅ Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="wide")

# ---- CONFIG ----
max_features = 20000  # Most common words
maxlen = 500  # Max sequence length
embedding_dim = 128
model_dir = "models"
tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")

# ---- LOAD IMDB DATASET ----
@st.cache_data
def load_imdb():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_imdb()

# ---- MODEL CREATION ----
def build_cnn():
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_han():
    input_layer = Input(shape=(maxlen,))
    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen)(input_layer)
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    attention_layer = Attention()([lstm_layer, lstm_layer])
    flatten_layer = Flatten()(attention_layer)
    dense_layer = Dense(128, activation='relu')(flatten_layer)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---- LOAD OR TRAIN MODELS ----
def load_or_train_model(model_name, build_fn):
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    
    if os.path.exists(model_path):
        st.sidebar.success(f"✅ {model_name} model loaded from disk!")
        return load_model(model_path)
    
    st.sidebar.warning(f"⚠️ {model_name} model not found! Training a new model...")
    model = build_fn()
    
    with st.spinner(f"🚀 Training {model_name} Model... Please wait!"):
        model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_path)
    
    st.sidebar.success(f"✅ {model_name} model trained and saved!")
    return model

cnn_model = load_or_train_model("cnn_model", build_cnn)
han_model = load_or_train_model("han_model", build_han)

# ---- TOKENIZER (For Text Processing) ----
if not os.path.exists(tokenizer_path):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(imdb.get_word_index().keys())
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
else:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

# ---- STREAMLIT UI ----
st.markdown("<h1 style='text-align: center; color: #3c6382;'>🎭 IMDB Sentiment Analysis (CNN vs HAN)</h1>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Select Model")
model_choice = st.sidebar.radio("Choose a Model:", ["CNN", "HAN"])

st.sidebar.header("📝 Enter Review for Sentiment Analysis")
user_input = st.sidebar.text_area("Enter your movie review:", height=150)

if st.sidebar.button("🔍 Predict Sentiment"):
    if user_input:
        user_seq = tokenizer.texts_to_sequences([user_input])
        user_padded = pad_sequences(user_seq, maxlen=maxlen)

        model = cnn_model if model_choice == "CNN" else han_model
        pred = model.predict(user_padded)[0][0]
        sentiment = "😊 Positive" if pred > 0.5 else "😞 Negative"

        st.success(f"✅ **Predicted Sentiment:** {sentiment} ({pred:.2f})")
    else:
        st.warning("⚠️ Please enter a review.")

# ---- MODEL EVALUATION ----
st.header("📊 Model Performance Comparison")

def evaluate_model(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

cnn_accuracy = evaluate_model(cnn_model, X_test, y_test)
han_accuracy = evaluate_model(han_model, X_test, y_test)

performance_df = pd.DataFrame({
    "Model": ["CNN", "HAN"],
    "Accuracy": [cnn_accuracy, han_accuracy]
})

st.table(performance_df)

# ---- PLOT ACCURACY ----
fig, ax = plt.subplots()
performance_df.set_index("Model").plot(kind="bar", ax=ax, color=["#3c6382", "#1e3799"])
plt.title("📈 CNN vs HAN Accuracy", fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

st.write("🎯 **Developed with Streamlit | IMDB Sentiment Analysis 🚀**")
