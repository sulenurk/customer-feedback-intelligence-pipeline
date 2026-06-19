import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Sayfa Tasarımı
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🤖", layout="centered")
st.title("🤖 Real-Time Sentiment Analysis")
st.write("Enter an English review below, and our Fine-Tuned DistilBERT model will analyze its sentiment!")

# 2. Model ve Tokenizer'ı Yükleme (Önbelleğe alıyoruz ki her tahminde tekrar yüklenmesin)
@st.cache_resource
def load_model():
    model_path = "./my_sentiment_model" # İndirdiğin klasörün yolu
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

try:
    tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. Kullanıcı Girişi
user_input = st.text_area("Your Review:", placeholder="Type something like: 'This movie was absolutely spectacular! I loved it.'")

# 4. Tahmin Butonu ve Mantığı
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Metni tokenize ediyoruz
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
        
        # Tahmin algoritması
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Logitleri olasılığa (Softmax) çeviriyoruz
            probs = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
        
        # Sonuçları Görselleştirme
        confidence = probs[0][prediction].item() * 100
        
        st.markdown("### Result:")
        if prediction == 1:
            st.success(f"🟢 **POSITIVE** (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"🔴 **NEGATIVE** (Confidence: {confidence:.2f}%)")