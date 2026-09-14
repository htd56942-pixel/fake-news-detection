import streamlit as st
import joblib

# Load mô hình và vectorizer từ SV2
model = joblib.load("model_fake_news.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📰 Ứng dụng phát hiện tin giả")

user_input = st.text_area("Nhập nội dung tin tức:", "")

if st.button("Kiểm tra"):
    if user_input.strip() != "":
        text_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(text_tfidf)[0]
        if prediction == "FAKE":
            st.error("❌ Đây là tin GIẢ")
        else:
            st.success("✅ Đây là tin THẬT")
          
