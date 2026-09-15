import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

# Load PhoBERT đã fine‑tune (thư mục phobert_fake_news phải nằm cùng repo)
tokenizer = AutoTokenizer.from_pretrained("phobert_fake_news")
model = AutoModelForSequenceClassification.from_pretrained("phobert_fake_news")

# Hàm tiền xử lý văn bản
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Hàm dự đoán
def predict(text):
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        output = model(enc['input_ids'], attention_mask=enc['attention_mask'])
        probs = torch.softmax(output.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
    return pred, probs

# Giao diện Streamlit
st.title("📰 Ứng dụng phát hiện tin giả tiếng Việt (PhoBERT Fine‑tune)")
st.write("Nhập nội dung bài viết để kiểm tra xem là **Tin thật** hay **Tin giả**.")

user_input = st.text_area("Nội dung bài viết:")

if st.button("Kiểm tra"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung trước khi kiểm tra.")
    else:
        pred, probs = predict(user_input)
        if pred == 1:
            st.error(f"🚨 Kết quả: Tin giả (Độ tin cậy {probs[1]*100:.2f}%)")
        else:
            st.success(f"✅ Kết quả: Tin thật (Độ tin cậy {probs[0]*100:.2f}%)")
import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Header
st.markdown("<h1 style='color:#1E90FF;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;color:gray;'>Ứng dụng phân biệt tin thật - tin giả</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Menu")
st.sidebar.radio("Chọn chế độ:", ["Kiểm tra một câu", "Kiểm tra bài báo", "Xem thống kê"])

# Input
text = st.text_area("Nhập nội dung cần kiểm tra:")

if st.button("Phân tích"):
    result = "FAKE NEWS 🚨"
    confidence = 92
    st.markdown(
        f"""
        <div style='background-color:#FF0000;padding:20px;border-radius:10px;'>
            <h2 style='color:white;'>BREAKING NEWS</h2>
            <h3 style='color:white;'>{result}</h3>
            <p style='color:white;'>Độ tin cậy: {confidence}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(confidence/100)


