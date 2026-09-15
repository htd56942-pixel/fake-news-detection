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


