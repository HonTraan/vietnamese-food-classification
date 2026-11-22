"""Streamlit interface for the trained Vietnamese-food classifier."""
import streamlit as st
from PIL import Image
from src.inference import predict

st.set_page_config(page_title="Vietnamese Food Classifier", page_icon="🍜")
st.title("🍜 Phân loại món ăn Việt Nam")
st.caption("Phân loại 30 món ăn Việt Nam")
threshold = st.sidebar.slider(
    "Ngưỡng độ tin cậy",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05,
)
uploaded = st.file_uploader("Tải ảnh món ăn", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded); st.image(image, caption="Ảnh đầu vào", use_container_width=True)
    try:
        predictions, _ = predict(image, top_k=3)
        label, confidence = predictions[0]
        if confidence < threshold:
            st.warning("Chưa xác định được món ăn.")
            st.caption(f"Độ tin cậy cao nhất {confidence:.1%}, dưới ngưỡng {threshold:.0%}.")
        else:
            st.success(f"Dự đoán: **{label}** ({confidence:.1%})")
            st.subheader("Top 3 dự đoán")
            st.table({"Hạng": list(range(1, len(predictions) + 1)),
                      "Món ăn": [item[0] for item in predictions],
                      "Confidence": [f"{item[1]:.1%}" for item in predictions]})
            st.bar_chart(dict(predictions))
    except FileNotFoundError:
        st.error("Chưa có models/best_model.pt. Hãy huấn luyện mô hình trước.")
