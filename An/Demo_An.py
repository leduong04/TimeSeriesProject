
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings("ignore")

# Load model và scaler
@st.cache_resource
def load_cnn_model():
    model = load_model(r"AN_model_stock.keras")  # Đường dẫn tới file CNN đã huấn luyện
    scaler = MinMaxScaler()
    return model, scaler

# Dự đoán theo kiểu autoregressive
def predict_with_cnn(model, context, prediction_steps, scaler):
    preds = []
    context = list(context.copy())

    for _ in range(prediction_steps):
        input_seq = np.array(context[-60:]).reshape(1, 60, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        preds.append(pred)
        context.append(pred)

    preds_array = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds_array).flatten()

# Streamlit UI
st.set_page_config(page_title="Stock Price Predictor (CNN)", layout="wide")
st.title("📈 Dự đoán giá cổ phiếu với mô hình 1D CNN")

with st.sidebar:
    st.header("Thông số đầu vào")
    ticker = st.text_input("Mã cổ phiếu", "AAPL").upper()
    end_date = st.date_input("Ngày kết thúc", datetime.now())
    start_date = st.date_input("Ngày bắt đầu", end_date - timedelta(days=365))
    prediction_days = st.slider("Số ngày dự đoán", 1, 30, 7)
    st.markdown("---")

try:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        st.error("Không tìm thấy dữ liệu cho mã cổ phiếu này!")
        st.stop()

    prices = df['Close'].values.astype(float)
    dates = df.index

    model, scaler = load_cnn_model()
    scaler.fit(prices.reshape(-1, 1))

    scaled_prices = scaler.transform(prices.reshape(-1, 1)).flatten()
    context = scaled_prices[-60:]  # Lấy 60 giá trị gần nhất

    preds = predict_with_cnn(model, context, prediction_days, scaler)

    last_date = dates[-1]
    pred_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates[-60:], prices[-60:], label='Giá thực tế', color='blue')
    ax.plot(pred_dates, preds, label='Dự đoán', color='red', linestyle='--')
    ax.axvline(last_date, color='gray', linestyle=':')
    ax.set_title(f"Dự đoán giá cổ phiếu {ticker} ({prediction_days} ngày)")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá ($)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Hiển thị bảng dữ liệu dự đoán
    st.subheader("Chi tiết dự đoán")
    pred_df = pd.DataFrame({
        "Ngày": pred_dates,
        "Giá dự đoán": preds,
        "Thay đổi hàng ngày (%)": np.concatenate([[0], np.diff(preds)/preds[:-1]*100])
    })
    st.dataframe(pred_df.style.format({
        "Giá dự đoán": "${:.2f}",
        "Thay đổi hàng ngày (%)": "{:.2f}%"
    }))

except Exception as e:
    st.error(f"Đã xảy ra lỗi: {str(e)}")
