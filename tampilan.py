import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import timedelta

# Load model dan scaler
model = load_model('model_lstm.h5')  # Gantilah dengan nama model Anda
scaler = pickle.load(open('skalar.pkl', 'rb'))  # Gantilah dengan nama file scaler Anda

# Fungsi untuk melakukan prediksi harga XAU/USD
def predict_price(data, model, scaler):
    # Ambil 60 data terakhir
    last_60_days = data[-60:].reshape((1, 60, 1))
    
    # Prediksi harga untuk 10 hari ke depan
    predictions = []
    for _ in range(10):
        pred = model.predict(last_60_days)
        predictions.append(pred[0, 0])
        last_60_days = np.append(last_60_days[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    
    # Kembalikan prediksi ke skala harga asli
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Upload file data XAU/USD
st.title('Prediksi Harga XAU/USD menggunakan LSTM')
uploaded_file = st.file_uploader("Upload Data XAU/USD CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca data dari file CSV
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Tampilkan data yang diupload
    st.write("Data XAU/USD yang Diupload:")
    st.write(df.tail())

    # Menyiapkan data untuk prediksi
    scaled_data = scaler.transform(df['Price'].values.reshape(-1, 1))
    
    # Prediksi harga 10 hari ke depan
    predictions = predict_price(scaled_data, model, scaler)
    
    # Tampilkan hasil prediksi
    start_date = df.index[-1]  # Tanggal terakhir dalam data
    future_dates = [start_date + timedelta(days=i) for i in range(1, 11)]
    
    result_df = pd.DataFrame({
        'Date': [date.strftime('%Y-%m-%d') for date in future_dates],
        'Prediksi (Price)': predictions.flatten()
    })
    
    st.write("Prediksi Harga XAU/USD 10 Hari Ke Depan:")
    st.write(result_df)

    # Visualisasi Prediksi
    st.subheader('Visualisasi Prediksi dan Data Aktual')
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], color='blue', label='Harga Aktual')
    plt.plot(result_df['Date'], result_df['Prediksi (Price)'], color='orange', label='Prediksi Harga')
    plt.title('Prediksi Harga XAU/USD (10 Hari ke Depan)', fontsize=20)
    plt.xlabel('Tanggal', fontsize=16)
    plt.ylabel('Harga XAU/USD', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    
    st.pyplot(plt)

