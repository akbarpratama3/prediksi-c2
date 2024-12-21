import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import timedelta
import streamlit as st

# Load model LSTM
model = load_model('model_lstm.h5')  # Gantilah dengan nama model Anda

# Fungsi untuk melakukan prediksi harga XAU/USD
def predict_price(data, model, scaler, prediction_days):
    # Ambil 60 data terakhir
    last_60_days = data[-60:].reshape((1, 60, 1))
    
    # Prediksi harga untuk sejumlah hari ke depan
    predictions = []
    for _ in range(prediction_days):
        pred = model.predict(last_60_days)
        predictions.append(pred[0, 0])
        last_60_days = np.append(last_60_days[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    
    # Kembalikan prediksi ke skala harga asli
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Streamlit UI
st.title('Prediksi Harga XAU/USD menggunakan LSTM')

# Upload file scaler
uploaded_scaler = st.file_uploader("Upload file skalar.pkl", type=["pkl"])

# Mengatur agar file scaler dapat dipilih oleh pengguna
if uploaded_scaler is not None:
    scaler = pickle.load(uploaded_scaler)  # Memuat scaler yang di-upload
    st.write("Scaler berhasil dimuat.")

    # Upload file data XAU/USD
    uploaded_file = st.file_uploader("Upload Data XAU/USD CSV", type=["csv"])

    if uploaded_file is not None:
        # Membaca data dari file CSV
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Tampilkan data yang di-upload
        st.write("Data XAU/USD yang Diupload:")
        st.write(df.tail())

        # Pastikan kolom 'Price' ada dan tidak kosong
        if 'Price' in df.columns and not df['Price'].isnull().all():
            # Pastikan data 'Price' adalah numerik
            price_data = pd.to_numeric(df['Price'], errors='coerce')  # Mengubah menjadi numerik dan ganti yang error menjadi NaN

            # Memastikan tidak ada nilai NaN atau Inf
            if np.any(np.isnan(price_data)) or np.any(np.isinf(price_data)):
                st.error("Data harga mengandung nilai NaN atau Inf. Harap periksa data Anda.")
            else:
                # Transformasikan data harga
                price_data = price_data.values.reshape(-1, 1)
                scaled_data = scaler.transform(price_data)

                # Input untuk jumlah hari prediksi
                st.subheader("Masukkan Jumlah Hari Prediksi")
                prediction_days = st.number_input(
                    "Jumlah hari untuk prediksi:",
                    min_value=1,
                    max_value=30,
                    value=10,
                    step=1
                )

                # Tombol untuk memulai prediksi
                if st.button("Lakukan Prediksi"):
                    # Prediksi harga sejumlah hari ke depan
                    predictions = predict_price(scaled_data, model, scaler, prediction_days)
                    
                    # Tentukan tanggal hasil prediksi
                    start_date = df.index[-1]  # Tanggal terakhir dalam data
                    future_dates = [start_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
                    
                    result_df = pd.DataFrame({
                        'Date': future_dates,
                        'Prediksi (Price)': predictions.flatten()
                    })

                    # Tampilkan hasil prediksi
                    st.write(f"Prediksi Harga XAU/USD {prediction_days} Hari Ke Depan:")
                    st.write(result_df)

                    # Visualisasi Prediksi
                    st.subheader('Visualisasi Prediksi dan Data Aktual')
                    plt.figure(figsize=(14, 7))
                    plt.plot(df.index, df['Price'], color='blue', label='Harga Aktual')
                    plt.plot(result_df['Date'], result_df['Prediksi (Price)'], color='orange', label='Prediksi Harga')
                    plt.title(f'Prediksi Harga XAU/USD ({prediction_days} Hari ke Depan)', fontsize=20)
                    plt.xlabel('Tanggal', fontsize=16)
                    plt.ylabel('Harga XAU/USD', fontsize=16)
                    plt.legend(fontsize=14)
                    plt.grid(True)

                    st.pyplot(plt)
        else:
            st.error("Kolom 'Price' tidak ditemukan atau data 'Price' kosong.")
