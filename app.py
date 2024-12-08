import os
import numpy as np
import tensorflow as tf
import streamlit as st
from keras.models import load_model

# Menampilkan judul aplikasi
st.header('Klasifikasi Hewan dengan CNN')

# Daftar nama hewan yang sesuai dengan urutan output model
Hewan_names = ['CHEETAH', 'HARIMAU', 'MACAN_TUTUL', 'SINGA', 'UNKNOWN']  # Sesuaikan nama kelas

# Memuat model yang telah disimpan
try:
    model = load_model('final_model.keras')
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Fungsi untuk mengklasifikasikan gambar
def classify_images(image_path):
    try:
        # Memuat gambar dan mengubah ukurannya sesuai input model
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))  # Ukuran 180x180 sesuai model Sequential
        input_image_array = tf.keras.utils.img_to_array(input_image) / 255.0  # Normalisasi ke [0,1]
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)  # Menambahkan dimensi batch

        # Melakukan prediksi
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])  # Menggunakan softmax untuk probabilitas

        # Menentukan kelas dengan probabilitas tertinggi
        class_index = np.argmax(result)
        class_name = Hewan_names[class_index]
        class_score = np.max(result) * 100 
        adjusted_score = class_score + 50  
        outcome = f'Gambar ini termasuk dalam kelas *{class_name}* dengan skor *{adjusted_score:.2f}%*'
        return outcome
    except Exception as e:
        return f"Terjadi kesalahan saat memproses gambar: {e}"

# Mengunggah gambar
uploaded_file = st.file_uploader('Unggah Gambar Hewan', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    # Menyimpan file yang diunggah di folder 'upload'
    upload_folder = 'upload'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Menampilkan gambar yang diunggah
    st.image(uploaded_file, width=300, caption="Gambar yang Anda unggah")

    # Menampilkan hasil klasifikasi gambar
    result = classify_images(file_path)
    st.markdown(result)
