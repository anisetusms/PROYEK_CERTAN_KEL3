import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

# Menampilkan judul aplikasi
st.header('Hewan Classification CNN Model')

# Daftar nama hewan yang sesuai dengan urutan output model
Hewan_names = ['CHEETAH', 'HARIMAU', 'MACAN_TUTUL', 'SINGA']

# Memuat model yang telah disimpan
model = load_model('Hewan_Recog_Model.keras')

# Fungsi untuk mengklasifikasikan gambar
def classify_images(image_path):
    # Memuat gambar dan mengubah ukurannya sesuai input model
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)  # Menambahkan dimensi batch

    # Melakukan prediksi
    predictions = model.predict(input_image_exp_dim)
    
    # Menggunakan softmax untuk mendapatkan probabilitas kelas
    result = tf.nn.softmax(predictions[0])
    
    # Menentukan kelas dengan probabilitas tertinggi
    outcome = f'Gambar ini termasuk dalam kelas {Hewan_names[np.argmax(result)]} dengan skor {np.max(result) * 100:.2f}%'
    return outcome

# Mengunggah gambar
uploaded_file = st.file_uploader('Unggah Gambar', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    # Menyimpan file yang diunggah di folder 'upload'
    upload_folder = 'upload'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Menampilkan gambar yang diunggah
    st.image(uploaded_file, width=200)

    # Menampilkan hasil klasifikasi gambar
    st.markdown(classify_images(file_path))
