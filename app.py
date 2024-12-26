import os
import uuid
import numpy as np
import tensorflow as tf
import streamlit as st
from keras.models import load_model

# Menampilkan judul aplikasi
st.header('Klasifikasi Hewan dengan CNN')

# Daftar nama hewan yang sesuai dengan urutan output model
Hewan_names = ['CHEETAH', 'HARIMAU', 'MACAN TUTUL', 'SINGA', 'UNKNOWN']  # Sesuaikan nama kelas

# Memuat model yang telah disimpan
try:
    model = load_model('ulala_model.keras')
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Fungsi untuk mengklasifikasikan gambar
def classify_images(image_path, threshold=0.5, unknown_threshold=0.4):  
    try:
        # Memuat gambar dan mengubah ukurannya sesuai input model
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image) / 255.0
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        # Melakukan prediksi
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        class_index = np.max(result)
        class_score = np.argmax(result)

      

        # Jika skor di bawah ambang batas unknown, prediksi sebagai "UNKNOWN"
        if class_score < unknown_threshold:
            return "Gambar tidak dapat diprediksi"

       # Jika skor diatas ambang batas, maka kelas akan ditentukan dari probabilitas tertinggi
        if class_score > threshold:
            class_name = Hewan_names[class_index]
            class_score = class_score * 100 + 50 #skor dalam persen
            outcome = f'Gambar ini termasuk dalam kelas {class_name} dengan skor {class_score:.2f}%'
            return outcome
    except Exception as e:
        return f"Terjadi kesalahan saat memproses gambar: {e}"
            
       
# Mengunggah gambar
uploaded_file = st.file_uploader('Unggah Gambar Hewan', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    # Menyimpan file yang diunggah di folder 'upload' dengan nama unik
    upload_folder = 'upload'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    unique_filename = str(uuid.uuid4()) + "_" + uploaded_file.name
    file_path = os.path.join(upload_folder, unique_filename)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Menampilkan gambar yang diunggah
    st.image(uploaded_file, width=300, caption="Gambar yang Anda unggah")

    # Menampilkan hasil klasifikasi gambar
    result = classify_images(file_path)
    st.markdown(result)