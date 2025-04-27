import streamlit as st
import numpy as np
from keras.layers import TFSMLayer
from PIL import Image
import tensorflow as tf
import os
import gdown
import zipfile

# Download model kalau belum ada
if not os.path.exists("saved_model"):
    url = "https://drive.google.com/uc?id=1ETPY-6YAxYr6VHIfpAv64-CzsPeKQW1L" 
    output = "model.zip"
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(".")

# Load model
model = TFSMLayer("saved_model", call_endpoint="serving_default")

# --- Class Labels ---
class_labels = [
    "kue_dadar_gulung", "kue_kastengel", "kue_klepon", "kue_lapis",
    "kue_lumpur", "kue_putri_salju", "kue_risoles", "kue_serabi"
]

# --- Resep Dictionary ---
resep_dict = {
    "Kue Dadar Gulung": {
        "deskripsi": "Dadar Gulung adalah salah satu makanan tradisional Indonesia berupa pancake isi kelapa dan gula jawa.",
        "bahan": [
            "Bahan Kue:",
            "- 200 gram terigu",
            "- 1 butir telur",
            "- 500 ml air",
            "- 1/2 sdm gula",
            "- 1/4 sdt garam",
            "- Sedikit vanili",
            "Bahan Vla:",
            "- 25 gram tepung maizena",
            "- Gula pasir secukupnya",
            "- 250 ml santan",
            "- 1 kuning telur",
            "- Garam secukupnya",
            "- 1/2 sdt vanili",
            "- 20 gram cokelat bubuk dilarutkan"
        ],
        "cara_membuat": [
            "Buat kulit: campur tepung terigu, garam, gula, lalu tambahkan air dan telur. Aduk rata.",
            "Masak kulit di wajan datar beroles margarin hingga matang.",
            "Buat vla: campur maizena, kuning telur, dan santan. Sisihkan.",
            "Masak santan, gula, garam, dan cokelat bubuk hingga mendidih. Matikan api, tambahkan vanili dan larutan maizena. Aduk hingga mengental.",
            "Isi kulit dadar dengan vla, lalu gulung." 
        ]
    },
    "Kue Kastengel": {
        "deskripsi": "Kue kering berbentuk batang kecil dengan rasa gurih keju, biasanya disajikan saat hari raya.",
        "bahan": [
            "- 200 gr mentega",
            "- 30 gr gula halus",
            "- 2 butir telur",
            "- 250 gr tepung terigu",
            "- 50 gr maizena",
            "- 30 gr susu bubuk",
            "- 150 gr keju gouda",
            "- 100 gr keju cheddar"
        ],
        "cara_membuat": [
            "Kocok mentega dan gula halus hingga rata.",
            "Masukkan kuning telur, aduk rata.",
            "Tambahkan tepung terigu, maizena, susu bubuk, keju gouda, dan keju cheddar. Aduk rata.",
            "Bentuk adonan persegi panjang, oles kuning telur, tabur keju, lalu oven suhu 140°C selama 40-50 menit."
        ]
    },
    "Kue Klepon": {
        "deskripsi": "Kue tradisional berbentuk bola berisi gula merah cair, berbalut kelapa parut.",
        "bahan": [
            "- 500 gr tepung ketan putih",
            "- 50 gr tepung beras",
            "- 3 lembar daun pandan",
            "- 300 gr gula merah",
            "- 200 ml air hangat",
            "- Pasta pandan secukupnya",
            "- 1/2 sdt garam",
            "- 1 sachet vanili",
            "- 250 gr kelapa parut",
            "- Daun pisang untuk sudi (opsional)",
            "- 1.5 lt air untuk merebus"
        ],
        "cara_membuat": [
            "Kukus kelapa parut dengan sedikit garam selama 30 menit, sisihkan.",
            "Campur air hangat, garam, vanili, dan pasta pandan.",
            "Campur tepung ketan dan tepung beras, tuangi campuran air sedikit demi sedikit, uleni hingga kalis.",
            "Isi adonan dengan gula merah sisir, bulatkan.",
            "Rebus klepon hingga mengapung, angkat, dan gulingkan di kelapa parut."
        ]
    },
    "Kue Lapis": {
        "deskripsi": "Kue tradisional dengan lapisan warna-warni cantik, kenyal, dan manis.",
        "bahan": [
            "- 125 gr tepung beras",
            "- 75 gr tepung tapioka",
            "- 125 gr gula pasir",
            "- 1/2 sdt garam",
            "- 200 ml santan kental",
            "- 500 ml air",
            "- 4 helai pandan",
            "- Vanili secukupnya",
            "- Pewarna makanan"
        ],
        "cara_membuat": [
            "Masak santan dengan garam dan daun pandan hingga mendidih. Sisihkan hingga hangat.",
            "Campur semua bahan, aduk rata.",
            "Tuang selapis demi selapis ke kukusan hingga habis.",
            "Kukus lapisan terakhir lebih lama hingga matang sempurna."
        ]
    },
    "Kue Lumpur": {
        "deskripsi": "Kue dengan tekstur sangat lembut, terbuat dari kentang kukus dan santan.",
        "bahan": [
            "- 250 gram kentang kukus",
            "- 250 gram tepung terigu",
            "- 2 butir telur",
            "- 550 ml santan",
            "- 3 sdm mentega",
            "- 100 gram gula pasir",
            "- 1/2 sdt garam",
            "- Kismis/keju"
        ],
        "cara_membuat": [
            "Haluskan kentang dengan santan, blender hingga lembut.",
            "Kocok telur dan gula hingga rata.",
            "Campurkan adonan kentang ke dalam kocokan telur, aduk rata.",
            "Panaskan cetakan, tuang adonan, beri topping, masak hingga matang."
        ]
    },
    "Kue Putri Salju": {
        "deskripsi": "Kue kering berbentuk bulan sabit bersalut gula halus, sangat populer saat hari raya.",
        "bahan": [
            "- 400 gram tepung terigu",
            "- 50 gram tepung maizena",
            "- 300 gram mentega",
            "- 100 gram gula halus",
            "- 2 kuning telur",
            "- 100 gram keju parut",
            "- Gula halus dan gula donat untuk taburan"
        ],
        "cara_membuat": [
            "Kocok mentega, gula, dan kuning telur hingga rata.",
            "Tambahkan keju parut, aduk rata.",
            "Masukkan tepung terigu dan maizena sedikit demi sedikit hingga adonan kalis.",
            "Cetak bentuk bulan sabit, oven 150°C selama 20 menit.",
            "Taburi gula halus dan gula donat setelah dingin."
        ]
    },
    "Kue Risoles": {
        "deskripsi": "Pastry goreng isi sayur atau daging, kulitnya tipis dan crispy setelah dilapisi panir.",
        "bahan": [
            "- 1 bungkus bihun kering",
            "- 3 siung bawang merah",
            "- 2 siung bawang putih",
            "- 5 buah cabai rawit",
            "- Sayur kol, wortel",
            "- 10 kulit lumpia",
            "- Kaldu jamur"
        ],
        "cara_membuat": [
            "Rendam bihun hingga lemas, tiriskan.",
            "Tumis bawang merah, bawang putih, cabai rawit, masukkan kol dan wortel.",
            "Masukkan bihun, beri kaldu jamur, aduk rata.",
            "Ambil kulit lumpia, beri isian, lipat dan gulung.",
            "Goreng hingga kecokelatan."
        ]
    },
    "Kue Serabi": {
        "deskripsi": "Pancake tradisional Indonesia, dibuat dari tepung beras dan santan, biasanya disajikan dengan kuah manis.",
        "bahan": [
            "- 150 gr terigu",
            "- 50 gr tepung beras",
            "- 1/2 sdt ragi",
            "- 1 sdt baking powder",
            "- 1/2 sdt garam",
            "- 2 sdm gula",
            "- 1 telur",
            "- 65 ml santan",
            "- 200 ml air pandan",
            "- Bahan kuah kinca: 200 gr gula merah, 3 sdm gula pasir, 300 ml air, 200 ml santan, 1/2 sdt garam, 1 lembar pandan"
        ],
        "cara_membuat": [
            "Campur bahan kering, aduk rata.",
            "Tambahkan bahan basah, aduk rata.",
            "Tutup adonan dan diamkan 30 menit.",
            "Masak adonan di teflon hingga bersarang.",
            "Buat kuah kinca: masak semua bahan hingga mendidih, saring.",
            "Sajikan serabi dengan kuah kinca."
        ]
    }
}

# --- Streamlit UI ---
st.set_page_config(page_title="Prediksi Kue Tradisional Indonesia", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #fff8f0;
    }
    h1 {
        color: #d2691e;
        font-family: 'Trebuchet MS', sans-serif;
        text-align: center;
    }
    .stButton>button {
        background-color: #d2691e;
        color: white;
        border-radius: 10px;
    }
    .st-expanderHeader {
        font-weight: bold;
        color: #d2691e;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍮 Prediksi dan Resep Kue Tradisional Indonesia")
st.markdown("---")

uploaded_file = st.file_uploader("📷 Upload foto kue tradisional Indonesia kamu di sini!", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='📸 Gambar yang Diupload', use_column_width=True)

    with st.spinner('⏳ Sedang menebak jenis kue tradisional...'):
        input_image = img.resize((150, 150))
        input_array = np.array(input_image) / 255.0
        input_array = np.expand_dims(input_array, axis=0).astype(np.float32)

        prediction_dict = model(input_array)
        prediction_tensor = list(prediction_dict.values())[0]
        prediction_array = prediction_tensor.numpy()[0]

        predicted_index = np.argmax(prediction_array)
        predicted_class = class_labels[predicted_index]
        confidence = np.max(prediction_array) * 100

        predicted_class_display = predicted_class.replace('_', ' ').title()

    st.markdown("---")

    if confidence < 50:
        st.warning("🚨 Hasil kurang pasti! Sepertinya kami belum bisa mengenali kuenya dengan baik. Coba upload foto dengan pencahayaan yang lebih bagus, ya!")
        st.info(f"🔎 Tingkat Keyakinan: **{confidence:.2f}%**")
    else:
        st.success(f"🎯 Kami yakin ini adalah: **{predicted_class_display}**")
        st.info(f"🔎 Tingkat Keyakinan: **{confidence:.2f}%**")

        resep = resep_dict.get(predicted_class_display, None)

        if resep:
            with st.expander("📖 Tentang Kue Ini"):
                st.write(resep['deskripsi'])

            with st.expander("🛒 Bahan-Bahan yang Diperlukan"):
                for item in resep['bahan']:
                    st.markdown(f"{item}")

            with st.expander("👩‍🍳 Cara Membuat Kue"):
                for idx, langkah in enumerate(resep['cara_membuat'], start=1):
                    st.markdown(f"{idx}. {langkah}")
        else:
            st.warning("Maaf, resep untuk kue ini belum tersedia.")

st.markdown("---")
st.caption("🇮🇩 Dibuat dengan cinta untuk kekayaan kuliner Nusantara | Powered by Streamlit")
