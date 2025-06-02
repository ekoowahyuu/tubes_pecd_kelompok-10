import cv2
import streamlit as st
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

# Load model dan komponen pendukung (pastikan file .pkl ada di folder yang sama)
model_data = joblib.load('cat_and_dog.pkl')
ensemble = model_data['model']
scaler = model_data['scaler']
selector = model_data['selector']
le = model_data['label_encoder']
selected_pets = model_data['classes']

# --- Styling dengan warna custom ---
st.set_page_config(
    page_title="Cat & Dog Pet Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Background utama - putih krem terang */
    .stApp {
        background-color: #F1EFEC;
        color: #030303;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 18px;
        line-height: 1.6;
    }

    /* Sidebar kiri - biru navy tua */
    [data-testid="stSidebar"] {
        background-color: #123458 !important;
        color: #F1EFEC !important;
        font-size: 18px !important;
    }
    [data-testid="stSidebar"] * {
        color: #F1EFEC !important;
    }

    /* Header utama & sidebar header - biru navy tua */
    .css-18e3th9, .css-1v3fvcr h1, .css-1v3fvcr h2, .css-1v3fvcr h3 {
        color: #123458 !important;
        font-weight: 700;
    }
    /* Judul dalam sidebar (header) */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1 {
        color: #F1EFEC !important;
        font-weight: 700;
    }

    /* Tombol prediksi - biru navy tua */
    .stButton>button {
        background-color: #123458;
        color: white;
        font-weight: 600;
        border-radius: 20px;
        padding: 14px 28px;
        font-size: 18px;
        border: none;
        box-shadow: 0 4px 8px rgba(18, 52, 88, 0.6);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0f2f4b;
        box-shadow: 0 6px 12px rgba(15, 47, 75, 0.8);
        cursor: pointer;
    }

    /* Tombol upload file custom - pakai warna navy yang sama */
    input[type="file"] {
        background-color: #123458 !important;
        color: #F1EFEC !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        cursor: pointer !important;
        border: none !important;
        font-size: 18px !important;
    }
    input[type="file"]::file-selector-button {
        background-color: #123458 !important;
        color: #F1EFEC !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        cursor: pointer !important;
        border: none !important;
        font-size: 18px !important;
    }
    input[type="file"]::file-selector-button:hover {
        background-color: #0f2f4b !important;
    }

    /* Label dari file uploader */
    div[data-testid="stFileUploader"] > label {
        color: #000000 !important;
        font-size: 40px !important;
        font-weight: 700 !important;
        margin-bottom: 8px;
    }

    /* Info, Error, Success, Warning box */
    .stInfo, .stError, .stSuccess, .stWarning {
        border-radius: 16px;
        padding: 20px;
        font-weight: 700;
        font-size: 18px;
        color: #030303;
    }

    /* Judul utama */
    .stTitle, h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        color: #123458;
    }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar dengan daftar kelompok dan info
with st.sidebar:
    st.header("Kelompok 10")
    st.write("""
    - Eko Wahyu Setiawan — 1301223135  
    - Rafelisha Ananfadya — 1301223466  
    - Fatah Fadhlur Rohman FN — 1301223298  
    """)

    st.markdown("---")

    st.header("About")
    st.write(
        """
        Upload an image of a cat or dog and get a prediction of its pet type.
        \n- Supported formats: JPG, PNG, JPEG
        \n- Images will be resized internally to 256x256.
        \n\nModel trained using classical features: HOG, LBP, Color, and Texture.
        """
    )

st.title("🐾 Cat & Dog Pet Classifier")

def extract_features(img_array):
    img = cv2.resize(img_array, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hog_features = hog(gray, orientations=12, pixels_per_cell=(32, 32),
                       cells_per_block=(3, 3), feature_vector=True)

    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    color_features = []
    for i in range(3):
        channel = lab[:, :, i]
        color_features.extend([
            np.mean(channel),
            np.std(channel),
            np.median(channel),
            np.min(channel),
            np.max(channel)
        ])

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    texture_features = [
        np.mean(sobelx),
        np.std(sobelx),
        np.mean(sobely),
        np.std(sobely)
    ]

    return np.concatenate([hog_features, lbp_hist, color_features, texture_features])

uploaded_file = st.file_uploader("Choose a cat or dog image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Processing image and predicting pet type..."):
            try:
                features = extract_features(img_array)
                X = [features]

                X_scaled = scaler.transform(X)
                X_selected = selector.transform(X_scaled)

                prediction = ensemble.predict(X_selected)
                predicted_pet = le.inverse_transform(prediction)[0]

                st.markdown(f"### Predicted Pet: **{predicted_pet}**")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")


