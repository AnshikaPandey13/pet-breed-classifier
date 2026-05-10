
import streamlit as st
import numpy as np
import joblib
from PIL import Image

from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Pet Breed Classifier",
    page_icon="🐾",
    layout="centered"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* Hide Streamlit Branding */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Background */
.stApp {
    background: linear-gradient(to bottom right, #d2b48c, #c19a6b, #a67b5b);
    color: white;
}

/* Main Container */
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

/* Top Glass Card */
.top-card {
    background: rgba(255,255,255,0.12);
    padding: 28px;
    border-radius: 30px;
    text-align: center;
    font-size: 52px;
    font-weight: 700;
    color: white;
    margin-bottom: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #f8f5f0;
    margin-bottom: 35px;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 15px;
}

/* Upload Text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div {
    color: #5b4636 !important;
    font-weight: 500;
}

/* Upload Button */
[data-testid="stFileUploader"] button {
    background-color: rgba(255,255,255,0.35) !important;
    color: #5b4636 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: bold !important;
}

/* Uploaded Image */
img {
    border-radius: 20px;
    margin-top: 15px;
}

/* Success Box */
.stSuccess {
    background-color: #f5f1eb !important;
    border-radius: 20px !important;
    padding: 18px !important;
    border: none !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}

.stSuccess p {
    color: #5b4636 !important;
    font-size: 28px !important;
    font-weight: bold !important;
    text-align: center;
}

/* Spinner */
.stSpinner > div {
    border-top-color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE CARD
# ---------------------------------------------------
st.markdown(
    """
    <div class="top-card">
        Pet Breed Classifier
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SUBTITLE
# ---------------------------------------------------
st.markdown(
    """
    <div class="subtitle">
        AI-powered breed recognition for cats and dogs
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD SAVED MODELS
# ---------------------------------------------------
@st.cache_resource
def load_saved_models():

    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    svm_model = joblib.load("svm_best.joblib")
    class_names = joblib.load("class_names.joblib")

    return scaler, pca, svm_model, class_names

# ---------------------------------------------------
# LOAD CNN MODELS
# ---------------------------------------------------
@st.cache_resource
def load_feature_extractors():

    vgg = VGG16(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )

    resnet = ResNet50(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )

    mobilenet = MobileNetV2(
        weights='imagenet',
        include_top=False,
        pooling='avg'
    )

    return vgg, resnet, mobilenet

# ---------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------
def extract_features(image, vgg, resnet, mobilenet):

    image = image.resize((224, 224))
    image = image.convert("RGB")

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # VGG16 Features
    vgg_input = vgg_pre(img_array.copy())
    vgg_features = vgg.predict(vgg_input, verbose=0).flatten()

    # ResNet50 Features
    res_input = res_pre(img_array.copy())
    res_features = resnet.predict(res_input, verbose=0).flatten()

    # MobileNetV2 Features
    mob_input = mob_pre(img_array.copy())
    mob_features = mobilenet.predict(mob_input, verbose=0).flatten()

    # Combine Features
    final_features = np.concatenate([
        vgg_features,
        res_features,
        mob_features
    ])

    return final_features

# ---------------------------------------------------
# LOAD EVERYTHING
# ---------------------------------------------------
scaler, pca, svm_model, class_names = load_saved_models()

vgg, resnet, mobilenet = load_feature_extractors()

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Pet Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Analyzing Image..."):

        try:

            # Extract Features
            features = extract_features(
                image,
                vgg,
                resnet,
                mobilenet
            )

            # Scale Features
            scaled_features = scaler.transform([features])

            # PCA Transform
            pca_features = pca.transform(scaled_features)

            # Predict
            prediction = svm_model.predict(pca_features)[0]

            # Breed Name
            breed = str(
                class_names[prediction]
            ).replace("_", " ").title()

            # Detect Cat or Dog
            breed_lower = breed.lower()

            cat_keywords = [
                "cat",
                "persian",
                "siamese",
                "maine",
                "ragdoll",
                "bengal",
                "british"
            ]

            is_cat = any(
                word in breed_lower
                for word in cat_keywords
            )

            pet_icon = "🐱" if is_cat else "🐶"

            # RESULT
            st.markdown("<h3 style='text-align:center; color:white;'>Predicted Breed</h3>", unsafe_allow_html=True)

            st.success(f"{pet_icon} {breed}")

        except Exception as e:

            st.error("Prediction Error")
            st.exception(e)