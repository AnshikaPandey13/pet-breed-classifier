import streamlit as st
import numpy as np
import joblib
import requests
from PIL import Image

from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="PET BREED CLASSIFIER",
    page_icon="🐾",
    layout="centered"
)

# ---------------------------------------------------
# SESSION STATE & HELPERS
# ---------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_data(ttl=86400)
def get_breed_info(breed_name, is_cat=False, version=2):
    import os
    try:
        pet_type = "cat" if is_cat else "dog"
        
        # Securely fetch API key
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        else:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            
        if not api_key:
            return {"description": "API Key is missing. Please configure secrets."}
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"
        
        system_prompt = (
            "You are a veterinary and pet expert AI. "
            "Provide accurate information about pet breeds. "
            "The breed name provided might be partial (e.g., 'English' instead of 'English Bulldog' or 'English Setter'). "
            "Use your best judgment to identify the specific breed of the given pet type. "
            "If ambiguous, discuss the most common breed that fits the partial name."
        )
        
        prompt = (
            f"Provide a concise, engaging summary of the {breed_name} {pet_type}. "
            "Write a brief introductory sentence. "
            "Then, provide exactly 3 short bullet points highlighting key traits (e.g., temperament, lifespan, care). "
            "IMPORTANT: Use the HTML <br> tag for line breaks. Use <b> tags for emphasis. "
            "Do NOT use markdown. Do NOT use markdown lists or asterisks. "
            "Use the '•' character for bullet points. Output only the raw HTML text without any code block wrappers."
        )
        
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 600}
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [])[0].get("text", "")
                if text:
                    # Clean up potential code block wrappers
                    text = text.replace("```html", "").replace("```", "").strip()
                    return {"description": text}
                    
    except Exception:
        pass
        
    return {
        "description": f"The {breed_name} is a known pet breed. Unfortunately, detailed traits could not be fetched from the AI right now."
    }   

def generate_report(breed, info):
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="PET BREED CLASSIFICATION REPORT", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"Predicted Breed: {breed}", ln=True)
        pdf.set_font("Arial", '', 12)
        import re
        clean_text = info['description'].replace('<br>', '\n').replace('<br/>', '\n')
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        safe_desc = clean_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=f"Description:\n{safe_desc}")
        out = pdf.output(dest='S')
        pdf_bytes = out.encode('latin-1') if hasattr(out, 'encode') else bytes(out)
        return pdf_bytes, "application/pdf", f"{breed}_report.pdf"
    except ImportError:
        import re
        clean_text = info['description'].replace('<br>', '\n').replace('<br/>', '\n')
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        report = f"PET BREED REPORT\n\nPredicted: {breed}\n\nDescription:\n{clean_text}\n"
        return report, "text/plain", f"{breed}_report.txt"

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.cdnfonts.com/css/sf-pro-display');

html, body, [class*="css"], .stApp {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
html, body, .stApp {
    scroll-behavior: smooth;
}

/* Clean Header & Sidebar Toggle */
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}
.stAppDeployButton, .stDeployButton {display: none !important;}
#MainMenu {display: none !important;}
footer {display: none !important;}

/* Ensure sidebar toggle is visible and styled */
[data-testid="collapsedControl"] {
    display: flex !important;
    color: #463f3a !important;
    background: #ffffff !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 8px rgba(70, 63, 58, 0.1) !important;
}

/* Modern Minimalist Background */
.stApp {
    background: #f4f3ee;
    color: #463f3a;
    padding-bottom: 80px; /* Space for footer */
}

/* Main Container */
.block-container {
    max-width: 900px;
    padding-top: 1rem;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #f4f3ee !important;
    border-right: 1px solid #bcb8b1;
}

/* Hero */
.hero-container {
    text-align: center;
    margin-bottom: 20px;
    animation: fadeInDown 0.8s ease-out;
}

.title-text {
    font-size: 56px;
    font-weight: 800;
    color: #463f3a;
    margin-bottom: 5px;
    letter-spacing: -2px;
}

.subtitle {
    font-size: 18px;
    color: #8a817c;
    font-weight: 400;
}

/* File Uploader Container */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #bcb8b1;
    border-radius: 12px;
    padding: 15px;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #463f3a;
    box-shadow: 0 4px 12px rgba(70, 63, 58, 0.08);
}

/* File Uploader Text Fix */
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color: #463f3a;
    font-weight: 500;
}



/* Uploaded Image rounding */
img {
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}

/* Spinner styling */
.stSpinner > div {
    border-top-color: #463f3a !important;
}

/* Footer */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: #ffffff;
    border-top: 1px solid #bcb8b1;
    text-align: center;
    padding: 15px;
    font-size: 14px;
    color: #8a817c;
    z-index: 100;
}
.footer a {
    color: #463f3a;
    text-decoration: none;
    font-weight: 600;
}

/* Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="title-text">PET BREED CLASSIFIER</div>
        <div class="subtitle">AI-powered breed recognition for cats & dogs</div>
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
# SIDEBAR HISTORY
# ---------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: #463f3a; text-align: center; font-size: 20px;'>Upload History</h2>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("<p style='color: #8a817c; text-align: center;'>No pets classified yet.</p>", unsafe_allow_html=True)
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            st.image(item['image'], use_container_width=True)
            st.markdown(f"<p style='color: #463f3a; text-align: center; font-weight: 600; margin-top: 5px;'>{item['breed']}</p>", unsafe_allow_html=True)
            if idx < len(st.session_state.history) - 1:
                st.markdown("<hr style='border: none; border-top: 1px solid #bcb8b1; margin: 15px 0;'>", unsafe_allow_html=True)

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Pet Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="hidden"
)

# ---------------------------------------------------
# LANDING PAGE (IF NO UPLOAD)
# ---------------------------------------------------
if uploaded_file is None:
    st.markdown("""
        <div style="margin-top: 10px; margin-bottom: 30px; animation: fadeUp 0.8s ease-out;">
            <div style="color: #463f3a; font-size: 20px; margin-bottom: 15px; font-weight: 700;">How it works</div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 200px; padding: 24px; background: #ffffff; border: 1px solid #bcb8b1; border-radius: 12px; box-shadow: 0 2px 8px rgba(70,63,58,0.04);">
                    <div style="width: 40px; height: 40px; border-radius: 8px; background: #e0afa0; display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                        <span style="font-size: 20px;">📸</span>
                    </div>
                    <div style="color: #463f3a; margin: 0 0 10px 0; font-size: 18px; font-weight: 700;">1. Upload Photo</div>
                    <p style="color: #8a817c; font-size: 14px; margin: 0; line-height: 1.5;">Upload a clear picture of a cat or dog. We support JPG and PNG formats.</p>
                </div>
                <div style="flex: 1; min-width: 200px; padding: 24px; background: #ffffff; border: 1px solid #bcb8b1; border-radius: 12px; box-shadow: 0 2px 8px rgba(70,63,58,0.04);">
                    <div style="width: 40px; height: 40px; border-radius: 8px; background: #bcb8b1; display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                        <span style="font-size: 20px;">🧠</span>
                    </div>
                    <div style="color: #463f3a; margin: 0 0 10px 0; font-size: 18px; font-weight: 700;">2. AI Analysis</div>
                    <p style="color: #8a817c; font-size: 14px; margin: 0; line-height: 1.5;">Our custom CNN ensemble models extract visual features instantly.</p>
                </div>
                <div style="flex: 1; min-width: 200px; padding: 24px; background: #ffffff; border: 1px solid #bcb8b1; border-radius: 12px; box-shadow: 0 2px 8px rgba(70,63,58,0.04);">
                    <div style="width: 40px; height: 40px; border-radius: 8px; background: #8a817c; display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                        <span style="font-size: 20px;">✨</span>
                    </div>
                    <div style="color: #463f3a; margin: 0 0 10px 0; font-size: 18px; font-weight: 700;">3. Get Results</div>
                    <p style="color: #8a817c; font-size: 14px; margin: 0; line-height: 1.5;">Discover the exact breed, confidence scores, and fascinating traits.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:
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

                # CUSTOM RESULT CARD
                st.markdown(f"""
                    <div style="
                        background: #ffffff;
                        border: 1px solid #bcb8b1;
                        padding: 30px;
                        border-radius: 12px;
                        text-align: center;
                        box-shadow: 0 4px 12px rgba(70,63,58,0.08);
                        margin-top: 20px;
                        margin-bottom: 20px;
                        animation: fadeUp 0.6s ease-out;
                    ">
                        <p style="color: #8a817c; font-size: 14px; font-weight: 500; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">
                            Analysis Complete
                        </p>
                        <h2 style="
                            color: #463f3a; 
                            margin: 0; 
                            font-size: 42px; 
                            font-weight: 800;
                            letter-spacing: -2px;
                        ">
                            {pet_icon} {breed}
                        </h2>
                    </div>
                """, unsafe_allow_html=True)

                # BREED TRAITS
                info = get_breed_info(breed, is_cat, version=2)
                st.markdown(f"""
                    <div style="margin-top: 30px; padding: 30px; background: #ffffff; border: 1px solid #bcb8b1; border-radius: 12px; box-shadow: 0 2px 8px rgba(70,63,58,0.04); animation: fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1);">
                        <h4 style="color: #463f3a; margin-top: 0; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 700;">
                            <span>✨</span> About {breed}
                        </h4>
                        <div style="color: #463f3a; font-size: 16px; margin: 0; line-height: 1.8; font-weight: 400;">
                            {info['description']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # ADD TO HISTORY
                if not any(item.get('name') == uploaded_file.name for item in st.session_state.history):
                    st.session_state.history.append({
                        "image": image,
                        "breed": breed,
                        "name": uploaded_file.name
                    })

                # DOWNLOAD REPORT
                st.markdown("<br>", unsafe_allow_html=True)
                report_data, mime_type, file_name = generate_report(breed, info)
                st.download_button(
                    label="📥 Download Prediction Report",
                    data=report_data,
                    file_name=file_name,
                    mime=mime_type,
                    use_container_width=True
                )

            except Exception as e:

                st.error("Oops! Something went wrong while analyzing the image. Please make sure you uploaded a valid pet photo.")
                with st.expander("Show technical details"):
                    st.exception(e)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Developed by <strong>Anshika Pandey</strong> | 
        <a href="https://github.com/AnshikaPandey13/pet-breed-classifier" target="_blank">View on GitHub</a> | 
        Version 1.0.0
    </div>
    """,
    unsafe_allow_html=True
)