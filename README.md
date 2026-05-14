# Hybrid Pet Breed Classifier

A modern, production-ready Streamlit web application that uses deep learning to classify dog and cat breeds from uploaded images. Beyond just predicting the breed, it leverages the **Google Gemini 2.5 Flash API** to instantly fetch and display concise, engaging insights (temperament, lifespan, care needs) about the predicted pet, and allows users to download a comprehensive PDF report of the analysis.

## ✨ Features

- **Advanced Machine Learning**: Uses an ensemble feature extraction pipeline (VGG16, ResNet50, MobileNetV2) combined with a highly accurate Support Vector Machine (SVM) classifier.
- **Modern, Premium UI**: Features a custom-designed, Next.js-inspired minimalist interface with clean typography (SF Pro), soft shadows, and a beautiful parchment color palette. 
- **AI-Powered Insights**: Integrates with the Google Gemini 2.5 Flash API to dynamically provide expert veterinary insights and facts about the classified breed.
- **Downloadable PDF Reports**: Generates an on-the-fly, downloadable PDF summary of the prediction and traits using `fpdf2`.
- **Session History**: Automatically keeps track of your recent uploads and predictions in the sidebar during your session.

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Custom Vanilla CSS
- **Machine Learning**: TensorFlow / Keras, Scikit-Learn, Joblib
- **Generative AI**: Google Gemini API (`generativelanguage.googleapis.com`)
- **Utilities**: `fpdf2` (PDF generation), `Pillow` (Image processing), `requests` (API calls)

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.9+ installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AnshikaPandey13/pet-breed-classifier.git
   cd pet-breed-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your API Keys:
   This application requires a Google Gemini API key to fetch breed information. 
   Create a `.streamlit/secrets.toml` file in the root directory and add your key:
   ```toml
   # .streamlit/secrets.toml
   GEMINI_API_KEY = "your_api_key_here"
   ```
   *(Alternatively, you can provide it via a `.env` file or export it as an OS environment variable).*

### Running the App

Run the Streamlit application locally:
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

## 📁 Repository Structure

- `app.py`: The main Streamlit application script containing the UI, prediction logic, and API calls.
- `requirements.txt`: Python package dependencies.
- `*.joblib`: Pre-trained models and transformers (`pca.joblib`, `scaler.joblib`, `svm_best.joblib`, `class_names.joblib`).
- `.streamlit/secrets.toml`: (Create this) Secure storage for your API keys.

## 📝 License

This project is licensed under the MIT License.
