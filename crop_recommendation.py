import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Smart Crop Recommender",
    page_icon="🌾",
    layout="wide"
)

# Robust model loading function
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Joblib loading failed: {str(e)}. Trying pickle alternatives...")
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except UnicodeDecodeError:
            with open(model_path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e:
            with open(model_path, 'rb') as f:
                return pickle.load(f, encoding='bytes')
            st.error(f"All loading methods failed: {str(e)}")
            raise

# Load the trained model
try:
    model = load_model('best_model_Random_Forest.pkl')
    # Success message hidden to prevent text contrast issues
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Crop dictionary - update this with your actual labels
crop_dict = {
    0: 'Apple', 1: 'Banana', 2: 'Blackgram', 3: 'Chickpea', 4: 'Coconut',
    5: 'Coffee', 6: 'Cotton', 7: 'Grapes', 8: 'Jute', 9: 'Kidneybeans',
    10: 'Lentil', 11: 'Maize', 12: 'Mango', 13: 'Mothbeans', 14: 'Mungbean',
    15: 'Muskmelon', 16: 'Orange', 17: 'Papaya', 18: 'Pigeonpeas',
    19: 'Pomegranate', 20: 'Rice', 21: 'Watermelon'
}

# Main content with clean, high-contrast design
st.markdown(
    """
    <style>
    /* Remove background image and set solid colors */
    .stApp {
        background-color: #f5f9f4;
    }
    
    .main-container {
        background-color: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        margin-top: 20px;
        color: #2d3436;
    }
    .header-title {
        color: #2e7d32 !important;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subheader {
        color: #388E3C !important;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .recommend-box {
        background: #e8f5e9 !important;
        border-left: 5px solid #4caf50;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        color: #1b5e20;
    }
    .parameter-box {
        background: #f1f8e9 !important;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        color: #2d3436;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #388E3C !important;
        transform: scale(1.05);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .footer {
        background-color: #2d3436 !important;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        color: #f5f6fa !important;
    }
    .stSlider > div > div > div {
        color: #2d3436 !important;
    }
    .stSlider label {
        font-weight: 600 !important;
        color: #2d3436 !important;
    }
    .stDataFrame {
        color: #2d3436 !important;
    }
    .stMarkdown {
        color: #2d3436 !important;
    }
    .divider {
        border-top: 2px solid #4CAF50;
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main content container
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header with high contrast
    st.markdown('<h1 class="header-title">🌾 Smart Agriculture Crop Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Optimize your farming with AI-powered crop suggestions</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Input columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌱 Soil Composition")
        n = st.slider('**Nitrogen (N) ratio**', 0, 150, 50, help="Essential for leaf growth")
        p = st.slider('**Phosphorous (P) ratio**', 0, 150, 30, help="Important for root development")
        k = st.slider('**Potassium (K) ratio**', 0, 200, 30, help="Vital for overall plant health")
        ph = st.slider('**pH Level**', 0.0, 14.0, 6.5, step=0.1, help="Soil acidity/alkalinity level")

    with col2:
        st.subheader("🌤️ Climate Conditions")
        temp = st.slider('**Temperature (°C)**', 0.0, 50.0, 25.0, step=0.5, help="Average daily temperature")
        humidity = st.slider('**Humidity (%)**', 0.0, 100.0, 60.0, step=0.5, help="Relative humidity level")
        rainfall = st.slider('**Rainfall (mm)**', 0.0, 500.0, 150.0, step=1.0, help="Annual rainfall amount")
    
    # Recommendation button
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        if st.button('**Get Crop Recommendation**', use_container_width=True):
            try:
                input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
                prediction = model.predict(input_data)
                crop_name = crop_dict.get(prediction[0], "Unknown Crop")
                
                # Display recommendation with high contrast
                st.markdown(f'<div class="recommend-box">'
                            f'<h2>Recommended Crop: <span style="color:#1b5e20; font-weight: 700;">{crop_name}</span></h2>'
                            f'<p>Based on your soil and climate parameters, this crop has the optimal growth potential</p>'
                            f'</div>', unsafe_allow_html=True)
                
                # Display parameters
                st.subheader("📊 Input Parameters Summary")
                params = pd.DataFrame({
                    'Parameter': ['Nitrogen (N)', 'Phosphorous (P)', 'Potassium (K)', 
                                 'Temperature', 'Humidity', 'pH Level', 'Rainfall'],
                    'Value': [f'{n} ppm', f'{p} ppm', f'{k} ppm', 
                             f'{temp} °C', f'{humidity}%', ph, f'{rainfall} mm']
                })
                st.dataframe(params.set_index('Parameter'), use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer with high contrast
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("### 🌍 Sustainable Agriculture Insights")
st.markdown("""
- **Nitrogen (N):** Essential for chlorophyll production and vegetative growth
- **Phosphorous (P):** Crucial for root development and energy transfer
- **Potassium (K):** Important for water regulation and disease resistance
""")
st.markdown("</div>", unsafe_allow_html=True)

st.caption("_Model: Random Forest Classifier | Trained on comprehensive agricultural dataset_")