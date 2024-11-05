import streamlit as st
import torch
import numpy as np
from PIL import Image
import tempfile
import cv2
import torch.nn as nn
import os
from streamlit_lottie import st_lottie
import requests

# Model definition
MODEL_URL = "https://drive.google.com/file/d/1ESk9azTVC8VwWtK_nEZs7unEIWtaWqVV/view?usp=sharing"  # e.g., direct download link from Google Drive

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
def download_model():
    model_path = "Final_weights.pth"
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        response = requests.get(MODEL_URL)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    return model_path

def load_model():
    model_path = download_model()
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image).transpose((2, 0, 1)) / 255.0
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (128, 128))
        frame_tensor = torch.tensor(resized_frame.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
        frames.append(frame_tensor)
        if len(frames) == 10:
            break
    cap.release()
    return frames

# Page configuration
st.set_page_config(
    page_title="3D Printing Defect Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.streamlit.io/support',
        'Report a bug': "https://www.streamlit.io/support/request",
        'About': "# 3D Printing Defect Detector\n\nA powerful AI tool for detecting defects in 3D printed objects. Analyze images and videos with ease!"
    }
)

# Custom CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    :root {{
        --primary-color: #7209b7;
        --secondary-color: #4361ee;
        --accent-color: #4cc9f0;
        --background-color: #f8f9fa;
        --text-color: #2b2d42;
    }}

    .stApp {{
        background: linear-gradient(135deg, #f6f8ff 0%, #ffffff 100%);
        font-family: 'Poppins', sans-serif;
    }}

    .main-header {{
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }}

    .main-header h1 {{
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        padding: 0;
        line-height: 1.2;
    }}

    .upload-section {{
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        margin-top: 2rem;
    }}

    .results-card {{
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        margin-top: 2rem;
    }}

    .stButton>button {{
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }}

    .status-success {{
        color: #10b981;
        font-weight: 600;
        font-size: 1.5rem;
    }}

    .status-error {{
        color: #ef4444;
        font-weight: 600;
        font-size: 1.5rem;
    }}

    .sidebar .sidebar-content {{
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Adjust default Streamlit spacing */
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }}
    
    /* Reduce space between elements */
    .stSelectbox, .stRadio {{
        margin-bottom: 0.75rem;
    }}
    
    /* Compact file uploader */
    .uploadedFile {{
        margin-bottom: 0.75rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header"><h1>🔍 3D Printing Defect Detector</h1></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    option = st.radio("Select Input Type", ["Image", "Video"], index=0)
    
    st.markdown("### About")
    st.markdown("""
    This AI-powered tool helps you detect defects in 3D printed objects.
    
    **Features:**
    - Real-time analysis
    - Support for images and videos
    - Advanced AI model
    """)

# Load model
model = load_model()

# Main content
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image for analysis:", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            with st.spinner("🔄 Processing Image..."):
                preprocessed_image = preprocess_image(image)
                with torch.no_grad():
                    prediction = model(preprocessed_image)
                    predicted_defect = torch.argmax(prediction, dim=1).item()

                result_text = "Defective ❌" if predicted_defect else "No Defect Detected ✅"
                result_class = "status-error" if predicted_defect else "status-success"
                st.markdown(f'<div class="results-card"><h2>Analysis Results</h2><p class="{result_class}">{result_text}</p></div>', unsafe_allow_html=True)

else:
    uploaded_file = st.file_uploader("Upload a video for analysis:", type=["mp4"], accept_multiple_files=False)

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.video(uploaded_file)

            with st.spinner("🔄 Processing Video..."):
                frames = preprocess_video(tmp_file.name)
                defect_counts = sum(torch.argmax(model(frame.unsqueeze(0)), dim=1).item() for frame in frames)
                is_defective = defect_counts > len(frames) / 2
                
                result_text = "Defective ❌" if is_defective else "No Defect Detected ✅"
                result_class = "status-error" if is_defective else "status-success"
                st.markdown(f'<div class="results-card"><h2>Analysis Results</h2><p class="{result_class}">{result_text}</p></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 2rem; padding: 1rem; background: white; border-radius: 20px; box-shadow: 0 6px 20px rgba(0,0,0,0.05);'>
    <p style='color: #666; font-size: 1.1rem; margin: 0;'>Developed by Group 10 | Anirudh, Asfaq, Ashmit, Nitish</p>
    <p style='color: #888; font-size: 1rem; margin: 0;'> AI MFG 2024 3D Printing Defect Detector</p>
</div>
""", unsafe_allow_html=True)
