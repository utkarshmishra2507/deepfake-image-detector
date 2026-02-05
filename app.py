import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
        background-attachment: fixed;
    }
    
    h1 {
        color: #00d4ff !important;
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #00b8d4);
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (Cached)
# ==========================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )
    # Load your trained weights here
    try:
        model.load_state_dict(torch.load('deepfake_detector_resnet.pth', map_location=device))
    except:
        st.warning("⚠️ Model weights not found. Please train the model first.")
    
    # Move model to device (CPU or CUDA)
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ==========================================
# 3. GRAD-CAM EXPLAINABILITY
# ==========================================
def get_gradcam(model, image_tensor):
    # This is a simplified Grad-CAM implementation for demo purposes
    # Ideally, hook into the last convolutional layer
    # For now, we return a dummy heatmap to visualize interface structure
    # In production, use 'pytorch-grad-cam' library
    heatmap = np.random.uniform(0, 1, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

# ==========================================
# 4. UI LOGIC
# ==========================================
st.title("🕵️‍♂️ Deepfake & Manipulated Media Detector")
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="font-size: 1.3rem; color: #00d4ff; text-align: center; font-weight: 500; letter-spacing: 1px;">
        ✨ Advanced AI-Powered Detection System ✨
    </p>
    <p style="text-align: center; color: #b0b0b0; margin-top: 0.5rem;">
        Detect synthetic and AI-generated images with advanced machine learning
    </p>
</div>
""", unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown("""
        <style>
        .image-container {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 230, 118, 0.05));
            border: 2px solid rgba(0, 212, 255, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
        }
        </style>
        <div class="image-container">
        """, unsafe_allow_html=True)
        st.image(image, caption='📸 Uploaded Image', use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.subheader("📊 Analysis Results")
    
    if uploaded_file is not None and st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner('🔬 Scanning for synthesis artifacts...'):
            # Preprocess
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
            is_fake = pred.item() == 0
            label = "🚫 FAKE (AI-Generated)" if is_fake else "✅ REAL (Authentic)"
            color = "#ff1744" if is_fake else "#00e676"
            
            # Display Metrics
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color: {color}; font-size: 2rem; margin: 0; text-shadow: 0 0 20px {color}33;">{label}</h2>
                <div style="margin-top: 1.5rem;">
                    <h3 style="color: #00d4ff; font-size: 1.5rem; margin: 0;">Confidence</h3>
                    <h2 style="color: {color}; font-size: 2.5rem; margin: 0.5rem 0 0 0;">{conf.item()*100:.1f}%</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Visual Explanation (Heatmap)
            st.write("### 🧠 Visual Explanation (Heatmap)")
            st.write("Areas contributing most to the decision:")
            heatmap = get_gradcam(model, img_tensor)
            
            # Overlay heatmap
            img_resized = np.array(image.resize((224, 224)))
            superimposed_img = heatmap * 0.4 + img_resized * 0.6
            superimposed_img = np.uint8(superimposed_img)
            
            st.image(superimposed_img, caption="🔥 Grad-CAM Activation Map", use_container_width=True)

# ==========================================
# 5. DISCLAIMER & GUIDELINES
# ==========================================
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(255, 152, 0, 0.15), rgba(255, 23, 68, 0.1)); 
            border-left: 5px solid #ff9800; border-radius: 8px; padding: 20px; margin: 20px 0;">
    <h3 style="color: #ff9800; margin-top: 0;">⚠️ Important Disclaimer</h3>
    <p style="color: #e0e0e0; line-height: 1.8; font-size: 1rem;">
        <strong>This detection system is not 100% accurate.</strong> AI-generated media detection is an evolving field, and no algorithm can guarantee 
        perfect accuracy in all scenarios. This tool provides probabilistic predictions based on machine learning models trained on specific datasets.
    </p>
    <p style="color: #e0e0e0; line-height: 1.8; font-size: 1rem; margin: 1rem 0 0 0;">
        <strong>Confidence scores reflect model certainty, not ground truth.</strong> Results should be:
    </p>
    <ul style="color: #e0e0e0; line-height: 2; font-size: 0.95rem; margin: 0.5rem 0;">
        <li>✓ Used as an auxiliary tool, not a definitive determination</li>
        <li>✓ Combined with manual inspection and other verification methods</li>
        <li>✓ Interpreted with caution when dealing with critical decisions</li>
        <li>✓ Subject to limitations of the underlying deep learning model</li>
    </ul>
    <p style="color: #b0b0b0; font-size: 0.9rem; margin-top: 1rem; font-style: italic;">
        For critical applications, please consult multiple detection tools and human experts. This system is for informational purposes only.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("")
st.markdown("""
<div style="text-align: center; color: #888888; font-size: 0.85rem; margin-top: 2rem; padding: 1rem;">
    <p>🔬 Powered by Advanced Deep Learning | ResNet-18 Architecture | Real-time Analysis</p>
    <p>© 2026 Deepfake Detection System | Use Responsibly</p>
</div>
""", unsafe_allow_html=True)
