import streamlit as st
import torch
import numpy as np
import cv2
import os
import segmentation_models_pytorch as smp
from PIL import Image
import io

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="Dal Lake AI Project", layout="wide")

# Force CPU if no GPU (Laptops usually run better on CPU for single images)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_model.pth'

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Must match the training config exactly
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    # Load weights (map_location handles CPU/GPU mismatch)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

try:
    model = load_model()
    st.sidebar.success(f"Model Loaded on {DEVICE.upper()}")
except FileNotFoundError:
    st.error("'best_model.pth' not found! Please place it in the same folder as app.py")
    st.stop()

# --- 3. HELPER FUNCTIONS ---
def process_image(image_input):
    """Resizes and normalizes image for the AI"""
    # Convert PIL to Numpy
    img = np.array(image_input)
    
    # Resize to 256x256 (Model Requirement)
    img_resized = cv2.resize(img, (256, 256))
    
    # Normalize (0-1) and Transpose (HWC -> CHW)
    x_tensor = np.transpose(img_resized, (2, 0, 1)).astype('float32') / 255.0
    x_tensor = torch.from_numpy(x_tensor).unsqueeze(0).to(DEVICE)
    
    return img_resized, x_tensor

def get_prediction(tensor, threshold=0.5):
    """Runs the model inference with dynamic threshold"""
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        # Apply the slider threshold
        pred_mask = (probs > threshold).float().cpu().squeeze().numpy()
    return pred_mask

def apply_overlay(bg_image, mask, color=(255, 0, 0), alpha=0.5):
    """Adds a colored overlay to the image"""
    overlay = bg_image.copy()
    colored_mask = np.zeros_like(bg_image)
    colored_mask[mask == 1] = color
    
    # Blend only where mask is 1
    overlay[mask == 1] = (1 - alpha) * bg_image[mask == 1] + alpha * colored_mask[mask == 1]
    return overlay

def get_error_map(true_mask, pred_mask):
    """Creates a visual map of where the AI was right or wrong"""
    h, w = true_mask.shape
    # Create black background
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. False Positive (Red): AI saw plastic, but there was none
    fp = np.logical_and(pred_mask == 1, true_mask == 0)
    error_map[fp] = [255, 0, 0]
    
    # 2. False Negative (Blue): AI missed the plastic
    fn = np.logical_and(pred_mask == 0, true_mask == 1)
    error_map[fn] = [0, 0, 255]
    
    # 3. True Positive (Green): AI correctly found plastic
    tp = np.logical_and(pred_mask == 1, true_mask == 1)
    error_map[tp] = [0, 255, 0]
    
    return error_map

# --- 4. UI LAYOUT ---
st.title("Dal Lake Plastic Segmentation")
st.markdown("Research Project: Semantic Segmentation of floating plastic debris using **U-Net (ResNet34)**.")

# Sidebar Controls
st.sidebar.header("Controls")
mode = st.sidebar.radio("Select Mode:", ["Browse Dataset", "Upload New Image"])
alpha = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.4)
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, help="Lower = More sensitive (detects more). Higher = Stricter (detects less).")

# --- MODE 1: BROWSE DATASET ---
if mode == "Browse Dataset":
    IMG_DIR = os.path.join('dataset', 'images')
    MASK_DIR = os.path.join('dataset', 'masks')
    
    if os.path.exists(IMG_DIR) and os.path.exists(MASK_DIR):
        files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))])
        
        if files:
            # Slider to pick image
            idx = st.sidebar.slider("Image Index", 0, len(files)-1, 0)
            selected_file = files[idx]
            
            # Load Data
            img_path = os.path.join(IMG_DIR, selected_file)
            mask_name = selected_file.replace('.jpg', '_mask.png')
            mask_path = os.path.join(MASK_DIR, mask_name)
            
            original_image = Image.open(img_path).convert("RGB")
            
            # Predict
            resized_img, tensor = process_image(original_image)
            pred_mask = get_prediction(tensor, threshold)
            
            # Load Ground Truth
            if os.path.exists(mask_path):
                true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                true_mask = cv2.resize(true_mask, (256, 256))
                _, true_mask = cv2.threshold(true_mask, 127, 1, cv2.THRESH_BINARY)
            else:
                true_mask = np.zeros((256, 256))
                st.warning("Ground Truth mask not found for this image.")

            # --- VISUALIZATION COLUMNS ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(resized_img, caption=f"Original: {selected_file}", use_container_width=True)
            
            with col2:
                vis_truth = apply_overlay(resized_img, true_mask, color=(0, 255, 0), alpha=alpha)
                st.image(vis_truth, caption="Ground Truth (Green)", use_container_width=True)
                
            with col3:
                vis_pred = apply_overlay(resized_img, pred_mask, color=(255, 0, 0), alpha=alpha)
                st.image(vis_pred, caption=f"AI Prediction (Red) @ {threshold}", use_container_width=True)

            # --- METRICS & ERROR MAP ---
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            iou = intersection / (union + 1e-6)
            
            st.divider()
            
            m_col1, m_col2 = st.columns([1, 2])
            
            with m_col1:
                st.subheader("Metrics")
                st.metric("IoU Accuracy Score", f"{iou:.4f}")
                st.info("Tip: Adjust the 'Confidence Threshold' in the sidebar to improve this score!")

            with m_col2:
                st.subheader("Error Analysis")
                error_map = get_error_map(true_mask, pred_mask)
                st.image(error_map, caption="Green=Correct | Red=False Positive (Noise) | Blue=False Negative (Missed)", use_container_width=True)

            # --- DOWNLOAD BUTTON ---
            # Save the AI Prediction image to bytes
            result_pil = Image.fromarray(vis_pred)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="⬇️ Download Prediction Image",
                data=byte_im,
                file_name=f"pred_{selected_file}",
                mime="image/png"
            )
            
        else:
            st.warning("No images found in dataset/images/")
    else:
        st.error("Dataset folder not found! Create a 'dataset' folder with 'images' and 'masks' inside.")

# --- MODE 2: UPLOAD IMAGE ---
elif mode == "Upload New Image":
    uploaded_file = st.file_uploader("Upload an image of Dal Lake", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Predict
        resized_img, tensor = process_image(image)
        pred_mask = get_prediction(tensor, threshold)
        
        # Visualize
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(resized_img, caption="Original Image", use_container_width=True)
            
        with col2:
            vis_pred = apply_overlay(resized_img, pred_mask, color=(255, 0, 0), alpha=alpha)
            st.image(vis_pred, caption="AI Detected Plastic (Red)", use_container_width=True)
            
        # Download
        result_pil = Image.fromarray(vis_pred)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Result",
            data=byte_im,
            file_name="ai_prediction.png",
            mime="image/png"
        )
            
        st.success("Analysis Complete!")