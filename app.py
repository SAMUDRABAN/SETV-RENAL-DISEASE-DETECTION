import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io
import h5py
import timm

# Function to create the TNT model
@st.cache_data
def create_tnt_model(num_classes):
    model = timm.create_model('tnt_s_patch16_224', pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model

# Function to load the model
@st.cache_data
def load_model(model_path, num_classes):
    model = create_tnt_model(num_classes)
    with h5py.File(model_path, 'r') as h5f:
        model_weights = {k: torch.tensor(v) for k, v in h5f.items()}
    model.load_state_dict(model_weights, strict=False)
    return model

# Image preprocessing function
def prepare_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# Display the logo
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "assets", "setv_global_cover.jpeg")
logo = Image.open(logo_path)
st.image(logo, width=200)

# Streamlit UI Code
st.title("SETV RENAL DISEASE DIAGNOSIS")

# Load model (update path as needed)
model_path = os.path.join(script_dir, "Model", "my_model (1).h5") # Adjust the path
model = load_model(model_path, num_classes=4)
model.eval()

class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        img = prepare_image(uploaded_file.getvalue())

        with torch.no_grad():
            prediction = model(img)
            predicted_idx = torch.argmax(prediction, 1).item()
            predicted_class = class_names[predicted_idx]

        st.write(f'Prediction: {predicted_class}')
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please try again with a different image file.")
