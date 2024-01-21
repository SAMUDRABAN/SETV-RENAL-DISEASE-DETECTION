import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import io
import h5py
import timm


def create_tnt_model(num_classes):
    # Create the TNT model with the specified number of classes
    model = timm.create_model('tnt_s_patch16_224', pretrained=False)
    # Replace the classifier head with the correct number of output classes
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    return model

def load_model(model_path, num_classes):
    # Create the model architecture
    model = create_tnt_model(num_classes)

    # Load weights from the HDF5 file
    with h5py.File(model_path, 'r') as h5f:
        model_weights = {k: torch.tensor(v) for k, v in h5f.items()}
    
    # Load the state dict into the model
    model.load_state_dict(model_weights, strict=False)
    return model

model_path = os.path.join("Model", "my_model (1).h5")
model = load_model(model_path, num_classes=4)


# Load model
model = load_model(model_path, num_classes=4)
model.eval()


# Image preprocessing function
def prepare_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# Class names as per your model's training
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Load model
model = load_model('C:\Kidney Disease Detection\Model\my_model (1).h5', num_classes=4) # Update path
model.eval()

def prepare_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# Streamlit app
st.title("SETV RENAL DISEASE DIGNOSIS")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg","webp"])

if uploaded_file is not None:
    try:
        # Attempt to open the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        # Prepare and classify the image
        img = prepare_image(uploaded_file.getvalue())  # Pass the bytes-like object directly

        with torch.no_grad():
            prediction = model(img)
            predicted_idx = torch.argmax(prediction, 1).item()
            predicted_class = class_names[predicted_idx]

        st.write(f'Prediction: {predicted_class}')
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please try again with a different image file.")