import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetB0,
    DenseNet121,
    VGG16,
    InceptionV3,
)
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

import pickle
import os
from sklearn.decomposition import PCA

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the saved fused model
loaded_model = load_model("./fused_model.h5")

# Load the saved classifier
with open("clf.pkl", "rb") as clf_file:
    loaded_clf = pickle.load(clf_file)

# Pre-trained models for feature extraction
models = {
    "ResNet50": ResNet50(weights="imagenet", include_top=False, pooling="avg"),
    "EfficientNet": EfficientNetB0(weights="imagenet", include_top=False, pooling="avg"),
    "DenseNet": DenseNet121(weights="imagenet", include_top=False, pooling="avg"),
    "VGG16": VGG16(weights="imagenet", include_top=False, pooling="avg"),
    #"AlexNet": None,  # AlexNet doesn't come with Keras, needs implementation
    "InceptionV3": InceptionV3(weights="imagenet", include_top=False, pooling="avg"),
}

# PCA for dimensionality reduction
pca = PCA(n_components=1)  # Adjust the number of components as needed

# Encoding classes
class_encoding_finger = {"fake": 0, "live": 1}
class_encoding_face = {"fake": 0, "live": 1}

# Decoding classes
class_decoding_face = {v: k for k, v in class_encoding_face.items()}
class_decoding_finger = {v: k for k, v in class_encoding_finger.items()}

def preprocess_face(image_path, threshold_value):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Check if any faces are detected
    if len(faces) > 0:
        # Take the first detected face
        x, y, w, h = faces[0]
        face_roi = image[y : y + h, x : x + w]

        # Apply Gaussian blur for noise reduction
        blurred_face = cv2.GaussianBlur(face_roi, (5, 5), 0)

        # Convert the blurred image to grayscale
        gray_blurred = cv2.cvtColor(blurred_face, cv2.COLOR_BGR2GRAY)

        # Calculate gradients using Sobel operators
        grad_x = cv2.Sobel(gray_blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Combine the gradients
        gradient = cv2.magnitude(grad_x, grad_y)

        # Binarize the gradient image using a threshold
        _, binary_image = cv2.threshold(gradient, threshold_value, 255, cv2.THRESH_BINARY)

        return binary_image.astype(np.uint8)  # Convert binary image to grayscale [0, 255]

    else:
        # If no face is detected, return a default black image
        return np.zeros_like(gray, dtype=np.uint8)
    
def preprocess_fingerprint(image_path, threshold_value):
    fingerprint_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Region of Interest (ROI) extraction using horizontal and vertical gradients
    horizontal_gradient = cv2.Sobel(fingerprint_image, cv2.CV_64F, 1, 0, ksize=3)
    vertical_gradient = cv2.Sobel(fingerprint_image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_differences = cv2.absdiff(horizontal_gradient, vertical_gradient)

    # Noise removal using a low-pass filter (3x3 normalized convolution box)
    blurred_fingerprint = cv2.boxFilter(gradient_differences, -1, (3, 3), normalize=True)

    # Binary operation using a threshold value
    _, binary_region_map = cv2.threshold(blurred_fingerprint, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_region_map.astype(np.uint8)  # Convert binary image to grayscale [0, 255]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize image to fit the model input size
    img = preprocess_input_resnet50(img)  # Preprocess the image according to ResNet50 model requirements
  # Preprocess the image according to model requirements
    return img

def extract_features(image_path, model_name):
    print(f"Model Name: {model_name}")  # Debugging statement
    if model_name in models:
        model = models[model_name]
        if model is not None:
            img = preprocess_image(image_path)
            print(f"Image Shape: {img.shape}")  # Debugging statement
            features = model.predict(np.expand_dims(img, axis=0))
            features = features.flatten()  # Flatten the features
            return features
        else:
            raise ValueError(f"Model {model_name} is not available.")
    else:
        raise ValueError(f"Model {model_name} not supported.")


def fuse_features(features_list):
    fused_features = np.concatenate(features_list, axis=0)
    # Resize or reshape the features to ensure the shape is (1, 6656)
    fused_features = fused_features[:6656]  # Trim or reshape to 6656 features
    return fused_features.reshape(1, -1)

st.title("Fingerprint-Face Liveness Detection")

# Dropdown for selecting the option
selected_option = st.selectbox("Select an option", ["Face", "Fingerprint"])

# Upload image through Streamlit
uploaded_file = st.file_uploader(f"Choose a {selected_option.lower()} image...", type=["jpg", "jpeg", "png", "bmp"])

threshold_value = 100  # Adjust this threshold value based on the characteristics of your data

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption=f"Original {selected_option} Image", use_column_width=True)

    # Save the uploaded image to a temporary file
    temp_image_path = f"temp_{selected_option.lower()}_image.jpg"
    with open(temp_image_path, "wb") as temp_image_file:
        temp_image_file.write(uploaded_file.getvalue())

    if selected_option == "Face":
        preprocessed_image = preprocess_face(temp_image_path, threshold_value)
    else:
        preprocessed_image = preprocess_fingerprint(temp_image_path, threshold_value)

    # Check if the preprocessed image is not None before converting to PIL Image
    if preprocessed_image is not None:
        # Convert the numpy array to a PIL Image
        pil_image = Image.fromarray(preprocessed_image)
        st.image(pil_image, caption=f"Preprocessed {selected_option} Image", use_column_width=True)

        # Extract features from the preprocessed image using different models
        feature_list = []
        for model_name in models.keys():
            features = extract_features(temp_image_path, model_name)
            feature_list.append(features)

        # Fuse the extracted features
        fused_features = fuse_features(feature_list)

        # Perform dimensionality reduction using PCA
        reduced_features = pca.fit_transform(fused_features.reshape(1, -1))
        fused_features=fused_features.reshape(1, -1)
        # Display the reduced features
        st.write("Reduced Features:")
        st.write(fused_features)

        if selected_option == "face":
            class_label = loaded_clf.predict(fused_features)
            decoded_class_label = class_decoding_face[class_label[0]]
        else:
           class_label = loaded_clf.predict(fused_features)
           decoded_class_label = class_decoding_finger[class_label[0]] 


        # Display the class label
        st.write(f"The detected  {selected_option} : {decoded_class_label}")

    else:
        st.warning("Error in preprocessing. Please check the input image.")

    # Remove the temporary image file
    os.remove(temp_image_path)
