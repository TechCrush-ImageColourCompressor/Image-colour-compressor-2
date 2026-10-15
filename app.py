import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

st.title("Image Colour Compressor")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
k = st.slider("Number of colours (K)", min_value=2, max_value=32, value=8)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.subheader("Original Image")
    st.image(image)

    h, w, c = img_array.shape
    pixels = img_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    new_colors = kmeans.cluster_centers_.astype("uint8")

    compressed_pixels = new_colors[labels]
    compressed_img_array = compressed_pixels.reshape(h, w, c)
    compressed_image = Image.fromarray(compressed_img_array)

    st.subheader(f"Compressed Image ({k} colours)")
    st.image(compressed_image)
