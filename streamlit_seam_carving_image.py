import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import logging
from icecream import ic


from argparse_seam_carving_image import (
    remove_horizontal_img,
    remove_vertical_img,
    add_horizontal,
    add_vertical,
)

# Streamlit app
st.title("Image Seam Carving")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    filename = uploaded_file.name

    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Original Dimensions: ", img_array.shape[:2])

    desired_height = st.number_input(
        "Desired Height",
        min_value=1,
        max_value=img_array.shape[0] * 2,
        value=img_array.shape[0],
    )
    desired_width = st.number_input(
        "Desired Width",
        min_value=1,
        max_value=img_array.shape[1] * 2,
        value=img_array.shape[1],
    )

    logging.info(
        f"Original Height: {img_array.shape[0]} | Original Width: {img_array.shape[1]}"
    )
    logging.info(f"Desired Height: {desired_height} | Desired Width: {desired_width}")

    if st.button("Resize Image"):
        if desired_height <= img_array.shape[0]:
            img_array = remove_horizontal_img(img_array, desired_height)
        else:
            img_array = add_horizontal(img_array, desired_height)

        if desired_width <= img_array.shape[1]:
            img_array = remove_vertical_img(img_array, desired_width)
        else:
            img_array = add_vertical(img_array, desired_width)

        st.image(img_array, caption="Resized Image", use_container_width=True)
        st.write("Resized Dimensions: ", img_array.shape[:2])

        basename, ext = os.path.splitext(filename)
        output_file = f"results/{basename}-result{ext}"
        cv2.imwrite(output_file, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        logging.info(f"Output image saved to {output_file}")
        st.write(f"Output image saved to {output_file}")
