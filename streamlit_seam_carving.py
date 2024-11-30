import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import logging
import tempfile
import ffmpeg
import base64
import subprocess
from argparse_seam_carving_image import (
    remove_horizontal_img,
    remove_vertical_img,
    add_horizontal,
    add_vertical,
)
from parallelized_seam_carving_video import main as video_seam_carving

import os


def convert_to_h264(input_path, output_path):
    try:
        # Open devnull to suppress output
        with open(os.devnull, "w") as devnull:
            # Run ffmpeg with output and error streams redirected to devnull
            command = [
                "ffmpeg",
                "-i",
                input_path,
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                "-preset",
                "medium",
                "-hide_banner",
                "-loglevel",
                "error",
                output_path,
            ]
            subprocess.run(command, stdout=devnull, stderr=devnull, check=True)

        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during conversion: {e.stderr}")
        st.error("Failed to convert video to H.264 format.")
        return None


st.title("Seam Carving for Images and Videos")

option = st.selectbox("Choose an option", ["Image", "Video"])


def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps


if option == "Image":
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


elif option == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        filename = uploaded_file.name

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, filename)

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            h264_input_path = input_path
            if not filename.endswith(".mp4"):
                h264_input_path = os.path.join(temp_dir, "input_converted.mp4")
                st.info("Converting input video to H.264 MP4 format...")
                h264_input_path = convert_to_h264(input_path, h264_input_path)

            if h264_input_path is None:
                st.error("Error converting input video to H.264 format.")

            cap = cv2.VideoCapture(h264_input_path)

            if not cap.isOpened():
                st.error("Error opening video file.")
            else:
                orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                st.video(h264_input_path)

                st.write(f"Original Video Dimensions: {orig_width}x{orig_height}")
                st.write(f"FPS: {fps}")

                # Get desired dimensions from user
                desired_height = st.number_input(
                    "Desired Height",
                    min_value=1,
                    max_value=orig_height,
                    value=orig_height,
                )
                desired_width = st.number_input(
                    "Desired Width", min_value=1, max_value=orig_width, value=orig_width
                )
                num_workers = st.slider(
                    "Number of Workers", min_value=1, max_value=8, value=2
                )

                if st.button("Resize Video"):
                    # Define the output video path
                    resized_path = os.path.join(temp_dir, "resized_video.avi")

                    # Class for video processing arguments
                    class Args:
                        filename = h264_input_path
                        desired_height = desired_height
                        desired_width = desired_width
                        num_workers = num_workers
                        output_file = resized_path

                    # Call the video processing function
                    logging.info("Starting video resizing...")
                    video_seam_carving(Args())

                    # Convert resized video to H.264 if not already
                    h264_resized_path = resized_path
                    if not resized_path.endswith(".mp4"):
                        h264_resized_path = os.path.join(temp_dir, "output_resized.mp4")
                        h264_resized_path = convert_to_h264(
                            resized_path, h264_resized_path
                        )

                    if h264_resized_path is None:
                        st.error("Error converting output video to H.264 format.")

                    st.success("Video resizing and conversion completed!")
                    st.video(h264_resized_path)

                    st.write(
                        f"Output video dimensions: {desired_width}x{desired_height}"
                    )
