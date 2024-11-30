import cv2
import numpy as np
import logging
import os
import argparse

logging.basicConfig(
    filename="evaluate_scaling_video.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    level=logging.INFO,
)


def compute_energy(frame):
    """Compute the energy of a frame using gradient magnitude."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.sum(energy)


def compute_video_energy(video_path):
    """Compute the total energy of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Unable to open video file: {video_path}")
        return None

    total_energy = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_energy += compute_energy(frame)
        frame_count += 1

    cap.release()
    logging.info(f"Computed energy for {frame_count} frames in {video_path}")
    return total_energy


def get_video_codec(output_path):
    """Determine the video codec based on the file extension."""
    ext = os.path.splitext(output_path)[1].lower()
    codecs = {
        ".avi": "XVID",
        ".mp4": "mp4v",
        ".mov": "mp4v",
        ".mkv": "X264",
    }
    if ext not in codecs:
        raise ValueError(f"Unsupported output video format: {ext}")
    return codecs[ext]


def create_uniformly_scaled_video(input_video_path, output_video_path, width, height):
    """Create a uniformly scaled version of the input video."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Unable to open video file: {input_video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*get_video_codec(output_video_path))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_LINEAR
        )
        out.write(resized_frame)

    cap.release()
    out.release()
    logging.info(f"Uniformly scaled video saved as: {output_video_path}")


def main(args):
    original_video_path = args.input_video
    resized_video_path = args.output_video
    temp_uniform_scaled_path = (
        f"uniform_scaled{os.path.splitext(original_video_path)[1]}"
    )

    # Validate input files
    if not os.path.exists(original_video_path):
        logging.error(f"Original video file does not exist: {original_video_path}")
        return
    if not os.path.exists(resized_video_path):
        logging.error(f"Resized video file does not exist: {resized_video_path}")
        return

    logging.info(
        f"Evaluating energy preservation for seam carving and uniform scaling."
    )

    # Get dimensions of the resized video
    resized_cap = cv2.VideoCapture(resized_video_path)
    if not resized_cap.isOpened():
        logging.error(f"Unable to open resized video file: {resized_video_path}")
        return

    resized_width = int(resized_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    resized_height = int(resized_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resized_cap.release()

    # Create a uniformly scaled version of the original video
    create_uniformly_scaled_video(
        original_video_path, temp_uniform_scaled_path, resized_width, resized_height
    )

    # Compute energy for original, seam-carved, and uniformly scaled videos
    original_energy = compute_video_energy(original_video_path)
    seam_carved_energy = compute_video_energy(resized_video_path)
    uniform_scaled_energy = compute_video_energy(temp_uniform_scaled_path)

    if (
        original_energy is None
        or seam_carved_energy is None
        or uniform_scaled_energy is None
    ):
        logging.error("Error in computing energy for one or more videos.")
        return

    # Calculate energy preservation ratios
    seam_carving_ratio = seam_carved_energy / original_energy
    uniform_scaling_ratio = uniform_scaled_energy / original_energy

    logging.info(f"Original Video Energy: {original_energy:.2f}")
    logging.info(f"Seam-Carved Video Energy: {seam_carved_energy:.2f}")
    logging.info(f"Uniform-Scaled Video Energy: {uniform_scaled_energy:.2f}")
    logging.info(f"Seam Carving Energy Preservation Ratio: {seam_carving_ratio:.4f}")
    logging.info(
        f"Uniform Scaling Energy Preservation Ratio: {uniform_scaling_ratio:.4f}"
    )

    print(f"Seam Carving Energy Preservation Ratio: {seam_carving_ratio:.4f}")
    print(f"Uniform Scaling Energy Preservation Ratio: {uniform_scaling_ratio:.4f}")

    # Clean up temporary file
    if os.path.exists(temp_uniform_scaled_path):
        os.remove(temp_uniform_scaled_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare energy preservation between seam carving and uniform scaling."
    )
    parser.add_argument(
        "-i",
        "--input_video",
        type=str,
        required=True,
        help="Path to the original input video (any supported format).",
    )
    parser.add_argument(
        "-o",
        "--output_video",
        type=str,
        required=True,
        help="Path to the seam-carved output video (any supported format).",
    )

    args = parser.parse_args()
    main(args)
