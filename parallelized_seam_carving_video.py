import cv2
import numpy as np
import concurrent.futures
import os
from glob import glob
import networkx as nx
import argparse
from icecream import ic
import logging

logging.basicConfig(
    filename="seam_carving.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    level=logging.INFO,
)


def usage():
    print(
        "Usage:python3 parallelized_seam_carving_video.py -f=<filename> -dh=<desired height> -dw=<desired width> -nw=<num workers> -o=<output file>"
    )


def remove_seam(image, seam):

    nrows, ncols, channels = image.shape
    # Initialize a reduced image with one less column
    reduced_image = np.zeros((nrows, ncols - 1, channels), dtype=image.dtype)

    for i in range(nrows):

        # If seam is not at the first column, copy pixels before the seam
        if seam[i] != 0:
            reduced_image[i, : seam[i]] = image[i, : seam[i]]
        # If seam is not at the last column, copy pixels after the seam
        if seam[i] < ncols - 1:
            reduced_image[i, seam[i] :] = image[i, seam[i] + 1 :]

    return reduced_image


def remove_seam_gray(gray_image, seam):
    nrows, ncols = gray_image.shape
    # Initialize a reduced image with one less column
    reduced_image = np.zeros((nrows, ncols - 1), dtype=gray_image.dtype)

    for i in range(nrows):
        # If seam is not at the first column, copy pixels before the seam
        if seam[i] > 0:
            reduced_image[i, : seam[i]] = gray_image[i, : seam[i]]
        # If seam is not at the last column, copy pixels after the seam
        if seam[i] < ncols - 1:
            reduced_image[i, seam[i] :] = gray_image[i, seam[i] + 1 :]

    return reduced_image


def find_seam(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5):
    rows, cols = gray_img1.shape

    # Constants
    a1, a2, a3, a4, a5 = 0.2, 0.2, 0.2, 0.2, 0.2
    inf = float("inf")

    # Energy map combining the inputs
    energy = (
        a1 * gray_img1
        + a2 * gray_img2
        + a3 * gray_img3
        + a4 * gray_img4
        + a5 * gray_img5
    )

    # Create a directed graph
    G = nx.DiGraph()

    # Adding nodes and edges
    for i in range(rows):
        for j in range(cols):
            node = (i, j)

            # Add edges to the next row
            if i < rows - 1:
                # Down
                G.add_edge(node, (i + 1, j), weight=energy[i, j])
                # Down-left
                if j > 0:
                    G.add_edge(node, (i + 1, j - 1), weight=energy[i, j])
                # Down-right
                if j < cols - 1:
                    G.add_edge(node, (i + 1, j + 1), weight=energy[i, j])

            # Add source and sink connections
            if i == 0:
                G.add_edge("source", node, weight=energy[i, j])
            if i == rows - 1:
                G.add_edge(node, "sink", weight=energy[i, j])

    # Compute the max flow (min cut) between source and sink
    flow_value, partition = nx.minimum_cut(G, "source", "sink", capacity="weight")
    reachable, non_reachable = partition

    # Extract the seam from the partition
    seam = np.zeros(rows, dtype=int)
    for i in range(rows):
        for j in range(cols):
            if (i, j) in reachable:
                seam[i] = j
                break

    return seam


def reduce_vertical(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img):
    seam = find_seam(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5)
    img = remove_seam(img, seam)
    gray_img1 = remove_seam_gray(gray_img1, seam)
    gray_img2 = remove_seam_gray(gray_img2, seam)
    gray_img3 = remove_seam_gray(gray_img3, seam)
    gray_img4 = remove_seam_gray(gray_img4, seam)
    gray_img5 = remove_seam_gray(gray_img5, seam)
    return img, gray_img1, gray_img2, gray_img3, gray_img4, gray_img5


def reduce_horizontal(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img):
    # Find seam on transposed images
    seam = find_seam(gray_img1.T, gray_img2.T, gray_img3.T, gray_img4.T, gray_img5.T)

    # Remove seam from all images and return the updated images
    img = remove_seam(img.transpose(1, 0, 2), seam).transpose(1, 0, 2)
    gray_img1 = remove_seam_gray(gray_img1.T, seam).T
    gray_img2 = remove_seam_gray(gray_img2.T, seam).T
    gray_img3 = remove_seam_gray(gray_img3.T, seam).T
    gray_img4 = remove_seam_gray(gray_img4.T, seam).T
    gray_img5 = remove_seam_gray(gray_img5.T, seam).T
    return img, gray_img1, gray_img2, gray_img3, gray_img4, gray_img5


def reduce_frame(frame1, frame2, frame3, frame4, frame5, v, h):
    img = frame1.copy()
    gray_img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray_img3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    gray_img4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
    gray_img5 = cv2.cvtColor(frame5, cv2.COLOR_BGR2GRAY)

    for _ in range(v):
        img, gray_img1, gray_img2, gray_img3, gray_img4, gray_img5 = reduce_vertical(
            gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img
        )

    for _ in range(h):
        img, gray_img1, gray_img2, gray_img3, gray_img4, gray_img5 = reduce_horizontal(
            gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img
        )
    return img


def create_video_from_images(image_dir, output_file, fps=30):
    # Get all images in the directory, sorted by file name
    images = sorted(glob(os.path.join(image_dir, "*.png")))

    if not images:
        logging.error("No images found in the directory.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape

    # Initialize VideoWriter for AVI
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for AVI
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        video.write(img)

    video.release()
    logging.info(f"Video saved as {output_file}")


def process_frame(args):
    frames, frame_id, max_frames, v, h = args
    frame1 = frames[frame_id]
    frame2 = frames[frame_id + 1] if frame_id + 1 < max_frames else frame1
    frame3 = frames[frame_id + 2] if frame_id + 2 < max_frames else frame2
    frame4 = frames[frame_id + 3] if frame_id + 3 < max_frames else frame3
    frame5 = frames[frame_id + 4] if frame_id + 4 < max_frames else frame4

    # print(f"Processing frame {frame_id + 1}/{max_frames}...")
    logging.info(f"Processing frame {frame_id + 1}/{max_frames}...")
    return reduce_frame(frame1, frame2, frame3, frame4, frame5, v, h)


def initialize_video_capture(in_file):
    cap = cv2.VideoCapture(in_file)
    if not cap.isOpened():
        logging.error("Unable to open input file.")
        return None, None, None, None, None

    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    return cap, max_frames, orig_width, orig_height, fps


def extract_frames(cap, max_frames):
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        logging.error("No frames extracted from video.")
        return None
    return frames


def process_video(frames, max_frames, orig_width, orig_height, ver, hor, num_workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(frames, i, max_frames, ver, hor) for i in range(max_frames)]
        out_frames = list(executor.map(process_frame, args))

    return out_frames


def save_output_video(out_frames, out_file, fps, orig_width, orig_height, ver, hor):
    output = cv2.VideoWriter(
        out_file,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (orig_width - ver, orig_height - hor),
    )

    for out_frame in out_frames:
        output.write(out_frame)

    output.release()
    logging.info(f"Output file saved as: {out_file}")


def main(args):
    in_file = args.filename
    num_workers = args.num_workers
    out_file = args.output_file

    cap, max_frames, orig_width, orig_height, fps = initialize_video_capture(in_file)
    if not cap:
        return

    ver = abs(orig_width - args.desired_width)
    hor = abs(orig_height - args.desired_height)

    logging.info(f"Original dimensions: {orig_width}x{orig_height}, FPS: {fps}")
    logging.info(f"Resizing to: {args.desired_width}x{args.desired_height}")

    frames = extract_frames(cap, max_frames)
    if not frames:
        return

    out_frames = process_video(
        frames, max_frames, orig_width, orig_height, ver, hor, num_workers
    )

    # out_file = in_file.split(".")[0] + "_result.avi"
    save_output_video(out_frames, out_file, fps, orig_width, orig_height, ver, hor)
    logging.info("Video processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduce video dimensions using seam carving based on sequential frames."
    )
    parser.add_argument(
        "-f", "--filename", type=str, help="Path to the input video file."
    )
    parser.add_argument(
        "-dh", "--desired_height", type=int, help="Desired height of the output video."
    )
    parser.add_argument(
        "-dw", "--desired_width", type=int, help="Desired width of the output video."
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        help="Number of workers for parallel processing.",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to the output video file."
    )

    args = parser.parse_args()
    main(args)
