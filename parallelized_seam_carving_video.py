import cv2
import numpy as np
import concurrent.futures
import os
from glob import glob
import networkx as nx


def usage():
    print(
        "Usage:python3 sequential_video.py <filename> <vertical cuts> <horizontal cuts>"
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
        print("No images found in the directory.")
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
    print(f"Video saved as {output_file}")


def process_frame(args):
    frames, frame_id, max_frames, v, h = args
    frame1 = frames[frame_id]
    frame2 = frames[frame_id + 1] if frame_id + 1 < max_frames else frame1
    frame3 = frames[frame_id + 2] if frame_id + 2 < max_frames else frame2
    frame4 = frames[frame_id + 3] if frame_id + 3 < max_frames else frame3
    frame5 = frames[frame_id + 4] if frame_id + 4 < max_frames else frame4

    print(f"Processing frame {frame_id + 1}/{max_frames}...")
    return reduce_frame(frame1, frame2, frame3, frame4, frame5, v, h)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        usage()
        sys.exit(-1)

    in_file = sys.argv[1]
    ver = int(sys.argv[2])
    hor = int(sys.argv[3])
    num_workers = int(sys.argv[4])

    cap = cv2.VideoCapture(in_file)
    if not cap.isOpened():
        print("Unable to open input file.")
        sys.exit(-1)

    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_file = in_file.split(".")[0] + "_result.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # out_frames = []
    # for i in range(max_frames):
    #     frame1 = frames[i]
    #     frame2 = frames[i + 1] if i + 1 < max_frames else frame1
    #     frame3 = frames[i + 2] if i + 2 < max_frames else frame2
    #     frame4 = frames[i + 3] if i + 3 < max_frames else frame3
    #     frame5 = frames[i + 4] if i + 4 < max_frames else frame4
    #     print(f"Processing frame {i+1}/{max_frames}...")

    #     out_frame = reduce_frame(frame1, frame2, frame3, frame4, frame5, ver, hor)
    # output_dir = "output_frames"
    # os.makedirs(output_dir, exist_ok=True)
    # frame_number = i
    # cv2.imwrite(f"{output_dir}/frame_{frame_number:04d}.png", out_frame)

    # out_frames.append(out_frame)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(frames, i, max_frames, ver, hor) for i in range(max_frames)]
        out_frames = list(executor.map(process_frame, args))

    output = cv2.VideoWriter(
        out_file, fourcc, fps, (orig_width - ver, orig_height - hor)
    )

    for out_frame in out_frames:
        output.write(out_frame)

    output.release()
    # image_directory = "output_frames"  # Path to directory containing images
    # output_video = "output_video.avi"  # Desired output video file name

    # create_video_from_images(image_directory, output_video, fps)

    print("Video processing complete.")
