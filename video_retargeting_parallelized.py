import cv2
import numpy as np
import threading
import sys
import maxflow
import os

# Global variables
ver = 1
hor = 0
frame_iter = 0
max_frames = 0
frames = []
out_frames = []
mutex_lock = threading.Lock()


def usage():
    print(
        "Usage: heuristic5 <filename> <vertical cuts> <horizontal cuts> <# of workers>"
    )
    sys.exit(1)


def remove_seam(image, seam):
    # Support both grayscale and RGB images
    nrows, ncols = image.shape[:2]  # Get rows and columns only
    if len(image.shape) == 3:  # RGB image
        reduced_image = np.zeros((nrows, ncols - 1, image.shape[2]), dtype=image.dtype)
    else:  # Grayscale image
        reduced_image = np.zeros((nrows, ncols - 1), dtype=image.dtype)

    for i in range(nrows):
        if seam[i] != 0:
            reduced_image[i, : seam[i]] = image[i, : seam[i]]
        if seam[i] != ncols - 1:
            reduced_image[i, seam[i] :] = image[i, seam[i] + 1 :]

    return reduced_image


def remove_seam_gray(gray_image, seam):
    nrows, ncols = gray_image.shape
    reduced_image = np.zeros((nrows, ncols - 1), dtype=np.uint8)

    for i in range(nrows):
        if seam[i] != 0:
            reduced_image[i, : seam[i]] = gray_image[i, : seam[i]]
        if seam[i] != ncols - 1:
            reduced_image[i, seam[i] :] = gray_image[i, seam[i] + 1 :]

    return reduced_image


import maxflow


def find_seam(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5):
    rows, cols = gray_img1.shape
    inf = 100000
    seam = np.zeros(rows, dtype=int)
    a1, a2, a3, a4, a5 = 0.2, 0.2, 0.2, 0.2, 0.2

    # Initialize graph
    g = maxflow.Graph[float]()
    node_ids = g.add_nodes(rows * cols)

    # Add edges to the graph
    for i in range(rows):
        for j in range(cols):
            node_index = i * cols + j
            if j == 0:  # Leftmost column
                g.add_tedge(node_index, inf, 0)  # Source edge
            elif j == cols - 1:  # Rightmost column
                g.add_tedge(node_index, 0, inf)  # Sink edge
            else:
                LR1 = gray_img1[i, j + 1] if j < cols - 1 else 0
                LR2 = gray_img2[i, j + 1] if j < cols - 1 else 0
                LR3 = gray_img3[i, j + 1] if j < cols - 1 else 0
                LR4 = gray_img4[i, j + 1] if j < cols - 1 else 0
                LR5 = gray_img5[i, j + 1] if j < cols - 1 else 0
                LR = a1 * LR1 + a2 * LR2 + a3 * LR3 + a4 * LR4 + a5 * LR5

                # Add edge between adjacent pixels
                if i < rows - 1:
                    g.add_edge(node_index, (i + 1) * cols + j, LR, inf)

    # Compute the maximum flow
    flow = g.maxflow()

    # Extract the seam from the graph
    for i in range(rows):
        for j in range(cols):
            if g.get_segment(i * cols + j) == 1:  # Part of the seam
                seam[i] = j
                break

    return seam


def reduce_vertical(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img):
    seam = find_seam(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5)
    return remove_seam(img, seam)


def reduce_horizontal(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img):
    # Similar to vertical but operating on rows instead of columns.
    seam = find_seam(gray_img1.T, gray_img2.T, gray_img3.T, gray_img4.T, gray_img5.T)
    return remove_seam(img.T, seam).T


def reduce_frame(frame1, frame2, frame3, frame4, frame5, v, h):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
    gray5 = cv2.cvtColor(frame5, cv2.COLOR_BGR2GRAY)

    if v > 0:
        frame1 = reduce_vertical(gray1, gray2, gray3, gray4, gray5, frame1)
        frame2 = reduce_vertical(gray1, gray2, gray3, gray4, gray5, frame2)
        frame3 = reduce_vertical(gray1, gray2, gray3, gray4, gray5, frame3)
        frame4 = reduce_vertical(gray1, gray2, gray3, gray4, gray5, frame4)
        frame5 = reduce_vertical(gray1, gray2, gray3, gray4, gray5, frame5)

    if h > 0:
        frame1 = reduce_horizontal(gray1, gray2, gray3, gray4, gray5, frame1)
        frame2 = reduce_horizontal(gray1, gray2, gray3, gray4, gray5, frame2)
        frame3 = reduce_horizontal(gray1, gray2, gray3, gray4, gray5, frame3)
        frame4 = reduce_horizontal(gray1, gray2, gray3, gray4, gray5, frame4)
        frame5 = reduce_horizontal(gray1, gray2, gray3, gray4, gray5, frame5)

    return frame1


def worker_thread():
    global frame_iter
    while True:
        with mutex_lock:
            if frame_iter >= max_frames:
                return
            frame_id = frame_iter
            frame_iter += 1

        print(f"Frame {frame_iter}/{max_frames}")

        frame1 = frames[frame_id]
        if frame_id < max_frames - 4:
            frame2 = frames[frame_id + 1]
            frame3 = frames[frame_id + 2]
            frame4 = frames[frame_id + 3]
            frame5 = frames[frame_id + 4]
        else:
            frame2 = frame3 = frame4 = frame5 = frame1

        out_frames[frame_id] = reduce_frame(
            frame1, frame2, frame3, frame4, frame5, ver, hor
        )
        output_dir = "output_frames"
        # os.makedirs(output_dir, exist_ok=True)
        frame_number = frame_iter
        cv2.imwrite(f"{output_dir}/frame_{frame_number:04d}.png", out_frames[frame_id])


def main():
    global frames, out_frames, max_frames
    print(sys.argv)
    if len(sys.argv) == 6:
        in_file = sys.argv[2]
        ver = int(sys.argv[3])
        hor = int(sys.argv[4])
        num_workers = int(sys.argv[5])
    else:
        usage()

    cap = cv2.VideoCapture(in_file)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        sys.exit(1)

    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_hei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = (orig_wid - ver, orig_hei - hor)
    # out_file = in_file.split(".")[0] + "-result.mov"
    # Update output file extension to MP4
    out_file = in_file.split(".")[0] + "-result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(out_file, fourcc, cap.get(cv2.CAP_PROP_FPS), size)

    frames = [cap.read()[1] for _ in range(max_frames)]
    out_frames = [None] * max_frames

    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(target=worker_thread)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for i in range(max_frames):
        output.write(out_frames[i])

    cap.release()
    output.release()


if __name__ == "__main__":
    main()
