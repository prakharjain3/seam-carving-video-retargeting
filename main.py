import cv2
import numpy as np
import maxflow

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def compute_energy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.abs(sobel_x) + np.abs(sobel_y)
    return energy

def find_vertical_seam_dp(energy):
    height, width = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, height):
        for j in range(0, width):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            elif j == width -1:
                idx = np.argmin(M[i-1, j-1:j+1])
                backtrack[i, j] = idx + j -1
                min_energy = M[i-1, idx + j -1]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j -1
                min_energy = M[i-1, idx + j -1]
            M[i, j] += min_energy

    seam = []
    j = np.argmin(M[-1])
    seam.append(j)
    for i in range(height-1, 0, -1):
        j = backtrack[i, j]
        seam.append(j)
    seam.reverse()
    return seam

def remove_seam(frame, seam):
    height, width, _ = frame.shape
    mask = np.ones((height, width), dtype=np.bool_)
    for y in range(height):
        mask[y, seam[y]] = False
    frame = frame[mask].reshape((height, width - 1, 3))
    return frame

def seam_carve_video(frames, num_seams_to_remove):
    carved_frames = frames.copy()
    previous_seam = None

    for _ in range(num_seams_to_remove):
        seams = []
        for i, frame in enumerate(carved_frames):
            energy = compute_energy(frame)
            if previous_seam is not None:
                # Modify energy to enforce temporal coherence
                for y in range(len(previous_seam)):
                    if 0 <= previous_seam[y] < energy.shape[1]:
                        energy[y, previous_seam[y]] += 1e5  # High penalty to stick to previous seam
            seam = find_vertical_seam_dp(energy)
            seams.append(seam)
            carved_frames[i] = remove_seam(carved_frames[i], seam)
        previous_seam = seams[0]  # Assuming all seams are similar; improve as needed
    return carved_frames

def main():
    input_video = 'input.mp4'
    output_video = 'output.mp4'
    frames = load_video(input_video)
    num_seams_to_remove = 50  # Number of columns to remove

    carved_frames = seam_carve_video(frames, num_seams_to_remove)
    save_video(carved_frames, output_video)

if __name__ == "__main__":
    main()
