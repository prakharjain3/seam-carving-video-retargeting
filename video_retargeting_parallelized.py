import cv2
import numpy as np
import networkx as nx
from threading import Thread, Lock
import argparse
import os
import logging

logging.basicConfig(filename='seam_carving.log', filemode='w', level=logging.INFO)


mutexLock = Lock()
frameIter = 0
maxFrames = 0
frames = []
outFrames = []
ver = 1
hor = 0


def reduce_frame(frame1, frame2, frame3, frame4, frame5, v, h):
    grayImg1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grayImg2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    grayImg3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    grayImg4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
    grayImg5 = cv2.cvtColor(frame5, cv2.COLOR_BGR2GRAY)

    min_val = min(v, h)
    diff = abs(v - h)

    for _ in range(min_val):
        frame1 = reduce_vertical(
            grayImg1, grayImg2, grayImg3, grayImg4, grayImg5, frame1)
        frame1 = reduce_horizontal(
            grayImg1, grayImg2, grayImg3, grayImg4, grayImg5, frame1)

    reduce_func = reduce_horizontal if h > v else reduce_vertical
    for _ in range(diff):
        frame1 = reduce_func(grayImg1, grayImg2, grayImg3,
                             grayImg4, grayImg5, frame1)

    # you can write the above code like this too
    # if h > v:
    #     for _ in range(diff):
    #         frame1 = reduce_horizontal(grayImg1, grayImg2, grayImg3, grayImg4, grayImg5, frame1)
    # else:
    #     for _ in range(diff):
    #         frame1 = reduce_vertical(grayImg1, grayImg2, grayImg3, grayImg4, grayImg5, frame1)

    return frame1


def reduce_vertical(grayImg1, grayImg2, grayImg3, grayImg4, grayImg5, img):
    seam = find_seam(grayImg1, grayImg2, grayImg3, grayImg4, grayImg5)
    img = remove_seam(img, seam)
    grayImg1 = remove_seam_gray(grayImg1, seam)
    grayImg2 = remove_seam_gray(grayImg2, seam)
    grayImg3 = remove_seam_gray(grayImg3, seam)
    grayImg4 = remove_seam_gray(grayImg4, seam)
    grayImg5 = remove_seam_gray(grayImg5, seam)
    return img


def reduce_horizontal(grayImg1, grayImg2, grayImg3, grayImg4, grayImg5, img):
    seam = find_seam(grayImg1.T, grayImg2.T,
                     grayImg3.T, grayImg4.T, grayImg5.T)
    img = remove_seam(img.T, seam)
    grayImg1 = remove_seam_gray(grayImg1.T, seam).T
    grayImg2 = remove_seam_gray(grayImg2.T, seam).T
    grayImg3 = remove_seam_gray(grayImg3.T, seam).T
    grayImg4 = remove_seam_gray(grayImg4.T, seam).T
    grayImg5 = remove_seam_gray(grayImg5.T, seam).T
    return img.T


def find_seam(grayImg1, grayImg2, grayImg3, grayImg4, grayImg5):
    rows, cols = grayImg1.shape
    g = nx.DiGraph()

    inf = float('inf')
    a1, a2, a3, a4, a5 = 0.2, 0.2, 0.2, 0.2, 0.2

    for i in range(rows):
        for j in range(cols):
            if j == 0:
                g.add_edge('s', (i, j), capacity=inf)
            if j == cols - 1:
                g.add_edge((i, j), 't', capacity=inf)

            if j == 0:
                LR = a1 * grayImg1[i, j+1] + a2 * grayImg2[i, j+1] + a3 * \
                    grayImg3[i, j+1] + a4 * \
                    grayImg4[i, j+1] + a5 * grayImg5[i, j+1]
                g.add_edge((i, j), (i, j + 1), capacity=LR)
                g.add_edge((i, j + 1), (i, j), capacity=inf)

            elif j != cols - 1:
                LR = a1 * abs(grayImg1[i, j + 1] - grayImg1[i, j - 1]) + a2 * abs(grayImg2[i, j + 1] - grayImg2[i, j - 1]) + a3 * abs(
                    grayImg3[i, j + 1] - grayImg3[i, j - 1]) + a4 * abs(grayImg4[i, j + 1] - grayImg4[i, j - 1]) + a5 * abs(grayImg5[i, j + 1] - grayImg5[i, j - 1])
                g.add_edge((i, j), (i, j + 1), capacity=LR)
                g.add_edge((i, j + 1), (i, j), capacity=inf)

            if i != rows - 1:

                if j == 0:
                    posLU = a1 * grayImg1[i, j] + a2 * grayImg2[i, j] + a3 * \
                        grayImg3[i, j] + a4 * \
                        grayImg4[i, j] + a5 * grayImg5[i, j]
                    negLU = a1 * grayImg1[i + 1, j] + a2 * grayImg2[i + 1, j] + a3 * \
                        grayImg3[i + 1, j] + a4 * \
                        grayImg4[i + 1, j] + a5 * grayImg5[i + 1, j]
                    g.add_edge((i, j), (i + 1, j), capacity=negLU)
                    g.add_edge((i + 1, j), (i, j), capacity=negLU)
                else:
                    posLU = a1 * abs(grayImg1[i, j] - grayImg1[i + 1, j - 1]) + a2 * abs(grayImg2[i, j] - grayImg2[i + 1, j - 1]) + a3 * abs(
                        grayImg3[i, j] - grayImg3[i + 1, j - 1]) + a4 * abs(grayImg4[i, j] - grayImg4[i + 1, j - 1]) + a5 * abs(grayImg5[i, j] - grayImg5[i + 1, j - 1])
                    negLU = a1 * abs(grayImg1[i + 1, j] - grayImg1[i, j - 1]) + a2 * abs(grayImg2[i + 1, j] - grayImg2[i, j - 1]) + a3 * abs(
                        grayImg3[i + 1, j] - grayImg3[i, j - 1]) + a4 * abs(grayImg4[i + 1, j] - grayImg4[i, j - 1]) + a5 * abs(grayImg5[i + 1, j] - grayImg5[i, j - 1])
                    g.add_edge((i, j), (i + 1, j - 1), capacity=negLU)
                    g.add_edge((i, j), (i + 1, j), capacity=posLU)
            if i != 0 and j != 0:
                g.add_edge((i, j), (i - 1, j - 1), capacity=inf)
            if i != rows - 1 and j != 0:
                g.add_edge((i, j), (i + 1, j - 1), capacity=inf)

    flow_value, flow_dict = nx.maximum_flow(g, 's', 't')
    seam = np.zeros(rows, dtype=int)
    for i in range(rows):
        for j in range(cols):
            if flow_dict['s'][(i, j)] == 0:
                seam[i] = j - 1
                break
            if j == cols - 1 and flow_dict[(i, j)]['t'] == 0:
                seam[i] = cols - 1
    return seam


def remove_seam(image, seam):
    nrows, ncols, _ = image.shape
    reduced_image = np.zeros((nrows, ncols - 1, 3), dtype=np.uint8)

    for i in range(nrows):
        if seam[i] != 0:
            reduced_image[i, :seam[i]] = image[i, :seam[i]]
        if seam[i] != ncols - 1:
            reduced_image[i, seam[i]:] = image[i, seam[i] + 1:]

    return reduced_image


def remove_seam_gray(gray_image, seam):
    nrows, ncols = gray_image.shape
    reduced_image = np.zeros((nrows, ncols - 1), dtype=np.uint8)

    for i in range(nrows):
        if seam[i] != 0:
            reduced_image[i, :seam[i]] = gray_image[i, :seam[i]]
        if seam[i] != ncols - 1:
            reduced_image[i, seam[i]:] = gray_image[i, seam[i] + 1:]

    return reduced_image


def reduce():
    global frameIter
    while True:
        mutexLock.acquire()
        if frameIter >= maxFrames:
            mutexLock.release()
            return
        frameId = frameIter
        frameIter += 1
        mutexLock.release()

        logging.info(f"Processing frame {frameIter}/{maxFrames}")

        print(f"Frame {frameIter}/{maxFrames}", end="\n")

        frame1 = frames[frameId]

        # Check if we are close to the end of the video and select the next frames accordingly
        if frameId < maxFrames - 4:
            frame2 = frames[frameId + 1]
            frame3 = frames[frameId + 2]
            frame4 = frames[frameId + 3]
            frame5 = frames[frameId + 4]
        elif frameId < maxFrames - 3:
            frame2 = frames[frameId + 1]
            frame3 = frames[frameId + 2]
            frame4 = frames[frameId + 3]
            frame5 = frame4
        elif frameId < maxFrames - 2:
            frame2 = frames[frameId + 1]
            frame3 = frames[frameId + 2]
            frame4 = frame3
            frame5 = frame3
        elif frameId < maxFrames - 1:
            frame2 = frames[frameId + 1]
            frame3 = frame2
            frame4 = frame2
            frame5 = frame2
        else:
            frame2 = frame1
            frame3 = frame1
            frame4 = frame1
            frame5 = frame1

        outFrames[frameId] = reduce_frame(
            frame1, frame2, frame3, frame4, frame5, ver, hor)

# check to ensure that the number of workers is a positive integer


def positive_int(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Invalid number of workers: {
                                         value}. Must be at least 1.")
    return ivalue


def main():
    global frames, outFrames, maxFrames, ver, hor

    parser = argparse.ArgumentParser(description='Seam Carving Video Resizer')
    parser.add_argument('-f', '--filename', type=str, help='Input video file')
    parser.add_argument('-vc', '--vertical_cuts', type=int,
                        help='Number of vertical seams to remove')
    parser.add_argument('-hc', '--horizontal_cuts', type=int,
                        help='Number of horizontal seams to remove')
    parser.add_argument('-nw', '--num_workers',
                        type=positive_int, help='Number of worker threads')
    args = parser.parse_args()

    inFile = args.filename
    ver = args.vertical_cuts
    hor = args.horizontal_cuts
    numWorkers = args.num_workers

    cap = cv2.VideoCapture(inFile)
    if not cap.isOpened():
        logging.error("Error opening video file.")
        return

    logging.info(f"Processing {inFile} with {numWorkers} workers...")

    maxFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    origWid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    origHei = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    S = (origWid - ver, origHei - hor)

    basename, ext = os.path.splitext(inFile)
    outFile = f"{basename}-result{ext}"

    output = cv2.VideoWriter(filename=outFile, fourcc=cv2.VideoWriter_fourcc(
        *'mp4v'), fps=cap.get(cv2.CAP_PROP_FPS), frameSize=S, isColor=True)

    frames = [cap.read()[1] for _ in range(maxFrames)]
    outFrames = [None] * maxFrames

    threads = []

    try:
        threads = [Thread(target=reduce) for _ in range(numWorkers)]
    except Exception as e:
        logging.error(f"Error creating threads: {e}")
        exit(1)

    # start the threads
    for thread in threads:
        try:
            thread.start()
        except Exception as e:
            logging.error(f"Error starting thread {thread}: {e}")
            exit(1)
    for thread in threads:
        try:
            thread.join()
        except Exception as e:
            logging.error(f"Error joining thread {thread}: {e}")
            exit(1)
    try:
        for frame in outFrames:
            if frame is not None:
                output.write(frame)
            else:
                logging.warning("Encountered a None frame while writing to output.")
    except Exception as e:
        logging.error(f"Error writing frame to output: {e}")
        exit(1)


if __name__ == "__main__":
    main()
