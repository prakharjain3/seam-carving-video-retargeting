import cv2
import sys


def usage():
    print("Usage: python video_info.py <video_file_path>")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()
        sys.exit(-1)

    file_path = sys.argv[1]

    vid = cv2.VideoCapture(file_path)
    if not vid.isOpened():
        print(f"Error: Unable to open video file '{file_path}'")
        sys.exit(-1)

    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"Video dimensions: {int(width)}x{int(height)}")

    vid.release()
