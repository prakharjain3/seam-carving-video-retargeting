import cv2
import numpy as np
import argparse
import os
import logging
from icecream import ic

logging.basicConfig(filename="seam_carving.log", filemode="w", level=logging.INFO)


def average(x, y):
    return (x + y) // 2


def calculate_energy(I):
    Y, X = I.shape
    energy = np.zeros((Y, X), dtype=int)

    for x in range(X):
        for y in range(Y):
            val = 0
            if x > 0 and x + 1 < X:
                val += abs(int(I[y, x + 1]) - int(I[y, x - 1]))
            elif x > 0:
                val += 2 * abs(int(I[y, x]) - int(I[y, x - 1]))
            else:
                val += 2 * abs(int(I[y, x + 1]) - int(I[y, x]))

            if y > 0 and y + 1 < Y:
                val += abs(int(I[y + 1, x]) - int(I[y - 1, x]))
            elif y > 0:
                val += 2 * abs(int(I[y, x]) - int(I[y - 1, x]))
            else:
                val += 2 * abs(int(I[y + 1, x]) - int(I[y, x]))

            energy[y, x] = val

    return energy


def remove_vertical(I, Xd):
    # global gray, energy, dp, dir
    X0 = I.shape[1]
    X = X0
    Y = I.shape[0]
    # print(f"pixel shape: {I[0, 0].shape}")
    if I.shape[-1] == 4:  # Check for RGBA
        I = cv2.cvtColor(I, cv2.COLOR_BGRA2BGR)

    # Reinitialize dp and dir with the correct size
    dp = np.zeros((X, Y), dtype=int)
    dir = np.zeros((X, Y), dtype=int)

    for k in range(X0 - Xd):
        logging.info(f"Removing vertical seam {k + 1}/{X0 - Xd}")
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        energy = calculate_energy(gray)

        dp[:X, 0] = energy[0, :X]

        for y in range(1, Y):
            for x in range(X):
                val = energy[y, x]
                dp[x, y] = -1

                if x > 0 and (dp[x, y] == -1 or val + dp[x - 1, y - 1] < dp[x, y]):
                    dp[x, y] = val + dp[x - 1, y - 1]
                    dir[x, y] = -1

                if dp[x, y] == -1 or val + dp[x, y - 1] < dp[x, y]:
                    dp[x, y] = val + dp[x, y - 1]
                    dir[x, y] = 0

                if x + 1 < X and (dp[x, y] == -1 or val + dp[x + 1, y - 1] < dp[x, y]):
                    dp[x, y] = val + dp[x + 1, y - 1]
                    dir[x, y] = 1

        best = dp[0, Y - 1]
        cur = 0

        for x in range(X):
            if dp[x, Y - 1] < best:
                best = dp[x, Y - 1]
                cur = x

        tmp = np.zeros((Y, X - 1, 3), dtype=np.uint8)

        for y in range(Y - 1, -1, -1):
            for i in range(X):
                if i < cur:
                    tmp[y, i] = I[y, i]
                elif i > cur:
                    tmp[y, i - 1] = I[y, i]

            if y > 0:
                cur = cur + dir[cur, y]

        I = tmp
        X -= 1

    return I


def remove_horizontal(I, Yd):
    logging.info(f"Removing horizontal seam {I.shape[0] - Yd}")
    I = cv2.transpose(I)
    I = remove_vertical(I, Yd)
    I = cv2.transpose(I)
    return I


def add_vertical(I, Xd):
    I0 = I.copy()
    X0 = I.shape[1]
    X = X0
    Y = I.shape[0]
    mark = np.zeros((Y, X), dtype=bool)
    pos = np.zeros((X, Y), dtype=int)

    for i in range(X):
        for j in range(Y):
            pos[i, j] = i

    dp = np.zeros((X, Y), dtype=int)
    dir = np.zeros((X, Y), dtype=int)

    for k in range(Xd - X0):
        logging.info(f"Adding vertical seam {k + 1}/{Xd - X0}")
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        energy = calculate_energy(gray)

        dp[:X, 0] = energy[0, :X]

        for y in range(1, Y):
            for x in range(X):
                val = energy[y, x]
                dp[x, y] = -1

                if x > 0 and (dp[x, y] == -1 or val + dp[x - 1, y - 1] < dp[x, y]):
                    dp[x, y] = val + dp[x - 1, y - 1]
                    dir[x, y] = -1

                if dp[x, y] == -1 or val + dp[x, y - 1] < dp[x, y]:
                    dp[x, y] = val + dp[x, y - 1]
                    dir[x, y] = 0

                if x + 1 < X and (dp[x, y] == -1 or val + dp[x + 1, y - 1] < dp[x, y]):
                    dp[x, y] = val + dp[x + 1, y - 1]
                    dir[x, y] = 1

        best = dp[0, Y - 1]
        cur = 0

        for x in range(X):
            if dp[x, Y - 1] < best:
                best = dp[x, Y - 1]
                cur = x

        tmp = np.zeros((Y, X - 1, 3), dtype=np.uint8)

        for y in range(Y - 1, -1, -1):
            for i in range(X):
                if i < cur:
                    tmp[y, i] = I[y, i]
                elif i > cur:
                    tmp[y, i - 1] = I[y, i]
                    pos[i - 1, y] = pos[i, y]
                else:
                    mark[y, pos[i, y]] = True

            if y > 0:
                cur = cur + dir[cur, y]

        I = tmp
        X -= 1

    tmp = np.zeros((Y, Xd, 3), dtype=np.uint8)

    for i in range(Y):
        cont = 0

        for j in range(X0):
            if mark[i, j]:

                aux = (
                    average(I0[i, j], I0[i, j + 1])
                    if j == 0
                    else (
                        average(I0[i, j], I0[i, j - 1])
                        if j == X0 - 1
                        else average(I0[i, j - 1], I0[i, j + 1])
                    )
                )

                # the above line is equivalent to the following code
                # if j == 0:
                #     aux = average(I0[i, j], I0[i, j + 1])
                # elif j == X0 - 1:
                #     aux = average(I0[i, j], I0[i, j - 1])
                # else:
                #     aux = average(I0[i, j - 1], I0[i, j + 1])

                tmp[i, cont] = aux
                cont += 1
                tmp[i, cont] = aux
                cont += 1
            else:
                tmp[i, cont] = I0[i, j]
                cont += 1

    I = tmp
    return I


def add_horizontal(I, Yd):
    logging.info(f"Adding horizontal seam {Yd - I.shape[0]}")
    I = cv2.transpose(I)
    I = add_vertical(I, Yd)
    I = cv2.transpose(I)
    return I


def main():
    parser = argparse.ArgumentParser(description="Image Seam Carving")
    parser.add_argument("-f", "--filename", help="Path to the input image")
    parser.add_argument(
        "-dh", "--desired_height", type=int, help="Desired height of the output image"
    )
    parser.add_argument(
        "-dw", "--desired_width", type=int, help="Desired width of the output image"
    )
    args = parser.parse_args()

    file = args.filename
    img = cv2.imread(file)

    if img is None:
        logging.error("Unable to open image file.")
        exit(1)

    orig_h, orig_w = img.shape[:2]

    desired_h = args.desired_height
    desired_w = args.desired_width

    logging.info(f"Original Height: {orig_h} | Original Width: {orig_w}")
    logging.info(f"Desired Height: {desired_h} | Desired Width: {desired_w}")

    dupImg = img.copy()
    if desired_h <= orig_h:
        dupImg = remove_horizontal(dupImg, desired_h)
    else:
        dupImg = add_horizontal(dupImg, desired_h)

    if desired_w <= orig_w:
        dupImg = remove_vertical(dupImg, desired_w)
    else:
        dupImg = add_vertical(dupImg, desired_w)

    base_name, ext = os.path.splitext(file)
    output_file = f"results/{base_name}-result{ext}"
    cv2.imwrite(output_file, dupImg)

    logging.info(f"Output image saved to {output_file}")


if __name__ == "__main__":
    main()
