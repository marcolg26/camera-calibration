import sys
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
from mpmath import *
from functions import *


def calibrate_camera(personal=0, grid_size_x=8, grid_size_y=11, square_size=11):

    if personal == 0:
        folderpath = os.path.dirname(os.path.abspath(__file__)) + "\\images\\"
    else:
        folderpath = os.path.dirname(
            os.path.abspath(__file__)) + "\\images\\personal\\"

    image_data = {}
    K_parameters = {}

    images_path = [os.path.join(folderpath, imagename) for imagename in os.listdir(
        folderpath) if (imagename.endswith(".tiff") or imagename.endswith(".tif"))]
    images_path.sort()

    Hk = []
    for p in images_path:

        img = os.path.basename(p).split('.')[0]
        image_data[img] = {}

        grid_size = (grid_size_x, grid_size_y)
        found, H, corners, real_coordinates = estimate_homographies(
            p, grid_size, square_size)

        image_data[img]["name"] = img

        if personal == 1:
            image_data[img]["personal"] = True
        else:
            image_data[img]["personal"] = False

        if found is True:
            Hk.append(H)
            image_data[img]["corners"] = corners
            image_data[img]["real_coordinates"] = real_coordinates
            image_data[img]["H"] = H

        else:
            image_data[img]["corners"] = False

    K = compute_K(Hk)

    for p in images_path:

        img = os.path.basename(p).split('.')[0]

        if image_data[img]["corners"] is not False:

            H = image_data[img]["H"]

            lambd = 1/(np.linalg.norm(np.linalg.inv(K)@H[:, 0]))
            image_data[img]["lambda"] = lambd

            R, t = compute_Rt(H, K)
            image_data[img]["t"] = t
            image_data[img]["R"] = R

            P = K @ np.hstack((R, t))
            image_data[img]["P"] = P

    K_parameters["alphau"] = K[0, 0]
    K_parameters["skew"] = acot(K[0, 1]/K_parameters["alphau"])
    K_parameters["alphav"] = K[1, 1]/sin(K_parameters["skew"])
    K_parameters["u0"] = K[0, 2]
    K_parameters["v0"] = K[1, 2]

    print(f'K = {K}')
    print(f'alphau = {K_parameters["alphau"]}')
    print(f'alphav = {K_parameters["alphav"]}')
    print(f'skew = {K_parameters["skew"]}')
    print(f'u0 = {K_parameters["u0"]}')
    print(f'v0 = {K_parameters["v0"]}')

    for p in images_path:

        img = os.path.basename(p).split('.')[0]

        image = cv2.imread(p)
        image2 = image.copy()

        if image_data[img]["corners"] is not False:

            compute_error(image_data[img])

            draw_points(image, image_data[img])
            xx = round(max(image_data[img]["real_coordinates"][:, 0])/2)
            yy = round(max(image_data[img]["real_coordinates"][:, 1])/2)

            superimpose_cylinder(image2, image_data[img], xx, yy, 30, 75)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        personal = int(sys.argv[1])
        grid_size_x = int(sys.argv[2])
        grid_size_y = int(sys.argv[3])
        square_size = int(sys.argv[4])

        calibrate_camera(personal, grid_size_x, grid_size_y, square_size)

    else:
        calibrate_camera()
