from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
from functions import *

personal = 1

if personal == 0:
    folderpath = os.path.dirname(os.path.abspath(__file__)) + "\\images\\"
else:
    folderpath = os.path.dirname(os.path.abspath(__file__)) + "\\images\\personal\\"

image_data = {}

images_path = [os.path.join(folderpath, imagename) for imagename in os.listdir(folderpath) if imagename.endswith(".tiff")]
images_path.sort()

HH=[]
for p in images_path:

    img = os.path.basename(p).split('.')[0]
    image_data[img] = {}

    grid_size = (8, 11)
    found, H, corners, real_coordinates = estimate_homographies(p, grid_size)

    image_data[img]["name"] = img
    
    if found is True:
        HH.append(H)
        image_data[img]["corners"] = corners
        image_data[img]["real_coordinates"] = real_coordinates
        image_data[img]["H"] = H
    
    else:
        image_data[img]["corners"] = False


K = compute_K(HH)

for p in images_path:

    img = os.path.basename(p).split('.')[0]

    if image_data[img]["corners"] is not False:

        H = image_data[img]["H"]

        lambd = 1/(np.linalg.norm(np.linalg.inv(K)@H[:,0]))
        image_data[img]["lambda"] = lambd

        R, t = compute_Rt(H, K)
        image_data[img]["t"] = t
        image_data[img]["R"] = R

        P = K @ np.hstack((R, t))
        image_data[img]["P"] = P

        alphau = K[0,0]
        alphav = K[1,1]
        u0 = K[0,2]
        v0 = K[1,2]

#print(image_data)
#print(image_data["image03"]["real_coordinates"])
#print(K)

for p in images_path:

    img = os.path.basename(p).split('.')[0]

    image = cv2.imread(p)
    image2=image.copy()

    if image_data[img]["corners"] is not False:

        error = compute_error(image_data[img])
        print(f'{image_data[img]["name"]} - {error}')

        draw_points(image, image_data[img])


        superimpose_cylinder(image2, image_data[img], 55, 35, 30, 75)

