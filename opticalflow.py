#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author - Mohit Rathore <mrmohitrathoremr@gmail.com>
# Licensed under The MIT License - https://opensource.org/licenses/MIT
# source - https://www.youtube.com/watch?v=kJouUVZ0QqU

from PIL import Image
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
from scipy import ndimage

from scipy.ndimage.filters import convolve

FOLDER_PATH, EPOCHS = "data/", 100

# first differntial equations
x_kernel = np.array([[-1, 1], [-1, 1]]) / 4.
y_kernel = np.array([[-1, -1], [1, 1]]) / 4.
t_kernel = np.array([[-1, -1], [-1, -1]]) / 4.

av_kernel = np.ones((3, 3)) / 9.

regularisation_constant = 1e-8

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float32" )
    return np.dot(data[...,:3], [0.299, 0.587, 0.114])


def main(path1, path2, epochs):

    img1 = load_image(path1)
    img2 = load_image(path2)    
    
    f_x = convolve(img1, x_kernel) + convolve(img2, x_kernel)
    f_y = convolve(img1, y_kernel) + convolve(img2, y_kernel)
    f_t = convolve(img1,t_kernel) + convolve(img2, -t_kernel)

    u = np.zeros_like(img1)
    v = np.zeros_like(img2)

    for _ in range(epochs):
        u_av = convolve(u, av_kernel)
        v_av = convolve(v, av_kernel)

        P = f_x * u_av + f_y * v_av + f_t
        D = regularisation_constant + np.square(f_x) + np.square(f_y)

        u = u_av - f_x * P / D
        v = v_av - f_y * P / D

    # fix direction and length of arrows
    u *= -4
    v *= -4
 
    ax = plt.figure(figsize=(10, 10)).gca()   

    step = 6
    for i in range(0,len(u), step):
        for j in range(0,len(v), step):
            ax.arrow(i, j, v[i,j], u[i,j], color='r', head_width=1.8)
 
    
    plt.imshow(img2.T,cmap = 'gray', animated=True)

    print("saving output image to output/" + path2.split("/")[-1])
    plt.savefig("output/" + path2.split("/")[-1])

    plt.close()


def homography(folder_path, epochs):
    files_names = sorted(os.listdir(folder_path))

    images = []
    for i in range(len(files_names)-1):
        main(folder_path + "/" + files_names[i], folder_path + "/" + files_names[i+1], epochs)
        images.append(imageio.imread("output/" +  files_names[i+1]))

    imageio.mimsave('output.gif', images)

if __name__ == "__main__":
    homography(FOLDER_PATH, EPOCHS)