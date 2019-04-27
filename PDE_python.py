import os
import sys
import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

def convoluteExperiment(u_, u, g, lambda_m, dt, dimCol, dimRow):
    #THIS IS NUMPY - thats why it is so fasttt
    z = u
    Ztop = z[0:-2, 1:-1]
    Ztop = np.pad(Ztop, ((2,0),(1,1)), 'constant', constant_values=(0,0))
    Zleft = z[1:-1, 0:-2]
    Zleft = np.pad(Zleft, ((1,1),(2,0)), 'constant', constant_values=(0,0))
    Zbottom = z[2:, 1:-1]
    Zbottom = np.pad(Zbottom, ((0,2),(1,1)), 'constant', constant_values=(0,0))
    Zright = z[1:-1, 2:]
    Zright = np.pad(Zright, ((1,1),(0,2)), 'constant', constant_values=(0,0))
    Zcenter = z
    #Zcenter = np.pad(Zcenter, ((1,1),(1,1)), 'constant', constant_values=(0,0))
    print("{} {} {} {} {}".format(Ztop.shape,Zleft.shape, Zbottom.shape, Zright.shape, Zcenter.shape))
    u_ = u + dt * (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) - dt * lambda_m* (u-g)
    print(type(u))
    return u_

def convolute(u_, u, g, lambda_m, dt, dimCol, dimRow):
    for i in range(1,dimCol-2):
        for j in range(1,dimRow-2):
            u_[i,j] = u[i,j] + dt * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) - dt * lambda_m[i,j]*(u[i,j]-g[i,j])

    return u_

def show_viz(i,original, masked, mask, inpainted):
    """Show Image using matplotlib"""
    plt.figure(i)
    plt.subplot(221), plt.imshow(original, 'gray')
    plt.title('original image')
    plt.subplot(222), plt.imshow(masked, 'gray')
    plt.title('source image')
    plt.subplot(223), plt.imshow(mask, 'gray')
    plt.title('mask image')
    plt.subplot(224), plt.imshow(inpainted, 'gray')
    plt.title('inpaint result')

    plt.tight_layout()
    plt.draw()

def show_ssim(original, masked, inpainted):
    """Show SSIM Difference"""
    print("SSIM : ")
    print("  Original vs. Original  : ", ssim(original,original))
    print("  Original vs. Masked    : ", ssim(original,masked))
    print("  Original vs. Inpainted : ", ssim(original,inpainted))

def inpaint(masked, mask):
    #Init Variable
    N = 10
    dt = 0.1
    lambda0 = 1
    DIM_COL, DIM_ROW = masked.shape
    lambda_m = np.multiply(mask,lambda0)
    u = masked.copy()

    t = time.time()
    i = 0

    #Iteration
    while i<N:
        u = convolute(u, u, masked, lambda_m, dt, DIM_COL,DIM_ROW)
        sys.stdout.write('\r')
        sys.stdout.write("Iteration : %d / %d" % (i+1, N))
        sys.stdout.flush()
        i=i+1

    sys.stdout.write('\r\n')
    sys.stdout.write("Execution Time :  %d s \n" % (time.time()-t))

    return u

if __name__ == '__main__':
    IMG_DIR = os.path.join(os.getcwd(), 'dataset')

    images = [x for x in next(os.walk(IMG_DIR))[2] if x.endswith('_.jpg')]
    for i in enumerate(images):
        i=i[0]+1
        original = cv2.imread(os.path.join(IMG_DIR, 'image{}_ori.jpg'.format(i)),0)
        masked = cv2.imread(os.path.join(IMG_DIR, 'image{}_.jpg'.format(i)),0)
        mask = cv2.imread(os.path.join(IMG_DIR, 'image{}_mask.jpg'.format(i)),0)

        # # Normalitation
        original = cv2.normalize(original, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        masked = cv2.normalize(masked, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask = 1-cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        print("\nFile : image{}_.png {}".format(i,masked.shape))
        inpainted = inpaint(masked,mask)

        show_viz(i,original,masked,mask,inpainted)
        show_ssim(original,masked,inpainted)

        #cv2.imwrite("output/image{}_inpaint.jpg".format(i),inpainted*255)

    plt.show()
