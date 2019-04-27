#Import libraries for simulation
import os
import time
import sys
import numpy as np
import cv2
import theano as th
from theano import tensor as T
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape([1,1]+ list(a.shape))
    return np.float32(a)

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
    # Init variable
    N = 2000

    #Variable for simulation
    input = T.tensor4('input')
    filters = T.tensor4('filters')
    kernel = make_kernel([[0.0, 1.0, 0.0],
                          [1.0, -4., 1.0],
                          [0.0, 1.0, 0.0]])

    theano_convolve2d = th.function([input, filters],
                                    T.nnet.conv2d(input, filters,
                                    border_mode='half',
                                    subsample=(1, 1)))

    # Create variables for simulation state
    U = masked
    G = masked
    M = np.multiply(mask,1)
    dt = 0.1

    """Discretized PDE update rules"""
    """u[i,j] = u[i,j] + dt * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) - dt * lambda_m[i,j]*(u[i,j]-g[i,j])"""

    # Run N steps of PDE
    t = time.time()
    for i in range(N):
        theano_convolved = theano_convolve2d(U.reshape(1, 1, U.shape[0], U.shape[1]),
                                             kernel.reshape(1, 1, 3, 3))
        U = U + dt * theano_convolved[0,0,:,:] - dt * M * (U-G)
        sys.stdout.write('\r')
        sys.stdout.write("Iteration : %d / %d" % (i+1, N))
        sys.stdout.flush()

    print("")
    print("Execution Time : {} s".format(time.time()-t))
    return np.float32(U)

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

        print("\nFile : image{}_.jpg {}".format(i,masked.shape))
        inpainted = inpaint(masked,mask)

        show_viz(i,original,masked,mask,inpainted)
        show_ssim(original,masked,inpainted)

        cv2.imwrite("output/image{}_inpaint_th.jpg".format(i),inpainted*255)

    plt.show()
