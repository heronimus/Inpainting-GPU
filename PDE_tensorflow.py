#Import libraries for simulation
import os
import time
import sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def heat_conv(input, kernel):
  """A simplified 2D convolution operation for Heat Equation"""
  input = tf.expand_dims(tf.expand_dims(input, 0), -1)

  result = tf.nn.depthwise_conv2d(input, kernel,
                                    [1, 1, 1, 1],
                                    padding='SAME')

  return result[0, :, :, 0]

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
    ROOT_DIR = os.getcwd()

    # Create variables for simulation state
    U = tf.Variable(masked)
    G = tf.Variable(masked)
    M = tf.Variable(np.multiply(mask,1))
    K = make_kernel([[0.0, 1.0, 0.0],
                     [1.0, -4., 1.0],
                     [0.0, 1.0, 0.0]])

    dt = tf.placeholder(tf.float32, shape=())

    """Discretized PDE update rules"""
    """u[i,j] = u[i,j] + dt * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) - dt * lambda_m[i,j]*(u[i,j]-g[i,j])"""

    #Tensorflow while_loop function, iterate the PDE N times.
    index_summation = (tf.constant(1), U, M, G, K)
    def condition(i, U, M, G, K):
        return tf.less(i, 2000)

    def body(i,U,M,G,K):
        U_ = U + 0.1 * heat_conv(U,K) - 0.1 * M * (U-G)
        # i = tf.Print(i, [i])
        return tf.add(i, 1), U_, M, G, K

    #Tensorflow Session
    with tf.Session():
        # Initialize state to initial conditions
        tf.global_variables_initializer().run()

        #Run PDE using tensorflow while_loop
        t = time.time()
        uf=tf.while_loop(condition, body, index_summation)[1]
        U = uf.eval()

    print("Execution Time : {} s".format(time.time()-t))

    return U

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
        cv2.imwrite("output/image{}_inpaint_tf.jpg".format(i),inpainted*255)

    plt.show()
