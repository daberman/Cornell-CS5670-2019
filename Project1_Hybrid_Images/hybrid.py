import sys
import cv2
import numpy as np


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    m, n = np.shape(kernel)
    hm = (m-1) / 2
    hn = (n-1) / 2

    newImg = np.zeros_like(img)

    # Pad based on 1-D vs 3-D
    if len(np.shape(img)) == 3: # If there are 3 values, then img is a w x y x 3 img
        img = np.pad(img, ((hm, hm), (hn, hn), (0, 0)), mode='constant', constant_values=0)
        # Also need to update kernel to be 3-D
        kernel = np.reshape(kernel, kernel.size)  # Flatten
        tempKernel = np.empty(3*kernel.size)
        tempKernel[0::3] = kernel  # Interlace kernel 3x
        tempKernel[1::3] = kernel
        tempKernel[2::3] = kernel
        kernel = np.reshape(tempKernel, (m, n, 3))  # Reshape to be 3-D kernel
    else: # otherwise, assume img is w x y
        img = np.pad(img, ((hm, hm), (hn, hn)), mode='constant', constant_values=0)

    # Apply kernel across the image
    for i in range(np.shape(newImg)[0]):
        for j in range(np.shape(newImg)[1]):
            newImg[i, j] = np.sum(img[i:i+m, j:j+n] * kernel, axis=(0, 1))

    return newImg


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    kernel = np.flip(kernel)

    return cross_correlation_2d(img, kernel)


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    kernel = np.zeros((height, width))

    mp = ((height-1)/2, (width-1)/2)

    for i in range(mp[0] + 1):
        for j in range(mp[1] + 1):
            # PDF = 1/(2*pi*sigma**2) * e^(-(x**2+y**2)/(2*sigma**))
            # PDF * pi = integrated value in circle of radius 1 pixel around pixel center;
            val = 1.0 / (2.0 * sigma**2) * np.exp(-(i**2 + j**2) / (2.0 * sigma**2))
            kernel[mp[0]+i, mp[1]+j] = val
            kernel[mp[0]+i, mp[1]-j] = val
            kernel[mp[0]-i, mp[1]+j] = val
            kernel[mp[0]-i, mp[1]-j] = val

    return kernel / np.sum(kernel)  # Normalize


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

