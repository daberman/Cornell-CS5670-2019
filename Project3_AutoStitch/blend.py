import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    minX = np.inf
    minY = np.inf
    maxX = -np.inf
    maxY = -np.inf
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            xy_trans = np.matmul(M, [x, y, 1])
            xy_trans /= xy_trans[2]
            if xy_trans[0] < minX:
                minX = xy_trans[0]
            elif xy_trans[0] > maxX:
                maxX = xy_trans[0]
            if xy_trans[1] < minY:
                minY = xy_trans[1]
            elif xy_trans[1] > maxY:
                maxY = xy_trans[1]
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    for ay in range(acc.shape[0]):
        for ax in range(acc.shape[1]):
            # Inverse warp acc to get img coords
            M_inv = np.linalg.inv(M)
            x, y, z = np.matmul(M_inv, [ax, ay, 1])
            x = x / z
            y = y / z
            # Check in bounds
            if x < 0 or x > img.shape[1]-1 or y < 0 or y > img.shape[0]-1:
                continue

            # Linear interpolation of img pixels for RGB value
            percent_x = x - np.floor(x)
            percent_y = y - np.floor(y)
            r, g, b = percent_x * percent_y * img[int(np.ceil(y)), int(np.ceil(x))]
            _r, _g, _b = (1-percent_x) * percent_y * img[int(np.ceil(y)), int(np.floor(x))]
            r += _r
            g += _g
            b += _b
            _r, _g, _b = percent_x * (1-percent_y) * img[int(np.floor(y)), int(np.ceil(x))]
            r += _r
            g += _g
            b += _b
            _r, _g, _b = (1-percent_x) * (1-percent_y) * img[int(np.floor(y)), int(np.floor(x))]
            r += _r
            g += _g
            b += _b

            # Check not black
            if r == 0 and g == 0 and b == 0:
                continue

            alpha = 1

            # Check if x within blendWidth
            if x < blendWidth:
                alpha = (x + 1.0) / (blendWidth + 1.0)
            elif img.shape[1] - x < blendWidth:
                alpha = float(img.shape[1] - x) / (blendWidth + 1.0)

            acc[ay, ax] += [r * alpha, g * alpha, b * alpha, alpha]


    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    img = np.zeros((acc.shape[0], acc.shape[1], 3))
    for ay in range(acc.shape[0]):
        for ax in range(acc.shape[1]):
            if acc[ay, ax, 3] > 0:
                img[ay, ax] = acc[ay, ax, :3] / acc[ay, ax, 3]
            else:
                img[ay, ax] = acc[ay, ax, :3]
    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        _minX, _minY, _maxX, _maxY = imageBoundingBox(img, M)
        minX = min(_minX, minX)
        minY = min(_minY, minY)
        maxX = max(_maxX, maxX)
        maxY = max(_maxY, maxY)
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

