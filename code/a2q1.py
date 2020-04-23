import numpy as np
import scipy.linalg as sp
import cv2 as cv


# Q1 a)
# Function does one dimensional linear interpolation to d times the input image size.
# source: the original image we want to interpolate.
# d: size want to increase.
def one_d_linear_interpolation(source, d):
    g_prime = np.zeros((source.shape[0], source.shape[1] * d))
    g = np.zeros((source.shape[0], source.shape[1] * d))

    # Replace points the position [i, j] in g_prime that i and j/d are integers with image pixel values
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            g_prime[i, j * d] = source[i, j]

    # Generate filter for averaging based on d
    h_filter = np.concatenate([np.arange(d), np.arange(d, -1, -1)]).astype(np.float) / d
    print h_filter

    for r in range(source.shape[0]):
        # do convolution of h_filter and each row of g prime
        g[r] = np.convolve(g_prime[r], h_filter, "same")

    return g


# Q1 b)
# Function does two dimensional linear interpolation to d times the input image size.
# source: the original image we want to interpolate.
# d: size want to increase.
def one_d_linear_interpolation(source, d):
    g_prime = np.zeros((source.shape[0] * d, source.shape[1] * d))

    # Replace points the position [i, j] in g_prime that i/d and j/d are integers with image pixel values
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            g_prime[i * d, j * d] = source[i, j]

    # Generate filter for averaging based on d
    one_d_filter = np.concatenate([np.arange(d), np.arange(d, -1, -1)]).astype(np.float) / d
    h_filter = np.multiply(one_d_filter.reshape(-1, 1), one_d_filter.reshape(1, -1))
    print h_filter

    # do convolution of h_filter and g prime
    g = cv.filter2D(g_prime, cv.CV_32F, h_filter)

    return g


if __name__ == '__main__':
    img = cv.imread('../bee.jpg')

    #Q1 a)
    one_d_result = np.zeros((4 * img.shape[0], 4 * img.shape[1], img.shape[2]))
    one_d_result[:, :, 0] = one_d_linear_interpolation(one_d_linear_interpolation(img[:, :, 0], 4).T, 4).T
    one_d_result[:, :, 1] = one_d_linear_interpolation(one_d_linear_interpolation(img[:, :, 1], 4).T, 4).T
    one_d_result[:, :, 2] = one_d_linear_interpolation(one_d_linear_interpolation(img[:, :, 2], 4).T, 4).T
    cv.imwrite("../results/q1a_1DQuadrupleSize.jpg", one_d_result)

    # Q1 b)
    two_d_result = np.zeros((4 * img.shape[0], 4 * img.shape[1], img.shape[2]))
    two_d_result[:, :, 0] = one_d_linear_interpolation(img[:, :, 0], 4)
    two_d_result[:, :, 1] = one_d_linear_interpolation(img[:, :, 1], 4)
    two_d_result[:, :, 2] = one_d_linear_interpolation(img[:, :, 2], 4)
    cv.imwrite("../results/q1b_2DQuadrupleSize.jpg", two_d_result)
