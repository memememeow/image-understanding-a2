import numpy as np
import scipy.linalg as sp
import cv2 as cv


def find_local_maximum(R, patch_size):
    k = patch_size // 2
    result = np.zeros((R.shape[0] + 2 * k, R.shape[1] + 2 * k))
    result[k:R.shape[0] + k, k:R.shape[1] + k] = R

    for i in range(k, R.shape[0]+k):
        for j in range(k, R.shape[1]+k):
            local_patch = result[i - k: i + k + 1, j - k: j + k + 1]
            local_max = local_patch.max()
            if result[i, j] != local_max:
                result[i, j] = 0

    return result[k: R.shape[0] + k, k: R.shape[1] + k]


# Q2 a)
#  This function detects corners in source image based on Harris and Stephens, 88 to compute R
def harris_corner_detect(source):
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY).astype(np.float32) * 0.15
    result = np.copy(source)
    result_nms = np.copy(gray) * 0.2

    blur = cv.GaussianBlur(gray, (5, 5), 7)

    # Compute gradients in 2 directions.
    Ix = cv.Sobel(blur, cv.CV_64F, 1, 0)
    Iy = cv.Sobel(blur, cv.CV_64F, 0, 1)

    # Compute Ix2, Iy2, IxIy
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    # Average Ix2, Iy2, IxIy with Gaussian filter.
    Ix2_blur = cv.GaussianBlur(Ix2, (7, 7), 10)
    Iy2_blur = cv.GaussianBlur(Iy2, (7, 7), 10)
    IxIy_blur = cv.GaussianBlur(IxIy, (7, 7), 10)

    # Compute R = det(M) - alpha * trace(M) ** 2
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur, IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    R = det - 0.05 * np.multiply(trace, trace)

    # Find points with larger R value, which is larger than 0.02 * R.max().
    R_copy = np.copy(R)
    R_copy[R < 0.02 * R.max()] = 0
    result_nms[find_local_maximum(R_copy, 9) != 0] = 255

    result[R > 0.02 * R.max(), 2] = 255

    return R, result, result_nms


# Q2 a)
#  This function detects corners in source image based on Brown, harmonic mean to compute R
def brown_corner_detect(source):
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY).astype(np.float32) * 0.15
    result = np.copy(source)
    result_nms = np.copy(gray) * 0.2

    blur = cv.GaussianBlur(gray, (5, 5), 7)
    # Compute gradients in 2 directions.
    Ix = cv.Sobel(blur, cv.CV_64F, 1, 0)
    Iy = cv.Sobel(blur, cv.CV_64F, 0, 1)

    # Compute Ix2, Iy2, IxIy
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    # Average Ix2, Iy2, IxIy with Gaussian filter.
    Ix2_blur = cv.GaussianBlur(Ix2, (7, 7), 10)
    Iy2_blur = cv.GaussianBlur(Iy2, (7, 7), 10)
    IxIy_blur = cv.GaussianBlur(IxIy, (7, 7), 10)

    # Compute R = det(M) / trace(M)
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur, IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    trace[trace == 0] = np.Inf
    R = det / trace

    # Find points with larger R value, which is larger than 0.15 * R.max().
    R_copy = np.copy(R)
    R_copy[R < 0.15 * R.max()] = 0
    result_nms[find_local_maximum(R_copy, 9) != 0] = 255
    result[R > 0.15 * R.max(), 2] = 255

    return R, result, result_nms


def guassian_pyramid(img):
    iterations = int(np.log2(min(img.shape))) - 2
    gaussian_pyramid = []
    image = img

    print iterations

    for i in range(iterations):
        gaussian_pyramid += [(image, i)]
        image = cv.pyrDown(image)

    return gaussian_pyramid


# Q2 c)
def sift_interest_points_detector(source, s = 3, sigma = 1.6, threshold = 0.3):
    k = np.power(2.0, 1.0 / s)
    for i in range(s):
        print sigma * (k ** i)

    sift_interest_points = []
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY).astype(np.float32)
    print gray.shape

    # 1) Compute a Gaussian image pyramid with one octave
    gaussian_pyramid = guassian_pyramid(gray)
    for image, scale in gaussian_pyramid:
        print scale

    # For each gaussian layer, construct octave
    for image, scale in gaussian_pyramid:
        row, col = image.shape
        print "gaussian pyd layer shape"
        print image.shape
        print "scale"
        print 2 ** scale

        # - compute gaussian filters for each images in this octave
        cur_octave = np.asarray([cv.GaussianBlur(image, (5, 5), sigma * (k ** i)) for i in range(s + 1)])

        # 2) Compute difference of Gaussians at every scale
        dog_pyramid = np.subtract(cur_octave[1: s + 1, :, :], cur_octave[0: s, :, :])
        print "dog pyd shape"
        print dog_pyramid.shape

        # 3) Find local maxima in scale and positions
        # - Find the neighbor pixels in same image level for each pixel, padding image with 0
        padding_dog_pyramid = np.zeros((s, row + 2, col + 2))
        padding_dog_pyramid[:, 1: row + 1, 1: col + 1] = dog_pyramid
        dog_patches = np.asarray([[[np.abs(padding_dog_pyramid[img, i - 1: i + 2, j - 1: j + 2]).reshape(9)
                                    for j in range(1, col + 1)] for i in range(1, row + 1)] for img in range(s)])
        print "dog patch"
        print dog_patches.shape

        # - Find extrema of each patches inside dog_pyramid
        patch_maxima = dog_patches.max(axis=3)

        # - Pruning of insignificant extrema
        for i in range(s):
            patch_maxima[i][patch_maxima[i] < patch_maxima[i].max() * threshold] = 0

        # - loop over all pixels is really slow
        for img in range(s):
            for i in range(row):
                for j in range(col):
                    if dog_pyramid[img, i, j] == patch_maxima[img, i, j]:
                        if img == 0:
                            if dog_pyramid[img, i, j] > patch_maxima[img + 1, i, j]:
                                sift_interest_points += [(i * (2 ** scale), j * (2 ** scale), sigma * (k ** img) * (2 ** scale), scale)]
                        elif img == s - 1:
                            if dog_pyramid[img, i, j] > patch_maxima[img - 1, i, j]:
                                sift_interest_points += [(i * (2 ** scale), j * (2 ** scale), sigma * (k ** img) * (2 ** scale), scale)]
                        else:
                            if dog_pyramid[img, i, j] > patch_maxima[img + 1, i, j] \
                                    and dog_pyramid[img, i, j] > patch_maxima[img - 1, i, j]:
                                sift_interest_points += [(i * (2 ** scale), j * (2 ** scale), sigma * (k ** img) * (2 ** scale), scale)]

    print "sift keypoints shape"
    print np.asarray(sift_interest_points).shape
    return sift_interest_points


if __name__ == '__main__':

    # Q2 a)
    img = cv.imread('../building.jpg')
    # img = cv.imread('../synthetic.png')

    R_matrix, harris_result, harris_result_nms = harris_corner_detect(img)
    cv.imwrite("../results/q2a_harrisCornerDetect.jpg", harris_result)
    cv.imwrite("../results/q2a_harrisCornerDetect_nms.jpg", harris_result_nms)
    cv.imwrite("../results/q2a_harrisRMatrix.jpg", R_matrix)

    mean, brown_result, brown_result_nms = brown_corner_detect(img)
    cv.imwrite("../results/q2a_brownCornerDetect.jpg", brown_result)
    cv.imwrite("../results/q2a_brownCornerDetect_nms.jpg", brown_result_nms)
    cv.imwrite("../results/q2a_brownHarmonicMean.jpg", mean)

    # Q2 b)
    rotate_img = cv.imread('../building_rotate.jpg')

    R_matrix, harris_result, harris_result_nms = harris_corner_detect(rotate_img)
    cv.imwrite("../results/q2b_harrisCornerDetect_rotate.jpg", harris_result)
    cv.imwrite("../results/q2b_harrisCornerDetect_nms_rotate.jpg", harris_result_nms)
    cv.imwrite("../results/q2b_harrisRMatrix_rotate.jpg", R_matrix)

    # Q2 c)
    sift_result = np.copy(img)
    sift_interest_points = sift_interest_points_detector(img, s=5, sigma=1.6, threshold=0.56)
    for x, y, rho, scale in sift_interest_points:
        sift_result[x, y] = [0, 0, 255]
        cv.circle(sift_result, (y, x), int(rho * (2 ** 0.5)), (0, 255, 255), 2)
    cv.imwrite("../results/q2c_SIFTInterestPoints.jpg", sift_result)

    g1, g2, g3, g4 = sift_interest_points_detector(img)
    cv.imwrite("../results/q2c_g1.jpg", g1)
    cv.imwrite("../results/q2c_g2.jpg", g2)
    cv.imwrite("../results/q2c_g3.jpg", g3)
    cv.imwrite("../results/q2c_g4.jpg", g4)

    d1, d2, d3 = sift_interest_points_detector(img)
    cv.imwrite("../results/q2c_d1.jpg", d1)
    cv.imwrite("../results/q2c_d2.jpg", d2)
    cv.imwrite("../results/q2c_d3.jpg", d3)


