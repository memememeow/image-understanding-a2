import numpy as np
import scipy.linalg as sp
import cv2 as cv
import matplotlib.pyplot as plt


# Q4 a)
def find_sift_interest_points(source):
    gray = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT()

    # find the keypoints and descriptors with SIFT
    keypoint, descriptor = sift.detectAndCompute(gray, None)
    img = cv.drawKeypoints(source, keypoint, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img


# Q4 b)
def compute_distance(x1, x2, k):
    return np.linalg.norm(x1 - x2, k)

def sift_matching(source1, source2, order):
    gray1 = cv.cvtColor(source1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(source2, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT()

    threshold = 0.8
    # find the keypoints and descriptors with SIFT
    keypoint_1, descriptor_1 = sift.detectAndCompute(gray1, None)
    keypoint_2, descriptor_2 = sift.detectAndCompute(gray2, None)

    n = descriptor_1.shape[0]
    m = descriptor_2.shape[0]
    print n
    print m

    matches_kp = []
    num_match = 0

    # Find all max and second max for all keypoints and descriptor of first image
    for i in range(n):
        distances_for_i = []
        for j in range(m):
            distance = compute_distance(descriptor_1[i], descriptor_2[j], order)
            distances_for_i.append((distance, j))

        distances_for_i.sort()
        ratio = float(distances_for_i[0][0]) / float(distances_for_i[1][0])
        if ratio < threshold:
            num_match += 1
            matches_kp.append((ratio, (keypoint_1[i].pt, keypoint_2[distances_for_i[0][1]].pt)))

    # print matches_kp.shape
    matches_kp.sort()
    print len(matches_kp)
    top_ten_matches = []

    for i in range(10):
        # print sorted_distance[i, 1]
        top_ten_matches.append(matches_kp[i][1])

    return top_ten_matches


def compare_threshold(source1, source2):
    gray1 = cv.cvtColor(source1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(source2, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT()

    thresholds = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

    # find the keypoints and descriptors with SIFT
    keypoint_1, descriptor_1 = sift.detectAndCompute(gray1, None)
    keypoint_2, descriptor_2 = sift.detectAndCompute(gray2, None)

    n = descriptor_1.shape[0]
    m = descriptor_2.shape[0]

    num_matches = []

    for thres in thresholds:
        num_match = 0
        # Find all max and second max for all keypoints and descriptor of first image
        for i in range(n):
            distances_for_i = []
            for j in range(m):
                distance = compute_distance(descriptor_1[i], descriptor_2[j], 2)
                distances_for_i.append((distance, j))

            distances_for_i.sort()
            ratio = float(distances_for_i[0][0]) / float(distances_for_i[1][0])
            if ratio < thres:
                num_match += 1

        num_matches.append(num_match / float(n))

    print num_matches
    return thresholds, num_matches


# Q4 d)
def add_random_noise(img):

    noisy_img = np.zeros(img.shape)
    normed_noisy_img = np.zeros(img.shape)
    # Normalized the image
    noisy_img = cv.normalize(img.astype(np.float), noisy_img, 0, 1, cv.NORM_MINMAX)
    # Add a gaussian noise
    gauss = np.random.normal(0, 0.08, img.shape)
    noisy_image = noisy_img + gauss

    cv.normalize(noisy_image, normed_noisy_img, 0, 1, cv.NORM_MINMAX)

    # clip the matrix to 0 and 1
    np.clip(normed_noisy_img, 0, 1)

    return normed_noisy_img


# Q4 e)
def sift_matching_rgb(source1, source2, order):
    sift = cv.SIFT()

    threshold = 0.
    m = descriptor_2.shape[0]
    # find the keypoints and descriptors with SIFT
    keypoint_1, descriptor_1 = sift.detectAndCompute(source1, None)
    keypoint_2, descriptor_2 = sift.detectAndCompute(source2, None)

    n = descriptor_1.shape[0]

    matches_kp = []
    num_match = 0

    # Find all max and second max for all keypoints and descriptor of first image
    for i in range(n):
        distances_for_i = []
        for j in range(m):
            distance = compute_distance(descriptor_1[i], descriptor_2[j], order)
            distances_for_i.append((distance, j))

        distances_for_i.sort()
        ratio = float(distances_for_i[0][0]) / float(distances_for_i[1][0])
        if ratio < threshold:
            num_match += 1
            matches_kp.append((keypoint_1[i].pt, keypoint_2[distances_for_i[0][1]].pt))

    return matches_kp


if __name__ == '__main__':
    img1 = cv.imread('../sample1.jpg')
    img2 = cv.imread('../sample2.jpg')

    # Q4 a)
    sift_result_1 = find_sift_interest_points(img1)
    cv.imwrite("../results/q4a_SIFTInterestPoints_sample_1.jpg", sift_result_1)
    sift_result_2 = find_sift_interest_points(img2)
    cv.imwrite("../results/q4a_SIFTInterestPoints_sample_2.jpg", sift_result_2)

    # Q4 b)
    top_ten = sift_matching(img1, img2, 2)
    new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])

    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    for a, b in top_ten:
        end1 = (int(a[0]), int(a[1]))
        end2 = (int(b[0]) + img1.shape[1], int(b[1]))
        cv.line(new_img, end1, end2, (0, 255, 255), 2)

    cv.imwrite("../results/q4b_SIFTTop10Matches.jpg", new_img)

    # Q4 b) find best threshold
    thresholds, number_matches = compare_threshold(img1, img2)
    plt.plot(thresholds, number_matches, 'ro')
    plt.axis([0, 1.0, 0, 0.5])
    plt.show()


    # Q4 c)
    # with one norm
    top_ten = sift_matching(img1, img2, 1)
    new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])

    new_img_one_norm = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img_one_norm[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img_one_norm[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    for a, b in top_ten:
        end1 = (int(a[0]), int(a[1]))
        end2 = (int(b[0]) + img1.shape[1], int(b[1]))
        cv.line(new_img_one_norm, end1, end2, (0, 255, 255), 2)

    cv.imwrite("../results/q4c_SIFTTop10Matches_1_norm.jpg", new_img_one_norm)

    # with three norm
    top_ten = sift_matching(img1, img2, 3)
    new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])

    new_img_three_norm = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img_three_norm[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img_three_norm[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    for a, b in top_ten:
        end1 = (int(a[0]), int(a[1]))
        end2 = (int(b[0]) + img1.shape[1], int(b[1]))
        cv.line(new_img_three_norm, end1, end2, (0, 255, 255), 2)

    cv.imwrite("../results/q4c_SIFTTop10Matches_3_norm.jpg", new_img_three_norm)


    # Q4 d)
    noisy_img1 = add_random_noise(img1)
    noisy_img2 = add_random_noise(img2)

    new_img_1 =  (noisy_img1 * 255).astype(np.uint8)
    new_img_2 =  (noisy_img2 * 255).astype(np.uint8)

    cv.imwrite("../results/q4d_source_1_gaussian_noise.jpg", new_img_1)
    cv.imwrite("../results/q4d_source_2_gaussian_noise.jpg", new_img_2)

    sift_result_1 = find_sift_interest_points(new_img_1)
    cv.imwrite("../results/q4d_SIFTInterestPoints_noisy_sample_1.jpg", sift_result_1)
    sift_result_2 = find_sift_interest_points(new_img_2)
    cv.imwrite("../results/q4d_SIFTInterestPoints_noisy_sample_2.jpg", sift_result_2)

    top_ten = sift_matching(new_img_1, new_img_2, 2)
    new_shape = (max(new_img_1.shape[0], new_img_2.shape[0]), new_img_1.shape[1] + new_img_2.shape[1], new_img_1.shape[2])

    new_img = np.zeros(new_shape, type(new_img_1.flat[0]))
    # Place images onto the new image.
    new_img[0:new_img_1.shape[0], 0:new_img_1.shape[1]] = new_img_1
    new_img[0:new_img_2.shape[0], new_img_1.shape[1]:new_img_1.shape[1] + new_img_2.shape[1]] = new_img_2

    for a, b in top_ten:
        end1 = (int(a[0]), int(a[1]))
        end2 = (int(b[0]) + new_img_1.shape[1], int(b[1]))
        cv.line(new_img, end1, end2, (0, 255, 255), 2)

    cv.imwrite("../results/q4d_SIFTTop10Matches_gaussian_noise.jpg", new_img)


    # Q4 e)
    img_1 = cv.imread('../colourTemplate.png')
    img_2 = cv.imread('../colourSearch.png')

    top_ten = sift_matching_rgb(img_1, img_2, 2)
    new_shape = (max(img_1.shape[0], img_2.shape[0]), img_1.shape[1] + img_2.shape[1], img_1.shape[2])

    new_img = np.zeros(new_shape, type(img_1.flat[0]))
    # Place images onto the new image.
    new_img[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
    new_img[0:img_2.shape[0], img_1.shape[1]:img_1.shape[1] + img_2.shape[1]] = img_2

    for a, b in top_ten:
        end1 = (int(a[0]), int(a[1]))
        end2 = (int(b[0]) + img_1.shape[1], int(b[1]))
        cv.line(new_img, end1, end2, (0, 255, 255), 2)

    cv.imwrite("../results/q4e_SIFTTop10Matches_rgb.jpg", new_img)