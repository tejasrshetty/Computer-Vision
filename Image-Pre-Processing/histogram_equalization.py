import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt


def create_histogram(img, low_level_pixel_value, high_level_pixel_value):
    hist = [0]*((high_level_pixel_value-low_level_pixel_value) +1)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # print(img[i][j])
            hist[img[i][j]] += 1

    return hist


def find_min_non_zero_val_hist(hist):
    min = 0
    for i in range(len(hist)):
        if hist[i] > 0:
            min = i
            break
    return min


def form_cumulative_histogram(hist):
    hist_cum = [0]*len(hist)

    hist_cum[0] = hist[0]
    for i in range(1, len(hist)):
        hist_cum[i] = hist_cum[i-1] + hist[i]
    return hist_cum


def equalize_hist(cum_hist,img):
    g_min = find_min_non_zero_val_hist(hist=cum_hist)
    h_min = cum_hist[g_min]
    eq_hist = [0]*len(cum_hist)
    denominator = (img.shape[0]*img.shape[1]- h_min)
    for i in range(len(cum_hist)):
        base_prod = (cum_hist[i]-h_min)/denominator
        normalized_prod = base_prod* (len(cum_hist)-1)
        eq_hist[i] = round(normalized_prod)
    return eq_hist


def equalize_image(img, eq_hist):
    hist_eq_img =  np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist_eq_img[i][j] = eq_hist[img[i][j]]
    return hist_eq_img


def eq_hist_and_plot_hist(input_img_loc, output_img_loc):
    img = cv2.imread(input_img_loc, 0)
    img_hist = create_histogram(img=img, high_level_pixel_value=255, low_level_pixel_value=0)
    cum_hist = form_cumulative_histogram(img_hist)
    init_eq_dist = equalize_hist(cum_hist,img)
    init_eq_img = equalize_image(img=img, eq_hist=init_eq_dist)
    init_eq_hist = create_histogram(img=init_eq_img, high_level_pixel_value=255, low_level_pixel_value=0)
    init_res = np.hstack((img, init_eq_img))  # stacking images side-by-side
    cv2.imwrite(output_img_loc, init_eq_img)
    factor = 0.04 / np.max(init_eq_hist)
    norms = copy.deepcopy(init_eq_hist)
    for i in range(0, len(init_eq_hist)):
        norms[i] = init_eq_hist[i] * (factor)
    index = np.arange(len(norms))
    plt.bar(index, norms)
    plt.legend(('Equalized Histogram for image ' + input_img_loc, ), loc='upper left')
    plt.show()
    factor = 0.04 / np.max(img_hist)
    norms_2 = img_hist
    for i in range(0, len(img_hist)):
        norms_2[i] = img_hist[i] * (factor)
    index = np.arange(len(norms))
    plt.bar(index, norms_2)
    plt.legend(('Original Histogram for image ' + input_img_loc,), loc='upper left')
    plt.show()


if __name__ == "__main__":
    # Equalize Histogram and plot it for Original image
    eq_hist_and_plot_hist('crowd.png','hist_eq_init_output.png')
    # Apply Histogram Equalization Again
    eq_hist_and_plot_hist('hist_eq_init_output.png','hist_eq_sec_output.png')
    eq_hist_and_plot_hist('low_contrast_image.jfif','low_contrast_output.png')
