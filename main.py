import math
import os
import glob

from PIL import Image
import numpy as np


def histogramify(img_list, is_hsi, threshold):
    freq_table = {}
    total_pixels = 0
    for image in img_list:
        img = Image.open(image)
        img_array = np.array(img)
        new_array = np.zeros_like(img_array, dtype=np.float64)
        for row in range(len(img_array)):
            for rgb in range(len(img_array[0])):
                r = img_array[row][rgb][0]
                g = img_array[row][rgb][1]
                b = img_array[row][rgb][2]
                if is_hsi:
                    (h, s, i) = hsiify(r, g, b)
                    new_array[row][rgb][0] = h
                    new_array[row][rgb][1] = s
                    new_array[row][rgb][2] = i
                else:
                    (y, cr, cb) = ycrcbify(r, g, b)
                    new_array[row][rgb][0] = y
                    new_array[row][rgb][1] = cr
                    new_array[row][rgb][2] = cb

        if is_hsi:
            for row in range(len(new_array)):
                for hsi in range(len(new_array[0])):
                    freq_table[(new_array[row][hsi][0], new_array[row][hsi][1])] = \
                        freq_table.get((new_array[row][hsi][0], new_array[row][hsi][1]), 0) + 1
        else:
            for row in new_array:
                for ycrcb in row:
                    freq_table[(ycrcb[1], ycrcb[2])] = freq_table.get((ycrcb[1], ycrcb[2]), 0) + 1

        # at this point, the freq_table should be full
        total_pixels += len(img_array) * len(img_array[0])

    for key, val in freq_table.items():
        freq_table[key] = val / total_pixels

    norm_dict = {}
    for k, v in freq_table.items():
        if v > threshold:
            norm_dict[k] = v

    return norm_dict


# these equations are from https://www.had2know.org/technology/hsi-rgb-color-converter-equations.html#:~:text=Equations%20to%20Convert%20RGB%20Values%20to%20HSI%20Values&text=I%20%3D%20(R%20%2B%20G%20%2B%20B)%2F3.
def hsiify(r, g, b):
    r = float(r)
    b = float(b)
    g = float(g)
    numer = (r - ((1 / 2) * g) - ((1 / 2) * b))
    denom = (r ** 2) + (g ** 2) + (b ** 2) - (r * g) - (r * b) - (g * b)
    if g >= b:
        h = math.acos(numer / math.sqrt(denom)) * (180 / math.pi) if denom > 0 else 90
    else:
        h = 360 - (math.acos(numer / math.sqrt(denom)) * (180 / math.pi)) if denom > 0 else 270

    i = sum([r, g, b]) / 3
    s = 1 - (min(r, g, b) / i) if i > 0 else 0
    return h, s, i


# these equations are from https://sistenix.com/rgb2ycbcr.html
def ycrcbify(r, g, b):
    y = 16 + ((65.738 / 256) * r) + ((129.057 / 256) * g) + ((25.064 / 256) * b)
    cb = 128 - ((37.945 / 256) * r) - ((74.494 / 256) * g) + ((112.439 / 256) * b)
    cr = 128 + ((112.439 / 256) * r) - ((94.154 / 256) * g) - ((18.285 / 256) * b)
    return y, cr, cb


def detect_skin(test_files, skin_colors, hsi):
    new_im_list = []
    for file in test_files:
        img_array = np.array(Image.open(file))
        convert_array = np.zeros_like(img_array)
        for row in range(len(img_array)):
            for rgb in range(len(img_array[0])):
                r = img_array[row][rgb][0]
                g = img_array[row][rgb][1]
                b = img_array[row][rgb][2]

                if hsi:
                    (h, s, i) = hsiify(r, g, b)
                    if (h, s) in skin_colors:
                        convert_array[row][rgb][0] = r
                        convert_array[row][rgb][1] = g
                        convert_array[row][rgb][2] = b
                else:
                    (y, cr, cb) = ycrcbify(r, g, b)
                    if (cr, cb) in skin_colors:
                        convert_array[row][rgb][0] = r
                        convert_array[row][rgb][1] = g
                        convert_array[row][rgb][2] = b

        new_im_list.append(convert_array)
    return new_im_list


def create_new_images(test_files, new_img_list, hsi):
    for i in range(len(new_img_list)):
        filename = test_files[i].split('/')[-1]
        image = Image.fromarray(new_img_list[i])
        if hsi:
            image.save("{}_skin_hsi.bmp".format(filename))
        else:
            image.save("{}_skin_ycrcb.bmp".format(filename))


def other_colorspace():
    training_files = list(glob.glob(os.path.join('./training_files', '*.bmp')))
    skin_colors = histogramify(training_files, False, 0.00001)
    test_files = list(glob.glob(os.path.join('./test_files', '*.bmp')))
    new_img_list = detect_skin(test_files, skin_colors, False)
    assert len(new_img_list) == len(test_files)

    create_new_images(test_files, new_img_list, False)


def main():
    training_files = list(glob.glob(os.path.join('./training_files', '*.bmp')))
    skin_colors = histogramify(training_files, True, 0.00000001)
    test_files = list(glob.glob(os.path.join('./test_files', '*.bmp')))
    new_img_list = detect_skin(test_files, skin_colors, True)
    assert len(new_img_list) == len(test_files)

    create_new_images(test_files, new_img_list, True)

    other_colorspace()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
