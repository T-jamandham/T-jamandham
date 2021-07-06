# libs to install, require python 3.7
# pip install opencv-python
# pip install numpy
# pip install imutils

import cv2 as cv
import numpy as np
import imutils


# opencv support to convert color from one format to another
# example: BGR, Grayscale, RGB, HSV, ...
def func1_ChangingColorSpace(img_path):
    # load image and show the original image
    img = cv.imread(img_path)
    cv.imshow('Original', img)

    # loop over each of the individual channels and display them
    # for (name, chan) in zip(('B', 'G', 'R'), cv.split(img)):
    #     cv.imshow(name, chan)

    # BGR -> Grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray Image', gray_img)

    # BGR -> HSV
    # hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # cv.imshow('HSV Image', hsv_img)

    # loop over each of the individual channels and display them
    # for (name, chan) in zip(('H', 'S', 'V'), cv.split(hsv_img)):
    #     cv.imshow(name, chan)

    cv.waitKey()
    cv.destroyAllWindows()

    # define where to save image
    path_to_save = 'Downloads/Multimedia/Downloaded/gray_img.png'
    save_img(path_to_save, gray_img)

def func2_Resize(img_path):
    img = cv.imread(img_path)   # BGR

    # using resize opencv
    # ratio = 200/img.shape[1]
    # dim = (200, int(img.shape[0]*ratio))
    # resize_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    # using resize method imutils, auto keep aspect ratio for us.
    resize_img = imutils.resize(img, height=300)  # Change height or width
    cv.imshow('Original', img)
    cv.imshow('Resized (Height)', resize_img)

    cv.waitKey()
    cv.destroyAllWindows()

    # define where to save image
    path_to_save = 'Downloads/Multimedia/Downloaded/resize_img.png'
    save_img(path_to_save, resize_img)

def func3_ImageTranslation(img_path):
    # load original image and show it to screen
    img = cv.imread(img_path)
    cv.imshow('Original', img)

    # shift the image 25 pixels to the right and 25 pixels down
    # Matrix M: [[1, 0, shiftX], [0, 1, shiftY]].
    # shiftX is horizontal (negative means shift left, positive means shift right)
    # shiftY is vertical (negative means go top, positive means go bottom)
    M = np.float32([[1, 0, 25], [0, 1, 25]])
    shifted_img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv.imshow('Shifted Right and Down', shifted_img)

    # Also use the imutils to shift image
    # imutils.translate(image, shiftX, shiftY)
    shifted_img2 = imutils.translate(img, 0, 100)
    cv.imshow('Shift Down', shifted_img2)

    cv.waitKey()
    cv.destroyAllWindows()

    # define where to save image
    path_to_save = 'Downloads/Multimedia/Downloaded/shifted_img.png'
    save_img(path_to_save, shifted_img)

# Rotate image
def func4_Rotation(img_path):
    # load original image and display it
    img = cv.imread(img_path)
    cv.imshow('Original', img)

    # grab the dimensions of the image and calculate the center of the image
    (h, w) = img.shape[:2]
    (cx, cy) = (w//2, h//2)

    # rotate our image by 45 degrees around the center of the image
    M = cv.getRotationMatrix2D((cx, cy), 45, 1.0)
    rotated_img = cv.warpAffine(img, M, (w, h))
    cv.imshow('Rotated Image by 45 Degrees', rotated_img)

    # use the imutils is the easy way to rotate image
    rotated_img2 = imutils.rotate(img, 90)
    cv.imshow('Rotated Image by imutils', rotated_img2)

    cv.waitKey()
    cv.destroyAllWindows()

    # define where to save image
    path_to_save = 'Downloads/Multimedia/Downloaded/rotated_img.png'
    save_img(path_to_save, rotated_img)


def func5_Crop(img_path):
    # load original image and display it
    img = cv.imread(img_path)
    cv.imshow('Original', img)

    # crop an image with OpenCV is just Numpy array slices in
    # startY:endY, startX:endX
    cat_face = img[30:420, 100:730]
    cv.imshow('Cat face', cat_face)

    cv.waitKey()
    cv.destroyAllWindows()

    # define where to save image
    path_to_save = 'Downloads/Multimedia/Downloaded/crop_img.png'
    save_img(path_to_save, cat_face)

# Blur image
def func6_Smoothing(img_path):
    # load original image and display it
    img = cv.imread(img_path)
    cv.imshow('Original', img)

    # average blur
    # define kernel size
    kernel_sizes = [(3, 3), (9, 9), (15, 15)]

    # loop through the kernel sizes:
    for (kx, ky) in kernel_sizes:
        # apply an 'average' blur to the image using the current kernel size
        blurred_img = cv.blur(img, (kx, ky))
        cv.imshow('Average ({}, {})'.format(kx, ky), blurred_img)

    # Gaussian Blur: similar to average blur but instead of using a simple mean
    # we are now using a weighted mean, where neighborhood pixels that are closer
    # to the central pixel contribute more weight to average
    # loop through the kernel sizes again
    for (kx, ky) in kernel_sizes:
        # apply a 'Gaussian' blur to the image
        blurred_img2 = cv.GaussianBlur(img, (kx, ky), 0)
        cv.imshow('Gaussian ({}, {})'.format(kx, ky), blurred_img2)

    cv.waitKey()
    cv.destroyAllWindows()

    # define where to save image
    path_to_save = 'Downloads/Multimedia/Downloaded/blur_img.png'
    save_img(path_to_save, blurred_img2)


def func7_EdgeDetection(img_path):
    # load original image and show it
    img = cv.imread(img_path)
    cv.imshow('Original', img)

    # edge detection with cv2.Canny
    # convert to grayscale and blur it slightly
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_img, (5, 5), 0)
    cv.imshow('Blurred', blurred)

    # compute a 'wide', 'mid-range', and 'tight' threshold for the edges
    # using the Canny edge detector
    wide = cv.Canny(blurred, 10, 200)
    mid = cv.Canny(blurred, 30, 150)
    tight = cv.Canny(blurred, 240, 250)

    # show the output Canny edge maps
    cv.imshow('Wide Edge Map', wide)
    cv.imshow('Mid Edge Map', mid)
    cv.imshow('Tight Edge Map', tight)

    cv.waitKey()
    cv.destroyAllWindows()

    # define where to save image
    path_to_save = 'Downloads/Multimedia/Downloaded/edge_detect.png'
    save_img(path_to_save, wide)


def func8_flipImage(img_path):
    # load original image and show it
    img = cv.imread(img_path)
    cv.imshow('Original', img)

    # flip the image horizontally
    flipped = cv.flip(img, 1)
    cv.imshow('Flipped Horizontally', flipped)

    # flip the image vertically
    flipped_img = cv.flip(img, 0)
    cv.imshow('Flipped Vertically', flipped_img)

    # flip the image both horizontally and vertically
    flipped_both = cv.flip(img, -1)
    cv.imshow('Flipped Horizontally & Vertically', flipped_both)

    cv.waitKey()
    cv.destroyAllWindows()

    # define where to save image
    path_to_save = 'Downloads/Multimedia/Downloaded/flip_img.png'
    save_img(path_to_save, flipped_img)


def save_img(path_to_save, img):
    print('Bạn có muốn lưu ảnh không? (y/n)')
    save = input()
    if save == 'y' or save == 'Y':
        cv.imwrite(path_to_save, img)


def menu(img_path):
    while True:
        print('Chon chuc nang[1-9]:')
        print('1. Thay đổi không gian màu')
        print('2. Thay đổi kích thước ảnh')
        print('3. Dịch chuyển vị trí ảnh')
        print('4. Xoay ảnh 1 góc')
        print('5. Cắt ảnh')
        print('6. Làm mờ ảnh')
        print('7. Phát hiện cạnh')
        print('8. Lật ảnh')
        print('9. Thoát chương trình')

        try:
            choice = int(input())
        except ValueError:
            print('Oops! Đó không phải là số hợp lệ. Thử lại')
            continue

        if choice == 1:
            func1_ChangingColorSpace(img_filepath)
        elif choice == 2:
            func2_Resize(img_path)
        elif choice == 3:
            func3_ImageTranslation(img_path)
        elif choice == 4:
            func4_Rotation(img_path)
        elif choice == 5:
            func5_Crop(img_path)
        elif choice == 6:
            func6_Smoothing(img_path)
        elif choice == 7:
            func7_EdgeDetection(img_path)
        elif choice == 8:
            func8_flipImage(img_path)
        elif choice == 9:
            exit()
        else:
            print('Yêu cầu không hợp lệ, vui lòng nhập lại')


if __name__ == '__main__':
    img_filepath = 'MultiMediaPic/13.jpg'
    menu(img_filepath)
