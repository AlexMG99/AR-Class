import numpy as np
import cv2


def BoxFilter(img):
    # Normalize image in order to have values from 0 to 1
    img = img/255

    # Kernel
    ksize = 11
    krn = np.zeros((ksize, ksize))
    krn[:,:] = 1.0 / (ksize*ksize)

    # Filter
    filtered = convolve(img, krn)

    return filtered

def convolve(img, krn):
    #Kernel
    ksize, _ = krn.shape
    krad = int(ksize/2)

    #Frame
    height, width, depth = img.shape
    framed = np.ones((height + 2*krad, width + 2*krad, depth))
    framed[krad:-krad, krad:-krad] = img

    #Filter
    filtered = np.zeros(img.shape)
    for i in range(0, height):
        for j in range(0, width):
            filtered[i, j] = (framed[i:i+ksize,j:j+ksize] * krn[:,:, np.newaxis]).sum(axis=(0,1))

    return filtered




def main():
    img = cv2.imread("yaoi.jfif", cv2.IMREAD_COLOR)
    patata = BoxFilter(img)
    cv2.imshow("Lenna", patata)
    cv2.imshow("Leanna", img)

    k = cv2.waitKey(0)

    if(k == 27):
        cv2.destroyAllWindows()


main()