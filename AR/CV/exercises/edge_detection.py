import numpy as np
import cv2

def Sobel(img, sobel_krn):
    # Normalize image in order to have values from 0 to 1
    img = img/255            

    # Filter
    filtered = convolve(img, sobel_krn)

    return filtered

def convolve(img, krn):
    #Kernel
    ksize, _ = krn.shape
    krad = int(ksize/2)

    #Frame
    height, width= img.shape
    framed = np.ones((height + 2*krad, width + 2*krad))
    framed[krad:-krad, krad:-krad] = img

    #Filter
    filtered = np.zeros(img.shape)
    for i in range(0, height):
        for j in range(0, width):
            filtered[i, j] = (framed[i:i+ksize,j:j+ksize] * krn[:,:]).sum(axis=(0,1))

    return filtered

def main():
    operator_hor = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

    operator_ver = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]]) 

    img = cv2.imread("sonic.jpg", 0)
    imgH = Sobel(img, operator_hor)
    imgV = Sobel(img, operator_ver)

    cv2.imshow("Normal", img)
    cv2.imshow("Horizontal", imgH)
    cv2.imshow("Vertical", imgV)

    imgFinal = np.sqrt(imgH**2 + imgV**2)
    imgFinal = imgFinal / imgFinal.max()


    cv2.imshow("Total", imgFinal)

    k = cv2.waitKey(0)

    if(k == 27):
        cv2.destroyAllWindows()


main()