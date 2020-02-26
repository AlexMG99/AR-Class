import numpy as np
import cv2


def BoxFilter(img):
    height, width, depth = img.shape

    out = np.zeros(img.shape)

    for i in range(0, height):
        for j in range(0, width):
            if(i-1>0 and i+1<height and j-1>0 and j+1<width):
                out[i,j] += img[i,j]*1/9
                out[i,j] += img[i+1,j]*1/9
                out[i,j] += img[i+1,j+1]*1/9
                out[i,j] += img[i+1,j-1]*1/9
                out[i,j] += img[i,j+1]*1/9
                out[i,j] += img[i,j-1]*1/9
                out[i,j] += img[i-1,j]*1/9
                out[i,j] += img[i-1,j+1]*1/9
                out[i,j] += img[i-1,j-1]*1/9         

    return out

def CalculateNeighbour(pixel):
    neighbour = arange(9)
    neighbour = neighbour.reshape(3,3)
    for i in range(0, 3):
        for j in range(0, 3):
            neighbour[i, j] = pixel[i, j]
    
    return neighbour.mean()


def main():
    img = cv2.imread("yaoi.jfif", cv2.IMREAD_COLOR)

    cv2.imshow("Lenna", BoxFilter(img/255))
    cv2.imshow("Leanna", img)

    k = cv2.waitKey(0)

    if(k == 27):
        cv2.destroyAllWindows()


main()