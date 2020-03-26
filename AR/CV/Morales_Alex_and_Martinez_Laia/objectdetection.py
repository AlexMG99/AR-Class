import cv2
import numpy as np

def ObjectDetection(inp, tgt):
    width = (inp.shape[0] - tgt.shape[0] + 1)
    height = (inp.shape[1] - tgt.shape[1] + 1)
    result_img = np.ones((width, height))

    for x in range(0, width):
        for y in range(0, height):
            sq_sum = 0
            for x_t in range(0, tgt.shape[0]):
                for y_t in range(0, tgt.shape[1]):
                    sq_sum += (tgt[x_t, y_t] - inp[x + x_t, y + y_t])**2
            result_img[x, y] = sq_sum

    result_img = result_img / result_img.max()
    return result_img

def IsFound(res, threshold):
    if((res.min() / res.max()) < 0.1):
        return 'FOUND'
    else:
        return 'NOT FOUND'

def main():
    # Image names
    tgt_name = 't1-img2.png'
    inp_name = 'img2.png'

    # Load image in grayscale
    target_img = cv2.imread(tgt_name, cv2.IMREAD_GRAYSCALE)
    input_img = cv2.imread(inp_name, cv2.IMREAD_GRAYSCALE)

    # Detects object
    result_img = ObjectDetection(input_img, target_img)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Load image in color
    tgt_img_col = cv2.imread(tgt_name)
    inp_img_col = cv2.imread(inp_name)
    cv2.imshow("Target", tgt_img_col)
    cv2.imshow("Matching Map", inp_img_col)

    # Checks if Target Found
    if(IsFound(result_img, 0.1) == 'FOUND'):
        imgFound = np.zeros((40,245,3), np.uint8)
        cv2.putText(imgFound, 'TARGET FOUND', (5,30), font, 1, (0,255,0),2)
    else:
        imgFound = np.zeros((40,315,3), np.uint8)
        cv2.putText(imgFound, 'TARGET NOT FOUND', (5,30), font, 1, (0,122,255),2)

    # Creates rectangle of matching icon
    rec_pos = np.where( result_img < 0.1)
    for pixel in zip(*rec_pos[::-1]):
        cv2.rectangle(inp_img_col, pixel, (pixel[0] + target_img.shape[0], pixel[1] + target_img.shape[1]), (0,255,0), 1)

    # Shows input result image with rectangle
    cv2.imshow("Input Img", inp_img_col)
    
    # Target image found
    cv2.imshow("Result", imgFound)

    # Closes application
    k = cv2.waitKey(0)

    if(k == 27):
        cv2.destroyAllWindows()

main()