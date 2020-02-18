import cv2
import numpy

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        out.write(frame)

        cv2.imshow('Sex Tape', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()