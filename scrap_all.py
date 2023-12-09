import os
import cv2

video = cv2.VideoCapture(r'C:\Users\USER\Videos\try_all.mp4')
currentframe = 0

if not os.path.exists("try_all_images"):
    os.makedirs("try_all_images")

while(True):
    success, frame = video.read()

    if success :
        cv2.imshow("The Video", frame)
        cv2.imwrite("./try_all_images/try_all"+str(currentframe)+".jpg", frame)
        currentframe+=1
    else :
        break

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break


video.release()
cv2.destroyAllWindows()