import os
import cv2

video = cv2.VideoCapture(r'C:\Users\USER\Videos\try_jojo_nana.mp4')
currentframe = 0

if not os.path.exists("try_jojo_nana_images"):
    os.makedirs("try_jojo_nana_images")

while(True):
    success, frame = video.read()

    if success :
        cv2.imshow("The Video", frame)
        cv2.imwrite("./try_jojo_nana_images/try_jojo_nana"+str(currentframe)+".jpg", frame)
        currentframe+=1
    else :
        break

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break


video.release()
cv2.destroyAllWindows()