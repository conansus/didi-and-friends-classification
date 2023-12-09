import os
import cv2

video = cv2.VideoCapture(r"C:\Users\USER\Videos\try_new_jojo.mp4")
currentframe=0

if not os.path.exists("try_new_jojo_images"):
    os.makedirs("try_new_jojo_images")
    

while(True):
    success, frame = video.read()

    if success :
        cv2.imshow("The Video", frame)
        cv2.imwrite("./try_new_jojo_images/try_new_jojo"+str(currentframe)+".jpg", frame)
        currentframe+=1
    else :
        break

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break

    
video.release()
cv2.destroyAllWindows()
