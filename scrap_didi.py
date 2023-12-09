import os
import cv2

video = cv2.VideoCapture(r"C:\Users\USER\Videos\try_didi.mp4")
currentframe=0

if not os.path.exists("try_didi_images"):
    os.makedirs("try_didi_images")
    

while(True):
    success, frame = video.read()

    if success :
        cv2.imshow("The Video", frame)
        cv2.imwrite("./try_didi_images/try_didi"+str(currentframe)+".jpg", frame)
        currentframe+=1
    else :
        break

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break

    
r"""
video.release()


vid = cv2.VideoCapture(r"C:\Users\USER\Videos\didi_2nd_video.mp4")
cf = 200
while(True):
    ret, frm = vid.read()

    if ret :
        cv2.imshow("The Video", frm)
        cv2.imwrite("./new_didi_images/didi"+str(currentframe)+".jpg", frm)
        currentframe+=1
    else :
        break

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break
"""


video.release()
cv2.destroyAllWindows()
