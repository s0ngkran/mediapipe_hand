# import the opencv library
import cv2

from datetime import datetime
import time
import copy
import os
def draw_text(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (100, 100)
    fontScale = 1
    color = (0, 255, 255)
    thickness = 2
    image = cv2.putText(image, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    return image

class MyText:
    def __init__(self):
        self.text = 'hello'
        self.t0 = time.time()
    def set_text(self, text):
        self.t0 = time.time()
        self.text = text
    def check_timeout(self):
        now = time.time()
        if now - self.t0 > 1:
            self.text = 'hello'

if not os.path.exists('./my_photo'):
    os.mkdir('./my_photo')
vid = cv2.VideoCapture(0)
my_text = MyText()
count = 0
while True:
    
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    ori_frame = copy.deepcopy(frame)

    # Display the resulting frame
    my_text.check_timeout()
    frame = draw_text(frame, my_text.text)
    cv2.imshow('frame', frame)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    key = cv2.waitKey(1)
    # print('key pressed', key)
    if key == ord('a'):
        break
    if key == 32: # spacebar
        # take photo
        now = datetime.now()
        name = now.strftime("%m%d%YT%H%M%S")
        count += 1
        cv2.imwrite('./my_photo/%s%d.jpg'%(name, count), ori_frame)
        my_text.set_text('%d ./my_photo/%s%d.jpg'%(count, name, count))
    

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

