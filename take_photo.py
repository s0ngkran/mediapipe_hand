# import the opencv library
import cv2

from datetime import datetime
# define a video capture object
vid = cv2.VideoCapture(0)

while True:
    
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
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
        cv2.imwrite('./my_photo/%s.jpg'%name, frame)
        print('saved','./my_photo/%s.jpg'%name)
    

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

