import cv2
import mediapipe as mp
import os
import json
from utils import get_list_folder, get_list_img, get_thai
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

######### config
PHOTO_FOLDER ='./TFS_training/' 
JSON_OUTPUT_PATH = './hand_mp_training_set.json'

# For static images:
IMAGE_FILES = []

# read imgs
i_img = 0
photo_folder =PHOTO_FOLDER 
for folder_name_ in get_list_folder(photo_folder):
    folder_name = os.path.join(photo_folder, folder_name_)
    for img_path in get_list_img(folder_name):
        img_path = os.path.join(folder_name, img_path)
        ground_truth = folder_name_
        IMAGE_FILES.append((img_path, ground_truth))
        i_img += 1
        print(i_img)

ans = []
count_empty = 0
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
  for idx, (file, ground_truth) in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    if image is None:
        output = {
            'img_path':file,
            'ground_truth': ground_truth,
            'handedness':'',
            'hand_landmarks': '',
        }
        ans.append(output)
        count_empty += 1
        continue
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    # print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
    #   print('hand_landmarks:', hand_landmarks)
     
      mp_drawing.draw_landmarks(
          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imwrite('./tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    print(idx)
    # # manage multi handedness
    result_multi_handedness = []
    for hand in results.multi_handedness:
        result_multi_handedness.append(str(hand))
    
    # manage multi hand landmarks
    result_multi_landmarks = []
    for hand_landmarks in results.multi_hand_landmarks:
        keypoints = []
        for data_point in hand_landmarks.landmark:
            keypoints.append({
                                'X': data_point.x,
                                'Y': data_point.y,
                                'Z': data_point.z,
                                })
        result_multi_landmarks.append(keypoints)

    output = {
        'img_path':file,
        'ground_truth': ground_truth,
        'handedness':result_multi_handedness,
        'hand_landmarks': result_multi_landmarks,
    }
    ans.append(output)
with open(JSON_OUTPUT_PATH, 'w') as f:
    json.dump(ans, f)
print('count empty', count_empty)
print('end')