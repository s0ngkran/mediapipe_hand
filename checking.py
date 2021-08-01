import json
print('start reading')
with open('hand_landmark.json', 'r') as f:
	dat = json.load(f)
print(dat[0].keys())
# print(len(dat[0]['hand_landmarks'][0]))
# print('xxx')
