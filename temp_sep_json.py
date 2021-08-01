import json

with open('hand_landmark.json', 'r') as f:
	data = json.load(f)

gt = 'first_time'
wait_list = []
for i, dat in enumerate(data):
	img_path = dat['img_path']
	ground_truth = dat['ground_truth']
	handedness = dat['handedness']
	hand_landmarks = dat['hand_landmarks']

	# find first index of that gt; i know my own data set
	if gt != ground_truth or gt == 'first_time':
		wait_list.append(i)
		gt = ground_truth

# fill dataset with wait list
training_set = []
validation_set = []
for i, dat in enumerate(data):
	if i in wait_list:
		validation_set.append(dat)
	else:
		training_set.append(dat)

assert len(data) == len(training_set)+len(validation_set)
# write json
with open('hand_landmark_mp_training_set.json', 'w') as f:
	json.dump(training_set, f)
print('write tr')
with open('hand_landmark_mp_validation_set.json', 'w') as f:
	json.dump(validation_set, f)

print('write va')
print('success')





