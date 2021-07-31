import json
print('start reading')
with open('hand_landmark.json', 'r') as f:
	dat = json.load(f)
	
for k, v in dat[0].items():
	print(k, v)
	print()
print('xxx')
