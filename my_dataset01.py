from torch.utils.data import Dataset
import json
import torch

class MyDataset(Dataset):
    def __init__(self, json_path, test_mode=False):
        with open(json_path, 'r') as f:
            dat = json.load(f)
        print('-------------n',len(dat))
        if test_mode:
            dat = dat[:50]
        
        self.img_path = []
        self.ground_truth = []
        self.handedness = []
        self.hand_landmarks = []

        for _dat in dat:
            _img_path = _dat['img_path']
            _ground_truth = _dat['ground_truth']
            _handedness = _dat['handedness']
            _hand_landmark = _dat['hand_landmarks']

            my_landmark = []
            if _hand_landmark == '':
                continue
            assert len(_hand_landmark[0]) == 21

            for point in _hand_landmark[0]:
                for k, v in point.items():
                    my_landmark.append(v)

            gt_dict = {
                '16': 0,
                '17': 1,
                '18': 2,
                '19': 3,
                '20': 4,
                '21': 5,
                '22': 6,
                '23': 7,
                '24': 8,
                '25': 9,
                'A': 10,
                'B': 11,
                'D': 12,
                'F': 13,
                'H': 14,
                'K': 15,
                'L': 16,
                'M': 17,
                'N': 18,
                'P': 19,
                'R': 20,
                'S': 21,
                'T': 22,
                'W': 23,
                'Y': 24,
            }
            self.img_path.append(_img_path)
            self.ground_truth.append(gt_dict[_ground_truth])
            self.handedness.append(_handedness)
            self.hand_landmarks.append(torch.FloatTensor(my_landmark))

        
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        ans = {
            'img_path': self.img_path[idx],
            'ground_truth': self.ground_truth[idx],
            'handedness': self.handedness[idx],
            'hand_landmarks': self.hand_landmarks[idx],
        }
        return ans
