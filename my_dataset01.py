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
                # number
                'Num01': 0,
                'Num02': 1,
                'Num03': 2,
                'Num04': 3,
                'Num05': 4,

                # addition
                'Add01': 5,
                'Add02': 6,
                'Add03': 7,
                'Add04': 8,
                'Add05': 9,

                # static pose
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

                # static pose for vowel
                'Vow01': 25,
                'Vow02': 26,
                'Vow03': 27,
                'Vow04': 28,
                'Vow05': 29,
            }
            thai_to_poses = {
                # static mode
                'ก':['K'],
                'ต':['T'],
                'ส':['P'],
                'ห':['H'],
                'บ':['B'],
                'ร':['R'],
                'ว':['W'],
                'ด':['D'],
                'ฟ':['F'],
                'ล':['L'],
                'ย':['Y'],
                'ม':['M'],
                'น':['N'],
                'อ':['A'],
                'อิ':['Vol01'],
                'โอ':['Vol03'],
                'ไอ':['Vol04'],
                'ใอ':['Vol05'],

                # dynamic mode
                'ข':['K','Num01'],
                'ค':['K','Num02'],
                'ฆ':['K','Num03'],

                'ถ':['T','Num01'],
                'ฐ':['T','Num02'],
                'ฒ':['T','Num03'],
                'ฑ':['T','Num04'],
                'ฏ':['T','Num05'],

                'ศ':['S','Num01'],
                'ษ':['S','Num02'],

                'ซ':['S','Add03'],

                'ป':['P','Num01'],
                'ผ':['P','Num02'],
                'ภ':['P','Num03'],

                'ฮ':['H','Num01'],
                'ฎ':['D','Num01'],
                'ฝ':['F','Num01'],
                'ฬ':['L','Num01'],
                'จ':['Add04','Add05'], # ผู้ฟังเห็น หน้ามือ แล้ว หลังมือ
                'ญ':['Y','Num01'],

                'ณ':['N','Num01'],
                'ง':['N','Add01'],

                'ท':['T','H'],
                'ธ':['T','H','Num01'],

                'ฉ':['Add02','H'],
                'ช':['Add02','H', 'Num01'],
                'ฌ':['Add02','H', 'Num02'],

                'อี':['Vow01','Num01'],
                'อึ':['Vow01','Vow02'],
                'อื':['Vow01','Num02'],
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
