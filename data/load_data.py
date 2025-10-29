from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', 
         '挂', '学', '警', '港', '澳',
         '领', '使', '临',
         '-'
         ]

# '挂', '学', '警', '港', '澳',
#          '领', '使', '临',

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, txt_path, imgSize, lpr_max_len, PreprocFun=None):
        
        self.img_paths = []
        self.img_labels = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        img_path, label_str = parts
                        self.img_paths.append("/home/pernod/CBLPRD-330k_v1/" + img_path)
                        try:
                            label = [CHARS_DICT[c] for c in label_str]
                        except KeyError as e:
                            print(f"Error: Character '{e.args[0]}' not found in CHARS_DICT. Skipping line: {line}")
                            continue
                        self.img_labels.append(label)

        
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        if Image is None:
            raise FileNotFoundError(f"Image not found or cannot be opened: {filename}")
        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        label = self.img_labels[index]
        # if len(label) == 8:
        #     if self.check(label) == False:
        #         print(filename)
        #         assert 0, "Error label ^~^!!!"
        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if label[2] != CHARS_DICT['D'] and label[2] != CHARS_DICT['F'] \
                and label[-1] != CHARS_DICT['D'] and label[-1] != CHARS_DICT['F']:
            print("Error label, Please check!")
            return False
        else:
            return True
