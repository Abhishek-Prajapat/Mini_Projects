import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import albumentations
import torch



class CelebTrain:

    def __init__(self, folds, img_height, img_width, mean, std):

        df = pd.read_csv("../input/train_folds.csv")
        df = df[df.kfold.isin(folds)].reset_index(drop=True)

        self.image_ids = df.image_id.values
        self.gender    = df.Male.values

        if len(folds) == 1:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True)
            ])

        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1, rotate_limit=5, 
                                                p=0.6),
                albumentations.Normalize(mean, std, always_apply=True)
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image = plt.imread(f'../input/haar_cascades/haar_cascades/{self.image_ids[item]}')
        image = self.aug(image = np.array(image))['image']
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        gender = self.gender[item]

        return {
            'image': torch.tensor(image, dtype=torch.float), 
            'gender' : torch.tensor(gender, dtype=torch.long),
        }




