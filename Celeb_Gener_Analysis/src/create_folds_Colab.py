import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
import os

# Give the project folder name as saved in google drive
PROJECT_NAME = ''

if __name__ == '__main__':

    df = pd.read_csv(f'/content/drive/MyDrive/{PROJECT_NAME}/input/list_attr_celeba.csv')
    selected_images = []
    for i in os.listdir(f'/content/drive/MyDrive/{PROJECT_NAME}/input/haar_cascades/haar_cascades/'):
        image = i.split('.')[0] + '.jpg'
        selected_images.append(image)

    df = df[df.image_id.isin(selected_images)].reset_index(drop = True)
    df = df[['image_id', 'Male']]
    
    df['Male'] = df['Male'].map({-1:0,1:1})
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=5)
    X = df.image_id.values
    y = df['Male']

    for fold, (trn_, val_) in enumerate(splitter.split(X, y)):

        df.loc[val_, 'kfold'] = fold

    df.to_csv(f'/content/drive/MyDrive/{PROJECT_NAME}/input/train_folds.csv', index=False)