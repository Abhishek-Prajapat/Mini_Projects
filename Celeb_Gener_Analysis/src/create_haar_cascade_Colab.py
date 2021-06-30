from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from joblib import Parallel, delayed
import joblib

# Give the project folder name as saved in google drive
PROJECT_NAME = ''


model_path = f'/content/drive/MyDrive/{PROJECT_NAME}/input/haarcascade_frontalface_default.xml'
image_dir = f'/content/drive/MyDrive/{PROJECT_NAME}/input/images/images/'
image_ids = os.listdir(image_dir)
faceDetector = cv2.CascadeClassifier(model_path)

def save_train_img(image_id):

    try:

        image_path = os.path.join(image_dir,image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faceRegions = faceDetector.detectMultiScale(image, scaleFactor=1.06, minNeighbors=6, minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faceRegions:
            image = image[y:y+h, x:x+w]

        image_id = image_id.split('.')[0]
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'/content/drive/MyDrive/{PROJECT_NAME}/input/haar_cascades/{image_id}.jpg', image)
        

    except Exception as e:
        pass



if __name__ == '__main__':

    Parallel(n_jobs=8, backend='multiprocessing')(
        delayed(save_train_img)(image_id) for image_id in tqdm(image_ids[:10000], total=10000))