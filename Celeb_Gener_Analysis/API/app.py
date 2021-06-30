import os
import albumentations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torchvision import models
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = '/media/naruto/Study/codes/Projects/Deep_Learning/Celeb-Gender-Analysis/API/static'
DEVICE = 'cpu'
MODEL = None

class GenderModel(nn.Module):

  def __init__(self):
    
    super(GenderModel, self).__init__()

    self.model = models.vgg16(pretrained=False)

    for param in self.model.parameters():
      param.requires_grad = False

    self.model.avgpool = nn.Flatten()

    self.model.classifier = nn.Sequential(
                        nn.Linear(512*4*4, 256),
                        nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(256, 2)
    )

    self.loss_fn = nn.CrossEntropyLoss()

  def forward(self, x):
    return self.model(x)

def predict(image_path, model):

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose([
                albumentations.Resize(128, 128, always_apply=True),
                albumentations.Normalize(mean, std, always_apply=True)
            ])

    img = Image.open(image_path)
    img = np.array(img)
    img = aug(image=img)['image']
    img = np.transpose(img, (2,0,1)).astype(np.float32)

    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    output = model(img).max(-1)[-1]

    return output.item()
    

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)
            pred = predict(image_path, MODEL)
            return render_template('index.html', prediction=pred, image_loc=image_file.filename)
    return render_template('index.html', prediction=-1, image_loc=None)

if __name__ == '__main__':
    MODEL = GenderModel()
    MODEL.load_state_dict(torch.load('/media/naruto/Study/codes/Projects/Deep_Learning/Celeb-Gender-Analysis/API/trained_model.bin'))
    MODEL.to(DEVICE)
    app.run(port=12000, debug=True)