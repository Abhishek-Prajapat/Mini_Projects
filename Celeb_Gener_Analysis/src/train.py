import os
import ast
from model_dispatcher import MODEL_DISPATCHER
from dataset import CelebTrain
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

DEVICE = 'cuda'
TRAINING_FOLDS_CSV = os.environ.get('TRAINING_FOLDS_CSV')
IMG_HEIGHT = int(os.environ.get('IMG_HEIGHT'))
IMG_WIDTH = int(os.environ.get('IMG_WIDTH'))
EPOCHS = int(os.environ.get('EPOCHS'))

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
VAL_BATCH_SIZE = int(os.environ.get('VAL_BATCH_SIZE'))

MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.environ.get('MODEL_STD'))

TRAINING_FOLDS = ast.literal_eval(os.environ.get('TRAINING_FOLDS'))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get('VALIDATION_FOLDS'))
BASE_MODEL = os.environ.get('BASE_MODEL')

writer = SummaryWriter(f'runs/on_fold_{VALIDATION_FOLDS[0]}')

def loss_fn(outputs, targets):
    loss = torch.nn.BCELoss()
    loss_value = loss(outputs, targets.reshape(-1,1))
    return loss_value



def train(dataloader, model, optimizer):

    model.train()
    final_loss = 0
    counter = 0

    for bi, d in enumerate(dataloader):
        image = d['image']
        gender = d['gender']

        image = image.to(DEVICE, dtype=torch.float)
        gender = gender.to(DEVICE, dtype=torch.float)

        optimizer.zero_grad()
        
        outputs = model(image)
        targets = gender
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()

        final_loss += loss
        counter += 1

    return final_loss/counter

def evaluate(dataloader, model):

    model.eval()
    final_loss = 0
    counter = 0

    for bi, d in enumerate(dataloader):
        image = d['image']
        gender = d['gender']

        image = image.to(DEVICE, dtype=torch.float)
        gender = gender.to(DEVICE, dtype=torch.float)
        
        outputs = model(image)
        targets = gender
        loss = loss_fn(outputs, targets)
        final_loss += loss
        counter += 1

    return final_loss/counter


def main():

    model = MODEL_DISPATCHER[BASE_MODEL](pretrained='imagenet')
    model.to(DEVICE)

    trainDataset = CelebTrain(
        folds = TRAINING_FOLDS,
        img_height = IMG_HEIGHT,
        img_width = IMG_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True,
        num_workers = 8
    )

    valDataset = CelebTrain(
        folds = VALIDATION_FOLDS,
        img_height = IMG_HEIGHT,
        img_width = IMG_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )

    val_loader = torch.utils.data.DataLoader(
        valDataset,
        batch_size = VAL_BATCH_SIZE,
        shuffle = True,
        num_workers = 8
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            patience=5, factor=0.3, verbose=True)

    for epoch in range(EPOCHS):

        train_loss = train(train_loader, model, optimizer)
        val_loss = evaluate(val_loader, model)
        scheduler.step(val_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        torch.save(model.state_dict(), f'{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin')
        print(f'Epoch: {epoch}, train_loss: {train_loss}, validation_loss: {val_loss}')


if __name__ == '__main__':
    main()
    writer.close()