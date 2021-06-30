export IMG_HEIGHT=128
export IMG_WIDTH=128

export EPOCHS=50
export TRAIN_BATCH_SIZE=64
export VAL_BATCH_SIZE=32

export BASE_MODEL='resnet34'

export MODEL_MEAN='(0.485, 0.456, 0.406)'
export MODEL_STD='(0.229, 0.224, 0.225)'

export TRAINING_FOLDS='(0, 1, 2, 3)'
export VALIDATION_FOLDS='(4,)'
python /content/drive/MyDrive/<PROJECT NAME>/src/train_Colab.py

export TRAINING_FOLDS='(0, 1, 2, 4)'
export VALIDATION_FOLDS='(3,)'
python /content/drive/MyDrive/<PROJECT NAME>/src/train_Colab.py

export TRAINING_FOLDS='(0, 1, 4, 3)'
export VALIDATION_FOLDS='(2,)'
python /content/drive/MyDrive/<PROJECT NAME>/src/train_Colab.py

export TRAINING_FOLDS='(0, 4, 2, 3)'
export VALIDATION_FOLDS='(1,)'
python /content/drive/MyDrive/<PROJECT NAME>/src/train_Colab.py

export TRAINING_FOLDS='(4, 1, 2, 3)'
export VALIDATION_FOLDS='(0,)'
python /content/drive/MyDrive/<PROJECT NAME>/src/train_Colab.py