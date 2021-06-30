import models

MODEL_DISPATCHER = {
    'resnet34'  : models.Resnet34,
    'resnet50'  : models.Resnet50,
    'resnet101' : models.Resnet101
}