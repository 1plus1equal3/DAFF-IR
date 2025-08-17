import torch
import torch.nn as nn

CCE = nn.CrossEntropyLoss()
BCE = nn.BCELoss()
MAE = nn.L1Loss()
MSE = nn.MSELoss()

# Reconstruction loss
def mse_dist(y_true, y_pred):
    return MSE(y_true, y_pred)

def l1_loss(y_true, y_pred):
    return MAE(y_true, y_pred)

def charbonnier_loss(y_true, y_pred, epsilon=1e-3):
    loss = torch.sqrt(torch.square(y_true - y_pred) + epsilon**2)
    loss = torch.mean(loss)
    return loss
    
# Classification loss
def classify_loss_fn(label, prediction):
    label = torch.argmax(label, dim=-1)
    # print(label.dtype, prediction.dtype)
    return CCE(prediction, label)

def level_loss_fn(label, prediction):
    return MSE(label, prediction)

# GAN loss
def discriminator_loss(real_img, fake_img):
    real_loss = torch.mean(real_img)
    fake_loss = torch.mean(fake_img)
    return fake_loss - real_loss

def generator_loss(fake_img):
    return -torch.mean(fake_img)