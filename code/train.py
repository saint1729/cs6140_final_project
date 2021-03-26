import time

import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import seaborn as sn  # for heatmaps
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import lab_distribution
from plot import imshow_torch, plot_image_channels
from network import Net
from custom_transforms import RGB2LAB, ToTensor

# ab_bins = np.load("./resources/pts_in_hull.npy")
# a_bins = ab_bins[:, 0]
# b_bins = ab_bins[:, 1]
# ab_bins_dict = {'ab_bins': ab_bins, 'a_bins': a_bins, 'b_bins': b_bins}

# data_dir = '../data/stl10_binary/train_X.bin'
# ab_bins_dict = lab_distribution.get_ab_bins_from_data(data_dir)
ab_bins_dict = np.load("./resources/prior_lab_distribution_train.npz", allow_pickle=True)
ab_bins, a_bins, b_bins = ab_bins_dict['ab_bins'], ab_bins_dict['a_bins'], ab_bins_dict['b_bins']

transform = transforms.Compose([transforms.Resize((96, 96)), RGB2LAB(ab_bins), ToTensor()])
trainset = torchvision.datasets.ImageFolder(root='../data/custom/train/', transform=transform)
testset = torchvision.datasets.ImageFolder('../data/custom/test/', transform=transform)

trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)

cnn = Net(ab_bins_dict)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnn = cnn.to(device)

cnn.set_rarity_weights(ab_bins_dict['w_bins'])
criterion = cnn.loss
optimizer = optim.Adam(cnn.parameters(), weight_decay=.001)

index = 0
start_time = time.time()
for epoch in range(5):
    for i, data in enumerate(trainloader):
        inputs, labels = data
        lightness, z_truth, original = inputs['lightness'], inputs[
            'z_truth'], inputs['original_lab_image']

        optimizer.zero_grad()
        outputs = cnn(lightness)
        ab_outputs = cnn.decode_ab_values()

        colorized_im = torch.cat((lightness, ab_outputs), 1)
        #    plot_image_channels(colorized_im.detach()[0, :, :, :], figure=20)
        loss = criterion(z_truth)
        loss.backward()
        optimizer.step()
        # if (i + 1) % 100 == 0:
        print(f"Epoch = {epoch + 1}, i = {i + 1}, loss = {loss.item()}, time taken = {time.time() - start_time}")

    # info = {'loss': loss}
    # for tag, value in info.items():
    #     print(value.detach())
    print(f"Epoch = {epoch + 1}, loss = {loss.item()}, time taken = {time.time() - start_time}")

    images = imshow_torch(colorized_im.detach()[0, :, :, :], figure=0)

print('parameter exploration')
for p in cnn.parameters():
    print(p.numel())

colorized_im = torch.cat((lightness, ab_outputs), 1)

imshow_torch(colorized_im.detach()[0, :, :, :], figure=1)

# plot_image_channels(colorized_im.detach()[0, :, :, :], figure=2)

imshow_torch(original[0, :, :, :], figure=3)
# plot_image_channels(original[0, :, :, :], figure=4)



with torch.no_grad():
    for b, (X_test, y_test) in enumerate(testloader):
        lightness, z_truth, original = X_test['lightness'], X_test['z_truth'], X_test['original_lab_image']
        outputs = cnn(lightness)
        ab_outputs = cnn.decode_ab_values()
        color_img = torch.cat((lightness, ab_outputs), 1)
        imshow_torch(color_img.detach()[0, :, :, :], figure=1, plot=True, color_space='lab')
        if b % 5 == 0:
            break

plt.show()
