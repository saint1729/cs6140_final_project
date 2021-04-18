import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
from torchvision import transforms, datasets
import numpy as np
from training_layers import PriorBoostLayer, NNEncLayer, NonGrayMaskLayer, decode
from data_loader import TrainImageFolder
from model import Color_model

original_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor()
])


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    train_set = TrainImageFolder(args.image_dir, original_transform)

    # Build data loader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers)

    test_data = TrainImageFolder('../../data/custom/test/test/', transform=transforms.Compose([]))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Build the models
    model = Color_model()
    # model.load_state_dict(torch.load('../model/models/model-171-216.ckpt'))
    encode_layer = NNEncLayer()
    boost_layer = PriorBoostLayer()
    nongray_mask = NonGrayMaskLayer()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduce=False)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    start_time = time.time()
    for epoch in range(args.num_epochs):
        for i, (images, img_ab) in enumerate(data_loader):
            try:
                # Set mini-batch dataset
                images = images.unsqueeze(1).float()
                img_ab = img_ab.float()
                encode, max_encode = encode_layer.forward(img_ab)
                targets = torch.Tensor(max_encode).long()
                boost = torch.Tensor(boost_layer.forward(encode)).float()
                mask = torch.Tensor(nongray_mask.forward(img_ab)).float()
                boost_nongray = boost * mask
                outputs = model(images)  # .log()
                output = outputs[0].cpu().data.numpy()
                out_max = np.argmax(output, axis=0)

                print('set', set(out_max.flatten()))
                loss = (criterion(outputs, targets) * (boost_nongray.squeeze(1))).mean()
                # loss=criterion(outputs,targets)
                # multi=loss*boost_nongray.squeeze(1)

                model.zero_grad()

                loss.backward()
                optimizer.step()

                # Print log info
                if i % args.log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, args.num_epochs, i + 1, total_step, loss.item()))

                # Save the model checkpoints
                if (i + 1) % args.save_step == 0:
                    torch.save(model.state_dict(), os.path.join(
                        args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            except:
                pass

    print(f"Total time: {time.time() - start_time}")

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.unsqueeze(1).float()
            y_val = model(X_test)
            color_img = decode(X_test, y_val)
            color_name = '../data/colorimg/' + str(b + 1) + '.jpeg'
            imageio.imsave(color_name, color_img * 255.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--image_dir', type=str, default='../data/CATS_DOGS/train/CAT', help='directory for resized images')
    parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=216, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    print(args)
    main(args)
