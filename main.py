import numpy as np
import argparse
import timeit
from sklearn import metrics
import cv2
import os

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

import datasets

class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device.'.format(device))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
    # Build the augmentations.
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Compose([
            get_color_distortion(),
            RandomGaussianBlur(),
            ]),
        transforms.ToTensor(),
        normalize,
        ])

    # Init the dataset and augmentations.
    train_dataset = datasets.OcularDiseaseDataset('train', transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = datasets.OcularDiseaseDataset('test', transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    net = torch.hub.load('facebookresearch/swav', 'resnet50')

    num_features = net.fc.in_features
    net.fc = nn.Sequential(
            nn.Linear(num_features, args.classes),
            nn.Sigmoid())

    if args.model_path and os.path.exists(args.model_path):
        print('Load the model state.')
        net.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))

    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    net.train()
    for epoch in range(1, args.epochs + 1):
        start_time = timeit.default_timer()
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        train_loss = 0
        for index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)

            loss = criterion(outputs, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, outputs.detach().cpu()))

            print(f'\repoch {epoch:3d}/{args.epochs:3d} batch {index:3d}/{len(train_loader):3d}', end='')
            print(f' loss {train_loss / index:6.4f}', end='')
            print(' {:5.1f}sec'.format(timeit.default_timer() - start_time), end='')

        aucs = []
        for i, name in enumerate(datasets.CLASSES):
            aucs.append(metrics.roc_auc_score(y_true[:, i], y_pred[:, i]))
            print('{:10s} {:5.3f}'.format(name, aucs[-1]))
        print('  mean AUC {:5.3f}'.format(np.mean(aucs)))

    torch.save(net.state_dict(), f'model/checkpoint.pth')

    net.eval()
    start_time = timeit.default_timer()
    y_true = torch.FloatTensor()
    y_pred = torch.FloatTensor()
    test_loss = 0
    for index, (images, labels) in enumerate(test_loader, 1):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = net(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        y_true = torch.cat((y_true, labels.cpu()))
        y_pred = torch.cat((y_pred, outputs.detach().cpu()))

        print(f'\rtest batch {index:3d}/{len(test_loader):3d}', end='')
        print(f' loss {test_loss / index:6.4f}', end='')
        print(' {:5.1f}sec'.format(timeit.default_timer() - start_time), end='')

    print('')

    aucs = []
    for i, name in enumerate(datasets.CLASSES):
        aucs.append(metrics.roc_auc_score(y_true[:, i], y_pred[:, i]))
        print('{:12s} {:5.3f}'.format(name, aucs[-1]))
    print('  mean AUC {:5.3f}'.format(np.mean(aucs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--classes', default=8, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float) # 5e-5
    parser.add_argument('--momentum', default=0.9, type=float)
    args = parser.parse_args()
    print(vars(args))

    main(args)
