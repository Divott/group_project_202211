from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch


from config import get_variables, load_df_mini
num_class, train_set_path, validation_set_path, image_path_prefix, train_number, validation_number, dataset_mean, dataset_std = get_variables(
    train_number=10000)


class clothing_dataset(Dataset):
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=num_class):
        df_train = load_df_mini(train_set_path, num_class, train_number)
        df_test = load_df_mini(validation_set_path,
                               num_class, validation_number)

        self.root = root
        self.transform = transform
        self.mode = mode

        self.train_labels = df_train['expression'].values.flatten()
        self.test_labels = df_test['expression'].values.flatten()

        self.x_train = df_train['face_x'].values.flatten()
        self.y_train = df_train['face_y'].values.flatten()
        self.w_train = df_train['face_width'].values.flatten()
        self.h_train = df_train['face_height'].values.flatten()

        self.x_test = df_test['face_x'].values.flatten()
        self.y_test = df_test['face_y'].values.flatten()
        self.w_test = df_test['face_width'].values.flatten()
        self.h_test = df_test['face_height'].values.flatten()

        if mode == 'all':
            self.train_imgs = df_train['subDirectory_filePath'].values.flatten(
            )
        elif self.mode == "labeled":
            train_imgs = df_train['subDirectory_filePath'].values.flatten()
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" %
                  (self.mode, len(self.train_imgs)))
        elif self.mode == "unlabeled":
            train_imgs = df_train['subDirectory_filePath'].values.flatten()
            pred_idx = (1 - pred).nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" %
                  (self.mode, len(self.train_imgs)))

        elif mode == 'test':
            self.test_imgs = df_test['subDirectory_filePath'].values.flatten()

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = image_path_prefix + '/' + self.train_imgs[index]
            target = self.train_labels[index]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            x = self.x_train[index]
            y = self.y_train[index]
            w = self.w_train[index]
            h = self.h_train[index]
            image = image.crop((x, y, x + w, y + h))
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = image_path_prefix + '/' + self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            x = self.x_train[index]
            y = self.y_train[index]
            w = self.w_train[index]
            h = self.h_train[index]
            image = image.crop((x, y, x + w, y + h))
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = image_path_prefix + '/' + self.train_imgs[index]
            target = self.train_labels[index]
            image = Image.open(img_path).convert('RGB')
            x = self.x_train[index]
            y = self.y_train[index]
            w = self.w_train[index]
            h = self.h_train[index]
            image = image.crop((x, y, x + w, y + h))
            img = self.transform(image)
            return img, target, img_path
        elif self.mode == 'test':
            img_path = image_path_prefix + '/' + self.test_imgs[index]
            target = self.test_labels[index]
            image = Image.open(img_path).convert('RGB')
            x = self.x_test[index]
            y = self.y_test[index]
            w = self.w_test[index]
            h = self.h_test[index]
            image = image.crop((x, y, x + w, y + h))
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        else:
            return len(self.train_imgs)


class clothing_dataloader():
    def __init__(self, root, batch_size, num_batches, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean,
                                 dataset_std),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean,
                                 dataset_std),
        ])

    def run(self, mode, pred=[], prob=[], paths=[]):
        if mode == 'warmup':
            warmup_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='all',
                                              num_samples=self.num_batches * self.batch_size * 2)
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return warmup_loader
        elif mode == 'train':
            labeled_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='labeled', pred=pred,
                                               probability=prob, paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            unlabeled_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='unlabeled', pred=pred,
                                                 probability=prob, paths=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader
        elif mode == 'eval_train':
            eval_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='all',
                                            num_samples=self.num_batches * self.batch_size)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
        elif mode == 'test':
            test_dataset = clothing_dataset(
                self.root, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1000,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
