from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from config import load_df_mini, img_loader


class dataset(Dataset):
    def __init__(self, mode, transform, num_class, train_set_path,
                 validation_set_path, image_path_prefix, train_number,
                 validation_number, pred=[]):
        self.mode = mode
        self.transform = transform
        self.image_path_prefix = image_path_prefix

        df_train = load_df_mini(train_set_path, num_class, train_number)
        df_test = load_df_mini(validation_set_path,
                               num_class, validation_number)

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
            print("%s data has a size of %d" %
                  (self.mode, len(self.train_imgs)))
        elif self.mode == "unlabeled":
            train_imgs = df_train['subDirectory_filePath'].values.flatten()
            pred_idx = (1 - pred).nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            print("%s data has a size of %d" %
                  (self.mode, len(self.train_imgs)))
        elif mode == 'test':
            self.test_imgs = df_test['subDirectory_filePath'].values.flatten()

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.image_path_prefix + '/' + self.train_imgs[index]
            target = self.train_labels[index]
            image = img_loader(img_path)
            x = self.x_train[index]
            y = self.y_train[index]
            w = self.w_train[index]
            h = self.h_train[index]
            image = image.crop((x, y, x + w, y + h))
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target
        elif self.mode == 'unlabeled':
            img_path = self.image_path_prefix + '/' + self.train_imgs[index]
            image = img_loader(img_path)
            x = self.x_train[index]
            y = self.y_train[index]
            w = self.w_train[index]
            h = self.h_train[index]
            image = image.crop((x, y, x + w, y + h))
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = self.image_path_prefix + '/' + self.train_imgs[index]
            target = self.train_labels[index]
            image = img_loader(img_path)
            x = self.x_train[index]
            y = self.y_train[index]
            w = self.w_train[index]
            h = self.h_train[index]
            image = image.crop((x, y, x + w, y + h))
            img = self.transform(image)
            return img, target
        elif self.mode == 'test':
            img_path = self.image_path_prefix + '/' + self.test_imgs[index]
            target = self.test_labels[index]
            image = img_loader(img_path)
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


class dataloader():
    def __init__(self, num_class, train_set_path, validation_set_path, image_path_prefix,
                 train_number, validation_number, resize, crop,
                 dataset_mean, dataset_std, batch_size, num_workers):
        # define attributes
        self.num_class = num_class
        self.train_set_path = train_set_path
        self.validation_set_path = validation_set_path
        self.image_path_prefix = image_path_prefix
        self.train_number = train_number
        self.validation_number = validation_number
        self.batch_size = batch_size
        self.num_workers = num_workers

        # define transforms
        self.transform_train = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
        # maybe there are other kinds of transforms waited to be added

    def run(self, mode, pred=[]):
        if mode == 'warm_up':
            warm_up_dataset = dataset(mode='all', transform=self.transform_train,
                                      num_class=self.num_class, train_set_path=self.train_set_path,
                                      validation_set_path=self.validation_set_path,
                                      image_path_prefix=self.image_path_prefix,
                                      train_number=self.train_number, validation_number=self.validation_number)
            warm_up_loader = DataLoader(
                dataset=warm_up_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return warm_up_loader
        elif mode == 'train':
            labeled_dataset = dataset(mode='labeled', transform=self.transform_train,
                                      num_class=self.num_class, train_set_path=self.train_set_path,
                                      validation_set_path=self.validation_set_path,
                                      image_path_prefix=self.image_path_prefix,
                                      train_number=self.train_number, validation_number=self.validation_number, pred=pred)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            unlabeled_dataset = dataset(mode='unlabeled', transform=self.transform_train,
                                        num_class=self.num_class, train_set_path=self.train_set_path,
                                        validation_set_path=self.validation_set_path,
                                        image_path_prefix=self.image_path_prefix,
                                        train_number=self.train_number, validation_number=self.validation_number, pred=pred)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader
        elif mode == 'eval':
            eval_dataset = dataset(mode='all', transform=self.transform_test,
                                   num_class=self.num_class, train_set_path=self.train_set_path,
                                   validation_set_path=self.validation_set_path,
                                   image_path_prefix=self.image_path_prefix,
                                   train_number=self.train_number, validation_number=self.validation_number)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
        elif mode == 'test':
            test_dataset = dataset(mode=mode, transform=self.transform_test,
                                   num_class=self.num_class, train_set_path=self.train_set_path,
                                   validation_set_path=self.validation_set_path,
                                   image_path_prefix=self.image_path_prefix,
                                   train_number=self.train_number, validation_number=self.validation_number)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*2,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
