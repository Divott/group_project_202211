import pandas as pd
from PIL import Image
import cv2


def get_variables(num_class=8, train_set_path='/home/tangb_lab/cse30013027/Data/AffectNet/training.csv',
                  validation_set_path='/home/tangb_lab/cse30013027/Data/AffectNet/validation.csv',
                  image_path_prefix='/home/tangb_lab/cse30013027/Data/AffectNet/Manually_Annotated_Images_AffectNet',
                  train_number=287651, validation_number=4000, resize=256, crop=224, dataset_mean=(0.5863, 0.4595, 0.4030),
                  dataset_std=(0.2715, 0.2424, 0.2366), seed=123, threshold_ini=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                  batch_size=32, num_workers=0, lr=0.002, num_epochs=5,
                  model_ini_path='/home/tangb_lab/cse30013027/zmj/checkpoint/model_initial.pth'):
    data = {}
    data['num_class'] = num_class
    data['train_set_path'] = train_set_path
    data['validation_set_path'] = validation_set_path
    data['image_path_prefix'] = image_path_prefix
    data['train_number'] = train_number
    data['validation_number'] = validation_number
    data['resize'] = resize
    data['crop'] = crop
    data['dataset_mean'] = dataset_mean
    data['dataset_std'] = dataset_std
    data['seed'] = seed
    data['threshold'] = threshold_ini
    data['batch_size'] = batch_size
    data['num_workers'] = num_workers
    data['lr'] = lr
    data['num_epochs'] = num_epochs
    data['model_ini_path'] = model_ini_path
    return data


def load_df_mini(path, num_class, num_load):
    df = pd.read_csv(path)
    df_out = df.loc[df['expression'] < num_class][:num_load]
    return df_out


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)
