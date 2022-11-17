import pandas as pd


def get_variables(num_class=8, train_set_path='/home/tangb_lab/cse30013027/Data/AffectNet/training.csv',
                  validation_set_path='/home/tangb_lab/cse30013027/Data/AffectNet/validation.csv',
                  image_path_prefix='/home/tangb_lab/cse30013027/Data/AffectNet/Manually_Annotated_Images_AffectNet',
                  train_number=287651, validation_number=4000, resize=256, crop=224, dataset_mean=(0.5863, 0.4595, 0.4030), dataset_std=(0.2715, 0.2424, 0.2366),
                  seed=123, threshold_ini=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], batch_size=32, num_workers=0):
    return num_class, train_set_path, validation_set_path, image_path_prefix, train_number, validation_number, resize, crop, dataset_mean, dataset_std, seed, threshold_ini, batch_size, num_workers


def load_df_mini(path, num_class, num_load):
    df = pd.read_csv(path)
    df_out = df.loc[df['expression'] < num_class][:num_load]
    return df_out
