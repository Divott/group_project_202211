def get_variables(num_class=8, train_set_path='/home/tangb_lab/cse30013027/Data/AffectNet/training.csv',
                  validation_set_path='/home/tangb_lab/cse30013027/Data/AffectNet/validation.csv',
                  image_path_prefix='/home/tangb_lab/cse30013027/Data/AffectNet/Manually_Annotated_Images_AffectNet',
                  train_number=100, validation_number=4000,
                  dataset_mean=(0.5863, 0.4595, 0.4030), dataset_std=(0.2715, 0.2424, 0.2366)):
    return num_class, train_set_path, validation_set_path, image_path_prefix, train_number, validation_number, dataset_mean, dataset_std
