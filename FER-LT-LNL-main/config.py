import pandas as pd
from PIL import Image
import cv2


def load_df_mini(path, num_class, num_load):
    df = pd.read_csv(path)
    df_out = df.loc[df['expression'] < num_class][:num_load]
    return df_out
