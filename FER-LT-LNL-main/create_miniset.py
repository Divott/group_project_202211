import pandas as pd

# 参数
num_class = 8

# 函数


def load_df_mini(path, num_class, num_load):
    df = pd.read_csv(path)
    df_out = df.loc[df['expression'] < num_class][:num_load]
    return df_out


# main
df_train = load_df_mini(
    '/home/tangb_lab/cse30013027/Data/AffectNet/training.csv',
    num_class, 100)
print(len(df_train))
