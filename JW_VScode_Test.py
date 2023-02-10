import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


data = pd.read_csv("results/Batch_8/output/Segmented_10cycles_Eggs_Day7_Batch_8_Calls_H3O_EN113_set1.csv")
data

df_mean = data.groupby(['group_id']).mean()
df_max = data.groupby(['group_id']).max()


def geometric_mean(x):return np.exp(np.mean(np.log(x)))
# group the dataframe by "group_id" and calculate the geometric mean
df_geomean = data.groupby('group_id').agg(geometric_mean(x))



print (df_mean)
print (df_max)
print (df_geomean)