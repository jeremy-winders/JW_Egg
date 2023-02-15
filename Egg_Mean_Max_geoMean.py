import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


data = pd.read_csv("Segmented_10cycles_Eggs_Day7_Batch_8_Calls_H3O_EN113_set1.csv")
data

df_mean = data.groupby(['group_id']).mean()
print(df_mean.head())


df_max = data.groupby(['group_id']).max()
print(df_max)

#Hey this works -Denn


def geometric_mean(x):
    value = np.exp(np.mean(np.log(x)))
    return value

test_array = np.array([1,2,3,4])

print(geometric_mean(test_array))

agg_funcs = ['mean', geometric_mean]
data_agg = data.groupby(['group_id']).agg(agg_funcs).reset_index()

data_filtered = data[data.columns.difference(['Unnamed: 1', 'AbsTime', 'RelTime', 'Cycle'])]
data_filtered_agg = data_filtered.groupby(['group_id']).agg(agg_funcs)
print(data_agg.head())

data_agg.columns


data_agg[data_agg[('group_id', '')] == '1']

data_agg[data_agg[('group_id', '')] == '1'][('E/N_Act [Td]', 'mean')]




data_agg[('E/N_Act [Td]', 'mean')]

# group the dataframe by "group_id" and calculate the geometric mean
# df_geomean = data.groupby('group_id').agg(geometric_mean(x))


#print(geometric_mean(np.array([0,2,3])))#