import pandas as pd
import numpy as np
import os
from os import listdir


#for file in os.dirlist('Segmented_data/O2/113'):pass

for file in os.listdir('Segmented_data/O2/113'):pass
#errors above move on



# Define the directory containing the files
directory = "Segmented_data/O2/113"

# List the files in the directory
files = os.listdir(directory)

# Process each file
for file in files:
    # Construct the full path to the file
    path = os.path.join(directory, file)

    # Process the file
    with open(path, "r") as f:
        data = f.read()
        # Do something with the file data













data1 = pd.read_csv("Segmented_data/H2O/113/Segmented_10cycles_Eggs_Day7_Batch_8_Calls_H3O_EN113_set1.csv")
data2 = pd.read_csv("Segmented_data/H2O/113/Segmented_10cycles_Eggs_Day7_Batch_8_Calls_H3O_EN113_set2.csv")
data4 = pd.read_csv("Segmented_data/H2O/113/Segmented_10cycles_Eggs_Day7_Batch_8_Calls_H3O_EN113_set4.csv")
data5 = pd.read_csv("Segmented_data/H2O/113/Segmented_10cycles_Eggs_Day7_Batch_8_Calls_H3O_EN113_set5.csv")
data6 = pd.read_csv("Segmented_data/H2O/113/Segmented_10cycles_Eggs_Day7_Batch_8_Calls_H3O_EN113_set6.csv")

data1
data2
data4
data5
data6

#Drop unneeded columns
data1_Drop = data1.drop(columns=['p-Drift_Act [mbar]','Unnamed: 1', 'AbsTime', 'RelTime', 'E/N_Act [Td]', 'Cycle'])
data1_Drop.head()



# Split eggs and blanks

# Define a function to use as the groupby
def group_key(x):
    if "Jar_BLK_" in x:
        return "Jar_BLK"
    else:
        return "Other"

# Split the DataFrame based on the group key
groups = data1_Drop.groupby(group_key)

# Extract the groups
jar_blk_group = groups.get_group("Jar_BLK")
other_group = groups.get_group("Other")













#data1_Drop.to_csv("Droptest_H3O_EN113_set1.csv")
# for each file process


#Export the mean of the data
df_mean = data.groupby(['group_id']).mean()
print(df_mean.head())
df_mean.to_csv("mean__Day7_Batch_8_Calls_H3O_EN113_set1.csv")

df_max = data.groupby(['group_id']).max()
print(df_max)





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