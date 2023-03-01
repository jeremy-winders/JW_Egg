import pandas as pd
import numpy as np
import os
import re

directory = '/Users/jeremy.winders/Documents/GitHub/JW_Egg/Segmented_data/H2O/113'


batch8dat = pd.DataFrame()





for file in os.listdir(directory):
    if file.strip().endswith('.csv'):
        data1 = pd.read_csv(os.path.join('/Users/jeremy.winders/Documents/GitHub/JW_Egg/Segmented_data/H2O/113', file))
        data1.drop(data1.columns[data1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        data1_drop = data1.drop(columns=['p-Drift_Act [mbar]', 'AbsTime', 'RelTime', 'E/N_Act [Td]', 'Cycle'])
        # Split eggs and blanks
        blanks = data1_drop.loc[data1_drop["group_id"].str.contains("Jar_BLK_*")]
        eggs = data1_drop.loc[~data1_drop["group_id"].str.contains("Jar_BLK_*")]
        print(blanks.columns)
        blanks_mean = blanks.select_dtypes(include=['float', 'int']).mean()






####### Check exports here #######
eggs.to_csv("Segmented_data/Test_Extraction/TEST_eggs.csv") #looks good
blanks.to_csv("Segmented_data/Test_Extraction/TEST_blanks.csv") #looks good
blanks_mean.to_csv("Segmented_data/Test_Extraction/TEST_blanks_mean.csv") #all columns are nows now but the mean of the blanks (1-12, should be 1-24!!)
#need to figure out where the missing blanks (13-24) went!



#subraction of blanks from eggs
        eggs_sub = eggs.sub(blanks_mean, axis='columns')  #loss of data in group_id column, some rows are negative now 
        eggs_sub.to_csv("Segmented_data/Test_Extraction/TEST_eggs_sub.csv") 
        clean_eggs = eggs_sub.groupby('group_id').mean() #loss of ALL data, only column names now
        clean_eggs.to_csv("Segmented_data/Test_Extraction/TEST_clean_eggs.csv")



#Removes extra info from columns and rounds masses to 1 decimal place
        pattern = r"m(\d+\.\d+)"
        matches = [re.search(pattern, col) for col in clean_eggs.columns]
        col_masses = [match.group(1) if match else None for match in matches]
        col_masses_round = [round(float(col), 1) for col in col_masses]
        col_masses_round  #looks good like it worked
        col_masses_round.to_csv("Segmented_data/Test_Extraction/TEST_col_masses_round.csv") #cannot export to csv as string "AttributeError: 'list' object has no attribute 'to_csv'"


#intergrate the rounded column masses into the dataframe


 



############################################################################
###---Start over single import and export---###



def column_round(col, ndigits=3):
    """
    Changes a column name to a rounded float of the mass
    Parameters
    ----------
    col: str
    ndigits: int
    Returns
    -------
    float
        rounded float of the mass
    """
    if col in ["AbsTime", "RelTime", "Cycle", "File"]:
        # keep the column the same if it's metadata
        return col
    else:
        # the first word in the column name
        col_mass = str(col).strip().split()[0]
        col_nums = ".".join(re.findall(r'\d+', col_mass))
        return round(float(col_nums), ndigits)













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
    if "Jar_BLK_*" in x:
        return "Jar_BLK"
    else:
        return "egg"



# Split the DataFrame based on the group id
df_data1_split = data1_Drop.group_key('group_id')   #broken

df_data1_split = data1_Drop.groupby('group_id').mean()
df_data1_split.head()

df_data1_splitOrder = df_data1_split.sort_values(by=['group_id', ascending=True])  #reorder/sort broken
df_data1_splitOrder

# Extract the groups
jar_blk_group = df_data1_split.groupby('Jar_BLK_*')
Egg_group = df_data1_split.get_group("Eggs")


#Try this way?
data1_1 = data1_Drop[data1_Drop['Jar_BLK_*'].notna()]
data1_







#data1_Drop.to_csv("Droptest_H3O_EN113_set1.csv")
# can't export because of the group_key error, so did it manually and then imported the file

data1_onlyEgg = pd.read_csv("Segmented_data/Test_Extraction/DropJars_H3O_EN113_set1.csv")
data1_onlyJar = pd.read_csv("Segmented_data/Test_Extraction/DropEggs_H3O_EN113_set1.csv")

data1_onlyEgg
data1_onlyJar



# Get mean
data1_onlyJar_mean = data1_onlyJar.groupby(['group_id']).mean()
data1_onlyJar_mean

#cannot get mean cause of group_id error, shows not in axis issue but I don't know how to fix it
#print(data1_onlyJar_mean.columns.tolist())
data1_onlyJar_mean.set_index()


#Drop unneeded columns
data1_onlyJar_mean.drop(columns=['group_id'])
Mean_all_Jars = data1_onlyJar_mean.drop(columns=['group_id'])
Mean_all_Jars.head()
Mean_all_Jars.to_csv("Segmented_data/Test_Extraction/mean_all_Jars_Day7_Batch_8_Calls_H3O_EN113_set1.csv")





#try different way to get mean of all blanks
blanks_mean_allBLK = blanks_mean.loc[blanks_mean["group_id"].str.contains("Jar_BLK_*")] #Key error on group_id
blanks_mean_allBLK
blanks_mean_allBLK = blanks_mean.drop('group_id', axis=1)
blanks_mean_allBLK.to_csv("Segmented_data/Test_Extraction/TEST_blanks_mean_allBLK.csv")
blanks_mean_allBLK = blanks_mean.drop('group_id', axis = 1)





        #blanks_mean = blanks.mean() ---> no idea why Adam wrote this (try the next line instead)
        #blanks_mean = blanks.groupby('group_id').mean()
        blanks_mean = blanks.select_dtypes(include=['float', 'int']).mean()

        # need to get mean of all blanks
        blanks_mean_allBLK = blanks_mean.drop['group_id', axis = 1] #issues with this line


#try different way to get mean of all blanks
blanks_mean_allBLK = blanks_mean.loc[blanks_mean["group_id"].str.contains("Jar_BLK_*")] #Key error on group_id
blanks_mean_allBLK
blanks_mean_allBLK = blanks_mean.drop('group_id', axis=1)
blanks_mean_allBLK.to_csv("Segmented_data/Test_Extraction/TEST_blanks_mean_allBLK.csv")
blanks_mean_allBLK = blanks_mean.drop('group_id', axis = 1)







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