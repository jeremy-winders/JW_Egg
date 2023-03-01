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
        eggs_sub = eggs.sub(blanks_mean, axis='columns')
        clean_eggs = eggs_sub.groupby('group_id').mean()
        batch8dat = pd.concat([batch8dat,clean_eggs])


batch8dat





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





#kMeans clustering testing


In [32]: kout1 = sklearn.cluster.KMeans(float(col_masses))
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Input In [32], in <cell line: 1>()
----> 1 kout1 = sklearn.cluster.KMeans(float(col_masses))

AttributeError: module 'sklearn' has no attribute 'cluster'

In [33]: kout1 = sklearn.cluster.KMeans(n_clusters=224).fit((float(col_masses))
    ...: 
    ...: 
    ...: )
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Input In [33], in <cell line: 1>()
----> 1 kout1 = sklearn.cluster.KMeans(n_clusters=224).fit((float(col_masses))
      2 
      3 
      4 )

AttributeError: module 'sklearn' has no attribute 'cluster'

In [34]: from sklearn.cluster import KMeans

In [35]: kout1 = sMeans(n_clusters=224).fitt((float(col_masses)))
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Input In [35], in <cell line: 1>()
----> 1 kout1 = sMeans(n_clusters=224).fitt((float(col_masses)))

NameError: name 'sMeans' is not defined

In [36]: kout1 = KMeans(n_clusters=224).fitt((float(col_masses)))
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Input In [36], in <cell line: 1>()
----> 1 kout1 = KMeans(n_clusters=224).fitt((float(col_masses)))

AttributeError: 'KMeans' object has no attribute 'fitt'

In [37]: kout1 = KMeans(n_clusters=224).fit((float(col_masses)))
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Input In [37], in <cell line: 1>()
----> 1 kout1 = KMeans(n_clusters=224).fit((float(col_masses)))

TypeError: float() argument must be a string or a number, not 'list'

In [38]: 

In [38]: 

In [38]: 

In [38]: fcols = [float(item) for item in col_masses]

In [39]: fcols
Out[39]: 
[103.036,
 105.016,
 109.055,
 111.074,
 117.052,
 143.096,
 145.08,
 149.01,
 15.007,
 15.97,
 16.978,
 17.009,
 18.017,
 18.887,
 18.939,
 19.007,
 19.44,
 19.735,
 20.007,
 203.904,
 204.919,
 205.907,
 206.995,
 21.006,
 217.894,
 223.026,
 224.03,
 225.006,
 226.981,
 228.961,
 26.996,
 266.95,
 27.985,
 28.988,
 281.024,
 282.023,
 282.999,
 286.948,
 29.972,
 299.016,
 30.984,
 31.964,
 32.97,
 329.784,
 33.003,
 33.965,
 330.79,
 331.836,
 340.985,
 341.976,
 342.979,
 355.034,
 356.089,
 357.041,
 358.038,
 358.992,
 359.986,
 36.636,
 36.913,
 37.001,
 37.661,
 37.748,
 371.055,
 372.078,
 373.044,
 374.041,
 375.033,
 376.015,
 38.0,
 38.998,
 41.003,
 41.998,
 429.048,
 43.01,
 43.974,
 430.055,
 431.052,
 432.05,
 433.036,
 44.978,
 445.069,
 446.076,
 447.06,
 448.057,
 449.05,
 45.958,
 46.988,
 49.971,
 50.992,
 53.004,
 55.007,
 57.031,
 59.022,
 60.022,
 60.999,
 61.997,
 62.984,
 69.034,
 71.042,
 73.022,
 74.02,
 75.009,
 76.008,
 77.003,
 78.012,
 79.012,
 81.009,
 83.048,
 85.052,
 87.02,
 88.04,
 89.024,
 90.913,
 91.012,
 92.999,
 94.004,
 95.015,
 96.018,
 97.048,
 103.044,
 117.061,
 149.025,
 18.018,
 18.94,
 19.015,
 19.454,
 19.739,
 20.008,
 203.96,
 204.933,
 205.922,
 21.009,
 217.913,
 223.045,
 225.025,
 27.001,
 28.991,
 281.052,
 283.022,
 286.981,
 29.975,
 30.986,
 31.974,
 32.973,
 329.81,
 33.005,
 33.969,
 330.843,
 331.863,
 341.023,
 355.128,
 356.139,
 357.085,
 358.077,
 359.037,
 37.015,
 37.642,
 37.692,
 371.124,
 372.129,
 373.085,
 374.076,
 375.062,
 38.002,
 39.002,
 41.006,
 42.002,
 429.098,
 43.017,
 43.975,
 430.094,
 431.09,
 432.082,
 44.994,
 445.108,
 446.11,
 447.1,
 448.091,
 449.089,
 45.962,
 46.993,
 50.999,
 53.013,
 55.014,
 57.037,
 59.027,
 60.025,
 61.016,
 62.992,
 69.041,
 71.05,
 73.028,
 75.023,
 77.009,
 78.018,
 81.019,
 83.058,
 88.05,
 89.043,
 90.924,
 91.019,
 93.009,
 94.015,
 95.026,
 96.028,
 97.059,
 101.057,
 103.045,
 105.032,
 107.032,
 109.059,
 111.078,
 117.06,
 125.093,
 143.113,
 149.027,
 15.01,
 15.973,
 151.034,
 16.982,
 18.886,
 19.016,
 19.737,
 20.009,
 207.005,
 21.008,
 217.909,
 223.046,
 224.05,
 225.026,
 228.995,
 26.999,
 266.983,
 268.971,
 27.988,
 281.054,
 282.058,
 283.028,
 285.015,
 286.988,
 297.068,
 299.047,
 30.989,
 31.976,
 329.812,
 33.968,
 330.846,
 331.86,
 341.026,
 342.022,
 343.009,
 355.131,
 356.143,
 357.094,
 358.08,
 359.04,
 36.641,
 36.917,
 360.035,
 37.017,
 37.663,
 37.752,
 371.127,
 372.127,
 373.088,
 374.079,
 375.072,
 376.054,
 38.004,
 41.007,
 429.097,
 43.016,
 430.093,
 431.089,
 432.081,
 433.068,
 445.114,
 446.115,
 447.099,
 448.096,
 45.002,
 47.035,
 48.022,
 48.979,
 49.978,
 53.014,
 59.023,
 60.023,
 61.011,
 62.003,
 63.002,
 65.03,
 71.048,
 73.032,
 74.027,
 75.019,
 77.01,
 78.017,
 79.022,
 81.017,
 83.059,
 85.058,
 87.031,
 88.048,
 89.035,
 90.925,
 91.02,
 93.013,
 94.016,
 95.027,
 96.029,
 97.057,
 15.881,
 16.214,
 17.016,
 17.055,
 17.554,
 18.062,
 18.632,
 18.723,
 18.781,
 18.97,
 19.474,
 19.54,
 19.943,
 20.05,
 20.82,
 203.008,
 203.98,
 205.084,
 205.968,
 21.049,
 218.001,
 22.916,
 25.355,
 28.034,
 28.765,
 28.793,
 28.859,
 29.028,
 29.746,
 30.017,
 31.721,
 31.772,
 31.998,
 328.554,
 329.556,
 329.885,
 33.037,
 33.886,
 33.999,
 330.895,
 332.034,
 332.69,
 35.387,
 355.145,
 356.248,
 357.165,
 358.143,
 358.713,
 359.096,
 36.737,
 36.797,
 37.058,
 37.29,
 37.626,
 37.777,
 371.191,
 372.285,
 373.171,
 374.177,
 375.156,
 38.042,
 38.476,
 39.038,
 41.053,
 429.204,
 43.045,
 431.181,
 44.025,
 445.239,
 445.875,
 447.21,
 45.024,
 46.016,
 47.04,
 50.011,
 51.019,
 53.035,
 55.029,
 57.076,
 59.074,
 61.061,
 63.04,
 69.062,
 72.947,
 73.077,
 75.075,
 76.948,
 78.068,
 81.035,
 83.112,
 88.113,
 89.096,
 91.049,
 93.069,
 94.059,
 95.054,
 96.054,
 101.056,
 103.041,
 105.018,
 109.052,
 111.079,
 117.056,
 123.075,
 143.107,
 145.087,
 149.02,
 15.006,
 16.98,
 17.007,
 18.016,
 19.043,
 19.089,
 19.109,
 19.729,
 20.006,
 202.911,
 203.966,
 204.248,
 204.589,
 204.939,
 205.432,
 205.703,
 205.918,
 207.002,
 21.005,
 217.906,
 223.041,
 225.02,
 266.978,
 281.039,
 283.013,
 286.971,
 29.971,
 30.983,
 31.969,
 32.074,
 32.968,
 330.84,
 331.858,
 332.186,
 332.836,
 341.011,
 355.111,
 356.122,
 357.073,
 358.059,
 359.03,
 36.91,
 37.012,
 37.123,
 37.656,
 37.749,
 37.999,
 371.113,
 372.112,
 373.073,
 374.064,
 375.05,
 376.043,
 38.999,
 41.999,
 429.081,
 43.013,
 43.971,
 430.082,
 431.078,
 432.07,
 44.973,
 445.104,
 446.105,
 447.089,
 448.086,
 48.981,
 49.976,
 50.991,
 51.997,
 53.006,
 55.009,
 57.035,
 59.025,
 60.02,
 61.001,
 61.999,
 62.993,
 65.008,
 67.023,
 69.037,
 71.044,
 73.024,
 75.011,
 77.008,
 78.015,
 79.017,
 81.006,
 83.055,
 85.059,
 87.023,
 88.043,
 89.027,
 91.014,
 94.011,
 95.022,
 96.021,
 99.047]

In [40]: kout1 = KMeans(n_clusters=224).fit(fcols)

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Input In [40], in <cell line: 1>()
----> 1 kout1 = KMeans(n_clusters=224).fit(fcols)

File ~/opt/miniconda3/envs/Eggs/lib/python3.8/site-packages/sklearn/cluster/_kmeans.py:1030, in KMeans.fit(self, X, y, sample_weight)
   1004 def fit(self, X, y=None, sample_weight=None):
   1005     """Compute k-means clustering.
   1006 
   1007     Parameters
   (...)
   1028         Fitted estimator.
   1029     """
-> 1030     X = self._validate_data(X, accept_sparse='csr',
   1031                             dtype=[np.float64, np.float32],
   1032                             order='C', copy=self.copy_x,
   1033                             accept_large_sparse=False)
   1035     self._check_params(X)
   1036     random_state = check_random_state(self.random_state)

File ~/opt/miniconda3/envs/Eggs/lib/python3.8/site-packages/sklearn/base.py:420, in BaseEstimator._validate_data(self, X, y, reset, validate_separately, **check_params)
    415     if self._get_tags()['requires_y']:
    416         raise ValueError(
    417             f"This {self.__class__.__name__} estimator "
    418             f"requires y to be passed, but the target y is None."
    419         )
--> 420     X = check_array(X, **check_params)
    421     out = X
    422 else:

File ~/opt/miniconda3/envs/Eggs/lib/python3.8/site-packages/sklearn/utils/validation.py:72, in _deprecate_positional_args.<locals>.inner_f(*args, **kwargs)
     67     warnings.warn("Pass {} as keyword args. From version 0.25 "
     68                   "passing these as positional arguments will "
     69                   "result in an error".format(", ".join(args_msg)),
     70                   FutureWarning)
     71 kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
---> 72 return f(**kwargs)

File ~/opt/miniconda3/envs/Eggs/lib/python3.8/site-packages/sklearn/utils/validation.py:619, in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
    617     # If input is 1D raise error
    618     if array.ndim == 1:
--> 619         raise ValueError(
    620             "Expected 2D array, got 1D array instead:\narray={}.\n"
    621             "Reshape your data either using array.reshape(-1, 1) if "
    622             "your data has a single feature or array.reshape(1, -1) "
    623             "if it contains a single sample.".format(array))
    625 # in the future np.flexible dtypes will be handled like object dtypes
    626 if dtype_numeric and np.issubdtype(array.dtype, np.flexible):

ValueError: Expected 2D array, got 1D array instead:
array=[103.036 105.016 109.055 111.074 117.052 143.096 145.08  149.01   15.007
  15.97   16.978  17.009  18.017  18.887  18.939  19.007  19.44   19.735
  20.007 203.904 204.919 205.907 206.995  21.006 217.894 223.026 224.03
 225.006 226.981 228.961  26.996 266.95   27.985  28.988 281.024 282.023
 282.999 286.948  29.972 299.016  30.984  31.964  32.97  329.784  33.003
  33.965 330.79  331.836 340.985 341.976 342.979 355.034 356.089 357.041
 358.038 358.992 359.986  36.636  36.913  37.001  37.661  37.748 371.055
 372.078 373.044 374.041 375.033 376.015  38.     38.998  41.003  41.998
 429.048  43.01   43.974 430.055 431.052 432.05  433.036  44.978 445.069
 446.076 447.06  448.057 449.05   45.958  46.988  49.971  50.992  53.004
  55.007  57.031  59.022  60.022  60.999  61.997  62.984  69.034  71.042
  73.022  74.02   75.009  76.008  77.003  78.012  79.012  81.009  83.048
  85.052  87.02   88.04   89.024  90.913  91.012  92.999  94.004  95.015
  96.018  97.048 103.044 117.061 149.025  18.018  18.94   19.015  19.454
  19.739  20.008 203.96  204.933 205.922  21.009 217.913 223.045 225.025
  27.001  28.991 281.052 283.022 286.981  29.975  30.986  31.974  32.973
 329.81   33.005  33.969 330.843 331.863 341.023 355.128 356.139 357.085
 358.077 359.037  37.015  37.642  37.692 371.124 372.129 373.085 374.076
 375.062  38.002  39.002  41.006  42.002 429.098  43.017  43.975 430.094
 431.09  432.082  44.994 445.108 446.11  447.1   448.091 449.089  45.962
  46.993  50.999  53.013  55.014  57.037  59.027  60.025  61.016  62.992
  69.041  71.05   73.028  75.023  77.009  78.018  81.019  83.058  88.05
  89.043  90.924  91.019  93.009  94.015  95.026  96.028  97.059 101.057
 103.045 105.032 107.032 109.059 111.078 117.06  125.093 143.113 149.027
  15.01   15.973 151.034  16.982  18.886  19.016  19.737  20.009 207.005
  21.008 217.909 223.046 224.05  225.026 228.995  26.999 266.983 268.971
  27.988 281.054 282.058 283.028 285.015 286.988 297.068 299.047  30.989
  31.976 329.812  33.968 330.846 331.86  341.026 342.022 343.009 355.131
 356.143 357.094 358.08  359.04   36.641  36.917 360.035  37.017  37.663
  37.752 371.127 372.127 373.088 374.079 375.072 376.054  38.004  41.007
 429.097  43.016 430.093 431.089 432.081 433.068 445.114 446.115 447.099
 448.096  45.002  47.035  48.022  48.979  49.978  53.014  59.023  60.023
  61.011  62.003  63.002  65.03   71.048  73.032  74.027  75.019  77.01
  78.017  79.022  81.017  83.059  85.058  87.031  88.048  89.035  90.925
  91.02   93.013  94.016  95.027  96.029  97.057  15.881  16.214  17.016
  17.055  17.554  18.062  18.632  18.723  18.781  18.97   19.474  19.54
  19.943  20.05   20.82  203.008 203.98  205.084 205.968  21.049 218.001
  22.916  25.355  28.034  28.765  28.793  28.859  29.028  29.746  30.017
  31.721  31.772  31.998 328.554 329.556 329.885  33.037  33.886  33.999
 330.895 332.034 332.69   35.387 355.145 356.248 357.165 358.143 358.713
 359.096  36.737  36.797  37.058  37.29   37.626  37.777 371.191 372.285
 373.171 374.177 375.156  38.042  38.476  39.038  41.053 429.204  43.045
 431.181  44.025 445.239 445.875 447.21   45.024  46.016  47.04   50.011
  51.019  53.035  55.029  57.076  59.074  61.061  63.04   69.062  72.947
  73.077  75.075  76.948  78.068  81.035  83.112  88.113  89.096  91.049
  93.069  94.059  95.054  96.054 101.056 103.041 105.018 109.052 111.079
 117.056 123.075 143.107 145.087 149.02   15.006  16.98   17.007  18.016
  19.043  19.089  19.109  19.729  20.006 202.911 203.966 204.248 204.589
 204.939 205.432 205.703 205.918 207.002  21.005 217.906 223.041 225.02
 266.978 281.039 283.013 286.971  29.971  30.983  31.969  32.074  32.968
 330.84  331.858 332.186 332.836 341.011 355.111 356.122 357.073 358.059
 359.03   36.91   37.012  37.123  37.656  37.749  37.999 371.113 372.112
 373.073 374.064 375.05  376.043  38.999  41.999 429.081  43.013  43.971
 430.082 431.078 432.07   44.973 445.104 446.105 447.089 448.086  48.981
  49.976  50.991  51.997  53.006  55.009  57.035  59.025  60.02   61.001
  61.999  62.993  65.008  67.023  69.037  71.044  73.024  75.011  77.008
  78.015  79.017  81.006  83.055  85.059  87.023  88.043  89.027  91.014
  94.011  95.022  96.021  99.047].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

In [41]: 

In [41]: import numpy as np

In [42]: np.ndarray((1,516), fcols)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Input In [42], in <cell line: 1>()
----> 1 np.ndarray((1,516), fcols)

TypeError: Field elements must be 2- or 3-tuples, got '103.036'

In [43]: np.ndarray((1,516), buffer=fcols)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Input In [43], in <cell line: 1>()
----> 1 np.ndarray((1,516), buffer=fcols)

TypeError: a bytes-like object is required, not 'list'

In [44]: np.ndarray((1,516), buffer=np.array(fcols))
Out[44]: 
array([[103.036, 105.016, 109.055, 111.074, 117.052, 143.096, 145.08 ,
        149.01 ,  15.007,  15.97 ,  16.978,  17.009,  18.017,  18.887,
         18.939,  19.007,  19.44 ,  19.735,  20.007, 203.904, 204.919,
        205.907, 206.995,  21.006, 217.894, 223.026, 224.03 , 225.006,
        226.981, 228.961,  26.996, 266.95 ,  27.985,  28.988, 281.024,
        282.023, 282.999, 286.948,  29.972, 299.016,  30.984,  31.964,
         32.97 , 329.784,  33.003,  33.965, 330.79 , 331.836, 340.985,
        341.976, 342.979, 355.034, 356.089, 357.041, 358.038, 358.992,
        359.986,  36.636,  36.913,  37.001,  37.661,  37.748, 371.055,
        372.078, 373.044, 374.041, 375.033, 376.015,  38.   ,  38.998,
         41.003,  41.998, 429.048,  43.01 ,  43.974, 430.055, 431.052,
        432.05 , 433.036,  44.978, 445.069, 446.076, 447.06 , 448.057,
        449.05 ,  45.958,  46.988,  49.971,  50.992,  53.004,  55.007,
         57.031,  59.022,  60.022,  60.999,  61.997,  62.984,  69.034,
         71.042,  73.022,  74.02 ,  75.009,  76.008,  77.003,  78.012,
         79.012,  81.009,  83.048,  85.052,  87.02 ,  88.04 ,  89.024,
         90.913,  91.012,  92.999,  94.004,  95.015,  96.018,  97.048,
        103.044, 117.061, 149.025,  18.018,  18.94 ,  19.015,  19.454,
         19.739,  20.008, 203.96 , 204.933, 205.922,  21.009, 217.913,
        223.045, 225.025,  27.001,  28.991, 281.052, 283.022, 286.981,
         29.975,  30.986,  31.974,  32.973, 329.81 ,  33.005,  33.969,
        330.843, 331.863, 341.023, 355.128, 356.139, 357.085, 358.077,
        359.037,  37.015,  37.642,  37.692, 371.124, 372.129, 373.085,
        374.076, 375.062,  38.002,  39.002,  41.006,  42.002, 429.098,
         43.017,  43.975, 430.094, 431.09 , 432.082,  44.994, 445.108,
        446.11 , 447.1  , 448.091, 449.089,  45.962,  46.993,  50.999,
         53.013,  55.014,  57.037,  59.027,  60.025,  61.016,  62.992,
         69.041,  71.05 ,  73.028,  75.023,  77.009,  78.018,  81.019,
         83.058,  88.05 ,  89.043,  90.924,  91.019,  93.009,  94.015,
         95.026,  96.028,  97.059, 101.057, 103.045, 105.032, 107.032,
        109.059, 111.078, 117.06 , 125.093, 143.113, 149.027,  15.01 ,
         15.973, 151.034,  16.982,  18.886,  19.016,  19.737,  20.009,
        207.005,  21.008, 217.909, 223.046, 224.05 , 225.026, 228.995,
         26.999, 266.983, 268.971,  27.988, 281.054, 282.058, 283.028,
        285.015, 286.988, 297.068, 299.047,  30.989,  31.976, 329.812,
         33.968, 330.846, 331.86 , 341.026, 342.022, 343.009, 355.131,
        356.143, 357.094, 358.08 , 359.04 ,  36.641,  36.917, 360.035,
         37.017,  37.663,  37.752, 371.127, 372.127, 373.088, 374.079,
        375.072, 376.054,  38.004,  41.007, 429.097,  43.016, 430.093,
        431.089, 432.081, 433.068, 445.114, 446.115, 447.099, 448.096,
         45.002,  47.035,  48.022,  48.979,  49.978,  53.014,  59.023,
         60.023,  61.011,  62.003,  63.002,  65.03 ,  71.048,  73.032,
         74.027,  75.019,  77.01 ,  78.017,  79.022,  81.017,  83.059,
         85.058,  87.031,  88.048,  89.035,  90.925,  91.02 ,  93.013,
         94.016,  95.027,  96.029,  97.057,  15.881,  16.214,  17.016,
         17.055,  17.554,  18.062,  18.632,  18.723,  18.781,  18.97 ,
         19.474,  19.54 ,  19.943,  20.05 ,  20.82 , 203.008, 203.98 ,
        205.084, 205.968,  21.049, 218.001,  22.916,  25.355,  28.034,
         28.765,  28.793,  28.859,  29.028,  29.746,  30.017,  31.721,
         31.772,  31.998, 328.554, 329.556, 329.885,  33.037,  33.886,
         33.999, 330.895, 332.034, 332.69 ,  35.387, 355.145, 356.248,
        357.165, 358.143, 358.713, 359.096,  36.737,  36.797,  37.058,
         37.29 ,  37.626,  37.777, 371.191, 372.285, 373.171, 374.177,
        375.156,  38.042,  38.476,  39.038,  41.053, 429.204,  43.045,
        431.181,  44.025, 445.239, 445.875, 447.21 ,  45.024,  46.016,
         47.04 ,  50.011,  51.019,  53.035,  55.029,  57.076,  59.074,
         61.061,  63.04 ,  69.062,  72.947,  73.077,  75.075,  76.948,
         78.068,  81.035,  83.112,  88.113,  89.096,  91.049,  93.069,
         94.059,  95.054,  96.054, 101.056, 103.041, 105.018, 109.052,
        111.079, 117.056, 123.075, 143.107, 145.087, 149.02 ,  15.006,
         16.98 ,  17.007,  18.016,  19.043,  19.089,  19.109,  19.729,
         20.006, 202.911, 203.966, 204.248, 204.589, 204.939, 205.432,
        205.703, 205.918, 207.002,  21.005, 217.906, 223.041, 225.02 ,
        266.978, 281.039, 283.013, 286.971,  29.971,  30.983,  31.969,
         32.074,  32.968, 330.84 , 331.858, 332.186, 332.836, 341.011,
        355.111, 356.122, 357.073, 358.059, 359.03 ,  36.91 ,  37.012,
         37.123,  37.656,  37.749,  37.999, 371.113, 372.112, 373.073,
        374.064, 375.05 , 376.043,  38.999,  41.999, 429.081,  43.013,
         43.971, 430.082, 431.078, 432.07 ,  44.973, 445.104, 446.105,
        447.089, 448.086,  48.981,  49.976,  50.991,  51.997,  53.006,
         55.009,  57.035,  59.025,  60.02 ,  61.001,  61.999,  62.993,
         65.008,  67.023,  69.037,  71.044,  73.024,  75.011,  77.008,
         78.015,  79.017,  81.006,  83.055,  85.059,  87.023,  88.043,
         89.027,  91.014,  94.011,  95.022,  96.021]])

In [45]: fcols2 = np.ndarray((1,516), buffer=np.array(fcols))

In [46]: kout1 = KMeans(n_clusters=224).fit(fcols2)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Input In [46], in <cell line: 1>()
----> 1 kout1 = KMeans(n_clusters=224).fit(fcols2)

File ~/opt/miniconda3/envs/Eggs/lib/python3.8/site-packages/sklearn/cluster/_kmeans.py:1035, in KMeans.fit(self, X, y, sample_weight)
   1005 """Compute k-means clustering.
   1006 
   1007 Parameters
   (...)
   1028     Fitted estimator.
   1029 """
   1030 X = self._validate_data(X, accept_sparse='csr',
   1031                         dtype=[np.float64, np.float32],
   1032                         order='C', copy=self.copy_x,
   1033                         accept_large_sparse=False)
-> 1035 self._check_params(X)
   1036 random_state = check_random_state(self.random_state)
   1038 # Validate init array

File ~/opt/miniconda3/envs/Eggs/lib/python3.8/site-packages/sklearn/cluster/_kmeans.py:958, in KMeans._check_params(self, X)
    956 # n_clusters
    957 if X.shape[0] < self.n_clusters:
--> 958     raise ValueError(f"n_samples={X.shape[0]} should be >= "
    959                      f"n_clusters={self.n_clusters}.")
    961 # tol
    962 self._tol = _tolerance(X, self.tol)

ValueError: n_samples=1 should be >= n_clusters=224.

In [47]: fcols2 = np.ndarray((516, 1), buffer=np.array(fcols))

In [48]: kout1 = KMeans(n_clusters=224).fit(fcols2)

In [49]: kout1
Out[49]: KMeans(n_clusters=224)

In [50]: kout1.getparams()
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Input In [50], in <cell line: 1>()
----> 1 kout1.getparams()

AttributeError: 'KMeans' object has no attribute 'getparams'

In [51]: kout1.get_params()
Out[51]: 
{'algorithm': 'auto',
 'copy_x': True,
 'init': 'k-means++',
 'max_iter': 300,
 'n_clusters': 224,
 'n_init': 10,
 'n_jobs': 'deprecated',
 'precompute_distances': 'deprecated',
 'random_state': None,
 'tol': 0.0001,
 'verbose': 0}

In [52]: kout1 = KMeans(n_clusters=224).fit_transform(fcols2)

In [53]: kout1
Out[53]: 
array([[ 83.301, 268.091,  42.051, ...,  59.011,  75.002, 255.023],
       [ 85.281, 266.111,  40.071, ...,  60.991,  76.982, 253.043],
       [ 89.32 , 262.072,  36.032, ...,  65.03 ,  81.021, 249.004],
       ...,
       [ 74.276, 277.116,  51.076, ...,  49.986,  65.977, 264.048],
       [ 75.287, 276.105,  50.065, ...,  50.997,  66.988, 263.037],
       [ 76.286, 275.106,  49.066, ...,  51.996,  67.987, 262.038]])

In [54]: kout1.shape
Out[54]: (516, 224)

In [55]: 