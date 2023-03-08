import pandas as pd
import numpy as np
import os
import re
from sklearn.cluster import KMeans
directory = '/Users/jeremy.winders/Documents/GitHub/JW_Egg/Segmented_data/H2O/113'


batch8dfs = []
max_cols = 0



for file in os.listdir(directory):
    if file.strip().endswith('.csv'):
        data1 = pd.read_csv(os.path.join(directory, file))
        data1.drop(data1.columns[data1.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        data1_drop = data1.drop(columns=['p-Drift_Act [mbar]', 'AbsTime', 'RelTime', 'E/N_Act [Td]', 'Cycle'])
        # Split eggs and blanks
        blanks = data1_drop.loc[data1_drop["group_id"].str.contains("Jar_BLK_*")]
        eggs = data1_drop.loc[~data1_drop["group_id"].str.contains("Jar_BLK_*")]
        blanks_mean = blanks.select_dtypes(include=['float', 'int']).mean()
        clean_eggs = eggs.groupby('group_id').mean()
        eggs_final = clean_eggs.sub(blanks_mean, axis='columns')
        eggs_final[eggs_final < 1] = 1
        cols = eggs_final.columns[eggs_final.mean() < 1000 ]
        eggs_final.drop(cols, axis=1, inplace=True)
        if eggs_final.shape[1] > max_cols:
            max_cols = eggs_final.shape[1]
        batch8dfs.append(eggs_final)


pattern = r"m(\d+\.\d+)"
mass_cols = []
for dat in batch8dfs: # open files
    mass_cols.extend(dat.columns) 
matches = [re.search(pattern, col) for col in mass_cols]
col_masses = [float(match.group(1)) if match else None for match in matches]
col_array = np.reshape(col_masses, (len(col_masses),1))


alist = list(col_array.flatten())

i1 = itertools.product(alist, repeat=2)
candidate = []
for val in i1:
    if abs(val[0] - val[1]) < 0.5 :
        candidate.append(abs(val[0] - val[1]))

np.mean(candidate) # 0.07
np.histogram(candidate)  # 0.11 looks like a good cut off

# use a distanve based clustering with a cutoff of 0.11, distance is euclidian but there is only 1 dim so it is like l1
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.11, linkage='ward').fit(col_array.reshape(-1,1))
#group_df.to_csv("/Users/jeremy.winders/Documents/GitHub/JW_Egg/Segmented_data/Test_Extraction/H30_113_grouped.csv")
group_df = pd.DataFrame({'group': clustering.labels_, 'floatval': list(col_masses), 'col_names' : list(mass_cols)})
mass_means = group_df.groupby('group').mean()
meandf = mass_means.iloc[list(group_df['group'])]
group_df["mean_mass"] = list(meandf['floatval'])
group_df["new_name"] = [ "EN113_H30_" + str(round(item, 3)) for item in list(group_df["mean_mass"])]
group_df = group_df.sort_values(by=['floatval'])

harmonized_dfs = []
for b8dat in batch8dfs:
    harmonized_dfs.append(b8dat.rename(columns=dict(zip(group_df['col_names'],group_df['new_name']))))

unified = pd.concat(harmonized_dfs)


#-------in group_df take the new_name columns and make them the column names in eggs_final------#
#           I've done tried a few ways to do this but I'm not sure how to do it (see below) 



#drop all columns but new roudned masses 
#new_col_names = group_df.drop(columns=['group', 'floatval', 'col_names', 'mean_mass'])

#replace columns names in df in eggs_final with new_col_names
eggs_final_1 = eggs_final.rename(columns=new_col_names.set_index('col_names')['new_name'].to_dict())

---> df = df.rename(columns=dict(zip(df.columns, new_col_names)))


