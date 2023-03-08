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
        eggs_final[eggs_final < 0] = 0
        if eggs_final.shape[1] > max_cols:
            max_cols = eggs_final.shape[1]
        batch8dfs.append(eggs_final)
        #batch8dat = pd.concat([batch8dat,eggs_final])


pattern = r"m(\d+\.\d+)"
mass_cols = []
for dat in batch8dfs:
    mass_cols.extend(dat.columns) 
matches = [re.search(pattern, col) for col in mass_cols]
col_masses = [float(match.group(1)) if match else None for match in matches]
col_array = np.reshape(col_masses, (len(col_masses),1))
kmeans = KMeans(n_clusters=max_cols, random_state=0).fit(col_array)
groups = kmeans.predict(col_array)

group_df = pd.DataFrame({'group': list(groups), 'floatval': list(col_masses), 'col_names' : list(mass_cols)})
mass_means = group_df.groupby('group').mean()
meandf = mass_means.iloc[list(group_df['group'])]
group_df["mean_mass"] = list(meandf['floatval'])
group_df["new_name"] = [ "EN113_H30_" + str(round(item, 2)) for item in list(group_df["mean_mass"])]








#drop all columns but new roudned masses 
#new_col_names = group_df.drop(columns=['group', 'floatval', 'col_names', 'mean_mass'])

#replace columns names in df in eggs_final with new_col_names
eggs_final_1 = eggs_final.rename(columns=new_col_names.set_index('col_names')['new_name'].to_dict())


#code that looks in rows of dataframe 'eggs_final_1'  to find values that match the  pattern = r"m(\d+\.\d+)"
#and IF those values match the column named 'col_names' in data frame 'group_df' it replaces the matched value in dataframe 'batch8dfs' with the row value of found in 'new_name' 

eggs_final_1 = eggs_final

for idx, row in eggs_final_1.iterrows():
    # Loop through the columns of the row
    for col_name, value in row.iteritems():
        # Check if the value matches the pattern
        match = re.match(pattern, value)
        if match:
            # If it does, check if the matched value is in the col_names column of group_df
            matched_value = match.group(1)
            if matched_value in group_df['col_names'].values:
                # If it is, replace the matched value with the corresponding new_name value from group_df
                new_name = group_df.loc[group_df['col_names'] == matched_value, 'new_name'].values[0]
                eggs_final_1.at[idx, col_name] = new_name

print(eggs_final_1)







#group_df.to_csv("/Users/jeremy.winders/Documents/GitHub/JW_Egg/Segmented_data/Test_Extraction/H30_113_grouped.csv")