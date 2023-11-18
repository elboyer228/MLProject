import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


df = pd.read_csv('Data/train.csv')

# number of repeted molecules in the data set 
df['Compound'] = df['Compound'].str.strip()
name_counts = df['Compound'].value_counts()
repeated_compounds = name_counts[name_counts > 1]
repeated_compounds_df = repeated_compounds.reset_index()
repeated_compounds_df.to_csv('Visualization_data/Repeated_Compounds_table.csv', index=False)

#number of data from labs
df['Lab'] = df['Lab'].str.strip()
lab_counts = df['Lab'].value_counts()
plt.figure(figsize=(14, 8))
plt.bar(lab_counts.index, lab_counts.values, color='lightblue')
plt.xticks(rotation=90)
plt.xlabel('Lab')
plt.ylabel('Counts')
plt.title('Counts of Molecules per Lab')
plt.tight_layout()
plt.savefig('Visualization_data/Lab_counts.png')



# analysis of the Retention time 
plt.figure(figsize=(10, 6))
retention_times = df['RT']

mean = np.mean(retention_times)
std_dev = np.std(retention_times)
z_scores = [(x - mean) / std_dev for x in retention_times] #to get the standardized 

plt.hist(retention_times, bins=10, density=True, alpha=0.6, color='lightblue', label='Retention Times')
plt.hist(z_scores, bins=10, density=True, alpha=0.6, color='lightgreen', label='Standardized Values')

# Generating the normal distribution curve
x = np.linspace(min(retention_times), max(retention_times), 100)
y = norm.pdf(x, mean, std_dev)  
plt.plot(x, y, 'b--', label='Normal Distribution')

#Standirdized nomal distribution curve 
x_std = np.linspace(min(z_scores), max(z_scores), 100)
y_std = norm.pdf(x_std, 0, 1) 
plt.plot(x_std, y_std, 'g--', label='Standard Normal Distribution')

plt.text(mean, 0.20, f'Mean={mean:.2f}', color='blue', fontsize=9, ha='left')
plt.text(mean, 0.17, f'Std Dev={std_dev:.2f}', color='blue', fontsize=9, ha='left')

                      
plt.xlabel('Retention Time')
plt.ylabel('Probability')
plt.title('Distribution of standardized and non-standardized retention times')
plt.legend()
plt.savefig('Visualization_data/RT_analysis.png')
