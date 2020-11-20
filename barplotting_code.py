## Python code to share for Paper

##Barcoding on the data

#packages
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rc('xtick', labelsize='small', direction = 'in')
plt.rc('ytick', labelsize='small', direction = 'in')
plt.rc('lines',  linewidth = 1.0)

save_to = r"" #directory to save images
save_file = r"" # directory to save barcoded data


##importing
files_1 = glob.glob(r'C:\Users\Liam Kelleher\Documents\ANMSA\Project - Saliva\Age_Sorted_New\All_means\*.txt')
df = pd.concat([pd.read_csv(file, sep= '\t', names = ['Wavenumber', file + ' Intensity']) for file in files_1], axis=1)
df1 = df.filter(like = 'Intensity')
df1.set_index(df.iloc[:,0].round(2), inplace = True)
df1 = df1 - df1.min()


##normalisation
df_norm =(df1-df1.min())/(df1.max()-df1.min())
df_norm.index = df1.index
df_norm.columns = df1.columns
df_mean = df_norm.mean(axis=1)
df_mean =(df_mean-df_mean.min())/(df_mean.max()-df_mean.min())
df_dif = df_norm


##smoothing
new_df = []
from scipy import signal
for n,m in enumerate(df_dif.columns):
    array = df_dif.iloc[:,n] # [2:,n]
    filtered = signal.savgol_filter(array, 21, 2, 2)#smoothing window
    filtered = abs(filtered)
    maxx = filtered.max()/1.2 #vary this value for the amount to filter (i.e 10/20/30%)
    filtered[filtered > maxx] = 1
    filtered[filtered < maxx] = 0
    new_df.append(filtered)
new_df = pd.DataFrame(np.stack(new_df, axis=1))
df_new_mean = new_df.mean(axis=1)
df_norm1 = df_new_mean##.iloc[5:-5]
df_norm1.index = df_mean.index##[5:-5]
new_df.index = df_mean.index


#Generating the barcodes
namez = df1.columns
names = [nam.split('\\')[-1] for nam in namez]
names = [nam.split(' ')[0] for nam in names]

for n, m in enumerate(names):
    thing = new_df.iloc[:,n]
    thing.to_csv('{}\\{}'.format(save_file, m), sep='\t', header = False)

df_norm_sum = new_df.sum(axis=1)
df_norm_sum.columns = ['Wavenumber', 'Values']
df_norm_new = df_norm1
df_norm_new[df_norm_new > 0] = 1

df_bar = df_norm_new[::-1]
df_mean = df_mean[::-1]


#Plotting the barcodes overlaid on the spectra
import matplotlib.gridspec as gridspec
import seaborn as sns
current_palette = sns.color_palette()
gs = gridspec.GridSpec(3, 1)


plt.figure()
ax=plt.subplot(gs[0,0])
plt.bar(x=df_bar.index, height=df_bar, color=current_palette[9]) # 'gray')#
plt.plot(df_mean.index, df_mean, c='k', label='Sample X')
plt.text(1100, 1.02, 'Sample X')
plt.ylabel('Intensity (a.u.)')
plt.ylim(0, 1)