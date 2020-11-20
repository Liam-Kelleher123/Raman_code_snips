##PCA analysis code

#######Go to set paramater at the bottom of the page

########################################################################################
############################ Import common libraries #################################
########################################################################################
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import SMOTE #needed to install imbalanced-learn
cmap = cm.get_cmap('Dark2') #Dark2, viridis, tab20b
#color = ['k', 'r', 'g', 'b', 'y', 'c']
color = ['k', 'k', 'k', 'k', 'r', 'r', 'r', 'r']

########################################################################################
################## Section 1 - Data importing and preprocessing ########################
########################################################################################

def set_directory(directory):
    os.chdir(directory)
    print(os.getcwd)
    return os.getcwd()

def get_data(yes): 
    thing = True
    if thing:
        names = glob.glob('*.txt') 
        df  = pd.concat([pd.read_csv(file, sep = '\t', skiprows=[0], names = ['Wavenumber', file + ' Intensity']) for file in names], axis = 1)
        cols = [c for c in df.columns if c.endswith('Intensity')]
        df1 = df[cols]
        df1.set_index(df.iloc[:,0].round(2), inplace = True)
        df1.plot(colormap = cmap)
        plt.legend(loc='upper right',  ncol= 1, fontsize = 'xx-small')
        df1 = df1.transpose()
        print(df1.isna().any().any())
        #df1 = df1.dropna(axis='index')
        names = list(df1.index)
        print('Data has been loaded and a second DataFrame of Intensity has been made. A plot of the loaded data has been made.')
        return names, df, df1
    else:
        print('No data has been added to DataFrame, and no data has been plotted.')
        return None, None, None
    
def encode_temp_catagorical_data(separate_1, separate_2, delete_1, delete_2):
    global X, y
    if separate_2 != 'no':
        index_names = list(df1.index)
        new_names_1 = [nam.split('{}'.format(separate_1), 1)[delete_1].split('{}'.format(separate_2), 1)[delete_2] for nam in index_names]
    elif separate_1 != 'no':
        index_names = list(df1.index)
        new_names_1 = [nam.split('{}'.format(separate_1), 1)[delete_1] for nam in index_names]
    else:
        new_names_1 = list(df1.index)
        print('No encoding has been carried out.')
    # Section that is encoding the reduced filenames
    df1.insert(loc = 0, column = 'Names', value = new_names_1)# if separate_2=='no' else new_names_2)
    df1.sort_values(by = ['Names'], inplace=True)
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    new_labels = labelencoder.fit_transform(df1['Names'].copy())
    #Append new filename variables to dataframe
    df1.insert(loc = 1, column = 'Encoded', value = new_labels)        
    #thing = df[['Wavenumber']].copy().iloc[:, 1].values
    X = df1.iloc[:,2:].values
    y = df1.iloc[:, 1].values
    print('Data has been encoded. {}'.format(y))
    X = X.astype(np.float64)
    #Section creating the dictionary for encoded values and names, used later in plots.
    value_1 = df1.Encoded.unique()
    value_2 = df1.Names.unique()
    dict_name = dict(zip(value_1, value_2))
    print(dict_name)
    return X, y, new_labels, df1, dict_name#,thing


########################################################################################
# Dimensionality reduction techniques LDA and PCA. Split between all data or training/test sets
########################################################################################
def apply_pca(components, pca_type, test_percent, cv_folds):
    global pca_tt, X_train_pca, X_test_pca, y_train_pca, y_test_pca, X_all_pca, cm_pca, y_pred_pca, pca_tt_loadings, pca_loadings, explained_variance, waven_val
    waven_val = df.iloc[:,0]
    #waven_val = data.iloc[:,0]
    if pca_type == 'pca_tt':
        #smote = SMOTE('not majority')
        #X_sm, y_sm = smote.fit_sample(X, y)
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X, y, test_size = test_percent)#, random_state = 0) # X_sm, y_sm
        pca_tt = PCA(n_components = components)
        X_train_pca = pca_tt.fit_transform(X_train_pca)
        X_test_pca = pca_tt.transform(X_test_pca)
        explained_variance_tt = pca_tt.explained_variance_ratio_
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train_pca, y_train_pca)
        scores = cross_val_score(LogisticRegression(), X, y, cv = cv_folds) #
        #print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        explained_variance = (pca_tt.explained_variance_ratio_*100).round(2)
        y_pred_pca = classifier.predict(X_test_pca)
        cm_pca = confusion_matrix(y_test_pca, y_pred_pca)
        pca_tt_loadings = pca_tt.components_.T * np.sqrt(pca_tt.explained_variance_)
        f, axarr = plt.subplots(components, sharex=True, figsize=(6,8))
        for i, a in enumerate(list(range(0, components)), 1):
            axarr[a].plot(waven_val, pca_tt_loadings[:,a], color='black')#, label = 'PCA Component {}'.format(i))
            axarr[a].set_title('PC{} ({}%)'.format(i, round(explained_variance[a]*100, 2)))
            axarr[a].set_ylabel('PCA Component {}'.format(i))
            axarr[a].yaxis.set_label_coords(-0.1, 0.5)
        plt.xlabel('Wavenumber (cm-1)')
        #f.align_ylabels()
        plt.tight_layout() 
        plt.savefig('{}\\PCA_Loading_Plot.pdf'.format(save_to), bbox_inches='tight')
        print('PCA has been performed for a training and test set (size = {}) of the data.\n'.format(test_percent), 
              'PC score percentage are {} for {} components.\n'.format(explained_variance_tt, components),
              "Accuracy: %0.4f (+/- %0.4f). \n" % (scores.mean(), scores.std() * 2),
              'PC loadings are {}'.format(pca_tt_loadings))
    elif pca_type == 'pca_all':
        test_percent = test_percent
        pca = PCA(n_components = components)
        X_all_pca = pca.fit_transform(X)
        pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        explained_variance = (pca.explained_variance_ratio_*100).round(2)
        fig = plt.figure(figsize=(2.55*2, 1.59*2))
        ax1 = fig.add_subplot(111)
        ax1.plot(waven_val, pca_loadings[:,0], color='k')
        ax1.set_ylabel('PC1 ({}%)'.format(explained_variance[0]))
        ax2 = ax1.twinx()
        ax2.plot(waven_val, pca_loadings[:,1], 'b-')
        ax2.set_ylabel('PC2 ({}%)'.format(explained_variance[1]), color='b')
        for tl in ax2.get_yticklabels():
            tl.set_color('b')
        ax1.set_xlabel('Wavenumber (cm-1)')
        #align_yaxis_np(ax1, ax2)
        plt.tight_layout() 
        plt.show()
        plt.savefig('{}\\PCA_Loading_Plot.pdf'.format(save_to), bbox_inches='tight')
        print('PCA has been performed all data.\n', 
              'PC scores are {} for {} components.\n'.format(explained_variance, components),
              'PC loadings are {}'.format(pca_loadings))
    elif pca_type == 'pca_with_plot':
        test_percent = test_percent
        pca = PCA(n_components = components)
        X_all_pca = pca.fit_transform(X)
        pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        explained_variance = pca.explained_variance_ratio_
        pca_plot = np.column_stack([pca_loadings, X.mean(axis=0).T])
        labels = ['PC1 ({}%)'.format(round(explained_variance[0]*100, 2)), 'PC2 ({}%)'.format(round(explained_variance[1]*100, 2)), 'Intensity']
        fig, axarr = plt.subplots(components+1, sharex=True, figsize=(7, 4.37),gridspec_kw = {'wspace':0, 'hspace':0})
        for i, a in enumerate(list(range(0, components+1)), 1):
            axarr[a].plot(waven_val, pca_plot[:,a], color='black')#, label = 'PCA Component {}'.format(i))
            #axarr[a].set_title('PCA Component {} ({}%)'.format(i, round(explained_variance[a]*100, 2)))
            axarr[a].set_ylabel(labels[i-1])
            axarr[a].yaxis.set_label_coords(-0.1, 0.5)
        plt.xlabel('Wavenumber (cm-1)')
        #fig.align_ylabels(axarr[:])
        plt.tight_layout() 
        plt.savefig('{}\\PCA_Loading_Plot.pdf'.format(save_to), bbox_inches='tight')
        print('PCA has been performed all data.\n', 
              'PC scores are {} for {} components.\n'.format(explained_variance, components),
              'PC loadings are {}'.format(pca_loadings))
    else:
        print('No PCA operations have been carried out.')


def plot_PCA(to_plot, plot_labels, title, xlabel, ylabel, ncol, colours):
    X_stack = np.column_stack((to_plot, plot_labels))
    X_stack = X_stack[np.argsort(X_stack[:,-1])]
    plt.figure(figsize=(1.59*2, 1.59*2))
    marker = itertools.cycle(('o', 's', '^', 'd', '*', '.', 'x', 'P'))
    #color=iter(cm.tab20b(np.linspace(0,1, X_stack[:,-1].max()+1 )))
    color = itertools.cycle(colours)#(('k', 'r', 'g', 'b', 'y')) ##(('k', 'k', 'k', 'k', 'r', 'r', 'r', 'r'))#
    for i in range(int(X_stack[:,-1].max()+1)):
        c=next(color)
        indicies = np.where(X_stack[:,-1] == i)#[0]
        plt.scatter(X_stack[indicies,0], X_stack[indicies,1], label=dict_name[i], s = 20, c=c, marker=next(marker))
    plt.ylabel('{} ({}%)'.format(ylabel, round(explained_variance[1], 2))) # Need to change number if PC scores are not PC1 vs PC2
    plt.xlabel('{} ({}%)'.format(xlabel, round(explained_variance[0], 2)))
    plt.title('{}'.format(title))
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.4, -0.22),  ncol= ncol)
    plt.tight_layout()
    plt.show()
    plt.savefig('{}\\PCA_Scores_Plot.pdf'.format(save_to), bbox_extra_artists=(lgd,), bbox_inches='tight')


###########################################################################################################
#####                   Set paramaters
###########################################################################################################
save_to = r' ' #copy and past the full file path

################# Importing and preprocessing data #################
set_directory(directory = r' ') # copy and past the full file path

names, df, df1 = get_data(1)

##X, y, new_labels, df2, dict_name = 
encode_temp_catagorical_data(separate_1 = '_', separate_2 = 'no', delete_1 = 0, delete_2 = 1) # remove excess file name paramters
#rename sample groups
dict_name = {0:'Sample X', 1:'Sample Y'}


#Results plotting PCA
apply_pca(components = 2, pca_type = 'pca_all', test_percent = 0.2, cv_folds=2) # pca_tt or pca_all
plot_PCA(to_plot = X_all_pca, plot_labels = y, title = 'Results', xlabel = 'PC1', ylabel = 'PC2', ncol =3, colours = ('k', 'r', 'g'))


