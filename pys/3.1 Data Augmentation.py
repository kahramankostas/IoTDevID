
# coding: utf-8

# ## This file first divides the files in the "csvs" folder into two as train and test. It then applies data augmentation for the required classes.

# ### Input & Output
# 
# `Input Files`: All files with the csv extension in the “./csvs/” folder is read.
# 
# `Output Files`: Divides input files as Train and Test. Creates augmented version of these train and test files.

# --------------

# ###  importing relevant libraries

# In[20]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import pandas as pd
import warnings
import os


# ### Discovering csv extension files under "csvs" folder.

# In[21]:


def find_the_way(path,file_format):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
name_list=find_the_way('./csvs','.csv')


# ### List of csv files to be processed

# In[22]:


name_list


# ### Split datasets train and test

# In[23]:


for name in name_list:    
    df=pd.read_csv(name)#,header=None) 
    X =df[df.columns[0:-1]]
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    y=df[df.columns[-1]]

    # setting up testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27,stratify=y)

    # concatenate our training data back together
    train = pd.concat([X_train, y_train], axis=1)


    file=name[0:-5]+"_"+"_train.csv"
    train.to_csv(file,index=False)


    test= pd.concat([X_test, y_test], axis=1)

    file=name[0:-5]+"_"+"_test.csv"
    test.to_csv(file,index=False)


# ### Discovering csv extension files under "csvs" folder dor augmentation.

# In[24]:


def find_the_way(path,file_format):
    files_add = []
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add
files_add=find_the_way('./csvs/','.csv')


# ### lists the labels and the number of label for the first file.

# In[25]:


df=pd.read_csv(files_add[0]) # 
devices=sorted(list(df[df.columns[-1]].unique()))
device_names={}
for i,ii in enumerate (devices):
    print(i,ii)
    device_names[i]=ii
#device_names
df.groupby("Label").size()


# # Data augmentation with  Resempling

# In[26]:


s=[]
for iii in files_add:
    print(iii)
    for ii in range (1):
        for i in device_names:
            print(device_names[i])
            df=pd.read_csv(iii)
            df_1= df[df['Label']==str(device_names[i])]
            df_1["Label"]=np.ones(df_1.shape[0])
            # upsample minority
            if "test" in iii:
                number=3000
            else:number=10000
            print(df_1.shape[0])
            #if sample<df_1.shape[0]:
            #if df_1.shape[0]>number:
            #    df_1 = df_1.sample(n=number)
           # else:
            df_1 = resample(df_1,
                                      replace=True, # sample with replacement
                                      n_samples=number, # match number in majority class
                                      random_state=27) # reproducible results
   
                
                
            y_train = df_1["Label"] #this section separates the label and the data into two separate pieces, as Label=y Data=X 
            del df_1["Label"]
            X_train = df_1
            st=device_names[i]
            s=[st for i in range(X_train.shape[0])]
            X_train["Label"]=s
            name=str(iii)+"_sk_resample.csv"
            X_train.to_csv(name, mode="a", index=False,header=False)

    


# # Data augmentation with SMOTE

# In[27]:


for iii in files_add:
    s=[]
    if "test" in iii:
        for ii in range (1):
            for i in device_names:
                print(device_names[i])
                df=pd.read_csv(iii)
                df_1= df[df['Label']==str(device_names[i])]
                df_1["Label"]=np.ones(df_1.shape[0])
                df_0= df[df['Label']!=(device_names[i])]
                df_0["Label"]=np.zeros(df_0.shape[0])
                number=3000
                df_0 = df_0.sample(n=number)
                df = pd.concat([df_1,df_0])
                df=df.reindex(np.random.permutation(df.index))
                y_train = df["Label"] #this section separates the label and the data into two separate pieces, as Label=y Data=X 
                del df["Label"]
                X_train = df
                sm = SMOTE(random_state=27, ratio=1.0)
                X_train, y_train = sm.fit_sample(X_train, y_train)
                X_train=pd.DataFrame(X_train)
                X_train["Label"]=y_train
                X_train= X_train[X_train['Label']==1]
                st=device_names[i]
                s=[st for i in range(X_train.shape[0])]

                X_train["Label"]=s
                print(X_train.shape)
                if X_train.shape[0]>number:
                    X_train = X_train.sample(n=number)
                print(X_train.shape)
                name=iii+"_smoote_resample.csv"
                X_train.to_csv(name, mode="a", index=False,header=False)

    else:
        for ii in range (1):
            for i in device_names:
                print(device_names[i])
                df=pd.read_csv(iii)
                df_1= df[df['Label']==str(device_names[i])]
                df_1["Label"]=np.ones(df_1.shape[0])
                df_0= df[df['Label']!=(device_names[i])]
                df_0["Label"]=np.zeros(df_0.shape[0])
                number=10000
                df_0 = df_0.sample(n=number)
                df = pd.concat([df_1,df_0])
                df=df.reindex(np.random.permutation(df.index))
                y_train = df["Label"] #this section separates the label and the data into two separate pieces, as Label=y Data=X 
                del df["Label"]
                X_train = df
                sm = SMOTE(random_state=27, ratio=1.0)
                X_train, y_train = sm.fit_sample(X_train, y_train)
                X_train=pd.DataFrame(X_train)
                X_train["Label"]=y_train
                X_train= X_train[X_train['Label']==1]
                st=device_names[i]
                s=[st for i in range(X_train.shape[0])]

                X_train["Label"]=s
                print(X_train.shape)
                if X_train.shape[0]>number:
                    X_train = X_train.sample(n=number)
                print(X_train.shape)
                name=iii+"_smoote_resample.csv"
                X_train.to_csv(name, mode="a", index=False,header=False)


    

