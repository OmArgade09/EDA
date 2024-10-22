# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:12:33 2024

@author: argad
"""

import pandas as pd
df=pd.read_csv("C:/5-data prep/ethnic diversity.csv.xls")

df.dtypes
##################################################################
df.Salaries=df.Salaries.astype(int)
df.dtypes

df.age=df.age.astype(float)
df.dtypes
###################################################################
df_new=pd.read_csv("C:/5-data prep/education.csv.xls")
duplicate=df_new.duplicated()
#output of this function is single column

duplicate
sum(duplicate)
#output will be 0
#Now let us import another dataset
df_new1=pd.read_csv("C:/5-data prep/mtcars_dup.csv.xls")
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)

df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
duplicate2
sum(duplicate2)

################################################################################

#Outliers
import pandas as pd
import seaborn as sns
df=pd.read_csv('C:/5-data prep/ethnic diversity.csv.xls')
sns.boxplot(df.Salaries)

sns.boxplot(df.age)

IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)

IQR


lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
#Now if we check lower limit salary ,its -19446.75
#Now if we check upper limit salary ,its 93992.8125

#TRIMMINGGGG'

import  numpy as np
import seaborn as sns 
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df] 
df.shape                    

df_trimmed.shape                     
                     

sns.boxplot(df_trimmed.Salaries)
                     
                     
 
#Replacement Technique
df_replaced=pd.DataFrame(np.where(df.Salaries > upper_limit,upper_limit,np.where(df.Salaries < lower_limit,lower_limit,df.Salaries)))                    

sns.boxplot(df_replaced[0])                     

#imp

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Salaries'])

df_t=winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])
#####################################################################################


import pandas as pd
df=pd.read_csv('C:/5-data prep/ethnic diversity.csv.xls')
df.var()
df.Salaries.var()==0
numeric_df= df.select_dtypes(include=[float,int])
numeric_df.var()

numeric_df.var(axis=0)==0

import pandas as pd
import numpy as np

df=pd.read_csv("C:/5-data prep/modified ethnic.csv.xls")
df.isna().sum()


#Creating impu

from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df['Salaries']= pd.DataFrame(mean_imputer.fit_transform(df[['Salaries']]))

df['Salaries'].isna().sum()
                     
#Median imputer
from sklearn.impute import SimpleImputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df['age']= pd.DataFrame(median_imputer.fit_transform(df[['age']]))

df['age'].isna().sum()
####
mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df['Sex']= pd.DataFrame(mode_imputer.fit_transform(df[['Sex']]))

df['Sex'].isna().sum()

df['MaritalDesc']= pd.DataFrame(mode_imputer.fit_transform(df[['MaritalDesc']]))

df['MaritalDesc'].isna().sum()

###################################################################

import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

X,y = make_classification(n_samples=1000, n_features=20,n_informative=2,n_redundant=10,n_clusters_per_class=1,weights=[0.99],flip_y=0,random_state=1)

'''
Parameters
                       n_samples=1000;
                       The total number of sample (data points) to generate
                       Here,1000 samples will be created
                       
                       n_features=20;
                       The total number of features (columns) in the dataset
                      Each sample will have 20 features
                      
                      n-informative=2;
                      the number of informative features
                      these featues are generated as random
                      
                      n_clusters_per class
                      


'''
print('Original Classification distribution',np.bincount(y))
smote = SMOTE(random_state=42)
X_res,y_res = smote.fit_resample(X,y)

print('resampled class distribution',np.bincount(y_res))

print('resampled class distribution : {np.bincount(y)}')

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

smote=SMOTE(random_state=42)
X_train_res,y_train_res=smote.fit_resample(X_train,y_train)

print(f'resampled class distribution : {np.bincount(y_train_res)}')

#####################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Generate a sample dataset
np.random.seed(0)
data = np.random.exponential(scale=2.0, size=1000)
df = pd.DataFrame(data, columns=['Value'])
#perform log tranformation
#gives the symmetrical distributed data
df['LogValue'] = np.log(df['Value'])
#Original data : left Skewed
fig, axes = plt.subplots(1, 2, figsize=(12,6))
#original data
axes[0].hist(df['Value'], bins=30, color='blue', alpha=0.7)
axes[0].set_title("Original data")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")

#Log-transformed data
axes[1].hist(df['LogValue'], bins=30, color='blue', alpha=0.7)
axes[1].set_title("Log transformed data")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
