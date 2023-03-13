# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/code/yaheaal/loan-status-with-different-models/notebook
Created on Mon Mar 13 12:07:46 2023

@author: TECHIE
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings     #https://docs.python.org/3/library/warnings.html#:~:text=Warning%20messages%20are%20typically%20issued,program%20uses%20an%20obsolete%20module.
warnings.filterwarnings('ignore')



#---------------------------------------------preprocessing----------------------------------------------------------


df = pd.read_csv(r'C:\Users\TECHIE\Desktop\Project\loan prediction\archive (1)\train_u6lujuX_CVtuZ9i.csv')

df.shape            #print the shape of dataframe

df.head()           # 1st 5 records of loaded data in df
# We got some categorical data, and it's a binary classification (Yes, NO)

df.info()
# We have missing data , we will handle them as we go

df.describe()       #https://www.w3schools.com/python/pandas/ref_df_describe.asp#:~:text=The%20describe()%20method%20returns,The%20average%20(mean)%20value.
# Describe the numerical data also check that Dtype is of type object(i.e.,'O')
# in this case Python object( dtype('O')) of numpy and pandas is equivalent to String(str) in Python datatype

df['Credit_History'].dtypes
df['Credit_History'] = df['Credit_History'].astype('O')  #as we want Credit_History to be categorical not numerical
# we will change the type of Credit_History to object becaues we can see that it is 1 or 0
df['Credit_History'].dtypes
#credit_history now converted to Python object(str)

df.describe(include='O')
# describe categorical data ("object") and print only the Pandas object(str) type of attributes

df.drop('Loan_ID', axis=1, inplace=True)
# we will drop ID because it's not important for our model and it will just mislead the model

df.duplicated().any()
# we got no duplicated rows(must be False to avoid redundant attributes.


#-------------------------------------------some visualizations-----------------------------------------------


plt.figure(figsize=(8,6))
sns.countplot(df['Loan_Status'])    #check for how many yes and how many is no

df['Loan_Status'].value_counts()    #print the unique values of the column ( 'Y' and 'N' here where there are 422 Y's and 192 N's)
                                    # more details: https://www.w3resource.com/pandas/series/series-value_counts.php#:~:text=The%20value_counts()%20function%20is,Excludes%20NA%20values%20by%20default.
#value_count()[0] gives 'Y's=422 as by default it's in decending order.And 
# let's look at the target percentage
print('The percentage of Y class : %.2f' % (df['Loan_Status'].value_counts()[0] / len(df))) #422/(422+192) 
print('The percentage of N class : %.2f' % (df['Loan_Status'].value_counts()[1] / len(df))) #192/(422+192)


#-------------------------------------let's look deeper in the data-------------------------------
# Credit_History

grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Credit_History')
# we didn't give a loan for most people who got Credit History = 0
# but we did give a loan for most of people who got Credit History = 1
# so we can say if you got Credit History = 1 , you will have better chance to get a loan
# important feature


# Gender

grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Gender')
#plt.figure(figsize=(15,5))
#sns.countplot(x='Gender', hue='Loan_Status', data=df)
# most males got loan and most females got one too so (No pattern)
# i think it's not so important feature, we will see later though it seems males have higher but we must know, females are lesser prone to take loans than male


# Married

plt.figure(figsize=(15,5))
sns.countplot(x='Married', hue='Loan_Status', data=df)
# most people who get married did get a loan
# if you'r married then you have better chance to get a loan :)
# good feature


# Dependents

plt.figure(figsize=(15,5))
sns.countplot(x='Dependents', hue='Loan_Status', data=df)
# first if Dependents = 0 , we got higher chance to get a loan ((very hight chance))
# good feature


# Education

grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Education')
# If you are graduated or not, you will get almost the same chance to get a loan (No pattern)
# Here you can see that most people did graduated, and most of them got a loan
# on the other hand, most of people who did't graduate also got a loan, but with less percentage from people who graduated
# not important feature


# Self_Employed

grid = sns.FacetGrid(df,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Self_Employed')
# No pattern (same as Education)


# Property_Area

plt.figure(figsize=(15,5))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df)
# We can say, Semiurban Property_Area got more than 50% chance to get a loan
# good feature


# ApplicantIncome

plt.scatter(df['ApplicantIncome'], df['Loan_Status'])
# No pattern


# the numerical data

df.groupby('Loan_Status').median() # median because Not affected with outliers
# we can see that when we got low median in CoapplicantInocme we got Loan_Status = N
# CoapplicantInocme is a good feature


#-----------------------------------------Simple process for the data----------------------------------------


#Missing values
#here i am just going to use a simple techniques to handle the missing data
df.isnull().sum().sort_values(ascending=False)
#isnull()-Detect missing values(if any value is missing gives True else False )
#sum()- gives the sumation of each attributes(concatenation if type is str)
#here isnull().sum() gives the summation of only those values where values are True, i.e., is null 


# We will separate the numerical columns from the categorical

cat_data = []
num_data = []
for i,c in enumerate(df.dtypes):
    if c == object:
        cat_data.append(df.iloc[:, i])
    else :
        num_data.append(df.iloc[:, i])
'''
i   c                   #cat_data has all object types as they are categorical
---------               #num_data has all numerical data
0 object
1 object
2 object
3 object
4 object
5 int64
6 float64
7 float64
8 float64
9 object
10 object
11 object
'''
'''
0 Series (614,)
1 Series (614,)
2 Series (614,)
3 Series (614,)
4 Series (614,)
5 Series (614,)
6 Series (614,)
7 Series (614,)
'''
cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()

cat_data.head()
num_data.head()

'''      Removing all missing data    '''

# cat_data
# If you want to fill every column with its own most frequent value you can use
cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
#Fill NA/NaN values using the specified method and fill all NAN/NA values with whichever is max in category or freq of NAN/NA
#say: 11	Male	Yes	2	graduate    NAN	1.0	Urban	Y
#to : 11	Male	Yes	2	Graduate	No	1.0	Urban	Y
# as at col 6, the highest number of values is No, NAN eplaced by No

cat_data.isnull().sum().any() # no more missing data out: False


# num_data
# fill every missing value with their previous value in the same column

num_data.fillna(method='bfill', inplace=True)
num_data.isnull().sum().any() # no more missing data out: False


#-------------------------------------Categorical data level encoded--------------------------------------


from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder()

# transform the target column

target_values = {'Y': 0 , 'N' : 1}
target = cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target = target.map(target_values)
#target:['Y', 'N', 'Y', 'Y', 'N'......] to [0, 1, 0, 0, 1........]
target.head()

#  level encode other columns

for i in cat_data:
    cat_data[i] = le.fit_transform(cat_data[i])

cat_data.head()

df = pd.concat([cat_data, num_data, target], axis=1)


#-----------------------------------------Training of data----------------------------------------------------------------


X = pd.concat([cat_data, num_data], axis=1)
y = target 

# we will use StratifiedShuffleSplit to split the data Taking into consideration that we will get the same ratio on the target column

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)