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


df = pd.read_csv(r'C:\Users\TECHIE\Desktop\Project\loan prediction\LoanPredictor\archive (1)\train_u6lujuX_CVtuZ9i.csv')

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


#------------------------------------------Model Training------------------------------------------------------------------


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    'SVC': SVC(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=42)
}
###########################################################################################################################
# loss                                                                                                                    #
                                                                                                                          #
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score                             #

def loss(y_true, y_pred, retu=False):                                                                                     #
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if retu:
        return pre, rec, f1, loss, acc
    else:
        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))
        
###########################################################################################################################

def train_eval_train(models, X, y):
    for name, model in models.items():
        print(name,':')
        model.fit(X, y)
        loss(y, model.predict(X))
        print('-'*30)
        
train_eval_train(models, X_train, y_train)


#--------------------------------------------------Coss Validation-----------------------------------------------------------------


# train_eval_cross
# in the next cell i will be explaining this function

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

def train_eval_cross(models, X, y, folds):
    # we will change X & y to dataframe because we will use iloc (iloc don't work on numpy array)
    X = pd.DataFrame(X) 
    y = pd.DataFrame(y)
    idx = [' pre', ' rec', ' f1', ' loss', ' acc']
    for name, model in models.items():
        ls = []
        print(name,':')

        for train, test in folds.split(X, y):
            model.fit(X.iloc[train], y.iloc[train]) 
            y_pred = model.predict(X.iloc[test]) 
            ls.append(loss(y.iloc[test], y_pred, retu=True))
        print(pd.DataFrame(np.array(ls).mean(axis=0), index=idx)[0])  #[0] because we don't want to show the name of the column
        print('-'*30)
        
train_eval_cross(models, X_train, y_train, skf)

# ohhh, as i said SVC is just memorizing the data, and you can see that here DecisionTreeClassifier is better than LogisticRegression 


#----------------------------------------------Feature Engineering-----------------------------------------------------------------------------

data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);



#--------------------------------------------Creating new features(Taken for nerds which are the field experts)---------------------------------

# I will try to make some operations on some features, here I just tried diffrent operations on diffrent features,
# having experience in the field, and having knowledge about the data will also help

X_train['new_col'] = X_train['CoapplicantIncome'] / X_train['ApplicantIncome']  
X_train['new_col_2'] = X_train['LoanAmount'] * X_train['Loan_Amount_Term'] 
#Test those new features
data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);

# new_col 0.03 , new_col_2, 0.047
# not that much , but that will help us reduce the number of features


#------------------------------------------Dropping some columns as told by experts--------------------------------------------------------------

X_train.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)

train_eval_cross(models, X_train, y_train, skf)
# ok, SVC is improving

#------------------------------------------------------------------------------------------------------------------------------------------------


# first lets take a look at the value counts of every label

for i in range(X_train.shape[1]):
    print(X_train.iloc[:,i].value_counts(), end='\n------------------------------------------------\n')


#----------------------------------------we will work on the features that have varied values----------------------------------------------------


# new_col_2

# we can see we got right_skewed
# we can solve this problem with very simple statistical teqniq , by taking the logarithm of all the values
# because when data is normally distributed that will help improving our model

from scipy.stats import norm

fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(X_train['new_col_2'], ax=ax[0], fit=norm)
ax[0].set_title('new_col_2 before log')

X_train['new_col_2'] = np.log(X_train['new_col_2'])  # logarithm of all the values

sns.distplot(X_train['new_col_2'], ax=ax[1], fit=norm)
ax[1].set_title('new_col_2 after log')    

# now we will evaluate our models, and i will do that continuously ,
#so i don't need to mention that every time

train_eval_cross(models, X_train, y_train, skf)

# wooow our models improved really good by just doing the previous step


# new_col

# most of our data is 0 , so we will try to change other values to 1

print('before:')
print(X_train['new_col'].value_counts())

X_train['new_col'] = [x if x==0 else 1 for x in X_train['new_col']]
print('-'*50)
print('\nafter:')
print(X_train['new_col'].value_counts())

train_eval_cross(models, X_train, y_train, skf)
# ok we are improving our models as we go 

for i in range(X_train.shape[1]):
    print(X_train.iloc[:,i].value_counts(), end='\n------------------------------------------------\n')
    
# looks better
    
    
    
#------------------------------------------------Outliers-----------------------------------------------------------------------------


#There is different techniques to handle outliers, here we are going to use IQR

# we will use boxplot to detect outliers

sns.boxplot(X_train['new_col_2']);
plt.title('new_col_2 outliers', fontsize=15);
plt.xlabel('')

threshold = 1.5  # this number is hyper parameter , as much as you reduce it, as much as you remove more points
                 # you can just try different values the deafult value is (1.5) it works good for most cases
                 # but be careful, you don't want to try a small number because you may loss some important information from the data .
                 
#The method of IQR

new_col_2_out = X_train['new_col_2']
q25, q75 = np.percentile(new_col_2_out, 25), np.percentile(new_col_2_out, 75) # Q25, Q75
print('Quartile 25: {} , Quartile 75: {}'.format(q25, q75))

iqr = q75 - q25
print('iqr: {}'.format(iqr))

cut = iqr * threshold
lower, upper = q25 - cut, q75 + cut
print('Cut Off: {}'.format(cut))
print('Lower: {}'.format(lower))
print('Upper: {}'.format(upper))

outliers = [x for x in new_col_2_out if x < lower or x > upper]
print('Nubers of Outliers: {}'.format(len(outliers)))
print('outliers:{}'.format(outliers))

data_outliers = pd.concat([X_train, y_train], axis=1)
print('\nlen X_train before dropping the outliers', len(data_outliers))
data_outliers = data_outliers.drop(data_outliers[(data_outliers['new_col_2'] > upper) | (data_outliers['new_col_2'] < lower)].index)

print('len X_train before dropping the outliers', len(data_outliers))

#Remove calculation of loan status from train to the result DataFrame which is y_train
X_train = data_outliers.drop('Loan_Status', axis=1)
y_train = data_outliers['Loan_Status']

#Visualizing again

sns.boxplot(X_train['new_col_2']);
plt.title('new_col_2 without outliers', fontsize=15);
plt.xlabel('')
# good :)


#-------------------------------------------------features selection-------------------------------------------------------------------------

# Self_Employed got really bad corr (-0.00061) , let's try remove it and see what will happen

data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True)

#X_train.drop(['Self_Employed'], axis=1, inplace=True)

train_eval_cross(models, X_train, y_train, skf)

data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True)

#----------------------------------------------evaluate the models on Test_data(FINALLY)----------------------------------------------------

X_test_new = X_test.copy()

#Processing like we have done to Train Data
x = []

X_test_new['new_col'] = X_test_new['CoapplicantIncome'] / X_test_new['ApplicantIncome']  
X_test_new['new_col_2'] = X_test_new['LoanAmount'] * X_test_new['Loan_Amount_Term']
X_test_new.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount'], axis=1, inplace=True)

X_test_new['new_col_2'] = np.log(X_test_new['new_col_2'])

X_test_new['new_col'] = [x if x==0 else 1 for x in X_test_new['new_col']]


#-------------------------------------------Trying with the model------------------------------------------------------------------------------


for name,model in models.items():
    print(name, end=':\n')
    loss(y_test, model.predict(X_test_new))
    print('-'*40)

