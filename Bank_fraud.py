#!/usr/bin/env python
# coding: utf-8


## This code assumes installation of numpy, pandas, sklearn and category_encoders
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.utils import resample
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# DATA LOADING

data_types = {'eventID':'object', 'accountNumber':'object', 'merchantId':'object', 'mcc':'object', 'merchantCountry':'object',           'merchantZip':'object', 'posEntryMode':'object'}
posEntryMode_val = {'0':'Entry Mode Unknown', '1':'POS Entry Mode Manual', '2':'POS Entry Model Partial MSG Stripe',                    '5':'POS Entry Circuit Card', '7':'RFID Chip (Chip card processed using chip)',                    '80':'Chip Fallback to Magnetic Stripe', '81': 'POS Entry E-Commerce',                    '90':'POS Entry Full Magnetic Stripe Read', '91':'POS Entry Circuit Card Partial'}
# There is a posEntryMode 79, not listed in the dict

df = pd.read_csv('./data-new/transactions_obf.csv', parse_dates=['transactionTime'], dtype=data_types)
df_labels = pd.read_csv('./data-new/labels_obf.csv', parse_dates=['reportedTime'])


# PREPROCESSING

def assign_fraud(row):
    if row.eventId in df_labels.eventId.values:
        fraud = 1
    else: 
        fraud = 0
    return fraud

df['fraud'] = df.apply(assign_fraud, axis=1) # Assigning labels
df = df.set_index('eventId').fillna('nonUK') # Filling in na in the merchantZip column
df['month'] = df.transactionTime.dt.month.astype('object') # Adding month as a categoical variable

# Checking data types and non-null counts
print(df.info())

print('Total data size:', df.shape)
print('Total fraud transactions:', df[df.fraud==1].shape)
print('Time range:', df.transactionTime.min(), df.transactionTime.max())

for i in df.columns:
    print('Unique', i, df[i].unique().shape)


# ANALYSIS

## Plotting some of the variables

fig, ax = plt.subplots(1,2, figsize=(15,5))
df.transactionAmount.hist(bins=[0,50,100,500,1000,1500], ax=ax[0])
df[df.fraud==1].transactionAmount.hist(bins=[0,50,100,500,1000,1500], ax=ax[1])
ax[0].set_title('All transactions by amount')
ax[1].set_title('Fraud by amount')
ax[0].set_xlabel('Amount')
ax[1].set_xlabel('Amount')
ax[0].set_ylabel('Count')
plt.show()

fig, ax = plt.subplots(1,2, figsize=(15,5))
df[(df.transactionTime>='2017-02')].groupby('month').month.hist(color='b', width=0.5, ax=ax[0])
df[(df.fraud==1) & (df.transactionTime>='2017-02')].groupby('month').month.hist(color='b', width=0.5, ax=ax[1])
ax[0].set_title('Total number of transactions through the months')
ax[1].set_title('Fraud through the months')
ax[0].set_xlabel('Month')
ax[1].set_xlabel('Month')
ax[0].set_ylabel('Count')
plt.show()

df.replace(posEntryMode_val)[(df.fraud==1)&(df.posEntryMode!='79')].groupby('posEntryMode').fraud.count().div(df.replace(posEntryMode_val)[df.posEntryMode!='79'].groupby('posEntryMode').fraud.count())   .plot(kind='bar',figsize=(18,5), align='edge')
plt.xticks(rotation=45, size=12)
plt.ylabel('Fraud as proporion of transactions', size=12)
plt.title('Fraud by Point of Sale Entry')
plt.show()

df[df.fraud==1].groupby('merchantCountry').fraud.count().div(df.groupby('merchantCountry').fraud.count()).plot(kind='bar',figsize=(18,5), align='center')
plt.xticks(rotation=90, size=12)
plt.ylabel('Fraud as proporion of transactions', size=15)
plt.title('Fraud by merchant country (UK = 826)')
plt.show()


# MODELLING

## Data splitting

X_train_v, X_test, y_train_v, y_test = train_test_split(df.drop(columns='fraud'), df.fraud, test_size=0.1,                                                        random_state=0, stratify=df[['fraud','month']])
X_train, X_valid, y_train, y_valid = train_test_split(X_train_v, y_train_v, test_size=0.1,                                                      random_state=0, stratify=pd.concat([y_train_v, X_train_v.month], axis=1))

data_dict = {'train':y_train, 'valid':y_valid,'test':y_test}
for i in data_dict:
    print('Number of samples in', i,':',data_dict[i].shape[0],', of which are fraud:',data_dict[i].sum())


## Setting up the pipeline

class fraud_prob:
    def __init__(self, data_X, data_y):
        self.X = data_X
        self.y = data_y
        self.str_cols = [col for col in data_X.columns if data_X[col].dtype=='O']
        self.enc = ce.TargetEncoder(cols=self.str_cols, min_samples_leaf=20, smoothing=10)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = GradientBoostingClassifier(n_estimators=150, learning_rate=1, min_samples_leaf=2, random_state=0)
        
    def train(self, data_X=None, data_y=None, samp=7000):
        """Fitting the encoder and scaler with the given data, then balances the data by resampling before fitting the model"""
        if data_X is not None and data_y is not None:
            self.X = data_X
            self.y = data_y
        self.enc.fit(X=self.X[self.str_cols], y=self.y)
        X_enc = self.enc.transform(X=self.X[self.str_cols], y=self.y)
        X_enc = self.X[['transactionAmount', 'availableCash']].merge(X_enc, on='eventId')
        self.scaler.fit(X_enc)
        X_sc = self.scaler.transform(X_enc)
        X_up, y_up = resample(X_sc[self.y.values==1], self.y[self.y.values==1], n_samples=samp, replace=True, random_state=0)
        X_down, y_down = resample(X_sc[self.y.values==0], self.y[self.y.values==0], n_samples=samp, replace=False, random_state=0)
        X_train_all = np.append(X_up, X_down, axis=0)
        y_train_all = np.append(y_up, y_down, axis=0)
        self.model.fit(X_train_all, y_train_all)
        
    def predict(self, X, y=None):
        """Passing the data through the pipeline (encode, scaler, model)"""
        X_enc = self.enc.transform(X[self.str_cols])
        X_enc = X[['transactionAmount', 'availableCash']].merge(X_enc, on='eventId')
        X_sc = self.scaler.transform(X_enc)
        prob = self.model.predict_proba(X_sc)
        if y is not None:
            y_pred = self.model.predict(X_sc)
            print('Confusion matrix:')
            print(pd.crosstab(pd.Series(y.values, name='Actual'), pd.Series(y_pred, name='Predicted')))
        return prob


test01 = fraud_prob(X_train, y_train)
test01.train(samp=14000)

print('Validation data, with', y_valid.sum(), 'fraudulent transactions:')
prob = test01.predict(X_valid, y_valid)
print('Test data, with', y_test.sum(), 'fraudulent transactions:')
prob = test01.predict(X_test, y_test)