# import your packages you need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
import seaborn as sns
import re 
import psycopg2

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

pd.options.display.float_format = '{:.1f}'.format

#need more description 

# NOTE: The data cleaning and feature engineering for this data set is done offline in a separate file
# and is very time consuming (it uses tsfresh to extract comprehensive features) and takes 20 minutes to run
# therefore, this was all done via jupyter notebook and saved to drive and then loaded here for downsampling,
# train/test split, random forest, and  
# this file takes in the final_data.csv file which is "wide" (one row per account number) with tons and tons
# of features (so it has 4000+ columns)

def downsample_data():  #this loads and downsamples the final cleaned data
    final_data=pd.read_csv(r'C:\Users\Laura\Documents\Insight 2019 Docs\final_data.csv',index_col=0, low_memory=False)
    
    # downsample the non-late rows to class balance data 
    df_notlate=final_data[final_data['late_label']=='Less than 3 late payments']   
    df_late=final_data[final_data['late_label']!='Less than 3 late payments'] 
    df_notlate_sample=df_notlate.sample(n=555, random_state=42)  
    df_final=pd.concat([df_notlate_sample, df_late]) 
    # data to pass to train test split and random forest
    df_final=df_final.sample(frac=1)  
    return df_final

#downsample your data
df_final=downsample_data()

def train_rfc(df_final):
    # Separate out X and Y for train test split and random forest
    # Where y is the late label
    X = df_final.drop('late_label',axis=1)
    y = df_final['late_label'] 
   
    # random forest classifier setup
    rfc = RandomForestClassifier(n_estimators=100, max_features='log2', max_depth=7, criterion='entropy')  #Initialize with whatever parameters you want to

    #initialize accuracy,AUC, f1 score lists
    rfc_acc_list=[]
    rfc_auc_list=[]
    f1score_list=[]

    # repeat the train/test + RFC procedure 10 times using this loop
    n = 10  
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
            X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=.20)
        # train model and make predictions and probabilities
            rfc.fit(X_train, y_train) 
            predictions = rfc.predict(X_cv)
            rf_probs = rfc.predict_proba(X_cv)[:, 1]
            
            # Get and append accuracy, AUC, f1 score
            rfc_acc=metrics.accuracy_score(y_cv, predictions)
            rfc_acc_list.append(rfc_acc)
            
            rfc_auc=(roc_auc_score(y_cv, rf_probs))
            rfc_auc_list.append(rfc_auc)

            f1score=f1_score(y_cv, predictions, average='weighted')
            f1score_list.append(f1score)

    # mean AUC             
    rfc_auc_mean=np.mean(rfc_auc_list)
    return predictions, rf_probs, rfc_acc_list, rfc_auc_list, f1score_list, X_cv, y_cv

#do train test split and random forest
predictions, rf_probs, rfc_acc_list, rfc_auc_list, f1score_list, X_cv, y_cv=train_rfc(df_final)

#undo dummies, put together table to display for predicted late accounts
def who_didnt_pay(X_cv, rf_probs, predictions, y_cv):
    
    pd.options.display.float_format = '{:.1f}'.format
    # undo dummies from X, which is super annoying and done manually
    reverse=X_cv[['meter_size_2.0', 'meter_size_3.0', 'meter_size_4.0','meter_size_6.0', 'meter_size_8.0']]
    reverse['meter_size_2.0'] =  reverse['meter_size_2.0'].apply(lambda x: 2.0 if x==1 else 0) 
    reverse['meter_size_3.0'] =  reverse['meter_size_3.0'].apply(lambda x: 3.0 if x==1 else 0)
    reverse['meter_size_4.0'] =  reverse['meter_size_4.0'].apply(lambda x: 4.0 if x==1 else 0)
    reverse['meter_size_6.0'] =  reverse['meter_size_6.0'].apply(lambda x: 6.0 if x==1 else 0)
    reverse['meter_size_8.0'] =  reverse['meter_size_8.0'].apply(lambda x: 8.0 if x==1 else 0)

    reverse['meter_size']=pd.DataFrame(reverse.sum(axis=1))  #row sums 
    # returns the meter size for the smallest meters (this is dropped during the initial dummies
    # generation and needs to be re-generated)
    reverse['meter_size']=reverse['meter_size'].apply(lambda x: 1.5 if x==0.0 else x)  

    # meters, probs, and preds are temporary data frames that we are only using to align
    # with our indexes for the purpose of merging them together to make the accounts dataframe
    meters=pd.DataFrame(reverse['meter_size'])
    meters=meters.reset_index()
    meters.columns = ['id', 'meter_size']
    meters.index.names = ['order']  #reindex

    probs=pd.DataFrame(rf_probs)  #probabilities
    probs.columns = ['probabilities']  #rename
    probs.index.names = ['order']  #reindex

    preds=pd.DataFrame(predictions)  #predictions
    preds.columns = ['predictions']  #rename
    preds.index.names = ['order']    #reindex

    # merge meters, probs, preds
    accounts=pd.merge(probs, preds, how='outer', on='order')  
    accounts=pd.merge(accounts, meters, how='outer', on='order') 

    # generate percent likely from the probabilities
    # the reason it is 1-probabilities is because pandas is intepreting higher probabilities as non-late 
    # instead of late, so it has to be flipped around
    accounts['Percent Likely']=round(((1-accounts['probabilities'])*100),0) # display percent likely
    
    # last 4 digits of meter id, this should be its own little function
    accounts['id']=accounts['id'].map(lambda x: '{:.0f}'.format(x)) # change its format
    accounts['Account Number']=accounts['id'].apply(str) # turn into string
    # slice last 4, add asterisks
    accounts['Account Number']='**************'+ accounts['Account Number'].str.slice(-4)  

    #generate the final dataframe to display 
    latepeople=pd.DataFrame(accounts[predictions=='3 or more late payments'][['Account Number','Percent Likely','meter_size' ]] ) #late folks as df

    #save down to csv for flask to load
    latepeople.to_csv(r'C:\Users\Laura\Documents\Insight 2019 Docs\latepeople.csv')
    print(latepeople.head())

    return accounts, latepeople

accounts, latepeople=who_didnt_pay(X_cv, rf_probs, predictions, y_cv)