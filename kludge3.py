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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

pd.options.display.float_format = '{:.1f}'.format

#ATTENTION: The data cleaning and feature engineering for this data set is done offline in a separate file
#and is very time consuming (it uses tsfresh to extract comprehensive features) and takes 20 minutes to run
#therefore, this was all done via jupyter notebook and saved to drive and then loaded here lols
def week3_demo_v2():
    #read in final data  #THIS IS JUST FOR THE UBUNTU INSTANCE DO NOT RUN THIS AT HOME
    final_data=pd.read_csv(r'/home/ubuntu/masterflask/app/data/final_data.csv',index_col=0, low_memory=False)
    print('data for model loaded')    
    #resample your data
    df_first=final_data[final_data['late_label']=='Less than 3 late payments']   #those without charges
    df_second=final_data[final_data['late_label']!='Less than 3 late payments'] #those with charges
    df_test=df_first.sample(n=555, random_state=42)  #resample the first 
    df_final=pd.concat([df_test, df_second])  #concatenate
    df_final=df_final.sample(frac=1)  #final to pass 

    #############
    #y is just the late label
    X = df_final.drop('late_label',axis=1)
    y = df_final['late_label']#for train, test in kf.split(X):

    print('starting random forest')
    rfc = RandomForestClassifier(n_estimators=100, max_features='log2', max_depth=7, criterion='entropy')  #Initialize with whatever parameters you want to

    rfc_acc_list=[]
    rfc_auc_list=[]
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
            X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=.20)
        # train model and make predictions
            rfc.fit(X_train, y_train) 
            predictions = rfc.predict(X_cv)
            rf_probs = rfc.predict_proba(X_cv)[:, 1]
            print('finished predictions')  
            #results_from_rfc=classification_report(y_cv,predictions)
            #rfc_cfm=confusion_matrix(y_cv,predictions)
            rfc_acc=metrics.accuracy_score(y_cv, predictions)
            rfc_acc_list.append(rfc_acc)
            #print("Accuracy:",rfc_acc)
        
            #some auc test train

            rfc_auc=(roc_auc_score(y_cv, rf_probs))
            rfc_auc_list.append(rfc_auc)
            
            #print("AUC Score:",rfc_auc)
    
    rfc_auc_mean=round(np.mean(rfc_auc_list),3)  
    #diagnostics
    #important features
    #using x columns since final data also has the final late status, so thats why this is X as opposed dto final data
    #feature_imp = pd.Series(rfc.feature_importances_,index=X_cv.columns).sort_values(ascending=False).head(10)

    #undo dummies, which is super annoying and done manually
    reverse=X_cv[['meter_size_2.0', 'meter_size_3.0', 'meter_size_4.0','meter_size_6.0', 'meter_size_8.0']]
    reverse['meter_size_2.0'] =  reverse['meter_size_2.0'].apply(lambda x: 2.0 if x==1 else 0) 
    reverse['meter_size_3.0'] =  reverse['meter_size_3.0'].apply(lambda x: 3.0 if x==1 else 0)
    reverse['meter_size_4.0'] =  reverse['meter_size_4.0'].apply(lambda x: 4.0 if x==1 else 0)
    reverse['meter_size_6.0'] =  reverse['meter_size_6.0'].apply(lambda x: 6.0 if x==1 else 0)
    reverse['meter_size_8.0'] =  reverse['meter_size_8.0'].apply(lambda x: 8.0 if x==1 else 0)

    reverse['meter_size']=pd.DataFrame(reverse.sum(axis=1))  #row sums 
    reverse['meter_size']=reverse['meter_size'].apply(lambda x: 1.5 if x==0.0 else x)  #returns the meter size
    reverse.head(20)

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

    accounts=pd.merge(probs, preds, how='outer', on='order')  #merge
    accounts=pd.merge(accounts, meters, how='outer', on='order') #merge
    accounts['Percent Likely']=round(((1-accounts['probabilities'])*100),0)  #display percent likely
    #accounts=pd.merge(accounts, y_test, how='outer', on='id') #
    #last 4 digits of meter id
    accounts['id']=accounts['id'].map(lambda x: '{:.0f}'.format(x)) #change its format
    accounts['Account Number']=accounts['id'].apply(str) #turn into string
    accounts['Account Number']='**************'+ accounts['Account Number'].str.slice(-4)  #slice last 4, add asterisks
    #print('These accounts are likely going to be late:')
    #print(accounts[predictions=='3 or more late payments '][['id','meter_size' ]])
    latepeople=pd.DataFrame(accounts[predictions=='3 or more late payments'][['Account Number','Percent Likely','meter_size' ]] ) #late folks as df
    latepeople=latepeople.reset_index(drop=True)
    #latepeople.to_html(r'/home/ubuntu/masterflask/app/data/latepeople.html')
    latepeople.to_csv(r'/home/ubuntu/masterflask/app/data/latepeople.csv')
 
    print(latepeople.head())

  
    return rfc_auc_mean, latepeople
