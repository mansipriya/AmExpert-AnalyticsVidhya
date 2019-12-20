# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:11:08 2019

@author: VE00YM015
"""
#importing required packages
import pandas as pd
import datetime
import numpy as np

#Loading data files
train = pd.read_csv('train.csv')
test = pd.read_csv('test_QyjYwdj.csv')
campaign_data = pd.read_csv('campaign_data.csv')
coupon_item_mapping = pd.read_csv('coupon_item_mapping.csv')
customer_demographics = pd.read_csv('customer_demographics.csv')
item_data = pd.read_csv('item_data.csv')
customer_transaction_data = pd.read_csv('customer_transaction_data.csv')

#merge train and test
data_merge = pd.concat([train[train.columns[0:4]],test], sort = False)

#changing date type of columns in campaign data
d = []
for date in campaign_data['start_date']:
    d.append(datetime.datetime.strptime(date,'%d/%m/%y'))   
campaign_data['start_date'] = d

d = []
for date in campaign_data['end_date']:
    d.append(datetime.datetime.strptime(date,'%d/%m/%y'))   
campaign_data['end_date'] = d

#getting duration of a campaign
d = []
for index, row in campaign_data.iterrows():
    d.append((row['end_date'] - row['start_date']).days)
campaign_data['duration'] = d


#merging campaign details in data merge
data_merge = pd.merge(data_merge,campaign_data,on = 'campaign_id')

#merging coupon item mapping with item data
coupon_item_mapping = pd.merge(coupon_item_mapping,item_data,on = 'item_id') 
#creating new features to put in train using item_id
unique = coupon_item_mapping.coupon_id.unique()
item_count = []
brand_count = []
i = 0
A = []
B = []
D = []
FL = []
FU = []
GA = []
GR = []
ME = []
MI = []
N = []
PA = []
PH = []
PR = []
R = []
SA = []
SE = []
SK = []
T = []
V = []
L = []
E = []
for id in unique:
    df = coupon_item_mapping[coupon_item_mapping['coupon_id'] == id]
    item_count.append(len(df['item_id']))
    brand_count.append(len(df.brand.unique()))
    l = e = 0
    a = b = d = fl = fu = ga = gr = me = mi = n = pa = ph = pr = r = sa = se = sk = t = v = 0
    for index, row in df.iterrows():
        if(row['brand_type'] == 'Established'):
            e = e + 1
        elif(row['brand_type'] == 'Local'):
            l = l + 1
        
        if(row['category'] == 'Flowers & Plants'):
            fl = fl + 1
        elif(row['category'] == 'Alcohol'):
            a = a + 1
        elif(row['category'] == 'Bakery'):
            b = b + 1
        elif(row['category'] == 'Dairy, Juices & Snacks'):
            d = d + 1
        elif(row['category'] == 'Fuel'):
            fu = fu + 1
        elif(row['category'] == 'Garden'):
            ga = ga + 1
        elif(row['category'] == 'Grocery'):
            gr = gr + 1
        elif(row['category'] == 'Meat'):
            me = me + 1
        elif(row['category'] == 'Miscellaneous'):
            mi = mi + 1
        elif(row['category'] == 'Natural Products'):
            n = n + 1
        elif(row['category'] == 'Packaged Meat'):
            pa = pa + 1
        elif(row['category'] == 'Pharmaceutical'):
            ph = ph + 1
        elif(row['category'] == 'Prepared Food'):
            pr = pr + 1
        elif(row['category'] == 'Restauarant'):
            r = r + 1
        elif(row['category'] == 'Salads'):
            sa = sa + 1
        elif(row['category'] == 'Seafood'):
            se = se + 1
        elif(row['category'] == 'Skin & Hair Care'):
            sk = sk + 1
        elif(row['category'] == 'Travel'):
            t = t + 1
        elif(row['category'] == 'Vegetables (cut)'):
            v = v + 1
    A.append(a)
    B.append(b)
    D.append(d)
    FL.append(fl)
    FU.append(fu)
    GA.append(ga)
    GR.append(gr)
    ME.append(me)
    MI.append(mi)
    N.append(n)
    PA.append(pa)
    PH.append(ph)
    PR.append(pr)
    R.append(r)
    SA.append(sa)
    SE.append(se)
    SK.append(sk)
    T.append(t)
    V.append(v)
    L.append(l)
    E.append(e)
    print(i)
    i = i +1
    
master = pd.DataFrame()
master['local_no'] = L
master['established_no'] = E
master['alcohol_no'] = A
master['bakery_no'] = B
master['dairy_no'] = D
master['flower_no'] = FL
master['fuel_no'] = FU
master['garden_no'] = GA
master['grocery_no'] = GR
master['meat_no'] = ME
master['misc_no'] = MI
master['natural_prod_no'] = N
master['packaged_no'] = PA
master['pharmaceutical_no'] = PH
master['prepared_no'] = PR
master['restaurant_no'] = R
master['salad_no'] = SA
master['seafood_no'] = SE    
master['skin_no'] = SK
master['travel_no'] = T
master['vegetable_no'] = V        
master['coupon_id'] = unique
########
#REMOVE        
##########       
data_merge.to_csv("data_merge.csv")
master.to_csv("master.csv")           
##########
#REMOVE        
##########  

data_merge = pd.merge(data_merge,master,on = 'coupon_id')

########
#REMOVE        
##########       
data_merge.to_csv("data_merge2.csv")          
##########
#REMOVE        
##########  
#####  

customer_mapping = pd.read_csv('CustomerMaster.csv')
data_merge = pd.read_csv('data_merge2.csv')
merge = pd.merge(customer_mapping,customer_demographics,on = 'customer_id',how = 'outer')
merge.family_size.fillna(2, inplace = True)
merge.no_of_children.fillna(0, inplace = True)
merge.income_bracket.fillna(5, inplace = True)
merge.marital_status.fillna("Not present", inplace = True)
merge.age_range.fillna("46-55", inplace = True)
merge.rented.fillna(0, inplace = True)
merge.income_bracket.value_counts()
merge2 = pd.merge(data_merge,merge,on = 'customer_id',how = 'outer')

items_bought = []

for index, row in merge2.iterrows():
    df_coupon = coupon_item_mapping[coupon_item_mapping['coupon_id'] == row['coupon_id']]
    df_cust_trans = customer_transaction_data[customer_transaction_data['customer_id'] == row['customer_id']]
    df_merge = pd.merge(df_coupon,df_cust_trans,how = 'inner', on = 'item_id')
    items_bought.append(len(df_merge))
    print(index)
    
merge2['items_bought'] = items_bought
        

merge2.drop(['Unnamed: 0_x'], axis = 1, inplace=True)
merge2.drop(['Unnamed: 0_y'], axis = 1, inplace=True)

merge2.drop(['campaign_id','coupon_id','customer_id','start_date','end_date'], axis = 1, inplace=True)

categorical = ['campaign_type','age_range','marital_status','rented']
for var in categorical:
    merge2 = pd.concat([merge2, pd.get_dummies(merge2[var], prefix=var)], axis=1)
    del merge2[var]
    
continuous = ['duration', 'local_no', 'established_no', 'alcohol_no',
       'bakery_no', 'dairy_no', 'flower_no', 'fuel_no', 'garden_no',
       'grocery_no', 'meat_no', 'misc_no', 'natural_prod_no', 'packaged_no',
       'pharmaceutical_no', 'prepared_no', 'restaurant_no', 'salad_no',
       'seafood_no', 'skin_no', 'travel_no', 'vegetable_no', 'cu_local_no',
       'cu_established_no', 'cu_alcohol_no', 'cu_bakery_no', 'cu_dairy_no',
       'cu_flower_no', 'cu_fuel_no', 'cu_garden_no', 'cu_grocery_no',
       'cu_meat_no', 'cu_misc_no', 'cu_natural_prod_no', 'cu_packaged_no',
       'cu_pharmaceutical_no', 'cu_prepared_no', 'cu_restaurant_no',
       'cu_salad_no', 'cu_seafood_no', 'cu_skin_no', 'cu_travel_no',
       'cu_vegetable_no', 'no_of_tran', 'no_of_days', 'qty', 'max_sp',
       'min_sp', 'avg_sp', 'max_od', 'min_od', 'avg_od', 'max_cd', 'min_cd',
       'avg_cd','income_bracket',
       'items_bought','rented_1.0']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

for var in continuous:
    merge2[var] = merge2[var].astype('float64')
    merge2[var] = scaler.fit_transform(merge2[var].values.reshape(-1, 1))

train_merge = merge2.iloc[:len(train)]
test_merge = merge2.iloc[len(train):]

train_merge.to_csv('train_merge.csv')
test_merge.to_csv('test_merge.csv')



from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [700,1000,850,1100],
    'max_features': ['auto', 'sqrt','log2'],
    'max_depth' : [7,8,9],
    'criterion' :['gini','entropy']
}
from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 2,verbose = 5, n_jobs = 2)
CV_rfc.fit(X_train, Y_train)
CV_rfc.best_params_
