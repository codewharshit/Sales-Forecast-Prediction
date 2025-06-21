import pandas as pd

df=pd.read_csv('sales_prediction.csv')
df.head()
X=df.drop(columns=['Item_Outlet_Sales'])
Y=df['Item_Outlet_Sales']
X.head()
X.isnull().sum()
mean=df['Item_Weight'].mean()
X_c=X.copy()
X_c.isnull().sum()
df[['Item_Type','Item_Weight']]

Pivot_weight=X_c.pivot_table(index='Item_Type',values='Item_Weight',aggfunc='mean').reset_index()
WeightType_map=dict(zip(Pivot_weight['Item_Type'],Pivot_weight['Item_Weight']))
WeightType_map

def insert_weights(data_frame):
    data_frame.loc[:,'Item_Weight']=data_frame.loc[:,'Item_Weight'].fillna(data_frame.loc[:,'Item_Type'].map(WeightType_map))
    return data_frame

X_c=insert_weights(X_c)
X_c.isnull().sum()
X_c.groupby(by=['Outlet_Size','Outlet_Location_Type']).size()
X_c.groupby(by=['Outlet_Size','Outlet_Type']).size()
OutletSize_Mapping = dict(X_c.groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode()[0]))
OutletSize_Mapping

def insert_size(data_frame):
    data_frame.loc[:,'Outlet_Size']=data_frame.loc[:,'Outlet_Size'].fillna(data_frame.loc[:,'Outlet_Type'].map(OutletSize_Mapping))
    return data_frame
  
X_c=insert_size(X_c)
X_c.isnull().sum()
X_c['Item_Fat_Content'].value_counts()
def fat_content(data_frame):
    data_frame['Item_Fat_Content']=data_frame['Item_Fat_Content'].replace({
        'LF':'Low Fat',        
        'reg': 'Regular',         
        'low fat' : 'Low Fat'    
    })
    return data_frame
  
X_c=fat_content(X_c)
X_c['Item_Fat_Content'].value_counts()
X_c['Item_Identifier'].str[:2].value_counts()
def gen_itemType(data_frame):
    data_frame['ItemType_gen']=data_frame['Item_Identifier'].str[:2]
    mapp={
        'FD':'Food',
        'NC':'Non-consumable',
        'DR':'Drink'
    }
    data_frame.replace(mapp,inplace=True)
    return data_frame
  
X_c=gen_itemType(X_c)
X_c.head()
X_c[['ItemType_gen','Item_Fat_Content']].value_counts()

def item_fat_content(data_frame):
    data_frame.loc[data_frame['ItemType_gen']=='Non-Consumable','Item_Fat_Content']='Non_Edible'
    return data_frame
X_c=item_fat_content(X_c)
X_c[['ItemType_gen','Item_Fat_Content']].value_counts()

def prepare_data(data_frame):
    data_frame=insert_weights(data_frame)
    data_frame=insert_size(data_frame)
    data_frame=fat_content(data_frame)
    data_frame=gen_itemType(data_frame)
    data_frame=item_fat_content(data_frame)
    return data_frame

from sklearn.model_selection import train_test_split
X=X_c
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X_train_c=X_train.copy()
X_train_c.head()

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X_train_c.dtypes
le_columns=X_train_c[['Outlet_Size','Outlet_Location_Type']]
le.fit(le_columns['Outlet_Size'])
le.fit(le_columns['Outlet_Location_Type'])

for column in X_train_c[['Outlet_Size','Outlet_Location_Type']]:
    X_train_c.loc[:,column]=le.fit_transform(X_train_c.loc[:,column])
X_train_c
cat_feats=X_train_c.select_dtypes(include='object').drop(columns=['Item_Identifier','Outlet_Identifier','Item_Type'])
num_feats=X_train.select_dtypes(exclude='object').reset_index()
ohe.fit(cat_feats)
ohe_features_names=ohe.get_feature_names_out(input_features=cat_feats.columns)
cat_feats_ohe=pd.DataFrame(ohe.transform(cat_feats).toarray(),columns=ohe_features_names)
cat_feats_ohe
X_train_final=pd.concat([cat_feats_ohe,num_feats],axis=1)
X_train_final.shape,X_test.shape
le_columns=X_test[['Outlet_Size','Outlet_Location_Type']]
le.fit(le_columns['Outlet_Size'])
le.fit(le_columns['Outlet_Location_Type'])

for column in X_test[['Outlet_Size','Outlet_Location_Type']]:
    X_test.loc[:,column]=le.fit_transform(X_test.loc[:,column])
    
cat_feats_test=X_test.select_dtypes(include='object').drop(columns=['Item_Identifier','Outlet_Identifier','Item_Type'])
num_feats_test=X_test.select_dtypes(exclude='object').reset_index()
ohe.fit(cat_feats_test)
ohe_features_names=ohe.get_feature_names_out(input_features=cat_feats_test.columns)
cat_feats_ohe_test=pd.DataFrame(ohe.transform(cat_feats_test).toarray(),columns=ohe_features_names)
X_test_final.isnull().sum()
X_test_final=pd.concat([cat_feats_ohe_test,num_feats_test],axis=1)
X_train_final=X_train_final.drop(columns='index')
X_test_final=X_test_final.drop(columns='index')
X_train_final.shape,X_test_final.shape

import matplotlib.pyplot as plt
import seaborn as sns
feat_imp = pd.Series(rf.feature_importances_, index=X_train_final.columns)
feat_imp.nlargest(15).plot(kind='barh')
plt.title("Top 15 Important Features")
plt.show()

from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor,GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def modelling(model,X_train,Y_train,X_test,Y_test,cv=10):
    cv_results=cross_validate(model,X_train,Y_train,cv=cv)
    model.fit(X_train,Y_train)
    y_pred=model.predict(X_test) 
    print('Model:', model)
    print("R²:", r2_score(Y_test,y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(Y_test,y_pred)))
  
from sklearn.model_selection import RandomizedSearchCV
def hyperparameter(model,parameters,cv):
    classifier=RandomizedSearchCV(model,parameters,cv=cv)
    classifier.fit(X_train_final,Y_train)
    print(model,classifier.best_params_)

params={
    'rf':{
    'n_estimators':[80,100,150],
    'max_depth':[3,5,10],
    'min_samples_leaf':[1,5,10]
},
'hgb':{
    'max_iter':[80,100,150,300],
    'max_depth':[3,5,7,10,50],
    'learning_rate':[0.1,0.3,0.5,0.8,0.9],
    'min_samples_leaf':[10,20,30,50]
},
'gbr':{
    'n_estimators':[80,100,150,300],
    'max_depth':[3,5,7,10,50],
    'learning_rate':[0.1,0.3,0.5,0.8,0.9],
    'min_samples_leaf':[1,5,10]
}
}
model_map = {
    'rf': RandomForestRegressor(),
    'gbr': GradientBoostingRegressor(),
    'hgb': HistGradientBoostingRegressor()
}
for key,model in model_map.items():
    hyperparameter(model,params[key],cv=5)
  
rf=RandomForestRegressor(max_depth=5, min_samples_leaf=5,random_state=42)
modelling(rf, X_train_final, Y_train,X_test_final,Y_test)
hgr=HistGradientBoostingRegressor(min_samples_leaf= 20, max_iter= 100,max_depth= 3,learning_rate= 0.1,random_state=42)
modelling(hgr,X_train_final, Y_train,X_test_final,Y_test)
gbr=GradientBoostingRegressor(n_estimators= 150, min_samples_leaf=5, max_depth=10,learning_rate= 0.1,random_state=42)
modelling(gbr,X_train_final, Y_train,X_test_final,Y_test)
