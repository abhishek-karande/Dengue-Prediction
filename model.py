import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
import xgboost
import math
#data loading
features=pd.read_csv("dengue_features_train.csv")
features=features.fillna(features.mean())
labels=pd.read_csv("dengue_labels_train.csv")
features=features.drop(["city","year","weekofyear","week_start_date"],axis=1)
#feature scaling
min_max_scaler= preprocessing.MinMaxScaler()
features_scaled=pd.DataFrame(min_max_scaler.fit_transform(features))
features_scaled.columns=features.columns
labels=labels.drop(["city",'year','weekofyear'],axis=1)
dataset=features_scaled.join(labels)
x=features['station_avg_temp_c']
y=dataset['total_cases']
plt.scatter(x,y)
corrmat=dataset.corr()
sns.heatmap(corrmat,square=True,cmap='YlGnBu')
x_train,x_test,y_train,y_test= train_test_split(features_scaled,labels,test_size=0.25)


xgb=xgboost.XGBRegressor()
xgb.fit(x_train,y_train)
prediction=xgb.predict(x_test)
for i in range (len(prediction)):
    prediction[i]=math.ceil(prediction[i])
    prediction[i]=int(prediction[i])
from sklearn.metrics import mean_absolute_error
acc=mean_absolute_error(y_test,prediction)   