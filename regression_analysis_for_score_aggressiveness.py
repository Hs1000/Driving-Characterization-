df=pd.read_csv('Final2.csv')
df=df.drop('Unnamed: 0',axis=1)
df = df[df.columns.drop(list(df.filter(regex='timestamp')))]
df=df.dropna(thresh=len(df)-2200, axis=1)
df=df.drop('bluetooth_vehicle_detected',axis=1)
df=df.drop('wifi_enabled',axis=1)
df=df.drop('accel_max_turn',axis=1)
df=df.drop('trip_start_reason',axis=1)
df=df.drop('trip_end_reason',axis=1)
df1=df

df1=df1.drop(['min_motion_user_accel_x_25%','min_motion_user_accel_x_50%','min_motion_user_accel_x_75%'],axis=1)

#importing libraries used in training and testing upto model creation
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestRegressor
scaler=StandardScaler()

#filling up the columns having nan values with mean of the columns
df['max_motion_user_accel_z_max'].fillna(df1['max_motion_user_accel_z_25%'].mean(skipna=True),inplace=True)
df1['max_motion_user_accel_z_50%'].fillna(df1['max_motion_user_accel_z_50%'].mean(skipna=True),inplace=True)
df1['max_motion_user_accel_z_75%'].fillna(df1['max_motion_user_accel_z_75%'].mean(skipna=True),inplace=True)
df1['max_motion_user_accel_z_max'].fillna(df1['max_motion_user_accel_z_max'].mean(skipna=True),inplace=True)
df1['max_motion_user_accel_z_min'].fillna(df1['max_motion_user_accel_z_min'].mean(skipna=True),inplace=True)
df1['max_motion_user_accel_z_mean'].fillna(df1['max_motion_user_accel_z_mean'].mean(skipna=True),inplace=True)
df1['max_motion_user_accel_z_std'].fillna(df1['max_motion_user_accel_z_std'].mean(skipna=True),inplace=True)

df1['max_motion_user_accel_x_25%'].fillna(df1['min_motion_user_accel_x_25%'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_x_50%'].fillna(df1['min_motion_user_accel_x_50%'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_x_75%'].fillna(df1['min_motion_user_accel_x_75%'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_x_max'].fillna(df1['min_motion_user_accel_x_max'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_x_min'].fillna(df1['min_motion_user_accel_x_min'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_x_mean'].fillna(df1['min_motion_user_accel_x_mean'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_x_std'].fillna(df1['min_motion_user_accel_x_std'].mean(skipna=True),inplace=True)

df1['abrupt_acceleration_event_25%'].fillna(df1['abrupt_acceleration_event_25%'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_50%'].fillna(df1['abrupt_acceleration_event_50%'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_75%'].fillna(df1['abrupt_acceleration_event_75%'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_count'].fillna(df1['abrupt_acceleration_event_count'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_freq'].fillna(df1['abrupt_acceleration_event_freq'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_max'].fillna(df1['abrupt_acceleration_event_max'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_min'].fillna(df1['abrupt_acceleration_event_min'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_mean'].fillna(df1['abrupt_acceleration_event_mean'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_std'].fillna(df1['abrupt_acceleration_event_std'].mean(skipna=True),inplace=True)
df1['abrupt_acceleration_event_top'].fillna(df1['abrupt_acceleration_event_top'].mean(skipna=True),inplace=True)

df1['instant_speed_25%'].fillna(df1['instant_speed_25%'].mean(skipna=True),inplace=True)
df1['instant_speed_50%'].fillna(df1['instant_speed_50%'].mean(skipna=True),inplace=True)
df1['instant_speed_75%'].fillna(df1['instant_speed_75%'].mean(skipna=True),inplace=True)
df1['instant_speed_count'].fillna(df1['instant_speed_count'].mean(skipna=True),inplace=True)
df1['instant_speed_freq'].fillna(df1['instant_speed_freq'].mean(skipna=True),inplace=True)
df1['instant_speed_min'].fillna(df1['instant_speed_min'].mean(skipna=True),inplace=True)
df1['instant_speed_mean'].fillna(df1['instant_speed_mean'].mean(skipna=True),inplace=True)
df1['instant_speed_std'].fillna(df1['instant_speed_std'].mean(skipna=True),inplace=True)
df1['instant_speed_top'].fillna(df1['instant_speed_top'].mean(skipna=True),inplace=True)
df1['instant_speed_max'].fillna(df1['instant_speed_max'].mean(skipna=True),inplace=True)



df1['max_motion_user_accel_x_count'].fillna(df1['max_motion_user_accel_x_count'].mean(skipna=True),inplace=True)
df1['max_motion_user_accel_y_count'].fillna(df1['max_motion_user_accel_y_count'].mean(skipna=True),inplace=True)
df1['max_motion_user_accel_z_count'].fillna(df1['max_motion_user_accel_z_count'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_x_count'].fillna(df1['min_motion_user_accel_x_count'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_y_count'].fillna(df1['min_motion_user_accel_y_count'].mean(skipna=True),inplace=True)
df1['min_motion_user_accel_z_count'].fillna(df1['min_motion_user_accel_z_count'].mean(skipna=True),inplace=True)


df1['motion_rot_rate_x_count'].fillna(df1['motion_rot_rate_x_count'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_y_count'].fillna(df1['motion_rot_rate_y_count'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_z_count'].fillna(df1['motion_rot_rate_z_count'].mean(skipna=True),inplace=True)

df1['motion_user_accel_x_count'].fillna(df1['motion_user_accel_x_count'].mean(skipna=True),inplace=True)
df1['motion_user_accel_y_count'].fillna(df1['motion_user_accel_y_count'].mean(skipna=True),inplace=True)
df1['motion_user_accel_z_count'].fillna(df1['motion_user_accel_z_count'].mean(skipna=True),inplace=True)


df1['motion_rot_rate_x_25%'].fillna(df1['motion_rot_rate_x_25%'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_x_50%'].fillna(df1['motion_rot_rate_x_50%'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_x_75%'].fillna(df1['motion_rot_rate_x_75%'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_x_max'].fillna(df1['motion_rot_rate_x_max'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_x_min'].fillna(df1['motion_rot_rate_x_min'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_x_mean'].fillna(df1['motion_rot_rate_x_mean'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_x_std'].fillna(df1['motion_rot_rate_x_std'].mean(skipna=True),inplace=True)

df1['motion_rot_rate_y_25%'].fillna(df1['motion_rot_rate_y_25%'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_y_50%'].fillna(df1['motion_rot_rate_y_50%'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_y_75%'].fillna(df1['motion_rot_rate_y_75%'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_y_max'].fillna(df1['motion_rot_rate_y_max'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_y_min'].fillna(df1['motion_rot_rate_y_min'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_y_mean'].fillna(df1['motion_rot_rate_y_mean'].mean(skipna=True),inplace=True)
df1['motion_rot_rate_y_std'].fillna(df1['motion_rot_rate_y_std'].mean(skipna=True),inplace=True)


df1['motion_user_accel_x_25%'].fillna(df1['motion_user_accel_x_25%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_x_50%'].fillna(df1['motion_user_accel_x_50%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_x_75%'].fillna(df1['motion_user_accel_x_75%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_x_max'].fillna(df1['motion_user_accel_x_max'].mean(skipna=True),inplace=True)
df1['motion_user_accel_x_min'].fillna(df1['motion_user_accel_x_min'].mean(skipna=True),inplace=True)
df1['motion_user_accel_x_mean'].fillna(df1['motion_user_accel_x_mean'].mean(skipna=True),inplace=True)
df1['motion_user_accel_x_std'].fillna(df1['motion_user_accel_x_std'].mean(skipna=True),inplace=True)

df1['motion_user_accel_y_25%'].fillna(df1['motion_user_accel_y_25%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_y_50%'].fillna(df1['motion_user_accel_y_50%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_y_75%'].fillna(df1['motion_user_accel_y_75%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_y_max'].fillna(df1['motion_user_accel_y_max'].mean(skipna=True),inplace=True)
df1['motion_user_accel_y_min'].fillna(df1['motion_user_accel_y_min'].mean(skipna=True),inplace=True)
df1['motion_user_accel_y_mean'].fillna(df1['motion_user_accel_y_mean'].mean(skipna=True),inplace=True)
df1['motion_user_accel_y_std'].fillna(df1['motion_user_accel_y_std'].mean(skipna=True),inplace=True)

df1['motion_user_accel_z_25%'].fillna(df1['motion_user_accel_z_25%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_z_50%'].fillna(df1['motion_user_accel_z_50%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_z_75%'].fillna(df1['motion_user_accel_z_75%'].mean(skipna=True),inplace=True)
df1['motion_user_accel_z_max'].fillna(df1['motion_user_accel_z_max'].mean(skipna=True),inplace=True)
df1['motion_user_accel_z_min'].fillna(df1['motion_user_accel_z_min'].mean(skipna=True),inplace=True)
df1['motion_user_accel_z_mean'].fillna(df1['motion_user_accel_z_mean'].mean(skipna=True),inplace=True)
df1['motion_user_accel_z_std'].fillna(df1['motion_user_accel_z_std'].mean(skipna=True),inplace=True)

df1['hard_braking_event_std'].fillna(df1['hard_braking_event_std'].mean(skipna=True),inplace=True)
df1['hard_braking_event_25%'].fillna(df1['hard_braking_event_25%'].mean(skipna=True),inplace=True)
df1['hard_braking_event_50%'].fillna(df1['hard_braking_event_50%'].mean(skipna=True),inplace=True)
df1['hard_braking_event_75%'].fillna(df1['hard_braking_event_75%'].mean(skipna=True),inplace=True)
df1['hard_braking_event_max'].fillna(df1['hard_braking_event_max'].mean(skipna=True),inplace=True)
df1['hard_braking_event_min'].fillna(df1['hard_braking_event_min'].mean(skipna=True),inplace=True)
df1['hard_braking_event_top'].fillna(df1['hard_braking_event_top'].mean(skipna=True),inplace=True)
df1['hard_braking_event_mean'].fillna(df1['hard_braking_event_mean'].mean(skipna=True),inplace=True)

df1['hard_cornering_event_std'].fillna(df1['hard_cornering_event_std'].mean(skipna=True),inplace=True)
df1['hard_cornering_event_25%'].fillna(df1['hard_cornering_event_25%'].mean(skipna=True),inplace=True)
df1['hard_cornering_event_50%'].fillna(df1['hard_cornering_event_50%'].mean(skipna=True),inplace=True)
df1['hard_cornering_event_75%'].fillna(df1['hard_cornering_event_75%'].mean(skipna=True),inplace=True)
df1['hard_cornering_event_max'].fillna(df1['hard_cornering_event_max'].mean(skipna=True),inplace=True)
df1['hard_cornering_event_min'].fillna(df1['hard_cornering_event_min'].mean(skipna=True),inplace=True)
df1['hard_cornering_event_top'].fillna(df1['hard_cornering_event_top'].mean(skipna=True),inplace=True)
df1['hard_cornering_event_mean'].fillna(df1['hard_cornering_event_mean'].mean(skipna=True),inplace=True)

df1['min_motion_user_accel_y_25%'].fillna(df1['min_motion_user_accel_y_25%'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_y_50%'].fillna(df1['min_motion_user_accel_y_50%'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_y_75%'].fillna(df1['min_motion_user_accel_y_75%'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_y_max'].fillna(df1['min_motion_user_accel_y_max'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_y_min'].fillna(df1['min_motion_user_accel_y_min'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_y_mean'].fillna(df1['min_motion_user_accel_y_mean'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_y_std'].fillna(df1['min_motion_user_accel_y_std'].astype('float32').mean(skipna=True),inplace=True)

df1['min_motion_user_accel_z_25%'].fillna(df1['min_motion_user_accel_z_25%'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_z_50%'].fillna(df1['min_motion_user_accel_z_50%'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_z_75%'].fillna(df1['min_motion_user_accel_z_75%'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_z_max'].fillna(df1['min_motion_user_accel_z_max'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_z_min'].fillna(df1['min_motion_user_accel_z_min'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_z_mean'].fillna(df1['min_motion_user_accel_z_mean'].astype('float32').mean(skipna=True),inplace=True)
df1['min_motion_user_accel_z_std'].fillna(df1['min_motion_user_accel_z_std'].astype('float32').mean(skipna=True),inplace=True)


df_final=df_final.drop('Unnamed: 0',axis=1)
a=df1.isnull().sum()

#defining the training variables from the dataset
x=df1[['abrupt_acceleration_event_count','abrupt_acceleration_events','accel_max_accel','accel_max_brake','average_speed','distance_km',
      'hard_braking_event_count','hard_braking_events','hard_cornering_event_count','hard_cornering_events','max_altitude','max_motion_user_accel_x_count','max_motion_user_accel_y_count','max_motion_user_accel_z_count','max_speed',
      'min_motion_user_accel_x_count','min_motion_user_accel_y_count','min_motion_user_accel_z_count',
      'motion_rot_rate_x_count','motion_rot_rate_y_count','motion_rot_rate_z_count','motion_user_accel_x_count',
      'motion_user_accel_y_count','motion_user_accel_z_count','abrupt_acceleration_event_25%','abrupt_acceleration_event_50%','abrupt_acceleration_event_75%','instant_speed_count',
      'abrupt_acceleration_event_freq', 'abrupt_acceleration_event_max','abrupt_acceleration_event_mean', 'abrupt_acceleration_event_min','abrupt_acceleration_event_std', 'abrupt_acceleration_event_top','instant_speed_25%',
      'instant_speed_50%','instant_speed_75%','instant_speed_freq','instant_speed_max','instant_speed_min','instant_speed_mean','instant_speed_std','instant_speed_top',
      'motion_rot_rate_x_25%','motion_rot_rate_x_50%','motion_rot_rate_x_75%','motion_rot_rate_x_max','motion_rot_rate_x_min','motion_rot_rate_x_std','motion_rot_rate_x_mean','hard_braking_event_25%','hard_braking_event_50%','hard_braking_event_75%',
      'hard_braking_event_max','hard_braking_event_min','hard_braking_event_mean','hard_braking_event_std','hard_braking_event_top','hard_cornering_event_top','hard_cornering_event_std','hard_cornering_event_25%','hard_cornering_event_50%','hard_cornering_event_75%',
      'hard_cornering_event_max','hard_cornering_event_min','hard_cornering_event_mean','motion_rot_rate_y_25%','motion_rot_rate_y_50%','motion_rot_rate_y_75%','motion_rot_rate_y_min','motion_rot_rate_y_max','motion_rot_rate_y_std','motion_rot_rate_y_mean',
      'motion_rot_rate_z_25%','motion_rot_rate_z_50%','motion_rot_rate_z_75%','motion_rot_rate_z_min','motion_rot_rate_z_max','motion_rot_rate_z_std','motion_rot_rate_z_mean','motion_user_accel_x_25%','motion_user_accel_x_50%','motion_user_accel_x_75%','motion_user_accel_x_mean',
      'motion_user_accel_x_max','motion_user_accel_x_min','motion_user_accel_x_std','motion_user_accel_y_25%','motion_user_accel_y_50%','motion_user_accel_y_75%','motion_user_accel_y_mean',
      'motion_user_accel_y_max','motion_user_accel_y_min','motion_user_accel_y_std','motion_user_accel_z_25%','motion_user_accel_z_50%','motion_user_accel_z_75%','motion_user_accel_z_mean','motion_user_accel_z_max','motion_user_accel_z_min','motion_user_accel_z_std']]

x=df1[['abrupt_acceleration_event_count','abrupt_acceleration_events','accel_max_accel','accel_max_brake','average_speed','distance_km',
      'hard_braking_event_count','hard_braking_events','hard_cornering_event_count','hard_cornering_events','max_altitude','max_motion_user_accel_x_count','max_motion_user_accel_y_count','max_motion_user_accel_z_count','max_speed',]]
#Target variable 
y=df1[['score_defensive_driving']]

#using train_test_split to split the data into training and testing set
(x_train,x_test,y_train,y_test)=train_test_split(x,y,test_size=0.2,random_state=42)

#creating a Random Forest Regression model and testing by calculating rmse
clf=RandomForestRegressor(min_samples_split=0.1,max_features='auto',max_depth=96)
clf.fit(x_train,y_train)
pred_train_Random_forest=clf.predict(x_train)
N = len(y_train)
rmse_train_Random_forest = np.sqrt(np.sum((np.array(y_train).flatten() - np.array(pred_train_Random_forest).flatten())**2)/N)

clf=RandomForestRegressor(min_samples_split=0.1,max_features='auto',max_depth=96)
clf.fit(x_train,y_train)
pred_test_Random_forest=clf.predict(x_test)
N = len(y_test)
rmse_test_Random_forest = np.sqrt(np.sum((np.array(y_test).flatten() - np.array(pred_test_Random_forest).flatten())**2)/N)

#visualizing through a scatter plot
plt.scatter(pred_test_Random_forest,y_test)
plt.ylabel('test_set')
plt.xlabel('predictions')
plt.title('comparison between real score and predicted score for score_defensive_driving')

#creating a Decision Tree Regression model and testing by calculating rmse
regressor = DecisionTreeRegressor(random_state = 0,min_samples_split=0.2,max_features='auto',max_depth=29)  
regressor.fit(x_train,y_train)
pred_train_Decision_trees=regressor.predict(x_train)
N = len(y_train)
rmse_train_Decision_trees = np.sqrt(np.sum((np.array(y_train).flatten() - np.array(pred_train_Decision_trees).flatten())**2)/N)

regressor = DecisionTreeRegressor(random_state = 0)  
regressor.fit(x_train,y_train)
pred_test_Decision_trees=regressor.predict(x_test)
N = len(y_test)
rmse_test_Decision_trees = np.sqrt(np.sum((np.array(y_test).flatten() - np.array(pred_test_Decision_trees).flatten())**2)/N)

#Visualizing through a scatter plot
plt.scatter(pred_test_Decision_trees,y_test)
plt.ylabel('test_set')
plt.xlabel('predictions')
plt.title('comparison between real score and predicted score for score_defensive_driving')


#creating a Logistic Regression model and testing by calculating rmse
log=LogisticRegression()
log.fit(x_train,y_train)       
pred_test_Logistic=log.predict(x_test)
N = len(y_test)
rmse_test_Logistic = np.sqrt(np.sum((np.array(y_test).flatten() - np.array(pred_test_Logistic).flatten())**2)/N)

log=LogisticRegression()
log.fit(x_train,y_train)       
pred_train_Logistic=log.predict(x_train)
N = len(y_train)
rmse_train_Logistic = np.sqrt(np.sum((np.array(y_train).flatten() - np.array(pred_train_Logistic).flatten())**2)/N)

#visualizing using a scatter plot
plt.scatter(pred_test_Logistic,y_test)
plt.ylabel('test_set')
plt.xlabel('predictions')
plt.title('comparison between real score and predicted score')

df1.to_csv('dataframe.csv',sep=',')
x.columns
df2=pd.read_csv('dataframe.csv')
df2=df2.drop('Unnamed: 0',axis=1)


