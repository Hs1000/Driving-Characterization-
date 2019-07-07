#importing Random Forests and Decision Trees Regressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

#Finding optimum values of parameters, basically hyperparameter tuning for random Forest Regressor
RF.fit(x_train,y_train)
pred_train_Random_forest = RF.predict(x_train)
pred_test_Random_forest = RF.predict(x_test)
RF = RandomForestRegressor(random_state = 1, n_jobs = -1) 
param_grid = { 
    'max_features' : ["auto", "sqrt", "log2"],
    'min_samples_split' : np.linspace(0.1, 1.0, 10),
     'max_depth' : [x for x in range(1,100)]}

#printing the value of hyperparameters
CV_rfc = RandomizedSearchCV(estimator=RF, param_distributions =param_grid, n_jobs = -1, cv= 10, n_iter = 50)
CV_rfc.fit(x_train, y_train)
CV_rfc.best_params_
CV_rfc.best_score_

DTR=DecisionTreeRegressor()
DTR.fit(x_train,y_train)
pred_train_Decision_trees = DTR.predict(x_train)
pred_test_Decision_trees = DTR.predict(x_test)
DTR = DecisionTreeRegressor(random_state = 1) 
param_grid = { 
    'max_features' : ["auto", "sqrt", "log2"],
    'min_samples_split' : np.linspace(0.1, 1.0, 10),
     'max_depth' : [x for x in range(1,100)]}

#parameter optimization for Decision Trees
CV_rfc = RandomizedSearchCV(estimator=DTR, param_distributions =param_grid, n_jobs = -1, cv= 10, n_iter = 50)
CV_rfc.fit(x_train, y_train)
CV_rfc.best_params_
CV_rfc.best_score_

#printing the value of hyperparameters
CV_rfc = RandomizedSearchCV(estimator=RF, param_distributions =param_grid, n_jobs = -1, cv= 10, n_iter = 50)
CV_rfc.fit(x_train, y_train)
CV_rfc.best_params_
CV_rfc.best_score_

#finding the score of each feature in the traing set for random Forest Regressor
clf1 = RandomForestRegressor(random_state=0,min_samples_split=0.4,max_features='auto',max_depth=34)
clf1 = clf.fit(x_train,y_train)
a1=(dict(zip(x.columns, clf1.feature_importances_)))

#finding the score of each feature in the traing set for Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(random_state=0,min_samples_split=0.1,max_features='auto',max_depth=17)
clf = clf.fit(x_train,y_train)
a1=(dict(zip(x.columns, clf.feature_importances_)))

#Visualizing the results
plt.subplots(figsize=(20,15))
plt.bar(*zip(*a1.items()))
plt.xticks( rotation='vertical')
plt.xlabel('importance_values')
plt.ylabel('name of features')
plt.show()

