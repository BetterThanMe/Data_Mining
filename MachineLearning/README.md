#Install package by command: $ sudo pip install ML-pkg-Minh

This package contain some machine learning model:
	- Support vector regression
	- Decision Tree Regression
	- Extra Tree Regresssion
	- Voting Tree Regression(with Decision and Extra Tree)

Input is a Dataframe your Bunch 

Features: is a array(or a number) of index columns of features in the dataframe

Target: is a index columns of feature to a output of the model

percent_train: percentage of train set in dataset: it's is random taking from dataset

The output will be a model(contain method like: predict,...) and score of test_data
	Except for VotingTreeRegression: the output is just model(with method like: 
					+ net(): return net of model
					+ score(X_test, y_test): return mean_squared_error
					+ predict(X_test): return y_predict
					...)

It will be develop more, if you have any trouble or suggestion, send email to my email address:
	
	minh.nq184293@sis.hust.edu.vn
		
