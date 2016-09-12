import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from scipy.sparse import hstack
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.externals import joblib
from sklearn.svm import SVC
import xgboost as xgb



import time

d = defaultdict(LabelEncoder)

rf_filename = "RF/rf_50.pkl"
ohe_filename = "encoder/ohe.pkl"



def load_data(train_or_test="train"):
	if train_or_test != "train":
		train_or_test = "test"


	people_fname = "Data/people.csv"
	event_fname = "Data/act_{}.csv".format(train_or_test)


	people_df = pd.read_csv(people_fname,parse_dates=["date"])

	new_name_dict = {"people_id":"people_id"}
	for col in people_df.columns:
		if col not in new_name_dict:
			new_name_dict[col] = "p_"+col

	people_df = people_df.rename(columns=new_name_dict)
	
	events_df = pd.read_csv(event_fname,parse_dates=["date"])
	events_df = pd.merge(events_df,people_df,on="people_id")
	events_df["weekday"] = events_df["date"].apply(pd.tslib.Timestamp.weekday)
	events_df["date_gap"] = (events_df["date"] - events_df["p_date"]).apply(lambda x: x.days)
	
	events_df.fillna("type 0",inplace=True)

	return events_df




def input_target_split(events_df):
	target = events_df["outcome"]
	inp = events_df.drop("outcome",axis=1)

	return inp, target



def convert_to_one_hot(inp,train_or_test="train"):

	if train_or_test != "train":
		train_or_test = "test"

	if "people_id" in inp.columns:
		inp.drop(["people_id","activity_id","date","p_date"],axis=1,inplace=True)

	categorical_cols = []
	for col in inp.columns:
		if inp.dtypes[col] in ['bool','object']:
			categorical_cols.append(col)
	

	cat_inp = inp[categorical_cols]
	
	inp.drop(categorical_cols,axis=1,inplace=True)
	
	#this function transforms the categorical features to integer encodings
	#while saving the transforms that did the work for use on the validation/test sets
	cat_inp = cat_inp.apply(lambda x: d[x.name].fit_transform(x))

	if train_or_test == "train":
		enc = OneHotEncoder(handle_unknown="ignore")
		enc.fit(cat_inp)
		joblib.dump(enc,ohe_filename)
	else:
		enc = joblib.load(ohe_filename)


	cat_inp = enc.transform(cat_inp)


	inp = hstack([inp,cat_inp])	

	return inp


def xgb_fit():

	train_inp,valid_inp,train_target,valid_target = prepare_input()

	dtrain = xgb.DMatrix(train_inp,label=train_target)
	dvalid = xgb.DMatrix(valid_inp)


	param = {'max_depth':10, 'eta':0.02, 'silent':1, 'objective':'binary:logistic' }
	param['nthread'] = 4
	param['eval_metric'] = 'auc'
	param['subsample'] = 0.7
	param['colsample_bytree']= 0.7
	param['min_child_weight'] = 0
	param['booster'] = "gblinear"

	watchlist  = [(dtrain,'train')]
	num_round = 300
	early_stopping_rounds=10
	bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds)


	train_pred = bst.predict(xgb.DMatrix(train_inp))
	valid_pred = bst.predict(xgb.DMatrix(valid_inp))

	return bst,train_pred,train_target,valid_pred,valid_target




def rf_fit():

	train_inp,valid_inp,train_target,valid_target = prepare_input()

	rf = RandomForestClassifier(random_state=31,n_jobs=-1,verbose=1,n_estimators=100,min_samples_split=5)
	start = time.time()

	rf.fit(train_inp,train_target)

	end = time.time()
	print "fitting took {:0.4} seconds".format(end-start)

	training_output = rf.predict_proba(train_inp)
	validation_output = rf.predict_proba(valid_inp)

	training_error = log_loss(train_target,training_output)
	validation_error = log_loss(valid_target,validation_output)

	print "Train error: {:02.4f}".format(training_error)
	print "Validation error: {:02.4f}".format(validation_error)


	joblib.dump(rf,rf_filename)


	return rf


def prepare_input():
	print "loading data"
	events_df = load_data()
	print "splitting input and target" 
	inp,target = input_target_split(events_df)
	print "converting to one hot"
	inp = convert_to_one_hot(inp)
	print "splitting training and validation"
	train_inp,valid_inp,train_target,valid_target = train_test_split(inp,target,train_size=.8,random_state=31)
	return train_inp,valid_inp,train_target,valid_target


def prepare_submission():
	print "loading data"
	edf = load_data(train_or_test="test")
	submission_df = edf[["activity_id"]]
	print "converting to one hot"
	inp = convert_to_one_hot(edf,train_or_test="test")
	print "loading random forest"
	rf = joblib.load(rf_filename)
	print "making predictions"
	predictions = pd.DataFrame(rf.predict_proba(inp),columns=[0,1],index=edf.index)
	submission_df["outcome"] = predictions[1]
	submission_df.to_csv("submission.csv",index=False)

	



def rf_training_curve():
	
	train_inp,valid_inp,train_target,valid_target = prepare_input()

	rf = RandomForestClassifier(random_state=31,n_jobs=-1)
	
	generate_learning_curve(rf,train_inp,train_target,valid_inp,valid_target,1500000)


def generate_learning_curve(model,training_input,training_target,validation_input,validation_target,max_n):

	#proceed by thousands, total number of games in training set is between 12 and 13k
	training_sizes = range(max_n/10,max_n,max_n/10)
	
	
	train_error_list = []
	validation_error_list = []

	#train models with different trainset sizes and store the resulting training and validation set errors
	for size in training_sizes:
		(train_error,validation_error) = train_predict(model,training_input[:size],training_target[:size],validation_input,validation_target)
		train_error_list.append(train_error)
		validation_error_list.append(validation_error)
	
	#prepare pretty plot, Figure 4 in final report
	plt.plot(training_sizes,train_error_list,"b")
	plt.plot(training_sizes,validation_error_list,"r")
	plt.xlabel("Training Observations")
	plt.ylabel("Log Loss Error")
	blue_patch = mpatches.Patch(color='blue',label="Training Error")
	red_patch = mpatches.Patch(color='red',label="Validation Error")
	plt.legend(handles=[blue_patch,red_patch],loc="upper right")
	plt.title("Model Learning Curves")

	plt.show()



def rf_grid_search():

	train_inp,valid_inp,train_target,valid_target = prepare_input()
	#set up scorer for grid search. log-loss is error, not score, so set greater_is_better to false,
	#and log-loss requires a probability
	log_loss_scorer = make_scorer(log_loss,greater_is_better=False,needs_proba=True)

	train_inp = train_inp[:100000]
	train_target = train_target[:100000]

	start = time.time()
	random_forest = RandomForestClassifier(random_state=31)
	# r_forest_parameters = {'n_estimators' : [120,300,500,800,1200],'max_depth':[5,8,15,25,30,None],'max_features':['log2','sqrt',None],
	# 'min_samples_split':[1,2,5,10,15,100],'min_samples_leaf':[1,2,5,10]}
	
	#75.1 minutes to run with these paramters - 72 fits

	r_forest_parameters = {'min_samples_split':[2,5,10,20,50,100],'min_samples_leaf':[1,2,5,10,50,100]}
	#grid search too slow to not use all cores, and wayyyy too slow to have no output.
	r_forest_grid_obj = GridSearchCV(random_forest,r_forest_parameters,log_loss_scorer,verbose=2,n_jobs=-1)
	r_forest_grid_obj = r_forest_grid_obj.fit(train_inp,train_target)
	random_forest = r_forest_grid_obj.best_estimator_
	print "Best params: " + str(r_forest_grid_obj.best_params_)	
	random_forest_train_error = log_loss(train_target,random_forest.predict_proba(train_inp))
	random_forest_validation_error = log_loss(valid_target,random_forest.predict_proba(valid_inp))
	print "Best random forest training error: {:02.4f}".format(random_forest_train_error)
	print "Best random forest validation error: {:02.4f}".format(random_forest_validation_error)
	end = time.time()
	print "RF grid search took {:02.4f} seconds".format(end-start)

	return random_forest


def svm_grid_search():

	#get data
	training_input,training_target,validation_input,validation_target = prepare_input()

	#set up scorer for grid search. log-loss is error, not score, so set greater_is_better to false,
	#and log-loss requires a probability
	log_loss_scorer = make_scorer(log_loss,greater_is_better=False,needs_proba=True)

	training_input = training_input[:100000]
	training_target = training_target[:100000]

	print training_input.shape[0]
	print training_target.shape[0]

	start = time.time()
	svm = SVC(random_state=31,probability=True)
	
	
	svm_parameters = {'C':[.001,.01,.1,1,10,100],'kernel':["rbf","sigmoid"]}
	svm_grid_obj = GridSearchCV(svm,svm_parameters,log_loss_scorer,verbose=2,n_jobs=-1)
	svm_grid_obj = svm_grid_obj.fit(training_input,training_target)
	svm = svm_grid_obj.best_estimator_
	print "Best params: " + str(svm_grid_obj.best_params_)	
	svm_train_error = log_loss(training_target,svm.predict_proba(training_input))
	svm_validation_error = log_loss(validation_target,svm.predict_proba(validation_input))
	print "Best SVM training error: {:02.4f}".format(svm_train_error)
	print "Best SVM validation error: {:02.4f}".format(svm_validation_error)
	end = time.time()
	print "RF grid search took {:02.4f} seconds".format(end-start)

	return svm

def visualizations(events_df):

	plt.subplots(1)

	value_per_day = events_df[["date","outcome"]].groupby("date").mean()

	plt.plot(value_per_day.index,value_per_day["outcome"])
	plt.show()

	outcome_per_weekday = events_df[["weekday","outcome"]].groupby("weekday").mean()
	plt.bar(outcome_per_weekday.index,outcome_per_weekday["outcome"])
	plt.show()


	aggregations = ['mean','count']
	outcome_per_date_gap = events_df[["date_gap","outcome"]].groupby("date_gap").agg(aggregations)
	

	f,axarr = plt.subplots(2,sharex=True)
	axarr[0].plot(outcome_per_date_gap.index.days,outcome_per_date_gap["outcome"]["mean"])
	axarr[1].plot(outcome_per_date_gap.index.days,np.log(outcome_per_date_gap["outcome"]["count"]))
	
	plt.show()




def train_predict(clf,training_input,training_target,validation_input,validation_target): 
	print "Training a {} using a training set of size {}".format(clf.__class__.__name__,training_input.shape[0])

	train_classifier(clf,training_input,training_target)

	train_error = predict_labels(clf,training_input,training_target)
	validation_error = predict_labels(clf,validation_input,validation_target)

	print "log-loss for training set: {:5.4}".format(train_error)
	print "log-loss for validation set: {:5.4}".format(validation_error)
	return (train_error,validation_error)

def train_classifier(clf,training_input,training_target):
	start = time.time()
	clf.fit(training_input,training_target)
	end = time.time()

	print "Trained model in {:0.2f} seconds".format(end-start)

def predict_labels(clf, features, target):
	start = time.time()
	pred = clf.predict_proba(features)
	end = time.time()

	print "Made predictions in {:0.4f} seconds.".format(end - start)
	return log_loss(target,pred)


