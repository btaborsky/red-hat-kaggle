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

import time

d = defaultdict(LabelEncoder)



def load_data():


	people_fname = "Data/people.csv"
	event_fname = "Data/act_train.csv"


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



def convert_to_one_hot(inp):

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

	
	enc = OneHotEncoder(handle_unknown="ignore")
	enc.fit(cat_inp)
	cat_inp = enc.transform(cat_inp)


	inp = hstack([inp,cat_inp])	

	return inp




def rf_fit(target,inp):

	train_inp,valid_inp,train_target,valid_target = train_test_split(inp,target,train_size=.8,random_state=31)

	rf = RandomForestClassifier(random_state=31,n_jobs=-1,verbose=1)
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


