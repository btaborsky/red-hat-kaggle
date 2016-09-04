import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict

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
	events_df["date_gap"] = events_df["date"] - events_df["p_date"]
	
	return events_df




def input_target_split(events_df):
	target = events_df["outcome"]
	inp = events_df.drop("outcome",axis=1)

	return inp, target



def convert_to_one_hot(inp):


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

	#some features have too many categories and will retain their integer values rather than
	#getting one-hot encoded.

	ordinal_cols = ["char_1","char_2","char_10","p_group_1","p_char_3","p_char_4","p_char_7"] 

	inp=pd.concat([inp,cat_inp[ordinal_cols]],axis=1)
	# print cat_inp.columns
	cat_inp.drop(ordinal_cols,inplace=True,axis=1)
	# print cat_inp.columns
	# print cat_inp.apply(pd.Series.nunique).sum()


	enc = OneHotEncoder(sparse=False)
	enc.fit(cat_inp)
	cat_inp = enc.transform(cat_inp)

	

	inp=pd.concat([inp,pd.DataFrame(cat_inp)],axis=1)

	return inp


def preprocess_inp(events_df):

	inp=events_df.drop(["people_id","activity_id","date","p_date"],axis=1)
	return inp




# def rf_fit(target,inp):



# 	inp_train,inp_valid,target_train,target_valid = train_test_split(inp,target,train_size=.8,random_state=31)

# 	rf = RandomForestClassifier(random_state = 31)



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







