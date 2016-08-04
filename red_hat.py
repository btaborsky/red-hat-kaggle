import pandas as pd
import pdb



def load_data():


	people_fname = "Data/people.csv"
	event_fname = "Data/act_train.csv"


	people_df = pd.read_csv(people_fname)

	events_df = pd.read_csv(event_fname)

	pdb.set_trace()
	return people_df,events_df
