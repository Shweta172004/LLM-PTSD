import pandas as pd
import numpy as np
import math
df1 = pd.read_csv("path_to_patients_info.csv")
df1['age'] = df1['age'].apply(round)
df1.to_csv("path_of_demographics.csv", index=False)

df2 = pd.read_csv("path_to_omr.csv")
df2.to_csv("path_to_medical_histories.csv" , index = False)

df3 = pd.read_csv("path_to_lab_events.csv")
df3.drop(['value' , 'hadm_id', 'order_provider_id'], axis =1)

df4 = pd.read_csv("path_to_emar.csv")
df4.drop(['poe_id', 'pharmacy_id','enter_provider_id'] , axis = 1)
df4.to_csv("path_to_medical_regimens.csv" , index = False)

