import pandas as pd
import numpy as np
import math

df1 = pd.read_csv("C:\\Users\\shweta patel\\OneDrive\\Desktop\\FINAL\\patient_info.csv")
df1['age'] = df1['age'].apply(round)
df1.to_csv("C:\\Users\\shweta patel\\OneDrive\\Desktop\\FINAL\\demographics.csv", index=False)

# print(df1.head())

df2 = pd.read_csv("C:\\Users\\shweta patel\\OneDrive\\Desktop\\FINAL\\omr.csv")
df2.to_csv("C:\\Users\\shweta patel\\OneDrive\\Desktop\\FINAL\\medical_histories.csv" , index = False)
# print(df2.head())

df3 = pd.read_csv("C:\\Users\\shweta patel\\OneDrive\\Desktop\\FINAL\\lab_events.csv")
df3.drop(['value' , 'hadm_id', 'order_provider_id'], axis =1)
df3.to_csv("C:\\Users\\shweta patel\\OneDrive\\Desktop\\FINAL\\lab_events.csv" , index = False)
# print(df3.head())

df4 = pd.read_csv("C:\\Users\\shweta patel\\OneDrive\\Desktop\\FINAL\\emar.csv")
df4.drop(['poe_id', 'pharmacy_id','enter_provider_id'] , axis = 1)
df4.to_csv("C:\\Users\\shweta patel\\OneDrive\\Desktop\\FINAL\\medical_regimens.csv" , index = False)

