import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle

dfval=pd.read_csv('Fertilizer2.csv')

Numerics=LabelEncoder()
inputs=dfval.drop('Fertilizer Name',axis='columns')
target=dfval['Fertilizer Name']

inputs['soil type_n']=Numerics.fit_transform(inputs['Soil Type'])
inputs['crop type_n']=Numerics.fit_transform(inputs['Crop Type'])

print(inputs)

inputs_n=inputs.drop(['Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous'],axis='columns')
print(inputs_n)

Classifier=GaussianNB()
Classifier.fit(inputs_n,target)

# pickle.dump(Classifier, open('model.pkl','wb'))
#Loading model to compare the results



print("The predicted fertilzer is",Classifier.predict([[20,60,43,8,10]]))
