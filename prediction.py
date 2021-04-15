from sklearn import datasets
from joblib import load
import numpy as np
import json
import pickle 

#load the model
my_model = load('lin_model.pkl')
#pickle.load(my_model)
class_names = ['Overall', 'PTS']

#with open('lin_model.pkl', 'rb') as f:
#  pickle.dump[f]
#class_names = my_model.NN['Overall', 'PTS']

def my_prediction(id):
  dummy = np.array(id)
  dummyT = dummy.reshape(1,-1)
  prediction = my_model.predict(dummyT)
  name = class_names[prediction]
  name = name.tolist()
  name_str = json.dumps(name)
  str = [name_str]
  return str
