import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))
# Read the datasets
import joblib
scaler_temp = joblib.load('scaler_temp.save') 


def proc_data(dataset,feature_inputs):
    if feature_inputs == 'ta':
        scaler = scaler_temp
    else:
        scaler = scaler_temp
    dataset = dataset.reshape(-1,1)
    X_final = scaler.transform(dataset)
    data_final = X_final.reshape(1,dataset.shape[0],1)

    return data_final,scaler
