
import numpy as np
from model import processing_data
import tensorflow as tf
from datetime import date

# Set GPU memory growth (optional, only needed if you're using a GPU)
#if tf.config.list_physical_devices('GPU'):
#    physical_devices = tf.config.list_physical_devices('GPU')
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def perform_forecasting(data_in):

    temp = data_in
    temp_model = tf.keras.models.load_model('model/temperature_stacked_lstm.h5')
    temp_norm,scaler_temp = processing_data.proc_data(temp,feature_inputs = 'ta')
    temp_pred = temp_model.predict(temp_norm)
    temp_pred_rev = scaler_temp.inverse_transform(temp_pred).reshape(24,1)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    final_result = temp_pred_rev.astype('float64')
    return final_result


if __name__ == "__main__":
    dataset = np.random.rand(1,168)
    out = perform_forecasting(dataset)
    print(out)
       
