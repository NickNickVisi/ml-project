# main.py
import numpy as np
from data_load import generate_patient_data
from plot import Heatmap
from print import printf
from model import train
from read_data import read
from missing_values import generate_missing_values
from missing_values import fill_missing_values

x = read()
df = generate_patient_data(int (x))

generate_missing_values(df)
fill_missing_values(df)

train_df, test_df, y_pred, y_test = train(df)

printf(y_test, y_pred)

Heatmap(train_df)