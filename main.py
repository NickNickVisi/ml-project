# main.py
from data_load import generate_patient_data
from plot import Heatmap
from print import printf
from model import train

# Generează datele
df = generate_patient_data(800)
train_df, test_df, y_pred, y_test = train(df)

printf(y_test, y_pred)

Heatmap(train_df)


