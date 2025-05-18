# main.py
from data_load import generate_patient_data
from plot import Heatmap
from print import printf
from model import train

# Generează datele
x = input ("Introdu numarul de teste dorite: ") 
while int(x) < 700:
    x = input ("Numar invalid de teste, alege un alt numar: ") 

df = generate_patient_data(int (x))
train_df, test_df, y_pred, y_test = train(df)

printf(y_test, y_pred)

Heatmap(train_df)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

