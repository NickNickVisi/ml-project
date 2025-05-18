# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from data_load import generate_patient_data
from plot import Heatmap

# Generează datele
df = generate_patient_data(800)

# Split train/test
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

# Encode categorice
combined = pd.concat([train_df, test_df])
for col in ['sex', 'activitate_fizica']:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
train_df = combined.iloc[:len(train_df)]
test_df = combined.iloc[len(train_df):]

# Train/test split
X_train = train_df.drop(columns=['risc_diabet'])
y_train = train_df['risc_diabet']
X_test = test_df.drop(columns=['risc_diabet'])
y_test = test_df['risc_diabet']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

Heatmap(train_df)


