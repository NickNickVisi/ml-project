from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def train(df):

# Split train/test
    train_df, test_df = train_test_split(df, test_size=0.285, random_state=42)

# Encode categorice
    combined = pd.concat([train_df, test_df])
    for col in ['activitate_fizica']:
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

    return train_df, test_df, y_pred, y_test