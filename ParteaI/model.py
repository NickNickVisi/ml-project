from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def train(df):

# Impartim datele in train si test
    train_df, test_df = train_test_split(df, test_size=0.285, random_state=42)

# Encodam datele
    combined = pd.concat([train_df, test_df])
    for col in ['activitate_fizica']:
       le = LabelEncoder()
       combined[col] = le.fit_transform(combined[col])
    train_df = combined.iloc[:len(train_df)]
    test_df = combined.iloc[len(train_df):]

# Splituim datele
    X_train = train_df.drop(columns=['risc_diabet'])
    y_train = train_df['risc_diabet']
    X_test = test_df.drop(columns=['risc_diabet'])
    y_test = test_df['risc_diabet']

# Antrenam modelul
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Afisam rezultatele
    return train_df, test_df, y_pred, y_test