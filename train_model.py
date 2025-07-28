import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("adult 3.csv")


df.replace('?', pd.NA, inplace=True)  
df.dropna(inplace=True)             
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop("income", axis=1)
y = df["income"]


model = RandomForestClassifier()
model.fit(X, y)


joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
