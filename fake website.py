import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.ensemble
from sklearn.metrics import accuracy_score

data = pd.read_csv("PhishingData.csv")

X = data.drop("Result", axis=1)
y = data["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = sklearn.ensemble.RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
