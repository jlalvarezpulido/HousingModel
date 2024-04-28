import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Housing.csv')
df = df.drop(columns=["sqft_lot15", "sqft_living15", "long", "lat", "date"])

X = df.iloc[:, 0:17]
y = df.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=16, test_size=0.3)

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

print(confusion_matrix(y_test, y_pred))
