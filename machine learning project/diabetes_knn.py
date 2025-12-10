from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("D:/machine learning project/diabetes.csv")


X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y= df ["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn= KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
score =accuracy_score(y_test,predictions)
print(score)