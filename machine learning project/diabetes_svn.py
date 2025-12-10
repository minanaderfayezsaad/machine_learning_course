import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("D:/machine learning project/diabetes.csv")

X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y= df ["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf=svm.SVC(kernel="linear")

clf.fit(X_train,y_train)

Y_pred=clf.predict(X_test)
print("accuracy:",clf.score(X_test,y_test))


