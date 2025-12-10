import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


df = pd.read_csv("D:/machine learning project/diabetes.csv")

X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
y= df ["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

score =accuracy_score(y_test,predictions)
print(score)

prediction = model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
print(prediction)

plt.figure(figsize=(30, 15))  # Increase figure size for better readability
tree.plot_tree(
    model, 
    feature_names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    class_names=["No Diabetes", "Diabetes"],
    filled=True,
    fontsize=12  # Increase font size for better readability
)
#Two  lines to make our compiler able to draw:
plt.savefig("decision_tree_plot.png")
sys.stdout.flush()
#---------------------