#import libraries

import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#load dataset

iris = pd.read_csv("Iris.csv")

#explore dataset

print("First 5 rows:\n", iris.head())
print("\nDataset Info:\n")
print(iris.info())
print("\nUnique Species:", iris["Species"].unique())

# 4. Data visualization

sns.pairplot(iris, hue="Species")
plt.show()

# 5. Split 

X = iris.drop(["Id", "Species"], axis=1)
y = iris["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 6. Train KNN model

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 7. Predictions

y_pred = knn.predict(X_test)

# 8. Evaluation

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# 9. Test on new 

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("\nPrediction for new sample:", prediction)