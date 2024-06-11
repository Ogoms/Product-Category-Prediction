# **Product Prediction Using Machine Learning**

**Overview**

This project focuses on predicting products using machine learning techniques. The project utilizes three main machine learning algorithms: K-Nearest Neighbors (KNN), Naive Bayes, and Decision Tree. The technologies used include Python, NumPy, pandas, and scikit-learn (sklearn).

**Table of Contents**

**.Installation**
**.Data Importation**
**.Data Cleaning**
**.Data Encoding**
**.Model Training**
    **.K-Nearest Neighbors (KNN)**
    **.Naive Bayes**
    **.Decision Tree**
**.Evaluation**
**.Usage**
**.Contributing**
**.License**

**Installation**

To get started with this project, clone the repository and install the required dependencies:

git clone https://github.com/Ogoms/Product-Category-Prediction.git

cd Product-Prediction-Using Machine-Learning

pip install -r requirements.txt


Ensure you have the following packages installed:

Python 3.x

NumPy

pandas

scikit-learn

**Data Importation**

The dataset is imported using pandas:

import pandas as pd

data = pd.read_csv('path_to_your_dataset.csv')


**Data Cleaning**

Data cleaning involves handling missing values, removing duplicates, and correcting inconsistencies

**Data Encoding**

Categorical variables are encoded to numerical values to be used in machine learning algorithms

**Model Training**

K-Nearest Neighbors (KNN)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = data.drop('target_column', axis=1)
y = data['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)


**Naive Bayes**

from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)


**Decision Tree**

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)


**Evaluation**

Evaluate the performance of each model using accuracy score and classification report:

from sklearn.metrics import accuracy_score, classification_report

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))


**Usage**

To use this project, simply run the Python scripts provided in the repository. You can modify the scripts to use your own dataset and adjust the parameters as needed.

**Contributing**

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

**License**
This project is licensed under the MIT License. See the LICENSE file for details.

**Contact**
For any inquiries or suggestions, please contact CoderOgoo at https://github.com/Ogoms