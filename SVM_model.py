import pandas as pd

data = pd.read_csv('diabetes.csv')
missing_values = data.isnull().sum()
print(missing_values)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score

# Choose the signatures and labels
selected_features = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
features = data[selected_features]
labels = data['Outcome']

# Divide the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardized data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Predict labels on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Evaluate the model Support Vector Machine:")
print(classification_report(y_test, y_pred))
print("Accuracy model SVM:", accuracy)
print("Mean Squared Error (MSE) model SVM:", mse)
print("Root Mean Squared Error (RMSE) model SVM:", rmse)
print("R2 Score model SVM:", r2)
