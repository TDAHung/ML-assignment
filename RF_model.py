import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score


# Reading the data of CSV
data = pd.read_csv('diabetes.csv')

# Choose the signatures and labels
features = data.drop('Outcome', axis=1)
labels = data['Outcome']

# Split the data into train and test test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict labels on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Evaluate the model Random Forest:")
print(classification_report(y_test, y_pred))
print("Accuracy model RF:", accuracy)
print("Mean Squared Error (MSE) model RF:", mse)
print("Root Mean Squared Error (RMSE) model RF:", rmse)
print("R2 Score model RF:", r2)
