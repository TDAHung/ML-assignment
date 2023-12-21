import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif

print("Matplotlib version:", matplotlib.__version__)

import pandas as pd

data = pd.read_csv('diabetes.csv')

# Attributes need to be checked for outliers
attributes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

#Check for outliers for each attribute
for attribute in attributes:
    # Q1, Q3 calculation
    Q1 = data[attribute].quantile(0.25)
    Q3 = data[attribute].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers based on IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Print outliers
    outliers = data[(data[attribute] < lower_bound) | (data[attribute] > upper_bound)]
    print(f"Outliers of '{attribute}':")
    print(outliers)
    print("\n")

# Count the number of 1 and 0 values ​​in the subclass attribute 'Outcome'
class_counts = data['Outcome'].value_counts()

# Print results
print("Numbers of 1:", class_counts[1])
print("Numbers of 0:", class_counts[0])

# Calculate the percentage of values ​​in the 'Outcome' subclass attribute
class_percentages = data['Outcome'].value_counts(normalize=True) * 100

# Print results
print("Value Ratio 1:", class_percentages[1])
print("Value Ratio 0:", class_percentages[0])

X = data.drop('Outcome', axis=1)
y = data['Outcome']
information_gain = mutual_info_classif(X, y, discrete_features=[False, False, False, False, False, False, False, False])
ig_df = pd.DataFrame({'Feature': X.columns, 'InformationGain': information_gain})
plt.figure(figsize=(10, 6))
sns.barplot(x='InformationGain', y='Feature', data=ig_df, palette='viridis')

plt.title('Information Gain of Features')
plt.show()

import seaborn as sns
sns.heatmap(data.corr(),annot=True,fmt='0.2f')

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('diabetes.csv')

# Calculate the correlation matrix
corr_matrix = data.corr()

# Get the correlation coefficient of column 'Outcome' with other columns
outcome_corr = corr_matrix['Outcome']

# Delete the correlation coefficient of column 'Outcome' with itself
outcome_corr = outcome_corr.drop('Outcome')

# Visualize the correlation coefficient
outcome_corr.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Attribute columns')
plt.ylabel('Correlation coefficients')
plt.title('Information gain between attribute and result columns')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score

# Specify labels
features = data.drop('Outcome', axis=1)
labels = data['Outcome']

# Divide the data into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardized data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Predict labels on the training set
y_train_pred = model.predict(X_train)

# Predict labels on the test set
y_pred = model.predict(X_test)


# Evaluate Model
# train
accuracy_train = accuracy_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_test, y_pred)
rmse_train = mean_squared_error(y_test, y_pred, squared=False)
r2_train = r2_score(y_test, y_pred)

print("Evaluate model Naive Bayes on the training set:")

print("Accuracy model NB train:", accuracy_train)

# test
accuracy_test = accuracy_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
r2_test = r2_score(y_test, y_pred)

print("\nEvaluate model Naive Bayes on the test set:")
print(classification_report(y_test, y_pred))
print("\nAccuracy model NB test:", accuracy_test)
print("\nMean Squared Error (MSE) model NB:", mse_test)
print("Root Mean Squared Error (RMSE) model NB:", rmse_test)
print("R2 Score model NB:", r2_test)

from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

import numpy as np

patient_info = np.array([[6,148,72,35,0,33.6,0.627,50]])
prediction = model.predict(patient_info)
if prediction == 0:
    print("No")
elif prediction == 1:
    print("Yes")
else:
    print("Error")
