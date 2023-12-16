import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('diabetes.csv')

# Chọn các đặc trưng và nhãn
features = data.drop('Outcome', axis=1)
labels = data['Outcome']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Xây dựng mô hình Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Dự đoán nhãn trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Evaluate Performance Of Decision Tree:")
print(classification_report(y_test, y_pred))
print("Accuracy DT model:", accuracy)
print("Mean Squared Error (MSE) DT model:", mse)
print("Root Mean Squared Error (RMSE) DT model:", rmse)
print("R2 Score DT model:", r2)

import seaborn as sns
sns.heatmap(data.corr(),annot=True,fmt='0.2f')
