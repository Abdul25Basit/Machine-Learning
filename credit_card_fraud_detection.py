import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download the Iris dataset from UCI Machine Learning Repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Define column names for the dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Read the dataset
try:
    iris_data = pd.read_csv(url, header=None, names=column_names)
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit()

# Data exploration and preprocessing
print(iris_data.head())
print(iris_data.tail())
print(iris_data.info())
print(iris_data.isnull().sum())

# Analyze class distribution
print(iris_data['class'].value_counts())

# Separate features and target variable
X = iris_data.drop(columns='class', axis=1)
Y = iris_data['class']

# Split data into training and testing sets (stratify ensures class balance)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Evaluate model performance
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)
