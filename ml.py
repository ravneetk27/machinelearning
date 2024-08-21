import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Importing data
path = "/Users/karmansingh/PycharmProjects/Day63_Udemy/pythonProject/day-63-starting-files-library-project/train_lyst1720633824458.csv"  # Specify your path here
data_train = pd.read_csv(path)

# Visualization
plt.hist(data_train["category"])
plt.show()

plt.plot(data_train["adview"])
plt.show()

# Remove videos with adview greater than 2000000 as outliers
data_train = data_train[data_train["adview"] < 2000000]

# Heatmap with only numeric data
numeric_data_train = data_train.select_dtypes(include=[np.number])

f, ax = plt.subplots(figsize=(10, 8))
corr = numeric_data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True,
            ax=ax, annot=True)
plt.show()

# Removing invalid data
data_train = data_train[data_train['views'] != 'F']
data_train = data_train[data_train['likes'] != 'F']
data_train = data_train[data_train['dislikes'] != 'F']
data_train = data_train[data_train['comment'] != 'F']

# Map category to numerical values
category = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
data_train["category"] = data_train["category"].map(category)

# Convert values to integers
data_train["views"] = pd.to_numeric(data_train["views"])
data_train["comment"] = pd.to_numeric(data_train["comment"])
data_train["likes"] = pd.to_numeric(data_train["likes"])
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"])
data_train["adview"] = pd.to_numeric(data_train["adview"])

# Encoding features
label_encoder = LabelEncoder()
data_train['duration'] = label_encoder.fit_transform(data_train['duration'])
data_train['vidid'] = label_encoder.fit_transform(data_train['vidid'])
data_train['published'] = label_encoder.fit_transform(data_train['published'])


# Function to convert duration to seconds
def checki(x):
    if isinstance(x, int):
        return x  # If the duration is already an integer, return it as is.

    y = x[2:]  # Assuming x is a string that starts with 'PT' (as in ISO 8601 duration format).
    h = m = s = mm = ''
    P = ['H', 'M', 'S']
    for i in y:
        if i not in P:
            mm += i
        else:
            if i == "H":
                h = mm
            elif i == "M":
                m = mm
            elif i == "S":
                s = mm
            mm = ''
    h = '00' if h == '' else h
    m = '00' if m == '' else m
    s = '00' if s == '' else s
    return f'{h}:{m}:{s}'


def func_sec(time_string):
    if isinstance(time_string, int):
        return time_string  # If already an integer, return it as is.

    h, m, s = time_string.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


# Ensure all values in the 'duration' column are treated as strings before applying 'checki'
data_train['duration'] = data_train['duration'].astype(str)
time = data_train['duration'].apply(checki)
data_train["duration"] = time.apply(func_sec)

# Split Data
Y_train = data_train["adview"]
X_train = data_train.drop(["adview", "vidid"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Normalise Data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Evaluation Metrics
def print_error(X_test, y_test, model_name):
    prediction = model_name.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
print_error(X_test, y_test, linear_regression)

# Support Vector Regressor
supportvector_regressor = SVR()
supportvector_regressor.fit(X_train, y_train)
print_error(X_test, y_test, supportvector_regressor)

# Decision Tree Regressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print_error(X_test, y_test, decision_tree)

# Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=15, min_samples_leaf=2)
random_forest.fit(X_train, y_train)
print_error(X_test, y_test, random_forest)

# ANN with Keras
ann = Sequential([
    Dense(6, activation="relu", input_shape=X_train.shape[1:]),
    Dense(6, activation="relu"),
    Dense(1)
])

optimizer = keras.optimizers.Adam()
loss = keras.losses.mean_squared_error
ann.compile(optimizer=optimizer, loss=loss, metrics=["mean_squared_error"])
history = ann.fit(X_train, y_train, epochs=100)
ann.summary()
print_error(X_test, y_test, ann)

# Save models
joblib.dump(decision_tree, "decisiontree_youtubeadview.pkl")
ann.save("ann_youtubeadview.h5")
