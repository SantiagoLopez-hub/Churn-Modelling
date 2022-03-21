import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X[:, :])

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(y_test)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Creating ANN
ann = tf.keras.models.Sequential()

# Input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Second layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# Compile ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100)



# Predict client action
print(
    ann.predict(
        sc.transform(
            [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
        )
    )
    # if more than 0.5, it is true, else, false
    > 0.5
)
