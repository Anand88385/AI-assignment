import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
df = pd.read_csv('https://drive.google.com/uc?export=download&id=1JQeF-UeGtlIl1zXpk2eDbIwn5d40gPFY')

# Drop irrelevant columns
df = df.drop(['Date'], axis=1)

# Fill missing values with mean
df = df.fillna(df.mean())

# Normalize numerical features
scaler = MinMaxScaler()
num_cols = ['Bedrooms', 'Bathrooms', 'Living Area', 'Lot Area', 'Floors',
            'Area (Excluding Basement)', 'Area of Basement', 'Built Year',
            'Renovation Year', 'Latitude', 'Longitude', 'Living Area after Renovation',
            'Lot Area after Renovation', 'Number of Schools Nearby', 'Distance from Airport']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Encode categorical feature
df = pd.get_dummies(df, columns=['Waterfront Present'])

# Split into training and testing sets
X = df.drop(['Price', 'Label', 'Count'], axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ANN model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
score = model.evaluate(X_test, y_test)
print('Test loss:', score)

