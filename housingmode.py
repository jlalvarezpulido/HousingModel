import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Read the dataset
df = pd.read_csv('Housing.csv')

# Selecting only relevant columns for this example
df = df[['bedrooms', 'price']]

# Extracting features and target variable
X = df[['bedrooms']]
y = df['price']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.3)

# Creating and training the model with feature names
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Function to predict price based on number of bedrooms
def predict_price(num_bedrooms):
    # Reshape input to match the model's expectations
    num_bedrooms = np.array([[num_bedrooms]])
    # Set feature names of input data to match X_train
    X_new = pd.DataFrame({'bedrooms': [num_bedrooms]})
    X_new.columns = X_train.columns
    # Predict the price
    predicted_price = model.predict(X_new)
    return predicted_price[0]

# Example usage of the function, erase the next "#" to put your input and the line after.
#num_bedrooms_input = int(input("Enter the number of bedrooms: "))
num_bedrooms_input = 5
predicted_price = predict_price(num_bedrooms_input)
print(f"Predicted price for {num_bedrooms_input} bedrooms: ${predicted_price/2:.2f}")
