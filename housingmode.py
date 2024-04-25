import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read the dataset
df = pd.read_csv('Housing.csv')

# Extracting features and target variable
X = df[["bathrooms","grade","waterfront","sqft_lot","sqft_living", "long", "lat", "bedrooms"]]
y = df["price"]
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=16, test_size =0.3)

model = RandomForestRegressor(max_depth=6,random_state=0, n_estimators=10)
model.fit(X_train, y_train)


def predict_price(bathrooms, grade, waterfront, sqft_lot, sqft_living, long, lat, bedrooms):
    userResponse = pd.DataFrame()
    userResponse['bathrooms'] = [bathrooms]  # 0-8
    userResponse['grade'] = [grade]#1-13 moves the price drastically
    userResponse['waterfront'] = [waterfront]#1 or 0
    userResponse['sqft_lot'] = [sqft_lot]#520 - 1650000
    userResponse['sqft_living'] = [sqft_living]#290 - 13500
    userResponse['long'] = [long]
    userResponse['lat'] = [lat]
    userResponse['bedrooms'] = [bedrooms]#1-33
    # Predict the price
    predicted_price = model.predict(userResponse)
    return predicted_price[0]


bathrooms = input("Enter Number of bathroms: ")
grade = input("Enter the grade of the home 1-13: ")
waterfront = input("Enter if the home is in a waterfront (0 or 1): ")
sqft_lot = input("Enter the square-footage of the lot (520-1650000): ")
sqft_living = input("Enter the square-footage of living area (290-13500): ")
long = input("Enter the longitude of the home: ")
lat = input("Enter the lattitude of the home: ")
bedrooms = input("Enter Number of bedrooms: ")
predicted_price = predict_price(bathrooms, grade, waterfront, sqft_lot, sqft_living, long, lat, bedrooms)
print(f"Predicted price for this Home is ${predicted_price/2:.2f}")
# Seattle: Latitude 47.6062, Longitude -122.3321
# Spokane: Latitude 47.6588, Longitude -117.4260
# Tacoma: Latitude 47.2529, Longitude -122.4443
# Vancouver: Latitude 45.6387, Longitude -122.6615
# Bellevue: Latitude 47.6101, Longitude -122.2015
# Everett: Latitude 47.9789, Longitude -122.2021
# Kent: Latitude 47.3809, Longitude -122.2348
# Renton: Latitude 47.4829, Longitude -122.2171
# Federal Way: Latitude 47.3223, Longitude -122.3126
