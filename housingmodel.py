import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read the dataset
df = pd.read_csv('Housing.csv')

# Extracting features and target variable
X = df[["bathrooms", "grade", "waterfront", "sqft_lot", "sqft_living", "lat", "long", "bedrooms"]]
y = df["price"]
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=16, test_size =0.3)

model = RandomForestRegressor(max_depth=6,random_state=0, n_estimators=10)
model.fit(X_train, y_train)


def cls() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def predict() -> None:
    userResponse = pd.DataFrame()
    bathrooms = input("Enter Number of bathroms: ")
    grade = input("Enter the grade of the home 1-13: ")
    waterfront = input("Enter if the home is in a waterfront (0 for no or 1 for yes): ")
    sqft_lot = input("Enter the square-footage of the lot (520-1650000): ")
    sqft_living = input("Enter the square-footage of living area (290-13500): ")
    lat = input("Enter the lattitude of the home: ")
    long = input("Enter the longitude of the home: ")
    bedrooms = input("Enter Number of bedrooms: ")
    userResponse['bathrooms'] = [bathrooms]  # 0-8
    userResponse['grade'] = [grade]#1-13 moves the price drastically
    userResponse['waterfront'] = [waterfront]#1 or 0
    userResponse['sqft_lot'] = [sqft_lot]#520 - 1650000
    userResponse['sqft_living'] = [sqft_living]#290 - 13500
    userResponse['lat'] = [lat]
    userResponse['long'] = [long]
    userResponse['bedrooms'] = [bedrooms]#1-33
    # Predict the price
    predicted_price = model.predict(userResponse)
    print(f"Predicted price for this home specification is ${predicted_price[0]:.2f}")
    input("press enter to return")
    cls()



def commands() -> None:
    cls()
    print("predict: Will begin a prediction query")
    print("city_list: Will print a list of popular cities in Washington with latitude and logitude values.")
    print("commands: display a list of valid commands")
    print("commands: display a list of valid commands")
    input("press enter to return")


def city_list() -> None:
    cls()
    print("While The values displayed are from Washington\nThe values could be extrapolated to any valid latitude and longitude values")
    print("Seattle: Latitude 47.60, Longitude -122.33")
    print("Spokane: Latitude 47.65, Longitude -117.42")
    print("Tacoma: Latitude 47.25, Longitude -122.44")
    print("Vancouver: Latitude 45.63, Longitude -122.66")
    print("Bellevue: Latitude 47.61, Longitude -122.20")
    print("Everett: Latitude 47.97, Longitude -122.20")
    print("Kent: Latitude 47.38, Longitude -122.23")
    print("Renton: Latitude 47.48, Longitude -122.21")
    print("Federal Way: Latitude 47.32, Longitude -122.31\n")


event_loop = True
count = 0

while event_loop:
    if count == 0:
        print("Welcome to the Housing Prediction Model Terminal application")
        count += 1
    user_input = input("To Start the model enter 'predict'\nFor a list of commands enter 'commands'\nTo Exit enter 'quit'\n")
    match user_input:
        case "predict":
            predict()
        case "city_list":
            city_list()
        case "commands":
            commands()
        case "quit":
            event_loop = False
