import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Read the dataset
df = pd.read_csv('Housing.csv')
df = df[["price", "bathrooms", "grade", "waterfront", "sqft_lot", "sqft_living", "lat", "long", "bedrooms"]]

# Extracting features and target variable
X = df[["bathrooms", "grade", "waterfront", "sqft_lot", "sqft_living", "lat", "long", "bedrooms"]]
y = df["price"]
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=16, test_size =0.3)

model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Performance metrics
mae_var = mean_absolute_error(y_test, y_pred)
r_square = r2_score(y_test, y_pred)


def cls() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def predict() -> None:
    userResponse = pd.DataFrame()
    bathrooms = int(input("Enter Number of bathroms: "))
    while bathrooms < 0:
        bathrooms = int(input("Invalid Entry\nEnter Number of bathroms: "))
    grade = int(input("Enter the grade of the home 1-13: (grade increases with more amenities)"))
    while 0 >= grade or grade > 13:
        grade = int(input("Invalid Entry\nEnter the grade of the home 1-13: (grade increases with more amenities) "))
    waterfront = int(input("Enter if the home is in a waterfront (0 for no or 1 for yes): "))
    while 0 > waterfront or waterfront > 1:
        waterfront = int(input("Invalid Entry\nEnter if the home is in a waterfront (0 for no or 1 for yes): "))
    sqft_lot = float(input("Enter the square-footage of the lot (520-1650000): "))
    while sqft_lot < 0:
        sqft_lot = float(input("Invalid Entry\nEnter the square-footage of the lot (520-1650000): "))
    sqft_living = float(input("Enter the square-footage of living area (290-13500): "))
    while sqft_living < 0:
        sqft_living = float(input("Invalid Entry\nEnter the square-footage of living area (290-13500): "))
    lat = float(input("Enter the lattitude of the home: "))
    while 90 < lat or lat < -90:
        lat = float(input("Invalid Entry\nEnter the lattitude of the home: "))
    long = float(input("Enter the longitude of the home: "))
    while 180 < long or long < -180:
        long = float(input("Enter the longitude of the home: "))
    bedrooms = int(input("Enter Number of bedrooms: "))
    while bedrooms < 0:
        bedrooms = int(input("Invalid Entry\nEnter Number of bedrooms: "))
    userResponse['bathrooms'] = [bathrooms]
    userResponse['grade'] = [grade]
    userResponse['waterfront'] = [waterfront]
    userResponse['sqft_lot'] = [sqft_lot]
    userResponse['sqft_living'] = [sqft_living]
    userResponse['lat'] = [lat]
    userResponse['long'] = [long]
    userResponse['bedrooms'] = [bedrooms]
    # Predict the price
    predicted_price = model.predict(userResponse)
    print(f"Predicted price for this home specification is ${predicted_price[0]:.2f}")
    log = input("would you like to log this prediction (enter 'y' for yes): ")
    if log == 'y':
        userResponse['predicted_price'] = predicted_price[0]
        userResponse.to_csv('log.csv', mode='a', index=False, header=False)
    input("press enter to return")
    cls()


def commands() -> None:
    cls()
    print("predict: Will begin a prediction query")
    print("city_list: Will print a list of popular cities in Washington with latitude and logitude values.")
    print("commands: display a list of valid commands")
    print("heatmap: displays descriptvie heatmap of the data")
    print("distribution: displays descriptvie price distribution of the data")
    print("scatterplot: displays descriptvie scatter plot of the price vs sqft_living")
    print("r2: displays the R-Squared value")
    print("mae: displays the mean absolute error")
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


def heatmap() -> None:
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.show()


def scatterplot() -> None:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='sqft_living', y='price', data=df)
    plt.title('price vs sqft_living')
    plt.xlabel('Living Area in Square feet')
    plt.ylabel('Price')
    plt.show()


def distribution() -> None:
    plt.figure(figsize=(10, 8))
    sns.histplot(df['price'])
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()


def r2() -> None:
    print(f'\nThe R-Squared value is: {r_square}')


def mae() -> None:
    print(f'\nThe mean absolute error is: ${mae_var:.2f}')


app_loop = True
count = 0

while app_loop:
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
            app_loop = False
        case "heatmap":
            heatmap()
        case "distribution":
            distribution()
        case "scatterplot":
            scatterplot()
        case "r2":
            r2()
        case "mae":
            mae()
