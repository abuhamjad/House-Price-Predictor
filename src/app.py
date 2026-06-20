import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
filename = r"C:\Users\abuhamjad\Desktop\Qskill\housePrice_predictor\data\dataset.csv"
df = pd.read_csv(filename)

# Create broader location categories
def get_city(address):
    address = str(address).lower()

    if "greater noida" in address:
        return "Greater Noida"
    elif "noida" in address:
        return "Noida"
    elif "gurgaon" in address or "gurugram" in address:
        return "Gurgaon"
    elif "ghaziabad" in address:
        return "Ghaziabad"
    elif "faridabad" in address:
        return "Faridabad"
    elif "delhi" in address:
        return "Delhi"
    else:
        return "Other"

df["City"] = df["Address"].apply(get_city)

# Drop irrelevant columns
df.drop(
    columns=[
        "Unnamed: 0",
        "desc",
        "Landmarks",
        "Price_sqft",
        "latitude",
        "longitude",
        "Address"
    ],
    inplace=True
)

# Define features and target
X = df.drop("price", axis=1)
y = df["price"]

# Identify categorical and numerical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# Build model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

print("\n===== HOUSE PRICE PREDICTOR =====")

while True:

    try:
        area = float(input("Area (sq ft): "))
        bedrooms = int(input("Bedrooms: "))
        bathrooms = int(input("Bathrooms: "))
        balcony = int(input("Balcony: "))
        parking = int(input("Parking Spaces: "))
        lift = int(input("Lift Count: "))

        print("\nAvailable Cities:")
        print("1. Delhi")
        print("2. Noida")
        print("3. Greater Noida")
        print("4. Gurgaon")
        print("5. Ghaziabad")
        print("6. Faridabad")

        city_choice = input("\nChoose City (1-6): ")

        city_map = {
            "1": "Delhi",
            "2": "Noida",
            "3": "Greater Noida",
            "4": "Gurgaon",
            "5": "Ghaziabad",
            "6": "Faridabad"
        }

        city = city_map.get(city_choice, "Delhi")

        status = input(
            "Status (Ready to move / Under Construction): "
        )

        age = input(
            "Property Age (New / Old): "
        )

        furnished = input(
            "Furnishing (Furnished / Semi-Furnished / Unfurnished): "
        )

        building_type = input(
            "Building Type (Apartment / Flat): "
        )

        sample_house = pd.DataFrame({
            "area": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Balcony": [balcony],
            "parking": [parking],
            "Lift": [lift],
            "City": [city],
            "Status": [status],
            "neworold": [age],
            "Furnished_status": [furnished],
            "type_of_building": [building_type]
        })

        prediction = model.predict(sample_house)

        print("\n" + "="*40)
        print(f"Model Accuracy (R²): {r2:.4f}")
        print(f"Estimated House Price: ₹{prediction[0]:,.2f}")
        print("="*40)

        again = input("\nPredict another house? (y/n): ")

        if again.lower() != "y":
            break

    except Exception as e:
        print(f"\nError: {e}")
        print("Please enter valid values.\n")