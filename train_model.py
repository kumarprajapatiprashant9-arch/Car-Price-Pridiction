import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("Loading dataset...")

data = pd.read_csv("car_data.csv")

print("✅ Dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())


# =========================================================
# 2. CHECK DATA
# =========================================================

print("\nChecking missing values...")

print(data.isnull().sum())


# =========================================================
# 3. REMOVE MISSING VALUES
# =========================================================

data = data.dropna()

print("\nAfter removing missing values:")
print("Dataset shape:", data.shape)


# =========================================================
# 4. INPUT FEATURES AND TARGET
# =========================================================

X = data.drop("Selling_Price", axis=1)

y = data["Selling_Price"]


# =========================================================
# 5. DEFINE COLUMNS
# =========================================================

categorical_features = [
    "Car_Name",
    "Company",
    "Fuel_Type",
    "Selling_Type",
    "Transmission"
]


numeric_features = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Owner"
]


# =========================================================
# 6. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),

        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# =========================================================
# 7. CREATE MACHINE LEARNING MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1
)


# =========================================================
# 8. CREATE PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================================================
# 9. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================================
# 10. TRAIN MODEL
# =========================================================

print("\n" + "=" * 50)
print("Training Random Forest model...")
print("=" * 50)

pipeline.fit(
    X_train,
    y_train
)

print("✅ Model training completed!")


# =========================================================
# 11. MAKE PREDICTIONS
# =========================================================

print("\nMaking predictions...")

y_pred = pipeline.predict(X_test)


# =========================================================
# 12. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(
    f"Mean Absolute Error : {mae:.2f} Lakh"
)

print(
    f"Mean Squared Error  : {mse:.2f}"
)

print(
    f"Root Mean Squared Error : {rmse:.2f} Lakh"
)

print(
    f"R2 Score            : {r2:.2f}"
)


# =========================================================
# 13. SAVE TRAINED MODEL
# =========================================================

model_filename = "car_price_model.pkl"

joblib.dump(
    pipeline,
    model_filename
)


print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print(
    f"✅ File created: {model_filename}"
)

print(
    "\nYou can now run the Streamlit application using:"
)

print(
    "streamlit run app.py"
)