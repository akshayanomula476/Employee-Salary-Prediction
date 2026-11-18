
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def train_model():

    # Load employee dataset
    data = pd.read_csv("employee_data.csv")

    # Input columns
    X = data[["experience_years", "education_level", "job_role"]]

    # Output column
    y = data["salary"]

    # Convert job roles into numerical values
    preprocessor = ColumnTransformer(
        transformers=[
            ("role", OneHotEncoder(handle_unknown="ignore"), ["job_role"])
        ],
        remainder="passthrough"
    )

    # Create Linear Regression model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate the model
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Train final model using all data
    model.fit(X, y)

    return model, mae, r2, data
