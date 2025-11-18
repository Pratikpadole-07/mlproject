import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Models
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


def train_model():

    print("🚀 Loading dataset...")
    df = pd.read_csv("data/stud.csv")

    target = "math_score"
    X = df.drop(columns=[target])
    y = df[target]

    # Identify columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    print("🔧 Building preprocessing...")
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # MODEL DICTIONARY
    models = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(),
        "CatBoost": CatBoostRegressor(verbose=False),
        "AdaBoost": AdaBoostRegressor()
    }

    # HYPERPARAMETERS
    params = {
        "Random Forest": {'n_estimators': [50, 100, 200]},
        "Decision Tree": {'criterion': ['squared_error', 'absolute_error']},
        "Gradient Boosting": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        "Linear Regression": {},
        "XGBoost": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        "CatBoost": {'iterations': [50, 100], 'depth': [6, 8]},
        "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    }

    best_score = -999
    best_model_name = None
    best_model = None

    # TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔥 Training all models...")

    for name, model in models.items():
        print(f"➡ Training {name}...")

        grid = GridSearchCV(model, params[name], cv=3, scoring='r2')
        grid.fit(X_train, y_train)

        preds = grid.predict(X_test)
        score = r2_score(y_test, preds)

        print(f"   {name} R2 Score: {score}")

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_model_name = name

    print("\n🏆 BEST MODEL:", best_model_name)
    print("🎯 BEST R2 SCORE:", best_score)

    # FINAL PIPELINE
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])

    print("🏗 Training final pipeline...")
    final_pipeline.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/model_pipeline.pkl", "wb") as f:
        pickle.dump(final_pipeline, f)

    print("\n✅ Model pipeline saved as artifacts/model_pipeline.pkl")


if __name__ == "__main__":
    train_model()
