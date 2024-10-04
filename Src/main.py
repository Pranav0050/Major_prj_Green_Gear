import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(C:\Users\HP\OneDrive\Desktop\Grp4 project\Front end\data.csv):
    data = pd.read_csv(C:\Users\HP\OneDrive\Desktop\Grp4 project\Front end\data.csv)
    print("Data loaded successfully. Shape:", data.shape)
    return data

def prepare_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    print(f"{model_name} trained successfully.")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print()

    return mse, r2

def main():
    file_path = input("C:\Users\HP\OneDrive\Desktop\Grp4 project\Front end\data.csv ")

    data = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(data)

    lr_model = LinearRegression()
    lr_mse, lr_r2 = train_and_evaluate(lr_model, X_train, X_test, y_train, y_test, "Linear Regression")

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_mse, rf_r2 = train_and_evaluate(rf_model, X_train, X_test, y_train, y_test, "Random Forest Regression")

    print("Model Comparison:")
    if lr_r2 > rf_r2:
        print("Linear Regression performed better in terms of R-squared score.")
    elif rf_r2 > lr_r2:
        print("Random Forest Regression performed better in terms of R-squared score.")
    else:
        print("Both models performed equally in terms of R-squared score.")

    print("\nNote: This is a simple comparison. In practice, you might want to consider other factors and perform more extensive evaluations.")

main()