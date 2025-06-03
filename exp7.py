import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_boston():
    b_housing = pd.read_csv('BostonHousing.csv')
    X = b_housing[['rm']]
    y = b_housing['medv']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    plt.scatter(X_test, y_test, color="blue", label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel("Average number of rooms (AveRooms)")
    plt.ylabel("Median value of homes ($100,000)")
    plt.title("Linear Regression - Boston Housing Dataset")
    plt.legend()
    plt.savefig('exp7-1.png')
    plt.show()
    
    print("Linear Regression - Boston Housing Dataset")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))
    
if __name__ == "__main__":
     print("Demonstrating Linear Regression and Polynomial Regression\n")
     linear_regression_boston()