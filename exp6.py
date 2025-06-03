from os import listdir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def kernel(point, xmat, k):
    m, n = xmat.shape
    weights = np.eye(m)
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(-(diff @ diff.T) / (2.0 * k**2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    XT_WX = xmat.T @ (wei @ xmat)
    XT_WX += np.eye(XT_WX.shape[0]) * 1e-5  # regularization
    W = np.linalg.inv(XT_WX) @ (xmat.T @ (wei @ ymat))
    return W

def localWeightRegression(xmat, ymat, k):
    m, n = xmat.shape
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] @ localWeight(xmat[i], xmat, ymat, k)
    return ypred

# Load data
data = pd.read_csv('tips.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)

# Prepare matrices
m = bill.shape[0]
one = np.ones((m, 1))
X = np.hstack((one, bill.reshape(-1, 1)))
y = tip.reshape(-1, 1)

# Predict using locally weighted regression
ypred = localWeightRegression(X, y, 0.3)

# Sort X and ypred for plotting
SortIndex = X[:, 1].argsort()
xsort = X[SortIndex]
ysort = ypred[SortIndex]

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='blue', label='Actual Data')
ax.plot(xsort[:, 1], ysort, color='red', linewidth=2, label='Predicted Regression Line')

plt.xlabel('Total bill', fontsize=14)
plt.ylabel('Tip', fontsize=14)
plt.title('Locally Weighted Regression', fontsize=18)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('exp6-1.png')
plt.show()