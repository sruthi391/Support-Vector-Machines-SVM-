import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load the dataset
df = pd.read_csv("preprocessed_breast_cancer.csv")

# Select features for 2D visualization
X = df[["radius_mean", "texture_mean"]].values
y = df["diagnosis"].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM with linear kernel
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_scaled, y)

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_scaled, y)

# Function to plot decision boundary
def plot_decision_boundary(clf, X, y, title, filename):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Plot and save decision boundaries
plot_decision_boundary(svm_linear, X_scaled, y, "SVM with Linear Kernel", "svm_linear_boundary.png")
plot_decision_boundary(svm_rbf, X_scaled, y, "SVM with RBF Kernel", "svm_rbf_boundary.png")

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 1],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_scaled, y)

print("Best parameters from GridSearchCV:", grid.best_params_)

# Cross-validation score of best model
scores = cross_val_score(grid.best_estimator_, X_scaled, y, cv=5)
print("Cross-validation accuracy:", scores.mean())
