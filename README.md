# Support-Vector-Machines-SVM-
# Support Vector Machines (SVM) for Binary Classification

This repository contains Python code demonstrating the application of Support Vector Machines (SVMs) for binary classification using the Breast Cancer dataset. The project covers data preprocessing, model training with different kernels, decision boundary visualization, and hyperparameter tuning using cross-validation.

## Objective

The main objectives of this project are to:
1.  Load and prepare a dataset for binary classification.
2.  Train SVM models with both linear and Radial Basis Function (RBF) kernels.
3.  Visualize decision boundaries for 2D data to understand how SVMs separate classes.
4.  Tune hyperparameters (`C` for linear, and `C`, `gamma` for RBF) to optimize model performance.
5.  Evaluate model performance using cross-validation.

## Dataset

The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, which can be downloaded [here](http://googleusercontent.com/url?q=http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data&sa=D&sntz=1&usg=AOvVaw2gB81P3w8F6g-lR_F-H5w7).
*Note: The dataset is expected to be named `breast-cancer.csv` in the same directory as the Python script.*

## Tools and Libraries

The following Python libraries are used in this project:
* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical operations.
* `matplotlib`: For plotting and visualization.
* `scikit-learn`: For machine learning algorithms, including SVMs, data splitting, scaling, and hyperparameter tuning.

## How to Run the Code

1.  **Download the Dataset:**
    Save the Breast Cancer dataset as `breast-cancer.csv` in the same directory where you will save the `svm_classification.py` file.

2.  **Install Dependencies:**
    Ensure you have all the necessary Python libraries installed. You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```

3.  **Run the Python Script:**
    Execute the `svm_classification.py` script from your terminal:
    ```bash
    python svm_classification.py
    ```

## Project Structure

* `svm_classification.py`: The main Python script containing all the code for data preprocessing, model training, visualization, and evaluation.
* `breast-cancer.csv`: The input dataset.
* `X_train_preprocessed.csv`: Saved preprocessed training features.
* `X_test_preprocessed.csv`: Saved preprocessed testing features.
* `y_train_preprocessed.csv`: Saved preprocessed training target variable.
* `y_test_preprocessed.csv`: Saved preprocessed testing target variable.
* `Linear_SVM_Decision_Boundary_2D.png`: Generated image file showing the decision boundary for the Linear SVM.
* `RBF_SVM_Decision_Boundary_2D.png`: Generated image file showing the decision boundary for the RBF SVM.

## What You'll Learn

* **Margin Maximization:** Understanding how SVMs find an optimal hyperplane to separate classes by maximizing the margin.
* **Kernel Trick:** Learning how non-linear data can be classified by implicitly mapping it into higher-dimensional feature spaces.
* **Hyperparameter Tuning:** The importance of adjusting parameters like `C` (regularization) and `gamma` (kernel coefficient for RBF) to optimize model performance and prevent overfitting.
* **Cross-validation:** A technique for robust model evaluation and hyperparameter selection.

## Interview Questions

Here are answers to common interview questions related to Support Vector Machines:

1.  **What is a support vector?**
    Support vectors are the data points that lie closest to the decision boundary (hyperplane). They are the critical elements in defining the decision boundary and are the only points that directly influence the construction of the SVM model.

2.  **What does the C parameter do?**
    The `C` parameter in SVM controls the trade-off between achieving a low training error and a large margin.
    * A **small C** creates a larger margin but allows more misclassifications (higher bias, lower variance). This can lead to underfitting.
    * A **large C** creates a smaller margin but aims for fewer misclassifications (lower bias, higher variance). This can lead to overfitting.

3.  **What are kernels in SVM?**
    Kernels are functions that take data as input and transform it into the desired format. In SVMs, the kernel function computes the dot product of the data points in a higher-dimensional feature space without explicitly calculating the coordinates in that space. This "kernel trick" allows SVMs to model non-linear relationships.

4.  **What is the difference between linear and RBF kernel?**
    * **Linear Kernel:** This kernel computes a linear decision boundary. It is suitable when the data is linearly separable, or when a simple, straight-line separation is sufficient. Its formula is a simple dot product: $K(x, x_i) = x \cdot x_i$.
    * **RBF (Radial Basis Function) Kernel:** This is a non-linear kernel that creates a non-linear decision boundary. It is effective when the data is not linearly separable and can capture complex relationships. The RBF kernel uses a Gaussian function to define similarity: $K(x, x_i) = \exp(-\gamma \cdot ||x - x_i||^2)$. The $\gamma$ parameter influences the "reach" or influence of a single training example, affecting the smoothness of the decision boundary.

5.  **What are the advantages of SVM?**
    * **Effective in high-dimensional spaces:** Performs well even with many features.
    * **Memory efficient:** Uses a subset of training points (support vectors) in the decision function.
    * **Versatile:** Different kernel functions can be specified for the decision function.
    * **Robust to overfitting (with proper tuning):** Maximizing the margin helps in better generalization.

6.  **Can SVMs be used for regression?**
    Yes, SVMs can be extended for regression tasks, known as Support Vector Regression (SVR). Instead of finding a hyperplane that separates classes, SVR finds a hyperplane that best fits the data points while minimizing the error within a certain margin (epsilon-insensitive tube).

7.  **What happens when data is not linearly separable?**
    When data is not linearly separable, a simple linear decision boundary cannot effectively separate the classes. In such cases, the kernel trick is used with non-linear kernels (like RBF, polynomial, or sigmoid) to implicitly map the data into a higher-dimensional feature space where it might become linearly separable.

8.  **How is overfitting handled in SVM?**
    Overfitting in SVMs can be handled by:
    * **Tuning the C parameter:** A smaller `C` value increases the margin and allows more misclassifications, which can prevent the model from fitting the training data too closely.
    * **Choosing the right kernel:** For some datasets, a simpler kernel (e.g., linear) might prevent overfitting compared to a complex non-linear kernel if the underlying relationship is simple.
    * **Tuning the gamma parameter (for RBF/other non-linear kernels):** A larger `gamma` value can lead to a more complex decision boundary and potentially overfitting. A smaller `gamma` creates a smoother boundary and reduces overfitting.
    * **Cross-validation:** Using cross-validation during hyperparameter tuning helps to find the optimal parameters that generalize well to unseen data, rather than just performing well on the training set.
