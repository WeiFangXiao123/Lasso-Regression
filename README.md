## Predicting MLB Player Salaries with Lasso Regression
This project applies **Lasso Regression** to predict **Major League Baseball (MLB) players' 1987 salaries** using historical performance and career statistics. The data includes 322 players and 19 predictor variables. The response variable is the opening day salary in thousands of dollars.
The analysis focuses on how Lasso improves prediction accuracy by performing **automatic variable selection** and **regularization**, helping to reduce overfitting when many correlated predictors are present.

### Objective
This project builds Lasso Regression models to predict player salaries, using **cross-validation** to select the optimal regularization parameter (```lambda```). It compares **mean-squared error** of ```lambda.large```, ```lambda.small```, and ```lambda.best```. 

### Methodology
**1. Data Preparation:**
- Remove NA rows in Salary
- Convert categorical predictors using model.matrix

**2. Lasso Model Fitting:**

```
fit <- cv.glmnet(x, y)
```
**3. Cross-Validation:**
```
lambda.large <- fit$cvm[which.max(fit$lambda)]
lambda.best <- fit$cvm[fit$lambda == lambda.best]
lambda.small <- fit$cvm[which.min(fit$lambda)]
```

**4. Model Interpretation:**
- Plot shrinkage paths
- Identify selected variables at optimal lambda

### Key Results
The smallest value is ```lambda.best``` with a cross-validation MSE of 109,665.3. This value is selected because it **minimizes the mean squared error during cross-validation**, making it the **most optimal balance** between bias and variance. In contrast, ```lambda.small``` corresponds to the smallest lambda value tested, which often leads to overfitting due to minimal regularization. On the other hand, ```lambda.large``` is associated with the highest lambda value, resulting in **underfitting due to excessive shrinkage of coefficients**.
