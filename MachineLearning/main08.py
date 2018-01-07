# !/usr/bin/python
# -*- coding: UTF-8 -*-

##########################
# Creator: Javy
# Crete Time: 20170512
# Email: xujavy@gmail.com
# Description: Machine Learning - Chapter ten
##########################

import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-database/housing/housing.data')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B'
                'LSTAT', 'MEDV']
df.head()

# Sample one
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
cols = ["LSTAT", 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

## sns.reset_orig()

# Sample two
import numpy as np
cm = np.corrcoef(de[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True,
                square=True, fmt='.2f', annot_kws={'size':15}
                tticklabels=cols, xticklabels=cols)
plt.show()

class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w__ = np.zeros(1 + X.shape[1])
        self.cost__ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            slef.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost__.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, Self.w_[1:] + self.w_[0])

    def predict(slef, X):
        return self.net_input(X)

# Sample three
X = df['RM'].values
y = df.['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (Standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (Standardized)')
plt.show()

num_rooms_std - sc.x.transformat([5.0])
price_std = lr.predict(num_rooms_std)
print("Price in $1000's: %.3f " sc.y.inverse_transform(price_std))
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.Intercept_)

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM] (Standardized))
plt.ylabel("Price in $1000\'s [MEDV] (Standardized)")
plt.show()

# Sample four
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(), max_trials=100,
                        min_samples=50, residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                        residual_threshold=5.0, random_state=0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='lightgreeen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Avaerage number of rooms [RM]')
plt.yabel('Price in #1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

print('Slope: %3.f' % ransac.estimator_.coef_[0])
print('Intercept: 5.3f' % ransac.estimator_.Intercept_)

# Sample Five
from sklearn.cross_validation import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_std = slr.predict(X_train)
y_test_std = sld.predict(y_test)

plt.scatter(y_train_std, y_train_std - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_std, y_test_std - y_test,
            c='lightgreeen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

from sklearn.metrics import mean_squared_error
print('MES train: %3.f, test: %3.f ' %
    mean_squared_error(y_train, y_train_std), mean_squared_error(y_test, y_test_std))

from sklearn.metrics import r2_score
print('R^2 train: %3.f, test: %3.f' %
        (r2_score(y_train, y_train_std)),
        (r2_score(y_test, y_test_std)))

from sklearn.linear_model inport Ridge
ridge = Ridge(alpha=1.0)

from sklearn.linear_model import Lasso
Lasso = Lasso(alpha=1.0)

from sklearn.linear_model import ElasticNet
Lasso = ElasticNet(alpha=1.0, 11_ratio=0.5)

# Sample six
from sklearn.preprocessing import PolynomialFeatures
X = np.array([258.0, 270.0, 294.0,
                     320.0, 342.0, 368.0,
                     396.0, 446.0, 480.0,
                     586.0])[:, np.newaxis]

y = np.array([236.4, 234.4, 252.8,
              298.6, 314.2, 342.2,
              360.8, 360.0, 391.2,
              390.8])

lr = LinearRegression()
pr = LinearRegression()

qudratic = PolynomialFeatures(degree=2)
X_quad = qudratic.fit_transform(X)

lr.fit(X, y)
X_fit = np.array(250, 600, 10)[:,np.newaxis]
y_lin_fit = lr.predict(X_fit)

pr.fit（X_quad, y）
y_quad_fit = pr.predict(qudratic.fit_transform(X_fit))
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit,
         label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit,
         label='qudratic fit')
plt.legend(loc='upper left')
plt.show()

y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print('Training MSE linear: {0}, qudratic: {1}'.format(mean_squared_error(y, y_lin_pred,
                                                       mean_squared_error(y, y_quad_pred))))
print('Training R^2 linear: {0}, qudratic: {1}'.format(r2_score(y, y_lin_pred,
                                                       r2_score(y, y_quad_pred))))

# Sample seven
X = df[['LSTAT']].values
Y = df[['MEDV']].values
regr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

X_fit = np.arange(X.min(), X.max())[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

plt.scatter(X, y
            label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit,
         lable='linear (d=1), $R^2={0}$'.format(linear_r2),
         color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit,
         lable='linear (d=2), $R^2={0}$'.format(quadratic_r2),
         color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit,
         lable='linear (d=3), $R^2={0}$'.format(cubic_r2),
         color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()

X_log = np.log(X)
y_sqrt = np.sqrt(y)

X_fit = np.arange(X_log.min()-1,
                  X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

plt.scatter(X_log, y_sqrt,
            label='linear (d=1), $R^2={0}$'.format(linear_r2),
            color='blue', lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]$')
plt.legend(loc='upper left')
plt.show()

# Sample eight
from sklearn.tree import DecisionTreeRegressor
X = df[['LSATA']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT])')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()

# Sample nine
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.ensemble import RandomForestRegree
forest = RandomForestRegree(n_estimators=1000,
                            criterion='mse',
                            random_state=1,
                            n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print('MSE train: {0}, test: {1}'.format(mean_squared_error(y_train, y_train_pred),
                                         mean_squared_error(y_test, y_test_pred)))
print('R^2 train: {0}, test: {1}'.format(r2_score(y_train, y_train_pred),
                                         r2_score(y_test, y_test_pred)))

plt.scatter(y_train_pred, y_train_pred - y_train,
            c='black', marker='o', s=35, alpha=0.5,
            label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
            c='lightgreeen', marker='s', s=35,
            alpha=0.7, label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
