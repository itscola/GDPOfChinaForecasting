import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Dataset = pd.read_csv('ChinaGDP.csv')
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values

# plt.scatter(X, y)
# plt.xlabel('Year')
# plt.ylabel('GDP of China')
# plt.title('GDP')
# plt.show()

from sklearn.preprocessing import PolynomialFeatures
Poly_regressor = PolynomialFeatures(degree=2)
Poly_X = Poly_regressor.fit_transform(X)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Poly_X,y)

plt.scatter(X,y)
plt.plot(X,regressor.predict(Poly_regressor.fit_transform(X)),color='red')
plt.xlabel('Year')
plt.ylabel('GDP of China')
plt.title('GDP')
plt.show()

print(regressor.predict(Poly_regressor.fit_transform([[2021]])))
print(regressor.predict(Poly_regressor.fit_transform([[2035]])))
