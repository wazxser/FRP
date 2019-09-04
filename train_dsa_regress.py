from sklearn import linear_model
import numpy as np
from scipy.stats import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error


robust_train = np.loadtxt('./robust_lenet1_sa.csv')
robust_test = np.loadtxt('./robust_lenet1_test.csv')[:1000]
sa_train = np.loadtxt('./dsa_lenet1_sa.csv')
sa_test = np.loadtxt('./dsa_lenet1.csv')[:1000]

pcc = .0
for i in range(10):
    polynomial = PolynomialFeatures(degree=i+1)
    x_transformed = polynomial.fit_transform(sa_train.reshape(-1, 1))

    regr = linear_model.LinearRegression()
    regr.fit(x_transformed, robust_train)

    results = regr.predict(polynomial.fit_transform(sa_test.reshape(-1, 1)))
    # np.savetxt('./results_dsa.csv', results, '%s')

    corr = stats.pearsonr(results, robust_test)
    if corr[0] > pcc:
        print('pcc: ', corr[0])
        print('mae: ', mean_absolute_error(robust_test, results))
        pcc = corr[0]
