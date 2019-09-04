import numpy as np
import scipy.stats as stats
from keras.models import load_model
from sklearn.metrics import mean_absolute_error


robust = np.loadtxt('./robust_lenet1_test.csv')[:1000]
features = np.loadtxt('./features_test.csv')

model = load_model('./features_regress.h5')
results = model.predict(features).reshape(1000,)

# np.savetxt('features_results.csv', results)

corr = stats.pearsonr(results, robust)[0]
print('pcc', corr)
print('mae: ', mean_absolute_error(robust, results))

