import statsmodels.api as sm
import numpy as np
for i in range(10):
    predictors_param_one = 200 * (i + 1)
    predictors_param_one_half = int(predictors_param_one / 2)
    predictors = np.random.random(predictors_param_one).reshape(predictors_param_one_half, 2)
    target = predictors.dot(np.array([0.4, 0.6])) + np.random.random(predictors_param_one_half)
    lmRegModel = sm.OLS(target, predictors)
    result = lmRegModel.fit()

    print(predictors_param_one, " - количество случайных значений.")
    print(predictors_param_one_half, " - shape.")
    print('Результаты регрессии.')
    print(result.summary())
