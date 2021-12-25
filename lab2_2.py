from sklearn import neighbors 
from sklearn.metrics import confusion_matrix
import numpy as np
for i in range(10):
    predictors = np.random.random(500 * (i + 1)).reshape(250 * (i+1),2) 
    target = np.around(predictors.dot(np.array([0.4, 0.6])) + np.random.random(250 * (i + 1))) 
    clf = neighbors.KNeighborsClassifier(n_neighbors = i + 1) 
    knn = clf.fit(predictors, target) 
    print('Метрика соответствий: ', knn.score(predictors, target))

    prediction = knn.predict(predictors)
    print('Матрица несоответствий:\n', confusion_matrix(target, prediction))

    print('\n', 500 * (i+1), '\n', 250*(i+1), '\n', i+1)
