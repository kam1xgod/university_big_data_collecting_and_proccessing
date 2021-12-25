import tarfile
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import load_svmlight_file
import numpy as np

uri = 'D:\\url_svmlight.tar.gz'
tar = tarfile.open(uri,"r:gz")
max_obs = 0
max_vars = 0
i = 0
split = 4
for tarinfo in tar:
    print(" extracting %s,f size %s" % (tarinfo.name, tarinfo.size))
    if tarinfo.isfile():
        f = tar.extractfile(tarinfo.name)
        X,y = load_svmlight_file(f)
        max_vars = np.maximum(max_vars, X.shape[0])
        max_obs = np.maximum(max_obs, X.shape[1])
    if i > split:
        break
    i+= 1
print("max X = %s, max y dimension = %s" % (max_obs, max_vars))

classes = [-1,1]
sgd = SGDClassifier(loss="log")
n_features=3231961
split = 4
i = 0
for tarinfo in tar:
    if i > split:
        break
    if tarinfo.isfile():
        f = tar.extractfile(tarinfo.name)
        X,y = load_svmlight_file(f,n_features=n_features)
        if i < split:
            sgd.partial_fit(X, y, classes=classes)
        if i == split:
            print (classification_report(sgd.predict(X),y))
i += 1
