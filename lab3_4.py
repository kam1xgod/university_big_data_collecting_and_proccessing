import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn import cluster
import pandas as pd
for i in range(1,11):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns = list(data.feature_names))
    print (X[:5])
    model   = cluster.KMeans(n_clusters=i*3, random_state=25*i)
    results = model.fit(X)
    X["cluster"] = results.predict(X)
    X["target"] = data.target
    X["c"] = "lookatmeIamimportant"
    classification_result=X[["cluster", "target","c"]].groupby(["cluster","target"]).agg("count")
    print(classification_result)

