from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
url = "D:\\heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(url, sep= ",")
X = data[[u'age', u'anaemia', u'creatinine_phosphokinase', u'diabetes', u'ejection_fraction', u'high_blood_pressure', u'platelets', u'serum_creatinine', u'serum_sodium', u'sex', u'smoking', u'time',]]
y = data.DEATH_EVENT
X = preprocessing.StandardScaler().fit(X).transform(X)
gnb = GaussianNB()
predicted_correct = []
for i in range(1,10):
    model = PCA(n_components = i)
    results = model.fit(X)
    Z = results.transform(X)
    fit = gnb.fit(Z,y)
    pred = fit.predict(Z)
    predicted_correct.append(confusion_matrix(pred,y).trace())
    print (predicted_correct)
    plt.plot(predicted_correct)
plt.show()
