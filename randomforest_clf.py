import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('krediVeriseti.csv', delimiter = ';')
df["evDurumu"].replace({"evsahibi": 1, "kiraci": 0}, inplace=True)
df["telefonDurumu"].replace({"var": 1, "yok": 0}, inplace=True)
df["KrediDurumu"].replace({"krediver": 1, "verme": 0}, inplace=True)
print(df.head())

X = np.array(df.iloc[:,:5])
y = np.array(df.iloc[:,5:]).reshape(len(df["KrediDurumu"]),)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)
#print(X_train.shape, X_test.shape)
#print(y_train.shape, y_test.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = RandomForestClassifier(max_depth = 5, n_estimators = 20)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(score)

y_tahmin = clf.predict(X_test)

print(confusion_matrix(y_test, y_tahmin))
