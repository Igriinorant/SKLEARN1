#mempersiapkan data
import pandas as pd
df1 = pd.read_csv('bensin2.csv')
print(df1)

#menampilkan data stastik dataset
df1.describe()

#memisahkan training data dan test data
import sklearn.model_selection as ms

liter = df1[['Liter']]
kilometer = df1[['Kilometer']]

X_train, X_test, y_train, y_test = ms.train_test_split(liter, kilometer, test_size=0.2,random_state=3)
print(X_train.size, X_test.size)

#Visualisasi
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(X_train, y_train,edgecolors='r')
plt.xlabel('Liter')
plt.ylabel('Kilometer')
plt.title('Konsumsi Bahan Bakar')
#menambahkan garis
x1 = np.linspace(0,45)
y1 = 4 +7 * x1
plt.plot(x1,y1)
plt.show()

#melatih model
import sklearn.linear_model as lm
model1 = lm.LinearRegression()
model1.fit (X_train, y_train)

#mengeluarkan nilai slope hasil perhitungan
model1.coef_

#mengeluarkan nilai intercept hasil perhitungan
model1.intercept_

#menambahkan garis dengan nilai slope dan intercept
plt.scatter(X_train, y_train,edgecolors='r')
plt.xlabel('Liter')
plt.ylabel('Kilometer')
plt.title('Konsumsi Bahan Bakar')
x1 = np.linspace(0,45)
y1 = 10.64 + 6.45 * x1
plt.plot(x1,y1)
plt.show()

#Scoring Model
r2 = model1.score(X_test, y_test)
print(r2)

#Melakukan prediksi dengan model
jarak = model1.predict([[60]])
print(jarak)