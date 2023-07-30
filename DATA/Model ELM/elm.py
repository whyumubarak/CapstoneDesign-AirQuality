import numpy as np
import pandas as pd
import joblib

# Memuat data dari file CSV menggunakan Pandas
df = pd.read_excel('D:\Data\Kuliah\TA\Projek-Capstone-Design\DATA\Classification\Data Latih.xlsx')
df['Kategori'] = df['Kategori'].astype(float)
df.info()

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

X = df.drop('Kategori', axis=1)
y = df['Kategori']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gunakan QuantileTransformer untuk melakukan transformasi quantile pada data
quantile_transformer = QuantileTransformer(output_distribution='normal', n_quantiles=200)
X_train_scaled = quantile_transformer.fit_transform(X_train)
X_test_scaled = quantile_transformer.transform(X_test)

import numpy as np
from numpy.linalg import pinv, inv
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

class elm():
    def __init__(self, hidden_units, activation_function, x, y, C, elm_type, one_hot=True, random_type='normal'):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = x
        self.y = y
        self.C = C
        self.class_num = np.unique(self.y).shape[0]
        self.beta = np.zeros((self.hidden_units, self.class_num))
        self.elm_type = elm_type
        self.one_hot = one_hot

        if elm_type == 'clf' and self.one_hot:
            self.one_hot_label = np.zeros((self.y.shape[0], self.class_num + 1))
            for i in range(self.y.shape[0]):
                self.one_hot_label[i, int(self.y[i])] = 1

        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=0, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=0, high=1, size=(self.hidden_units, 1))
        if self.random_type == 'normal':
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    def __input2hidden(self, x):
        #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
        self.temH = np.dot(self.W, x.T) + self.b

        if self.activation_function == 'sigmoid':
            self.H = 1 / (1 + np.exp(-self.temH))

        if self.activation_function == 'relu':
            self.H = self.temH * (self.temH > 0)

        if self.activation_function == 'sin':
            self.H = np.sin(self.temH)

        if self.activation_function == 'tanh':
            self.H = (np.exp(self.temH) - np.exp(-self.temH)) / (np.exp(self.temH) + np.exp(-self.temH))

        if self.activation_function == 'leaky_relu':
            self.H = np.maximum(0, self.temH) + 0.1 * np.minimum(0, self.temH)

        return self.H

    def __hidden2output(self, H):
        self.output = np.dot(H.T, self.beta)
        return self.output

    def fit(self, algorithm):
        self.time1 = time.perf_counter()
        self.H = self.__input2hidden(self.x)
        if self.elm_type == 'clf':
            if self.one_hot:
                self.y_temp = self.one_hot_label
            else:
                self.y_temp = self.y
        if self.elm_type == 'reg':
            self.y_temp = self.y

        if algorithm == 'no_re':
            self.beta = np.dot(pinv(self.H.T), self.y_temp)

        if algorithm == 'solution1':
            self.tmp1 = inv(np.eye(self.H.shape[0]) / self.C + np.dot(self.H, self.H.T))
            self.tmp2 = np.dot(self.tmp1, self.H)
            self.beta = np.dot(self.tmp2, self.y_temp)

        if algorithm == 'solution2':
            self.tmp1 = inv(np.eye(self.H.shape[0]) / self.C + np.dot(self.H, self.H.T))
            self.tmp2 = np.dot(self.H.T, self.tmp1)
            self.beta = np.dot(self.tmp2.T, self.y_temp)
        self.time2 = time.perf_counter()

        self.result = self.__hidden2output(self.H)

        if self.elm_type == 'clf':
            self.result = np.exp(self.result) / np.sum(np.exp(self.result), axis=1).reshape(-1, 1)

        if self.elm_type == 'clf':
            self.y_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]
            self.correct = 0
            for i in range(self.y.shape[0]):
                if self.y_[i] == self.y[i]:
                    self.correct += 1
            self.train_score = self.correct / self.y.shape[0]
        if self.elm_type == 'reg':
            self.train_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y)) / self.y.shape[0])
        train_time = str(self.time2 - self.time1)
        return self.beta, self.train_score, train_time

    def predict(self, x):
        #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
        self.H = self.__input2hidden(x)
        self.y_ = self.__hidden2output(self.H)
        if self.elm_type == 'clf':
            self.y_ = np.where(self.y_ == np.max(self.y_, axis=1).reshape(-1, 1))[1]

        return self.y_

    def predict_proba(self, x):
        #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
        self.H = self.__input2hidden(x)
        self.y_ = self.__hidden2output(self.H)
        if self.elm_type == 'clf':
            self.proba = np.exp(self.y_) / np.sum(np.exp(self.y_), axis=1).reshape(-1, 1)
        return self.proba

    def score(self, x, y):
        self.prediction = self.predict(x)
        if self.elm_type == 'clf':
            self.correct = 0
            for i in range(y.shape[0]):
                if self.prediction[i] == y[i]:
                    self.correct += 1
            self.test_score = self.correct / y.shape[0]
        if self.elm_type == 'reg':
            self.test_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y)) / self.y.shape[0])
        return self.test_score
    
import random
import numpy as np

random.seed(42)
np.random.seed(42)

model = elm(hidden_units=500, activation_function='sigmoid', C=1000, elm_type='clf', random_type='normal',
            x=X_train_scaled, y=y_train.astype(int).values.reshape(-1, 1))# Mengubah dimensi y_train menjadi (n_samples, 1)

# Melatih model dan menyimpannya ke dalam file .sav
beta, train_accuracy, running_time = model.fit('solution2')

# Melakukan prediksi dengan model yang telah dilatih
prediction = model.predict(X_test_scaled)

print('-----------------------------------------------------')
print("classifier train accuracy:", train_accuracy)
print('classifier test accuracy:', model.score(X_test_scaled, y_test.values.reshape(-1, 1)))  # Mengubah dimensi y_test menjadi (n_samples, 1)
print("\nclassifier test prediction:", prediction)
print('classifier running time:', running_time)
print('Akurasi model dengan membuang 15% data:', np.round(np.mean(prediction == y_test) * 100, 2), '%')

cm = confusion_matrix(y_test, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

from sklearn.metrics import classification_report
# Generate classification report
report = classification_report(y_test, prediction)
print(report)

import pickle
filename = 'ELMM.sav'
pickle.dump(model, open(filename,'wb'))

#python elm.py
