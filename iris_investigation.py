import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#%%
iris = pd.read_csv('iris_all3.csv')
iris.head()
#%%
sns.FacetGrid(iris, hue='species', size=4).map(plt.scatter, 'sepal_length',
                                               'sepal_width').add_legend(loc=1)
plt.show()
# %%
sns.set_style('whitegrid')
sns.pairplot(iris, hue='species', size=3)
plt.show()
#%%
x = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris['species'].values
#%%
scaler_x = MinMaxScaler()
scaler_x.fit(x)
x_scaled = scaler_x.transform(x)
#%%
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y,
                                                    test_size=0.2,
                                                    random_state=0)
#%%
enc = OneHotEncoder(categories='auto')
y_enc_train = enc.fit_transform(y_train[:, np.newaxis]).toarray()
y_enc_test = enc.fit_transform(y_test[:, np.newaxis]).toarray()
# %%
model = Sequential()
model.add(Dense(4, input_dim=4, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
#%%
print(model.summary())
#%%
history_callback = model.fit(x_train, y_enc_train,
                                 batch_size=5,
                                 epochs=500,
                                 verbose=0,
                                 validation_data=(x_test, y_enc_test),)

score = model.evaluate(x_test, y_enc_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Test loss: 0.16588877141475677
# Test accuracy: 1.0
#%%
pred = model.predict(x_test)
#%%
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
val_acc = history_callback.history['val_accuracy']
val_loss = history_callback.history['val_loss']
ax1.plot(val_acc)
ax2.plot(val_loss)
ax1.set_ylabel('validation accuracy')
ax2.set_ylabel('validation loss')
ax2.set_xlabel('epochs')
# ax1.legend()
# ax2.legend()
plt.show()
#%%
clf = AdaBoostClassifier(n_estimators=500)
clf.fit(x_train, y_train)
#%%
clf.score(x_test, y_test)
# 0.9
#%%
ada_pred = clf.predict(x_test)
accuracy_score(y_test, ada_pred)
#%%
confusion_matrix(y_test, ada_pred)
"""
array([[11,  0,  0],
       [ 0, 13,  0],
       [ 0,  3,  3]], dtype=int64)
"""
#%%
class_rep = classification_report(y_test, ada_pred)
"""
precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        11
  versicolor       0.81      1.00      0.90        13
   virginica       1.00      0.50      0.67         6

    accuracy                           0.90        30
   macro avg       0.94      0.83      0.85        30
weighted avg       0.92      0.90      0.89        30
"""
#%%