
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
import tensorflow as tf
import pickle
from am.Settings import  TARGET_COL
from sklearn.model_selection import train_test_split


def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def classificador(trainX, trainY, testX):
  classifier = KNeighborsClassifier(n_neighbors=8, leaf_size=1, p=1 )
  classifier.fit(trainX, trainY)
  y_pred = classifier.predict(testX)
  return y_pred


def matrix_confusion(y_test,y_pred):
  cm = confusion_matrix(y_test, y_pred)
  return cm

#Gerando calculador de metricas
def metrics (y_test, y_pred):
  ac = accuracy_score(y_test, y_pred)
  rec = recall_score(y_test,y_pred, average= 'weighted')
  prec = precision_score(y_test,y_pred, average='weighted')
  return (ac, rec, prec)


def printResult(y_test,y_pred):
  acuracia, recall, precisao = metrics(y_test, y_pred)
  stringSaida = "acuracia = {ac} , recall = {rec} , precisao = {prec} ".format(ac=acuracia, rec=recall, prec=precisao)
  print(stringSaida)
  print("\n")


with tf.device('/GPU:1'):
    # change name model
    df = pickle.load(open('modelo','rb'))

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model = KNeighborsClassifier(n_neighbors=8, leaf_size=1, p=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

y_pred = classificador(X_train, y_train, X_test)


printResult(y_test, y_pred)



