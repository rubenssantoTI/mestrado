from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
from am.Settings import TARGET_COL
import matplotlib.pyplot as plt
from sklearn import svm

def classificadorSVM(trainX, trainY):
  model = svm.SVC(kernel='poly', C=1, gamma=1)
  model.fit(trainX, trainY)

  return model


def matrix_confusion(y_test,y_pred):
  cm = confusion_matrix(y_test, y_pred)
  return cm


def metrics (y_test, y_pred):
  ac = accuracy_score(y_test, y_pred)
  rec = recall_score(y_test,y_pred, average= 'weighted')
  prec = precision_score(y_test,y_pred, average='weighted')
  return (ac, rec, prec)


def print_result(y_test,y_pred):
  acuracia, recall, precisao = metrics(y_test, y_pred)
  stringSaida = "acuracia = {ac} , recall = {rec} , precisao = {prec} ".format(ac=acuracia, rec=recall, prec=precisao)
  print(stringSaida)
  print("\n")

def cm_to_inch(value):
    return value/2.54

with tf.device('/GPU:1'):
    df = pickle.load(open('modelo-AM_60', 'rb'))

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    model = svm.SVC(kernel='poly', C=5, gamma=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_result(y_test, y_pred)

   
