# Importando Biliotecas Importantes
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from matplotlib import pyplot
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt
import glob
import pandas as pd

DATA_COLS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "target"]
window = 5
features = 1
HZ = 11
XYZ = 3
look_back = window * HZ

#3867

def loadFiles(path):
    l = [pd.read_csv(filename, names=DATA_COLS) for filename in glob.glob("..//data//" + str(path) + "//*.csv")]
    dataframe = pd.concat(l, axis=0)
    values = dataframe.values
    return values

def dadosY(v, time):
    target = []
    for t in range(int(time)):
        a = v[t:(t + look_back), 34][0]
        target.append(a)

    return target


def dadosX(v, time):
    treino = list()
    start = 0
    for t in range(int(time)):
        xyz = []
        a = []
        if t == 0:
            end = look_back
        else:
            end = (t + 1) * look_back

        if (start + look_back) < len(v):
            a = v[start:end, 0:34]
        else:
            a = v[start:len(v) - 1, 0:34]

        for i in range(len(a)):
            sub = a[i]
            for s in range(len(sub)):
                xyz.append(sub[s])
        treino.append(xyz)
        start = end
    return treino


def toTarget(target, teste):
    for i in range(len(target)):
        teste.append(target[i])


def defineWindowY(values):
    teste = []
    loop = int(values.shape[0] / (look_back * features))  # 208 X 468 X 5
    start = 1
    # HZ * features * XYZ
    for i in range(loop):
        calc = int(values.shape[0] / loop)
        if (start) < len(values):
            v = values[start: (i + 1) * calc - 1, 34][0]
            teste.append(v)
        else:
            v = values[len(values) - 1][34]
            teste.append(v)

        start = start + calc
    return np.array(teste)


def defineWindowX(values):
    treino = list()
    size = int(values.shape[0] / features)
    end = size
    for i in range(features):
        v = values[(i * size):end]
        treino.append(np.array(dadosX(v, len(v[0:end]) / (int(look_back)))))
        end += size
    return np.array(treino)


###TREINO
train = loadFiles("model3")
trainX = defineWindowX(train)
trainX = np.dstack(trainX)
trainY = defineWindowY(train)

test = loadFiles("model3//teste")
testX = defineWindowX(test)
testX = np.dstack(testX)
testY = defineWindowY(test)

verbose, epochs, batch_size = 1, 15, 64
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], 1
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(LSTM(units=8, return_sequences=True, dropout=0.5))
model.add(LSTM(units=8, dropout=0.5))
model.add(Dense(units=25, activation="relu"))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='categorical_hinge', metrics=['accuracy'])

print(model.summary())

history = model.fit(x=trainX,
                        y=trainY,
                        batch_size=1,
                        epochs=100,
                        verbose=verbose,
                        callbacks=[callback],
                        validation_data=(testX, testY),
                        shuffle=True
                        )


pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss LSTM')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

