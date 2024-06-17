import numpy as np
import pandas as pd
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from keras.models import Model
from tensorflow.keras import layers
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Input, TimeDistributed, GRU
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from keras.optimizers import Adam, Adamax, RMSprop, Nadam, Adadelta
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from keras.utils import to_categorical
#from torch.optim import AdamW
#from kdd_processing import load_data
from ton_processing import load_data

#from torch.optim import AdamW

# 加载数据
def calculate_class_accuracies(true_labels, pred_labels):
    # 获取混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    # 计算每个类别的准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    return class_accuracies

#train_all, train_all_label, test_all, test_all_label = load_data()
train = np.load('data/encoded_train_tonb.npy')  # (175341, 32)
train_label = np.load('data/train_label_tonb.npy')  # (175341, 10)
test = np.load('data/encoded_test_tonb.npy')  # (175341, 10)
test_label = np.load('data/test_label_tonb.npy')  #train shape:  (204210, 197)， (204317, 197)， (204424, 197) test shape: (51053, 197)




# 利用TimesereisGenerator生成序列数据
time_steps = 1
batch_size = 1024

# 先把训练集划分出一部分作为验证集


train1 = train_label[(time_steps-1):-1, :]
test1 = test_label[(time_steps-1):-1, :]

np.save('data/botb_train_label.npy', train1)
np.save('data/botb_test_label.npy', test1)




#print(val_data.shape[0])
# 数据集生成器
train_label_ = np.insert(train_label, 0, 0, axis=0)
test_label_ = np.insert(test_label, 0, 0, axis=0)

train_generator = TimeseriesGenerator(train, train_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(test, test_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)

input_traffic = Input(shape=(time_steps, 38 ))
#input_traffic = Input(shape=(38,1))
# 1 lstm layer, stateful=True
#gru1 = Bidirectional(GRU(units=29, activation='tanh',
#                          return_sequences=True, recurrent_dropout=0.1))(input_traffic)
lstm1 =  Bidirectional(LSTM(units=19, activation='tanh',
                           return_sequences=True))(input_traffic)
#gru_drop1 = Dropout(0.5)(gru1)
lstm_drop1 = Dropout(0.3)(lstm1)
#print(lstm_drop1.shape)
# 2 lstm layer, stateful=True
#gru2 =Bidirectional (GRU(units=19, activation='tanh', return_sequences=False,
#                           recurrent_dropout=0.1))(gru_drop1)
lstm2 =  Bidirectional(LSTM(units=19, activation='tanh', return_sequences=False,
                           recurrent_dropout=0.3))(lstm_drop1)
#gru_drop2 = Dropout(0.5)(gru2)
lstm_drop2 = Dropout(0.3)(lstm2)
#lstm3 = Bidirectional(LSTM(units=32, activation='tanh', return_sequences=False,
         #                   recurrent_dropout=0.1))(lstm_drop2)
#lstm_drop3 = Dropout(0.3)(lstm3)
# mlp
mlp = Dense(units=10, activation='relu')(lstm_drop2)
#mlp1 = Dense(units=10, activation='relu')(gru_drop2)
#mlp1 = Dense(units=19, activation='relu')(mlp)
mlp2 =  Dense(units=2, activation='sigmoid')(mlp)
classifier = Model(input_traffic, mlp2)

#optimize = Adadelta(learning_rate=0.9, rho=0.4)
#optimleize = AdamW(params=classifier.trainable_weights, lr=0.001, weight_decay=0.001)
optimize =Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
import timeit

start_time = timeit.default_timer()
#classifier.compile(optimizer=optimize, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
classifier.compile(optimizer=optimize, loss='binary_crossentropy', metrics=['binary_accuracy'])
elapsed = timeit.default_timer() - start_time
print(str(elapsed) + ' seconds')
# 设置一些callbacks
save_dir = os.path.join(os.getcwd(), 'models')
filepath="best_modelbotb1.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='binary_accuracy', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True,
                         write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
reduc_lr = ReduceLROnPlateau(monitor='val_binary_accuracy', patience=20, mode='max', factor=0.1, min_delta=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
# 拟合及预测
history = classifier.fit_generator(train_generator, epochs=100, verbose = 2,
                                   callbacks=[checkpoint, tbCallBack, reduc_lr, early_stopping],
                                   validation_data=test_generator)



classifier.load_weights('./models/best_modelbotb1.hdf5')
train_probabilities = classifier.predict_generator(train_generator)

train_pred = train_probabilities
print("shape is: ", train_pred.shape) # (68490, 5) (72519, 5) (77051, 5)


test_probabilities = classifier.predict_generator(test_generator)


test_pred = test_probabilities

np.save('data/train_botb1.npy', train_pred)

np.save('data/test_botb1.npy', test_pred)