
# coding: utf-8

# In[11]:


MAX_LEN = 20
feature_SIZE = 1000
import numpy as np
import keras
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.utils import np_utils


# In[ ]:


size = 87898
with open("./data/PUBG_data.txt", encoding = 'utf-8') as data:
    data_padded0 = data.readlines()[0:size]
with open("./data/PUBG_label.txt", encoding = 'utf-8') as label:
    label_padded0 = label.readlines()[0:size]
print(data_padded0.shape)
print(label_padded0.shape)
for i in range(len(data_padded0)):
    data_padded0[i] = (data_padded0[i].split())
for i in range(len(label_padded0)):
    label_padded0[i] = (label_padded0[i].split())

for i in data_padded0:
    data_padded = keras.preprocessing.sequence.pad_sequences(data_padded0, maxlen=MAX_LEN, dtype='int32',
                                           padding='post', truncating='post', value=0.)
for i in label_padded0:
    label_padded = keras.preprocessing.sequence.pad_sequences(label_padded0, maxlen=MAX_LEN, dtype='int32',
                                           padding='post', truncating='post', value=0.)
# one-hot      
train_data = []
#train_label = []
for i in range(len(data_padded)):
    #train_data.append(np_utils.to_categorical(data_padded[i], feature_SIZE + 2))
    train_label.append(np_utils.to_categorical(label_padded[i], feature_SIZE + 2))
#train_data = np.array(train_data)
train_label = np.array(train_label)


# In[2]:


BATCH_SIZE = 100
NUM_LAYERS = 5
HIDDEN_DIM = 128
EPOCHS = 10


# In[3]:


def create_UniLSTM(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    MODEL = Sequential()
    MODEL.add(Embedding(X_vocab_len+1, 300, input_length=X_max_len, mask_zero=True))
    MODEL.add(LSTM(hidden_size))
    MODEL.add(RepeatVector(X_max_len))
    MODEL.add(LSTM(hidden_size, return_sequences=True))
    MODEL.add(TimeDistributed(Dense(y_vocab_len, activation='softmax')))
    return MODEL


# In[12]:


from random import randint
def generate_batch_data_random(x, y, batch_size):
    xlen = len(x)
    loopcount = xlen
    while (True):
        i = randint(0,loopcount-2*batch_size)
        x1 = x[i :i + batch_size]
        y1 = y[i :i + batch_size]
        xp = []
        yp = []
        for i in range(batch_size):
            xp.append(x1[i])
            yp.append(np_utils.to_categorical(y1[i], VOCAB_SIZE + 2))
        yield np.array(xp),np.array(yp)

def generate_valid_data_random(x, y, batch_size):
    xlen = len(x)
    loopcount = xlen
    while (True):
        i = randint(0,loopcount-2*batch_size)
        x1 = x[i :i + batch_size]
        y1 = y[i :i + batch_size]
        xp = []
        yp = []
        for i in range(batch_size):
            xp.append(x1[i])
            yp.append(np_utils.to_categorical(y1[i], VOCAB_SIZE + 2))
        yield np.array(xp),np.array(yp)


# define model
train = create_UniLSTM(feature_SIZE+2, MAX_LEN, feature_SIZE+2, MAX_LEN, HIDDEN_DIM, NUM_LAYERS)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
train.summary()


# In[ ]:


# train model
tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
epochsave = keras.callbacks.ModelCheckpoint('./save',period=10)

train.fit_generator(generate_batch_data_random(feature_padded[0:(int(0.8*size)-1)], winrates_padded[0:(int(0.8*size)-1)], BATCH_SIZE),                                                      
    steps_per_epoch = 200,
    epochs = 100, 
    validation_data = generate_valid_data_random(feature_padded[int(0.8*size):(size-1)], winrates_padded[int(0.8*size):(size-1)], BATCH_SIZE), 
    validation_steps = 10, 
    initial_epoch = 0,
    callbacks = [tensorboard,epochsave])


# In[ ]:


# TO-DO
import keras
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding

from random import randint
def generate_batch_data_random(x, y, batch_size):
    xlen = len(x)
    loopcount = xlen
    while (True):
        i = randint(0,loopcount-2*batch_size)
        x1 = x[i :i + batch_size]
        y1 = y[i :i + batch_size]
        xp = []
        yp = []
        for i in range(batch_size):
            xp.append(x1[i])
            yp.append(np_utils.to_categorical(y1[i], VOCAB_SIZE + 2))
        yield np.array(xp),np.array(yp)

def generate_valid_data_random(x, y, batch_size):
    xlen = len(x)
    loopcount = xlen
    while (True):
        i = randint(0,loopcount-2*batch_size)
        x1 = x[i :i + batch_size]
        y1 = y[i :i + batch_size]
        xp = []
        yp = []
        for i in range(batch_size):
            xp.append(x1[i])
            yp.append(np_utils.to_categorical(y1[i], VOCAB_SIZE + 2))
        yield np.array(xp),np.array(yp)


# define model
train = create_UniLSTM(VOCAB_SIZE+2, MAX_LEN, VOCAB_SIZE+2, MAX_LEN, HIDDEN_DIM, NUM_LAYERS)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
train.summary()
# train model
tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
epochsave = keras.callbacks.ModelCheckpoint('./save1',period=10)

train.fit_generator(generate_batch_data_random(data_padded[0:(int(0.8*size)-1)], label_padded[0:(int(0.8*size)-1)], BATCH_SIZE),                                                      
    steps_per_epoch = 200,
    epochs = 100, 
    validation_data = generate_valid_data_random(data_padded[int(0.8*size):(size-1)], label_padded[int(0.8*size):(size-1)], BATCH_SIZE), 
    validation_steps = 10, 
    callbacks = [tensorboard,epochsave])
train.save('model_without_attention.h5')





