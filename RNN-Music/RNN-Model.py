import numpy as np
import os
import tensorflow as tf
import more_itertools
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

def rolling_window(a, T,s):
    #In: 'a' list or and array
    #    'window' The size of the rolling window
    #    's' The stride of the rolling window
    #Out: An array with all the rolling windows of size T and stride s of an array a 
    aux = more_itertools.windowed(a,n=T, step=s,fillvalue=-1)    
    return np.asarray(list(aux))

def change_tone(p,t):
    #In: 'p' the piano roll notation of a melody
    #    't' the number of tones to transpose
    n = len(p)
    p_aux = p.copy()
    for i in range(n):
        if p_aux[i]!=-1.:
            p_aux[i]=  p[i]+t
    return p_aux
            
def pianoroll_onehot(p):
  #In: 'p' a piano roll
  #Out: the one hot encoder of the piano roll  
  n = len(p)
  I = np.identity(88)
  X = np.zeros([88,n])
  for i in range(n):
    if p[i]!=-1:
      X[:,i] = I[int(p[i]),:]
  return X

class voice:
    def __init__(self, name):
        self.name = name
        self.note2index = {}
        self.note2count = {}
        self.index2note = {}
        self.n_notes = 0  # Count the number of notes

    def add(self, inp):      
      for note in inp:
            self.addNote(note)

    def addNote(self, note):
        #add Note if its not already in the 'vocabulary'
        if note not in self.note2index:
            self.note2index[note] = self.n_notes
            self.note2count[note] = 1
            self.index2note[self.n_notes] = note
            self.n_notes += 1
        else:
            self.note2count[note] += 1
    
    def note2index_f(self,input):
      aux = []
      for note in input:
        aux.append(self.note2index[note])
      return aux

class LSTM_Armonia(tf.keras.Model):
  def __init__(self,voice_1_tokens,voice_2_tokens,embedding_dim, l, hidden_state_dimensions):
    super().__init__()
    self.l = l
    #An embedding layer per voice
    self.embedding_voice1 = tf.keras.layers.Embedding(voice_1_tokens,embedding_dim)
    self.embedding_voice2 = tf.keras.layers.Embedding(voice_2_tokens,embedding_dim)
    #The number LSTM layers in the model
    self.LSTM = []
    for i in range(l):
        self.LSTM.append(tf.keras.layers.LSTM(hidden_state_dimensions[i],
                                       return_sequences=True,
                                       return_state=True,
                                       activation ='tanh'))
        
    self.dense = tf.keras.layers.Dense(voice_2_tokens)

  def call(self, inputs,states = None, return_state=False, training=False):
    #A many to many RNN model
    aux_n = inputs.shape[1]//2
    voice1 = inputs[:,0:aux_n]
    voice2 = inputs[:,aux_n:]
    voice1 = self.embedding_voice1(voice1,training=training)
    voice2 = self.embedding_voice2(voice2,training=training)
    x  = tf.keras.layers.Concatenate(axis=-1)([voice1, voice2])
    if states is None:
        states = self.l*[None]
    for i in range(self.l):
        if states[i] is None:
            states[i] = self.LSTM[i].get_initial_state(x)
        x, states_h,states_c = self.LSTM[i](x, initial_state=states[i], training=training)
        states[i] = [states_h,states_c]
    x = self.dense(x)
    if return_state:
      return x, states
    else:
      return x





#Piano roll and rolling window data generator

T = 32
s = 1
voice_1 = []
voice_2 = []
voice_2_input = []

voice_1_tokenizer = voice('voice1')
voice_2_tokenizer = voice('voice2')

for invent in os.listdir('PianoRoll/Inventions/'):
    if invent !='.DS_Store':
        aux = np.loadtxt('PianoRoll/Inventions/'+invent).tolist()
        aux.append(aux[1].copy())
        aux[-1].pop()
        aux[-1].insert(0,-1)
        X = rolling_window(aux[0],T,s)
        Y = rolling_window(aux[1],T,s)
        Y_input = rolling_window(aux[2],T,s)
        for i in zip(X,Y,Y_input):
            for j in range(-6,6):
                voice_1_aux = change_tone(i[0],j)
                voice_1_tokenizer.add(voice_1_aux)
                voice_1.append(voice_1_tokenizer.note2index_f(voice_1_aux))
                
                voice_2_aux = change_tone(i[1],j)
                voice_2_tokenizer.add(voice_2_aux)
                voice_2.append(voice_2_tokenizer.note2index_f(voice_2_aux))
                
                
                voice_2_input_aux = change_tone(i[2],j)
                voice_2_tokenizer.add(voice_2_input_aux)
                voice_2_input.append(voice_2_tokenizer.note2index_f(voice_2_input_aux))
                
for invent in os.listdir('PianoRoll/Sinfonias/'):
    if invent !='.DS_Store':
        aux = np.loadtxt('PianoRoll/Sinfonias/'+invent).tolist()[0:2]
        aux.append(aux[1].copy())
        aux[-1].pop()
        aux[-1].insert(0,-1)
        X = rolling_window(aux[0],T,s)
        Y = rolling_window(aux[1],T,s)
        Y_input = rolling_window(aux[2],T,s)
        for i in zip(X,Y,Y_input):
            for j in range(-6,6):
                voice_1_aux = change_tone(i[0],j)
                voice_1_tokenizer.add(voice_1_aux)
                voice_1.append(voice_1_tokenizer.note2index_f(voice_1_aux))
                
                voice_2_aux = change_tone(i[1],j)
                voice_2_tokenizer.add(voice_2_aux)
                voice_2.append(voice_2_tokenizer.note2index_f(voice_2_aux))
                
                
                voice_2_input_aux = change_tone(i[2],j)
                voice_2_tokenizer.add(voice_2_input_aux)
                voice_2_input.append(voice_2_tokenizer.note2index_f(voice_2_input_aux))

for invent in os.listdir('PianoRoll/Sinfonias/'):
    if invent !='.DS_Store':
        aux = np.loadtxt('PianoRoll/Sinfonias/'+invent).tolist()[1:3]
        aux.append(aux[1].copy())
        aux[-1].pop()
        aux[-1].insert(0,-1)
        X = rolling_window(aux[0],T,s)
        Y = rolling_window(aux[1],T,s)
        Y_input = rolling_window(aux[2],T,s)
        for i in zip(X,Y,Y_input):
            for j in range(-6,6):
                voice_1_aux = change_tone(i[0],j)
                voice_1_tokenizer.add(voice_1_aux)
                voice_1.append(voice_1_tokenizer.note2index_f(voice_1_aux))
                
                voice_2_aux = change_tone(i[1],j)
                voice_2_tokenizer.add(voice_2_aux)
                voice_2.append(voice_2_tokenizer.note2index_f(voice_2_aux))
                
                
                voice_2_input_aux = change_tone(i[2],j)
                voice_2_tokenizer.add(voice_2_input_aux)
                voice_2_input.append(voice_2_tokenizer.note2index_f(voice_2_input_aux))

with open('Harmony/VoiceClass/voice1Tokenizer.pickle', 'wb') as file:
    pickle.dump(voice_1_tokenizer, file) 
with open('Harmony/VoiceClass/voice2Tokenizer.pickle', 'wb') as file:
    pickle.dump(voice_2_tokenizer, file) 


#Creation of the dataset in tensorflow
dataset_in =  np.concatenate((voice_1,voice_2_input),axis = 1)
dataset_out = voice_2

X_train, X_val, y_train, y_val = train_test_split(dataset_in, dataset_out, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = np.asarray(X_train),np.asarray(X_val),np.asarray(y_train),np.asarray(y_val)
dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))

#Training settings
BATCH_SIZE = 128
BUFFER_SIZE = 10000
dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

#Hyperparameters of all model to do a grid search.

hyperparameters = [{'l': 1, 'hidden': [32]},{'l': 1, 'hidden': [64]},{'l': 1, 'hidden': [128]},{'l': 1, 'hidden': [256]},{'l': 1, 'hidden': [512]},
                  {'l': 2, 'hidden': [64,64]},{'l': 2, 'hidden': [128,128]},{'l': 2, 'hidden': [256,256]},{'l': 2, 'hidden': [512,512]},
                  {'l': 3, 'hidden': [128,64,128]},{'l': 3, 'hidden': [256,128,256]},{'l': 3, 'hidden': [512,256,512]}]

metrics = []

for hp in hyperparameters:
  LSTM_Bach = LSTM_Armonia(voice_1_tokenizer.n_notes,voice_2_tokenizer.n_notes,30,hp['l'],hp['hidden'])
  for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = LSTM_Bach(input_example_batch)

  LSTM_Bach.summary()
  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

  example_batch_loss = loss(target_example_batch, example_batch_predictions)
  mean_loss = example_batch_loss.numpy().mean()
  print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
  print("Mean loss:        ", mean_loss)

  LSTM_Bach.compile(optimizer='adam', loss=loss,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


  EPOCHS = 1
  history = LSTM_Bach.fit(dataset, epochs=EPOCHS,validation_data=(X_val, y_val))
  metrics.append({'layers': hp['l'],
                  'hidden': hp['hidden'],
                  'train loss': history.history['loss'][-1],
                  'train accuaricy': history.history['sparse_categorical_accuracy'][-1],
                  'val_loss': history.history['val_loss'][-1],
                  'val_sparse_categorical_accuracy': history.history['val_sparse_categorical_accuracy'][-1]
                  })
  LSTM_Bach.save(str(hp['l'])+' + '+str(hp['hidden']))
  
metrics = pd.DataFrame(metrics)  

print(metrics.iloc[metrics['val_sparse_categorical_accuracy'].argmax()])


