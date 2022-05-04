import tensorflow as tf
import music21
import numpy as np
import pickle
import matplotlib.pyplot as plt

def music21_dict(c):
    #In: A music Stream object from music21 library
    #Out: A dictionary with the notes and duration of a melody
    num_voice = len(c.parts)
    x = []
    for i in range(num_voice):
          x.append([])
          for n in c.parts[i].flat:
              if isinstance(n,music21.note.Note):
                  x[i].append({'note': int((n.pitch.midi)),'duration':n.duration.quarterLength})
              else:
                  if isinstance(n,music21.note.Rest):
                      x[i].append({'note': -1,'duration':n.duration.quarterLength})
    return(x)

def dict_pianoroll(d):
    #In: A dictionary with the notes and duration of a melody
    #Out: The pianoroll representation of the melody
    n = len(d[0])
    l = []
    aux = 0
    for i in range(n):
        aux = aux + d[0][i]['duration']
        if aux == 0.125:
          continue
        else:
          duration = int(aux/0.25)
          for j in range(duration):
            l.append(int(d[0][i]['note']))
          aux = 0
    return l

class OneStep(tf.keras.Model):
  def __init__(self, model, voice_1_tokenizer, voice_2_tokenizer, temperature=2.2):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.voice_1_tokenizer = voice_1_tokenizer
    self.voice_2_tokenizer = voice_2_tokenizer



  def generate_one_step(self,voice_1_note,voice_2_note=-1, states = None):
    # Convert strings to token IDs.
   
    
    voice_1_id = np.asarray(self.voice_1_tokenizer.note2index_f([voice_1_note]))
    voice_2_id = np.asarray(self.voice_2_tokenizer.note2index_f([voice_2_note]))
    

    inputs_ids = np.concatenate((voice_1_id,voice_2_id))
    inputs_ids = inputs_ids.reshape((1,2))

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states= self.model(inputs=inputs_ids,states = states,return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature


    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    #predicted_ids = tf.squeeze(predicted_ids, axis=-1)
    # Convert from token ids to characters
    aux = int(predicted_ids)
    predicted_chars = self.voice_2_tokenizer.index2note[aux]
    # Return the characters and model state.
    return predicted_chars, states

def change_tone(p,t):
    #In: 'p' the piano roll notetion of a melody
    #    't' the number of tones to transpose
    n = len(p)
    p_aux = p.copy()
    for i in range(n):
        if p_aux[i]!=-1.:
            p_aux[i]=  p[i]+t
    return p_aux

def pianoroll_dict(p):
    aux = p[0]
    t = 1
    voice = []
    for n in p[1:]:
      if aux == n:
        t = t+1
      else:
        voice.append({'note': aux,'duracion': t})
        aux = n
        t = 1 
    return voice

def dict_stream(p1,p2):
  s = music21.stream.Score(id='mainScore')

  for p in [p1,p2]:
    stream = music21.stream.Part(id='part0')
    for d in p:
      print(d)
      if d['note'] ==-1:
        r = music21.note.Rest()
        r.duration.quarterLength = d['duracion']/4
        stream.append(r)
      else:
        n = music21.note.Note(d['note'],quarterLength=d['duracion']/4)
        stream.append(n)
    s.append(stream)
  return s

def pianoroll_plot(p):
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
  def __init__(self,voice_1_tokens,voice_2_tokens,embedding_dim, l, hidden_state_dimensions):
    super().__init__()
    self.l = l
    self.embedding_voice1 = tf.keras.layers.Embedding(voice_1_tokens,embedding_dim)
    self.embedding_voice2 = tf.keras.layers.Embedding(voice_2_tokens,embedding_dim)
    self.LSTM = []
    for i in range(l):
        self.LSTM.append(tf.keras.layers.LSTM(hidden_state_dimensions[i],
                                       return_sequences=True,
                                       return_state=True,
                                       activation ='tanh'))
        
    self.dense = tf.keras.layers.Dense(voice_2_tokens)

  def call(self, inputs,states = None, return_state=False, training=False):
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

def generate2voice(model,p):
    #In: 'model' the inference model.
    #    'p' a melody in piano roll to harmonize
    #Out: A second 'harmoius voice'

    states = None
    next_char = -1
    result = [next_char]
    voice1 = p
    voice2 = []
    for i in voice1:
      if i!=-1:
        voice2.append(music21.note.Note(i).pitch.midi)
      else:
        voice2.append(-1)
    for n in voice2:
      next_char, states= one_step_model.generate_one_step(n,next_char, states = states)
      result.append(next_char)
    return result[1:]

with open('Harmony/VoiceClass/voice1Tokenizer.pickle', 'rb') as file:
    voice_1_tokenizer = pickle.load(file) 

with open('Harmony/VoiceClass/voice2Tokenizer.pickle', 'rb') as file:
    voice_2_tokenizer = pickle.load(file) 


#Load model

LSTM_Bach = tf.keras.models.load_model('Harmony/Models/saved_model')
BACH = LSTM_Armonia(voice_1_tokenizer.n_notes,voice_2_tokenizer.n_notes,30,2,[256,256])

dataset_in =  np.concatenate((32*[3],32*[3])).reshape(-1)
dataset_out = np.asarray((32*[3])).reshape(-1)
dataset = tf.data.Dataset.from_tensor_slices(([dataset_in],[dataset_out]))

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = BACH(tf.reshape(input_example_batch,(1,64)))


BACH.set_weights(LSTM_Bach.weights)
one_step_model = OneStep(BACH, voice_1_tokenizer, voice_2_tokenizer)

#Recurrent harmonizer
prelude = music21.converter.parse('test/PreludeBach.mxl')
prelude = music21_dict(prelude)
prelude = dict_pianoroll(prelude)
prelude = change_tone(prelude,20)
result = generate2voice(one_step_model,prelude)

#Piano roll to midi format
voice1 = pianoroll_dict(prelude)
voice2 = pianoroll_dict(result)
stream = dict_stream(voice1,voice2)
mf = music21.midi.translate.streamToMidiFile(stream)
mf.open('BachPreludeHarmony.mid', 'wb')
mf.write()
mf.close()