import music21
import numpy as np
import os



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
    return x

def dict_pianoroll(d):
    #In: A dictionary with the notes and duration of a melody
    #Out: The pianoroll representation of the melody
    n = len(d)
    l = []
    aux = 0
    for i in range(n):
        print(i)
        aux = aux + d[i]['duration']
        if aux == 0.125:
          continue
        else:
          duration = int(aux/0.25)
          for j in range(duration):
            l.append(int(d[i]['note']))
          aux = 0
    return l

def onehot_encoder(p):
  #In: a piano roll representation
  #Out: the onehot representation of the pianoroll
  n = len(p)
  I = np.identity(88)
  X = np.zeros([88,n])
  for i in range(n):
    if p[i]!=-1:
      X[:,i] = I[p[i],:]
  return X

def generate_data_inv():
    invention_dict = []
    invention_pianoroll = []
    for invencion in os.listdir('Dataset/invenciones_musescore'):
        if invencion!= '.DS_Store':
            invent_aux = music21.converter.parse('invenciones_musescore/'+invencion)
            invention_dict.append(music21_dict(invent_aux))
            invention_pianoroll.append([])
            for i in invention_dict[-1]:
                aux = dict_pianoroll(i)
                invention_pianoroll[-1].append(aux.copy())
            np.savetxt('PianoRoll/Inventions/'+invencion[:-4] + '.txt',invention_pianoroll[-1],fmt='%i')
                
def generate_data_sinf():
    invention_dict = []
    invention_pianoroll = []
    for invencion in os.listdir('Dataset/sinfonias_musescore'):
        if invencion!= '.DS_Store':
            print(invencion)
            invent_aux = music21.converter.parse('sinfonias_musescore/'+invencion)
            invention_dict.append(music21_dict(invent_aux))
            invention_pianoroll.append([])
            for i in invention_dict[-1]:
                aux = dict_pianoroll(i)
                invention_pianoroll[-1].append(aux.copy())
            np.savetxt('PianoRoll/Sinfonias/'+invencion[:-4] + '.txt',invention_pianoroll[-1],fmt='%i')                
                

            
generate_data_inv()
generate_data_sinf()