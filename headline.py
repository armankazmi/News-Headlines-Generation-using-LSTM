#LSTM Model to generate Headlilnes using character based modelling.

#Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import pandas as pd

#Data
df = pd.read_csv("headlines.csv")
data = df['title']

#Pre-Processing of text

def pre_process(text):
    s = '-'
    pos = text.find(s)
    text = text[:pos]
    return text

#Pre-processed data in variable clean
clean = data.apply(pre_process)

#Finding the longest sequence available in the training data
max = 0
for i in range(len(clean)):
  l = len(clean[i])
  if l>= max:
    max = l
  else:
    max = max

#In continous text form seprated by '\n'
text = clean.str.cat(sep='\n')

#Total length of vocabulary
chars = list(set(text))
data_size, vocab_size = len(text), len(chars)    

#Creating dictionary to map charachters to indices and vice versa
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }

#Converting training data to input form of sequence length
seq_length = max
dataX = []
dataY = []
for i in range(0, data_size - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_ix[char] for char in seq_in])
    dataY.append(char_to_ix[seq_out])
n_patterns = len(dataX)

#X and y
X = np.reshape(dataX, (n_patterns, seq_length, 1))  #shape of X = (n_patterns,max,)
# normalize
X = X / float(vocab_size)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#Model:
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Fitting on 50 epochs
model.fit(X, y, epochs=50, batch_size=128)

#Generating Random Headlines
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Starting:")
print ("\"", ''.join([ix_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(vocab_size)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = ix_to_char[index]
	seq_in = [ix_to_char[value] for value in pattern]
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
  
#Printing the pattern 
print ("\"", ''.join([ix_to_char[value] for value in pattern]), "\"")
