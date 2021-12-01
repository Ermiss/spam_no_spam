from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read the dataset
df = pd.read_csv('data\spam_or_not_spam\spam_or_not_spam.csv', encoding='UTF-8')

# remove the rows with nulls
df = df.loc[df['email'].notnull()]
y = df['label']

text = ' '.join(df['email'])
text = text.split()
freq_comm = pd.Series(text).value_counts()
rare = freq_comm[freq_comm.values == 1]


# clean the dataset from misspelling words
def get_clean_text(x):
    if type(x) is str:
        x = x.lower()
        x = ' '.join([t for t in x.split() if t not in rare])
        return x
    else:
        return x


df['email'] = df['email'].apply(lambda x: get_clean_text(x))

text = df['email'].tolist()
# tokenize each word
token = Tokenizer()
token.fit_on_texts(text)

vocab_size = len(token.word_index) + 1
encoded_text = token.texts_to_sequences(text)

max_length = 0
for i in range(0, len(encoded_text)):
    current_length = len(encoded_text[i])
    if current_length > max_length:
        max_length = current_length
        num = i

X = pad_sequences(encoded_text, maxlen=max_length, padding='post')
# dictionary for the vectors
glove_vectors = dict()
# open glove vectors 200 dimentions
file = open(r'C:\Users\ermis\Downloads\glove.twitter.27B\glove.twitter.27B.200d.txt', encoding='utf-8')
# fill the dictionary with the vectors
for line in file:
    values = line.split()
    word = values[0]
    vectors = np.asarray(values[1:])
    glove_vectors[word] = vectors
file.close()
word_vector_matrix = np.zeros((vocab_size, 200))

# take the vector for each word
for word, index in token.word_index.items():
    vector = glove_vectors.get(word)
    if vector is not None:
        word_vector_matrix[index] = vector

# split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25, stratify=y)
vec_size = 200


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# creating the neural network
model = Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_length, weights=[word_vector_matrix], trainable=False))

model.add(Conv1D(64, 8, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

# model.add(Dense(16, activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy',
              metrics=['acc', f1_m, precision_m, recall_m])

history = model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))

loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
# print the metrics
print("F1 score:", f1_score)
print("Precision:", precision)
print("Recall:", recall)

