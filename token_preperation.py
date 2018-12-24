from os import listdir
from numpy import array
from numpy import argmax
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu
from pickle import load
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalMaxPooling2D
from text_procces import load_clean_descriptions , load_photo_features ,train_test_split , load_doc , load_set


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, desc, image, max_length):
	Ximages, XSeq, y = list(), list(),list()
	vocab_size = len(tokenizer.word_index) + 1
	# integer encode the description
	seq = tokenizer.texts_to_sequences([desc])[0]
	# split one sequence into multiple X,y pairs
	for i in range(1, len(seq)):
		# select
		in_seq, out_seq = seq[:i], seq[i]
		# pad input sequence
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		# encode output sequence
		out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		# store
		Ximages.append(image)
		XSeq.append(in_seq)
		y.append(out_seq)
	# Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
	return [Ximages, XSeq, y]



# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
