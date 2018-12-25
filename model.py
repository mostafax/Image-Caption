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
from token_preperation import create_sequences , create_tokenizer , word_for_id


# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor (encoder)
	inputs1 = Input(shape=(7, 7, 512))
	fe1 = GlobalMaxPooling2D()(inputs1)

	fe2 = Dense(128, activation='relu')(fe1)

	fe3 = RepeatVector(max_length)(fe2)

	# embedding encoder
	inputs2 = Input(shape=(max_length,))
	emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
	emb3 = LSTM(256, return_sequences=True)(emb2)
	emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
	# merge inputs
	merged = concatenate([fe3, emb4])
	# language model (decoder)
	lm2 = LSTM(500)(merged)
	lm3 = Dense(500, activation='relu')(lm2)
	outputs = Dense(vocab_size, activation='softmax')(lm3)
	# Merging the models together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)

	print("Wights loaded")

	model.load_weights('weghits/weghits.now.h5')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, features, tokenizer, max_length, n_step):
	# loop until we finish training
	while 1:
		# loop over photo identifiers in the dataset
		keys = list(descriptions.keys())
		for i in range(0, len(keys), n_step):
			Ximages, XSeq, y = list(), list(),list()
			for j in range(i, min(len(keys), i+n_step)):
				image_id = keys[j]
				# retrieve photo feature input
				image = features[image_id][0]
				# retrieve text input
				desc = descriptions[image_id]
				# generate input-output pairs
				in_img, in_seq, out_word = create_sequences(tokenizer, desc, image, max_length)
				for k in range(len(in_img)):
					Ximages.append(in_img[k])
					XSeq.append(in_seq[k])
					y.append(out_word[k])
			# yield this batch of samples to the model
			yield [[array(Ximages), array(XSeq)], array(y)]


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text
from scipy.misc import imshow


import cv2
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, desc in descriptions.items():
		# generate description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# store actual and predicted
		actual.append([desc.split()])
		predicted.append(yhat.split())
		print('Actual:    %s' % desc)
		print('Predicted: %s' % yhat)
		im = cv2.imread('Flickr8K_Data/'+key+'.jpg')
		cv2.imshow('ff',im)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		if len(actual) >= 4:
			break
	# calculate BLEU score
	bleu = corpus_bleu(actual, predicted)
	return bleu

# load data set
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
dataset = load_set(filename)
#print('Dataset: %d' % len(dataset))
# train-test split
train, test = train_test_split(dataset)
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
test_descriptions = load_clean_descriptions('descriptions.txt', test)
#print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))
# photo features
train_features = load_photo_features('features.pkl', train)
test_features = load_photo_features('features.pkl', test)
#print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
#print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max(len(s.split()) for s in list(train_descriptions.values()))
#print('Description Length: %d' % max_length)

# define experiment
model_name = 'Finalmodel'
verbose = 2
n_epochs = 1
n_photos_per_update = 2
n_batches_per_epoch =int(len(train) / n_photos_per_update)
n_repeats = 1

# run experiment
train_results, test_results = list(), list()
for i in range(n_repeats):
	# define the modelkjj
	model = define_model(vocab_size, max_length)

	# fit model
	#model.fit_generator(data_generator(train_descriptions, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch=n_batches_per_epoch, epochs=n_epochs, verbose=verbose)
	#model.save_weights('weghits/weghits.now.h5')
	print("New Weight saved")
	# evaluate model on training data
	train_score = evaluate_model(model, train_descriptions, train_features, tokenizer, max_length)
	test_score = evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
