import numpy as np
def read_glove_vector(glove_vec):
	with open(glove_vec, 'r', encoding='UTF-8') as f:
		words = set()
		word_to_vec_map = {}
		for line in f:
			w_line = line.split()
			curr_word = w_line[0]
			word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)



	return word_to_vec_map

def compute_frequencies(vectorize, features):
	dictionary = vectorize.vocabulary_
	frequencies = np.asarray(features.sum(axis = 0)).ravel()
	word_frequencies = {word: frequencies[idx] for word, idx in dictionary.items()}
	sorted_word_frequencies = dict(sorted(word_frequencies.items(), key = lambda item: item[1], reverse = True))
	return sorted_word_frequencies