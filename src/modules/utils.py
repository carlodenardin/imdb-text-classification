def compute_frequencies(vectorize, features):
	dictionary = vectorize.vocabulary_
	frequencies = np.asarray(features.sum(axis = 0)).ravel()
	word_frequencies = {word: frequencies[idx] for word, idx in dictionary.items()}
	sorted_word_frequencies = dict(sorted(word_frequencies.items(), key = lambda item: item[1], reverse = True))
	return sorted_word_frequencies