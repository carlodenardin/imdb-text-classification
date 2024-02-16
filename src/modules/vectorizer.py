from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from typing import Callable, Tuple


class Vectorizer:
	"""
	The Vectorizer class is used to convert a dataset composed of strings into a numerical representation using different vectorization techniques.
	"""

	def _vectorize_int(self, dataset, max_features: int):
		tokenizer = Tokenizer(num_words = max_features)
		return tokenizer, tokenizer.fit_on_texts(dataset)
	
	def _vectorize_tf_idf(
			self,
			dataset,
			max_features: int,
			ngram_range: Tuple[int, int],
			preprocessor: Callable
		):
		
		vectorize = TfidfVectorizer(
			preprocessor = preprocessor,
			ngram_range = ngram_range,
			max_features = max_features,
		)
		return vectorize, vectorize.fit_transform(dataset)

	def _vectorize_count(
			self,
			dataset,
			max_features: int,
			ngram_range: Tuple[int, int],
			preprocessor: Callable
		):

		vectorize = CountVectorizer(
			preprocessor = preprocessor,
			ngram_range = ngram_range,
			max_features = max_features,
		)
		return vectorize, vectorize.fit_transform(dataset)

	def vectorize_data(
			self,
			dataset,
			output_mode: str = 'int',
			max_features: int = None,
			ngram_range: Tuple[int, int] =  (1, 1),
			preprocessor: Callable = None
		):
		"""
		Converts a dataset composed of strings into a numerical representation.

		Parameters:
			dataset (List[str]): The dataset composed of strings.
			output_mode (str): The output mode specifying the vectorization technique to use. Options: 'int', 'tf-idf', 'count'.
			max_features (int): The maximum number of features to consider.
			ngram_range (Tuple[int, int]): The range of n-grams to consider.
			preprocessor (Callable): The preprocessor function to apply.

		Returns:
			The vectorizer and the vectorized dataset.
		"""
		assert output_mode in ['int', 'tf-idf', 'count'], f'Invalid output mode: {output_mode}'

		if output_mode == 'int':
			return self._vectorize_int(dataset, max_features)
		elif output_mode == 'tf-idf':
			return self._vectorize_tf_idf(dataset, max_features, ngram_range, preprocessor)
		else:
			return self._vectorize_count(dataset, max_features, ngram_range, preprocessor)