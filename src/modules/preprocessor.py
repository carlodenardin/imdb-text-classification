from nltk.corpus import stopwords

import re
import tensorflow as tf

HTML_PATTERN = r'<.*?>'
PUNCTUATION_PATTERN = r'[^\w\s]' 
DIGITS_PATTERN = r'\d+'

class Preprocessor:
	"""
	A class to perform text preprocessing tasks such as removing HTML tags, punctuation, and stopwords.
	"""

	def __init__(self, stopwords_language: str = 'english'):
		"""
		Initializes the Preprocessor class with a specified language for stopwords.

		Parameters:
			stopwords_language (str): The language of stopwords to use. Defaults to 'english'.
		"""
		self.stop_words = stopwords.words(stopwords_language)

	def perform_soft_preprocessing(self, input: str):
		"""
		Performs soft preprocessing on the input text, which includes removing HTML tags and punctuation,
		and converting the text to lowercase.

		Parameters:
			input (str): The input text to preprocess.

		Returns:
			str: The preprocessed text.
		"""
		return re.sub(f'{HTML_PATTERN}|{PUNCTUATION_PATTERN}', ' ', input).lower()
	
	def perform_strong_preprocessing(self, input: str):
		"""
		Performs strong preprocessing on the input text, which includes removing HTML tags, punctuation,
		digits, converting the text to lowercase and removing stopwords.

		Parameters:
			input (str): The input text to preprocess.

		Returns:
			str: The preprocessed text.
		"""
		output = re.sub(f'{HTML_PATTERN}|{PUNCTUATION_PATTERN}|{DIGITS_PATTERN}', ' ', input).lower()
		output = ' '.join([word for word in output.split() if word not in self.stop_words])
		return output