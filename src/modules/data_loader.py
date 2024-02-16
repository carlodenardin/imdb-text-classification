import os
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

class DataLoader:
	"""
	A class to load train and test datasets for text classification tasks.
	"""

	def __init__(self, seed: int):
		"""
		Initializes the DataLoader class with a specified seed for randomization.

		Parameters:
			seed (int): The seed value for randomization.
		"""
		self.seed = seed

	

	def _load_train_data_no_validation(self, train_dir: str, shuffle: bool):
		"""
		Loads the training data without validation split.

		Parameters:
			train_dir (str): The directory containing the training data.
			shuffle (bool): Whether to shuffle the data. Default is True.

		Returns:
			The training data and target.
		"""
		train_dataset = load_files(
			train_dir,
			shuffle = shuffle,
			random_state = self.seed
		)
		return train_dataset.data, train_dataset.target

	def _load_train_data_with_validation(self, train_dir: str, validation_split: float, shuffle: bool):
		"""
		Loads the training data with validation split.

		Parameters:
			train_dir (str): The directory containing the training data.
			validation_split (float): The proportion of the training data to use for validation.
			shuffle (bool): Whether to shuffle the data. Default is True.

		Returns:
			the training data and target, and the validation data and target.
		"""
		train_dataset = load_files(
			train_dir,
			shuffle = shuffle,
			random_state = self.seed
		)

		train_data, val_data, train_target, val_target = train_test_split(
			train_dataset.data, train_dataset.target, 
			test_size = validation_split,
			random_state = self.seed
		)

		return train_data, train_target, val_data, val_target
	
	def load_train_data(self, train_dir: str, validation_split: float = 0, shuffle: bool = True):
		"""
		Loads the training data from the specified directory, with an option for validation split.

		Parameters:
			train_dir (str): The directory containing the training data.
			validation_split (float): The proportion of the training data to use for validation.
									Default is 0 (no validation split).
			shuffle (bool): Whether to shuffle the data. Default is True.

		Returns:
			The training data and target, and the validation data and target (if validation_split > 0).
		"""
		if validation_split == 0:
			return self._load_train_data_no_validation(train_dir, shuffle = shuffle)
		else:
			assert 0 < validation_split < 1, f'{validation_split} is outside [0, 1]'
			return self._load_train_data_with_validation(train_dir, validation_split, shuffle)

	def load_test_data(self, test_dir: str, shuffle: bool = False):
		"""
		Loads the test data from the specified directory.

		Parameters:
			test_dir (str): The directory containing the test data.
			shuffle (bool): Whether to shuffle the data. Default is False.

		Returns:
			the test data and target.
		"""
		test_data = load_files(
			test_dir,
			shuffle = shuffle,
			random_state = self.seed
		)
		return test_data.data, test_data.target