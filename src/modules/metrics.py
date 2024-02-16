import tensorflow as tf

class Metrics(tf.keras.metrics.Metric):

	"""
		Not used in the current implementation
	"""

	def __init__(self, name='metrics', **kwargs):
		super(Metrics, self).__init__(name=name, **kwargs)
		self.true_positives = self.add_weight(name='tp', initializer='zeros')
		self.false_positives = self.add_weight(name='fp', initializer='zeros')
		self.false_negatives = self.add_weight(name='fn', initializer='zeros')

	def update_state(self, y_true, y_pred, sample_weight=None):
		y_pred = tf.cast(y_pred > 0.5, tf.float32)
		y_true = tf.cast(y_true, tf.float32)

		true_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
		false_pos = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), tf.float32))
		false_neg = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), tf.float32))

		self.true_positives.assign_add(true_pos)
		self.false_positives.assign_add(false_pos)
		self.false_negatives.assign_add(false_neg)

	def result(self):
		precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
		recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
		f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
		return {
			'precision': precision,
			'recall': recall,
			'f1_score': f1
		}
