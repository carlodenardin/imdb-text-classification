import matplotlib.pyplot as plt
import seaborn as sns

class Aesthetic:

	@staticmethod
	def plot_balance_of_reviews(positive_count, negative_count):
		data = [positive_count, negative_count]
		labels = ['Positive', 'Negative']
		colors = ['#52b788', '#e5383b']

		plt.pie(data, labels = labels, colors = colors, autopct = '%1.1f%%', startangle = 90)
		plt.axis('equal')
		plt.title('Balance of Reviews')
		plt.show()

	@staticmethod
	def plot_dl_history(history):
		# Plot loss
		plt.figure(figsize=(8, 6))
		plt.subplot(1, 2, 1)
		plt.plot(history.history['loss'], label='train_loss')
		plt.plot(history.history['val_loss'], label='val_loss')
		plt.title('Model Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()

		# Plot accuracy
		plt.subplot(1, 2, 2)
		plt.plot(history.history['accuracy'], label='train_accuracy')
		plt.plot(history.history['val_accuracy'], label='val_accuracy')
		plt.title('Model Accuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()

		plt.tight_layout()
		plt.show()

	@staticmethod
	def plot_most_important_words(positive_words_count, negative_words_count):
		keys_positive = list(positive_words_count.keys())
		values_positive = list(positive_words_count.values())
		keys_negative = list(negative_words_count.keys())
		values_negative = list(negative_words_count.values())

		sns.set_theme(style="whitegrid")

		fig, axes = plt.subplots(1, 2, figsize=(12, 4))

		palette_positive = sns.color_palette("Greens_r", len(keys_positive))
		ax1 = sns.barplot(ax=axes[0], x=values_positive, y=keys_positive, hue = keys_positive, palette=palette_positive)
		ax1.set_title('Positive Reviews')
		ax1.set_xlabel('Count', fontdict={'fontname': 'Arial', 'fontsize': 12})
		ax1.set_ylabel('Words')
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)
		ax1.xaxis.grid(False)

		palette_negative = sns.color_palette("Reds_r", len(keys_negative))
		ax2 = sns.barplot(ax=axes[1], x=values_negative, y=keys_negative, hue = keys_negative, palette=palette_negative)
		ax2.set_title('Negative Reviews')
		ax2.set_xlabel('Count', fontdict={'fontname': 'Arial', 'fontsize': 12})
		ax2.set_ylabel('Words')
		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)
		ax2.xaxis.grid(False)

		plt.tight_layout()

		plt.show()

	@staticmethod
	def plot_single_most_important_word(word_frequencies):
		keys_positive = list(word_frequencies.keys())
		values_positive = list(word_frequencies.values())

		sns.set_theme(style="whitegrid")

		fig, ax = plt.subplots(1, 1, figsize=(6, 4))

		palette = sns.color_palette("Blues_r", len(keys_positive))
		ax = sns.barplot(ax = ax, x = values_positive, y = keys_positive, hue = keys_positive, palette = palette)
		ax.set_title('Reviews')
		ax.set_xlabel('Count', fontdict={'fontname': 'Arial', 'fontsize': 12})
		ax.set_ylabel('Words')
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.xaxis.grid(False)

		plt.tight_layout()

		plt.show()

	@staticmethod
	def plot_words_distribution(lengths, interval):
		fig, ax = plt.subplots(ncols=1, figsize=(6, 4))

		sns.kdeplot(lengths, color = '#1D70A2', ax = ax, legend = False)

		xs = ax.lines[0].get_xdata()
		ys = ax.lines[0].get_ydata()

		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		ax.fill_between(xs, 0, ys, facecolor='#258ED0', alpha=0.2)
		ax.fill_between(xs, 0, ys, where=(xs <= interval), facecolor='#258ED0', alpha=0.5)

		ax.set_xlim(0, 1000)

		ax.set_title('Lenght of reviews distribution')
		ax.set_xlabel('Lenght of reviews')
		ax.set_ylabel('Density')

		plt.show()