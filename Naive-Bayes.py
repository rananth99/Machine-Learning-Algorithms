###   MACHINE LEARNING ALGORITHM PYTHON CODE  ####

--------------------------------------------------------------------------------------------

############################## NAIVE BAYES ########################

import os
import re
import string
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
DATA_DIR = 'enron'
target_names = ['ham', 'spam']
 
def get_data(DATA_DIR,x):
	#if(x==1):
	#	print("For datasets 1,2,3")
	#	subfolders = ['enron%d' % i for i in range(1,4)]
	#else:
	#	print("For datasets 4,5,6")
	#	subfolders = ['enron%d' % i for i in range(4,7)]
	subfolders = ['enron%d' % i for i in range(1,7)]
 
	data = []
	target = []
	for subfolder in subfolders:
		# spam
		spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
		for spam_file in spam_files:
			with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
				data.append(f.read())
				target.append(1)
 
		# ham
		ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
		for ham_file in ham_files:
			with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
				data.append(f.read())

				target.append(0)
 
	return data, target

class SpamDetector(object):
	"""Implementation of Naive Bayes for binary classification"""
	def clean(self, s):
		translator = str.maketrans("", "", string.punctuation)
		return s.translate(translator)
 
	def tokenize(self, text):
		text = self.clean(text).lower()
		return re.split("\W+", text)
 
	def get_word_counts(self, words):
		word_counts = {}
		for word in words:
			word_counts[word] = word_counts.get(word, 0.0) + 1.0
		return word_counts
	
	def fit(self, X, Y):
		self.num_messages = {}
		self.log_class_priors = {}
		self.word_counts = {}
		self.vocab = set()
	 
		n = len(X)
		self.num_messages['spam'] = sum(1 for label in Y if label == 1)
		self.num_messages['ham'] = sum(1 for label in Y if label == 0)
		#self.log_class_priors['spam'] = math.log(self.num_messages['spam'] / n)
		#self.log_class_priors['ham'] = math.log(self.num_messages['ham'] / n)
		
		self.log_class_priors['spam'] = self.num_messages['spam'] / n
		self.log_class_priors['ham'] = self.num_messages['ham'] / n
		self.word_counts['spam'] = {}
		self.word_counts['ham'] = {}
	 
		for x, y in zip(X, Y):
			c = 'spam' if y == 1 else 'ham'
			counts = self.get_word_counts(self.tokenize(x))
			for word, count in counts.items():
				if word not in self.vocab:
					self.vocab.add(word)
				if word not in self.word_counts[c]:
					self.word_counts[c][word] = 0.0
	 
				self.word_counts[c][word] += count
			
			
	def predict(self, X,n):
		result = []
		for x in X:
			counts = self.get_word_counts(self.tokenize(x))
			spam_score = 0
			ham_score = 0
			for word, _ in counts.items():
				if word not in self.vocab: continue
				
				# add Laplace smoothing
				log_w_given_spam = math.log((self.word_counts['spam'].get(word, 0.0) + n) / (self.num_messages['spam'] +n*len(self.vocab)))
				log_w_given_ham =  math.log((self.word_counts['ham'].get(word, 0.0) + n) / (self.num_messages['ham'] + n*len(self.vocab)))
				
				#log_w_given_spam = (self.word_counts['spam'].get(word, 0.0) + n) / (self.num_messages['spam'] +n*len(self.vocab))
				#log_w_given_ham =  (self.word_counts['ham'].get(word, 0.0) + n) / (self.num_messages['ham'] + n*len(self.vocab))
	 
				spam_score += log_w_given_spam
				ham_score += log_w_given_ham
	 
			spam_score += self.log_class_priors['spam']
			ham_score += self.log_class_priors['ham']
	 
			if spam_score > ham_score:
				result.append(1)
			else:
				result.append(0)
		return result

if __name__ == '__main__':
	
	for j in [1,2,3]:
		print("Add-{} smoothing".format(str(j)))
		#print("Without log")
		X, y = get_data(DATA_DIR,j)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
		MNB = SpamDetector()
		MNB.fit(X_train, y_train)
 
		pred = MNB.predict(X_test,j)
		true = y_test
 
		accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) / float(len(pred))
		print("{0:.4f}".format(accuracy))
		print(confusion_matrix(true,pred)) 


-------------------------------------------------------------------------------