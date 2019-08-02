import jieba

SENSITIVE_THRESHOLD = 0.1

# 敏感词过滤功能
class SensitiveFilter:
	def __init__(self):
		with open('./utils/sensitive.txt', 'r', encoding='utf-8') as infile:
			data = infile.readline()
			self.words = data.split('|')
		print(' - sensitive filter initialized.')

	def is_sensitive(self, sentence):
		words = jieba.lcut(sentence)
		sensitive_count = 0
		for wd in words:
			if  wd in self.words:
				sensitive_count += 1
		if sensitive_count / len(words) > SENSITIVE_THRESHOLD:
			return True
		else:
			return False

