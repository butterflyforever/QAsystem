from collections import deque
MAX_MSG_QUEUE_LENGTH = 3
REPEAT_THRESHOLD = 2
IMAGE_MODE_THRESHOLD = 2

# 管理每个用户的会话上下文队列
class Q:
	def __init__(self):
		self.q = deque([])
		self.mode = 'normal'

	def append(self, elem):
		if len(self.q) >= MAX_MSG_QUEUE_LENGTH:
			self.q.popleft()
		self.q.append(elem)

	def is_repeat(self, elem):
		if self.q.count(elem) > REPEAT_THRESHOLD:
			return True
		else:
			return False

	def should_image_mode(self):
		if self.q.count('EMOJI') > IMAGE_MODE_THRESHOLD:
			return True
		else:
			return False

