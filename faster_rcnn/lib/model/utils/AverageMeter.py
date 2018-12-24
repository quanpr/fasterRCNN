
class AverageMeter():
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.num = 0

	def update(self, val, times=1):
		self.val += val
		self.num += 1*times
		self.avg = self.val/self.num

	def clear(self):
		self.val = 0
		self.avg = 0
		self.num = 0