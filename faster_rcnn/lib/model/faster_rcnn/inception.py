from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import math
import pdb

__all__ = ['Inception3', 'inception_v3']

def inception_v3(pretrained=False, **kwargs):
	r"""Inception v3 model architecture from
	`"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	if pretrained:
		model_path = 'data/pretrained_model/inception_v3_caffe.pth'
		model = Inception3(**kwargs)
		print("Loading pretrained weights from %s" %(model_path))
		state_dict = torch.load(model_path)
		model.load_state_dict({k:v for k,v in state_dict.items() if k in model.state_dict()})
		return model

	return Inception3(**kwargs)


class Inception3(nn.Module):

	def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, shrink=1, mimic=False):
		super(Inception3, self).__init__()
		self.shrink = shrink
		self.mimic = mimic

		self.aux_logits = aux_logits
		self.transform_input = transform_input
		self.Conv2d_1a_3x3 = BasicConv2d(3, 32//self.shrink, kernel_size=3, stride=2)
		self.Conv2d_2a_3x3 = BasicConv2d(32//self.shrink, 32//self.shrink, kernel_size=3)
		self.Conv2d_2b_3x3 = BasicConv2d(32//self.shrink, 64//self.shrink, kernel_size=3, padding=1)
		self.Conv2d_3b_1x1 = BasicConv2d(64//self.shrink, 80//self.shrink, kernel_size=1)
		self.Conv2d_4a_3x3 = BasicConv2d(80//self.shrink, 192//self.shrink, kernel_size=3)
		self.Mixed_5b = InceptionA(192, pool_features=32, shrink=self.shrink)
		self.Mixed_5c = InceptionA(256, pool_features=64, shrink=self.shrink)
		self.Mixed_5d = InceptionA(288, pool_features=64, shrink=self.shrink)
		self.Mixed_6a = InceptionB(288, shrink=self.shrink)
		self.Mixed_6b = InceptionC(768, channels_7x7=128, shrink=self.shrink)
		self.Mixed_6c = InceptionC(768, channels_7x7=160, shrink=self.shrink)
		self.Mixed_6d = InceptionC(768, channels_7x7=160, shrink=self.shrink)
		self.Mixed_6e = InceptionC(768, channels_7x7=192, shrink=self.shrink)
		if aux_logits:
			self.AuxLogits = InceptionAux(768, num_classes)
		self.Mixed_7a = InceptionD(768, shrink=self.shrink)
		self.Mixed_7b = InceptionE(1280, shrink=self.shrink)
		self.Mixed_7c = InceptionE(2048, shrink=self.shrink)
		self.fc = nn.Linear(2048, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				import scipy.stats as stats
				stddev = m.stddev if hasattr(m, 'stddev') else 0.1
				X = stats.truncnorm(-2, 2, scale=stddev)
				values = torch.Tensor(X.rvs(m.weight.numel()))
				values = values.view(m.weight.size())
				m.weight.data.copy_(values)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		if self.transform_input:
			x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
			x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
			x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
			x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
		# 299 x 299 x 3
		x = self.Conv2d_1a_3x3(x)
		# 149 x 149 x 32
		x = self.Conv2d_2a_3x3(x)
		# 147 x 147 x 32
		x = self.Conv2d_2b_3x3(x)
		# 147 x 147 x 64
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		# 73 x 73 x 64
		x = self.Conv2d_3b_1x1(x)
		# 73 x 73 x 80
		x = self.Conv2d_4a_3x3(x)
		# 71 x 71 x 192
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		# 35 x 35 x 192
		x = self.Mixed_5b(x)
		# 35 x 35 x 256
		x = self.Mixed_5c(x)
		# 35 x 35 x 288
		x = self.Mixed_5d(x)
		# 35 x 35 x 288
		x = self.Mixed_6a(x)
		# 17 x 17 x 768
		x = self.Mixed_6b(x)
		# 17 x 17 x 768
		x = self.Mixed_6c(x)
		# 17 x 17 x 768
		x = self.Mixed_6d(x)
		# 17 x 17 x 768
		x = self.Mixed_6e(x)
		# 17 x 17 x 768
		if self.training and self.aux_logits:
			aux = self.AuxLogits(x)
		# 17 x 17 x 768
		x = self.Mixed_7a(x)
		# 8 x 8 x 1280
		x = self.Mixed_7b(x)
		# 8 x 8 x 2048
		x = self.Mixed_7c(x)
		# 8 x 8 x 2048
		x = F.avg_pool2d(x, kernel_size=8)
		# 1 x 1 x 2048
		x = F.dropout(x, training=self.training)
		# 1 x 1 x 2048
		x = x.view(x.size(0), -1)
		# 2048
		x = self.fc(x)
		# 1000 (num_classes)
		if self.training and self.aux_logits:
			return x, aux
		return x


class InceptionA(nn.Module):

	def __init__(self, in_channels, pool_features, shrink):
		super(InceptionA, self).__init__()

		self.branch1x1 = BasicConv2d(in_channels//shrink, 64//shrink, kernel_size=1)

		self.branch5x5_1 = BasicConv2d(in_channels//shrink, 48//shrink, kernel_size=1)
		self.branch5x5_2 = BasicConv2d(48//shrink, 64//shrink, kernel_size=5, padding=2)

		self.branch3x3dbl_1 = BasicConv2d(in_channels//shrink, 64//shrink, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv2d(64//shrink, 96//shrink, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = BasicConv2d(96//shrink, 96//shrink, kernel_size=3, padding=1)

		self.branch_pool = BasicConv2d(in_channels//shrink, pool_features//shrink, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch5x5 = self.branch5x5_1(x)
		branch5x5 = self.branch5x5_2(branch5x5)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionB(nn.Module):

	def __init__(self, in_channels, shrink):
		super(InceptionB, self).__init__()
		self.branch3x3 = BasicConv2d(in_channels//shrink, 384//shrink, kernel_size=3, stride=2)

		self.branch3x3dbl_1 = BasicConv2d(in_channels//shrink, 64//shrink, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv2d(64//shrink, 96//shrink, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = BasicConv2d(96//shrink, 96//shrink, kernel_size=3, stride=2)

	def forward(self, x):
		branch3x3 = self.branch3x3(x)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

		outputs = [branch3x3, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionC(nn.Module):

	def __init__(self, in_channels, channels_7x7, shrink):
		super(InceptionC, self).__init__()
		self.branch1x1 = BasicConv2d(in_channels//shrink, 192//shrink, kernel_size=1)

		c7 = channels_7x7//shrink
		self.branch7x7_1 = BasicConv2d(in_channels//shrink, c7, kernel_size=1)
		self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7_3 = BasicConv2d(c7, 192//shrink, kernel_size=(7, 1), padding=(3, 0))

		self.branch7x7dbl_1 = BasicConv2d(in_channels//shrink, c7, kernel_size=1)
		self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7dbl_5 = BasicConv2d(c7, 192//shrink, kernel_size=(1, 7), padding=(0, 3))

		self.branch_pool = BasicConv2d(in_channels//shrink, 192//shrink, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch7x7 = self.branch7x7_1(x)
		branch7x7 = self.branch7x7_2(branch7x7)
		branch7x7 = self.branch7x7_3(branch7x7)

		branch7x7dbl = self.branch7x7dbl_1(x)
		branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionD(nn.Module):

	def __init__(self, in_channels, shrink):
		super(InceptionD, self).__init__()
		self.branch3x3_1 = BasicConv2d(in_channels//shrink, 192//shrink, kernel_size=1)
		self.branch3x3_2 = BasicConv2d(192//shrink, 320//shrink, kernel_size=3, stride=2)

		self.branch7x7x3_1 = BasicConv2d(in_channels//shrink, 192//shrink, kernel_size=1)
		self.branch7x7x3_2 = BasicConv2d(192//shrink, 192//shrink, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7x3_3 = BasicConv2d(192//shrink, 192//shrink, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7x3_4 = BasicConv2d(192//shrink, 192//shrink, kernel_size=3, stride=2)

	def forward(self, x):
		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)

		branch7x7x3 = self.branch7x7x3_1(x)
		branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

		branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
		outputs = [branch3x3, branch7x7x3, branch_pool]
		return torch.cat(outputs, 1)


class InceptionE(nn.Module):

	def __init__(self, in_channels, shrink):
		super(InceptionE, self).__init__()
		self.branch1x1 = BasicConv2d(in_channels//shrink, 320//shrink, kernel_size=1)

		self.branch3x3_1 = BasicConv2d(in_channels//shrink, 384//shrink, kernel_size=1)
		self.branch3x3_2a = BasicConv2d(384//shrink, 384//shrink, kernel_size=(1, 3), padding=(0, 1))
		self.branch3x3_2b = BasicConv2d(384//shrink, 384//shrink, kernel_size=(3, 1), padding=(1, 0))

		self.branch3x3dbl_1 = BasicConv2d(in_channels//shrink, 448//shrink, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv2d(448//shrink, 384//shrink, kernel_size=3, padding=1)
		self.branch3x3dbl_3a = BasicConv2d(384//shrink, 384//shrink, kernel_size=(1, 3), padding=(0, 1))
		self.branch3x3dbl_3b = BasicConv2d(384//shrink, 384//shrink, kernel_size=(3, 1), padding=(1, 0))

		self.branch_pool = BasicConv2d(in_channels//shrink, 192//shrink, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = [
			self.branch3x3_2a(branch3x3),
			self.branch3x3_2b(branch3x3),
		]
		branch3x3 = torch.cat(branch3x3, 1)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = [
			self.branch3x3dbl_3a(branch3x3dbl),
			self.branch3x3dbl_3b(branch3x3dbl),
		]
		branch3x3dbl = torch.cat(branch3x3dbl, 1)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

	def __init__(self, in_channels, num_classes):
		super(InceptionAux, self).__init__()
		self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
		self.conv1 = BasicConv2d(128, 768, kernel_size=5)
		self.conv1.stddev = 0.01
		self.fc = nn.Linear(768, num_classes)
		self.fc.stddev = 0.001

	def forward(self, x):
		# 17 x 17 x 768
		x = F.avg_pool2d(x, kernel_size=5, stride=3)
		# 5 x 5 x 768
		x = self.conv0(x)
		# 5 x 5 x 128
		x = self.conv1(x)
		# 1 x 1 x 768
		x = x.view(x.size(0), -1)
		# 768
		x = self.fc(x)
		# 1000
		return x


class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)

class inception(_fasterRCNN):
	def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, shrink=1, mimic=False):
		self.shrink = shrink
		self.pretrained = pretrained
		self.class_agnostic = class_agnostic
		self.dout_base_model = 768//shrink 

		_fasterRCNN.__init__(self, classes, class_agnostic, shrink, mimic)

	def _init_modules(self):

		if self.pretrained == True:
			inception = inception_v3(pretrained=True, aux_logits=False, transform_input=False, shrink=self.shrink)
		else:
			inception = inception_v3(pretrained=False, aux_logits=False, transform_input=False, shrink=self.shrink)

		# Build resnet.
		self.RCNN_base = nn.Sequential(inception.Conv2d_1a_3x3,
										inception.Conv2d_2a_3x3,
										inception.Conv2d_2b_3x3,
										nn.MaxPool2d(kernel_size=3, stride=2),

										inception.Conv2d_3b_1x1,
										inception.Conv2d_4a_3x3,
										nn.MaxPool2d(kernel_size=3, stride=2),

										inception.Mixed_5b,
										inception.Mixed_5c,
										inception.Mixed_5d,
										inception.Mixed_6a,
										inception.Mixed_6b,
										inception.Mixed_6c,
										inception.Mixed_6d,
										inception.Mixed_6e)

		self.RCNN_top = nn.Sequential(inception.Mixed_7a,
										inception.Mixed_7b,
										inception.Mixed_7c)

		self.RCNN_cls_score = nn.Linear(2048//self.shrink, self.n_classes)
		if self.class_agnostic:
			self.RCNN_bbox_pred = nn.Linear(2048//self.shrink, 4)
		else:
			self.RCNN_bbox_pred = nn.Linear(2048//self.shrink, 4 * self.n_classes)

		# Fix blocks
		for n in range(0,7):
			for p in self.RCNN_base[n].parameters(): p.requires_grad=False

		assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
		if cfg.RESNET.FIXED_BLOCKS >= 3:
			for p in self.RCNN_base[9].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 2:
			for p in self.RCNN_base[8].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 1:
			for p in self.RCNN_base[7].parameters(): p.requires_grad=False

		def set_bn_fix(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
				for p in m.parameters(): p.requires_grad=False

		self.RCNN_base.apply(set_bn_fix)
		self.RCNN_top.apply(set_bn_fix)

	def train(self, mode=True):
		# Override train so that the training mode is set as we want
		nn.Module.train(self, mode)
		if mode:
			# Set fixed blocks to be in eval mode
			self.RCNN_base.eval()
			self.RCNN_base[8:].train()

			def set_bn_eval(m):
				classname = m.__class__.__name__
				if classname.find('BatchNorm') != -1:
					m.eval()

			self.RCNN_base.apply(set_bn_eval)
			self.RCNN_top.apply(set_bn_eval)

	def _head_to_tail(self, pool5):
		fc7 = self.RCNN_top(pool5).mean(3).mean(2)
		return fc7

class tiny_inception(_fasterRCNN):
	def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, shrink=1, mimic=False):
		self.shrink = shrink
		self.class_agnostic = class_agnostic
		self.dout_base_model = 768//shrink 

		_fasterRCNN.__init__(self, classes, class_agnostic, shrink, mimic)

	def _init_modules(self):
		print('initial tiny inception with shrinking parameters {}'.format(self.shrink))
		inception = inception_v3(pretrained=False, aux_logits=False, transform_input=False, shrink=self.shrink)

		# Build inception.
		self.RCNN_base = nn.Sequential(inception.Conv2d_1a_3x3,
										inception.Conv2d_2a_3x3,
										inception.Conv2d_2b_3x3,
										nn.MaxPool2d(kernel_size=3, stride=2),

										inception.Conv2d_3b_1x1,
										inception.Conv2d_4a_3x3,
										nn.MaxPool2d(kernel_size=3, stride=2),

										inception.Mixed_5b,
										inception.Mixed_5c,
										inception.Mixed_5d,
										inception.Mixed_6a,
										inception.Mixed_6b,
										inception.Mixed_6c,
										inception.Mixed_6d,
										inception.Mixed_6e)

		self.RCNN_top = nn.Sequential(inception.Mixed_7a,
										inception.Mixed_7b,
										inception.Mixed_7c)

		self.RCNN_cls_score = nn.Linear(2048//self.shrink, self.n_classes)
		if self.class_agnostic:
			self.RCNN_bbox_pred = nn.Linear(2048//self.shrink, 4)
		else:
			self.RCNN_bbox_pred = nn.Linear(2048//self.shrink, 4 * self.n_classes)

		in_channels = self.dout_base_model
		self.transf_layer = nn.Sequential(
										nn.Conv2d(in_channels, 768,
											kernel_size=3, stride=1,padding=1,
											dilation=1, bias=True),
										nn.BatchNorm2d(768)
										)

		# initialize transform layer
		for m in self.transf_layer:
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()	


	def _head_to_tail(self, pool5):
		fc7 = self.RCNN_top(pool5).mean(3).mean(2)
		return fc7