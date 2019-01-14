# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import sys
sys.path.append('/home/prquan/github/faster-rcnn.pytorch')
sys.path.append('/home/prquan/github/faster-rcnn.pytorch/lib')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
			adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.evaluator import evaluator
from model.utils.AverageMeter import AverageMeter

import pdb
import copy
import math

def count_para(model):
	num = 0
	for p in model.parameters():
		n = 1
		for s in p.shape:
			n *= s
		num += n
	return num

def trainer(model, dataloader, optimizer, args):

	name_to_save = args.save_model[:-4] if args.save_model[-4:] == '.pth' else args.save_model

	fasterRCNN = model
	fasterRCNN.train()
	best_mAP = 0
	lr = args.lr

	evl_rec = args.evl_rec

	# initilize the tensor holder here.
	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)

	# ship to cuda
	if args.cuda:
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if args.use_tfboard:
		from tensorboardX import SummaryWriter
		logger = SummaryWriter("logs")

	avg_rpn_cls, avg_rpn_bbox, avg_rcnn_cls, avg_rcnn_bbox = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
	for epoch in range(args.start_epoch, args.max_epochs + 1):
		# setting to train mode
		fasterRCNN.train()
		loss_temp = 0
		start = time.time()
		# current_mAP = evaluator(copy.deepcopy(fasterRCNN.module), args, evl_rec=evl_rec)
		if epoch % (args.lr_decay_step + 1) == 0:
				adjust_learning_rate(optimizer, args.lr_decay_gamma)
				lr *= args.lr_decay_gamma

		data_iter = iter(dataloader)

		for step in range(len(dataloader)):
			data = next(data_iter)
			im_data.data.resize_(data[0].size()).copy_(data[0])
			im_info.data.resize_(data[1].size()).copy_(data[1])
			gt_boxes.data.resize_(data[2].size()).copy_(data[2])
			num_boxes.data.resize_(data[3].size()).copy_(data[3])

			fasterRCNN.zero_grad()
			rois, cls_prob, bbox_pred, \
			rpn_loss_cls, rpn_loss_box, \
			RCNN_loss_cls, RCNN_loss_bbox, \
			rois_label= fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
			
			if args.evl_rec:
				loss = rpn_loss_cls.mean() + rpn_loss_box.mean() 
			else:
				loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
						 + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

			loss_temp += loss.item()
			avg_rpn_cls.update(rpn_loss_cls.mean().item())
			avg_rpn_bbox.update(rpn_loss_box.mean().item())
			avg_rcnn_cls.update(RCNN_loss_cls.mean().item())
			avg_rcnn_bbox.update(RCNN_loss_bbox.mean().item())

			# backward
			optimizer.zero_grad()
			loss.backward()

			if args.net in ('vgg16, inception'):
					clip_gradient(fasterRCNN, 10.)
			optimizer.step()

			if step % args.disp_interval == 0:
				end = time.time()
				if step > 0:
					loss_temp /= (args.disp_interval + 1)

				if args.mGPUs:
					loss_rpn_cls = rpn_loss_cls.mean().item()
					loss_rpn_box = rpn_loss_box.mean().item()
					loss_rcnn_cls = RCNN_loss_cls.mean().item()
					loss_rcnn_box = RCNN_loss_bbox.mean().item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt
				else:
					loss_rpn_cls = rpn_loss_cls.item()
					loss_rpn_box = rpn_loss_box.item()
					loss_rcnn_cls = RCNN_loss_cls.item()
					loss_rcnn_box = RCNN_loss_bbox.item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt

				print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
																% (args.session, epoch, step, len(dataloader), loss_temp, lr))
				print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
				if args.evl_rec:
					print("\t\t\trpn_cls: %.4f, rpn_box: %.4f" \
												% (avg_rpn_cls.avg, avg_rpn_bbox.avg))
				else:
					print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
												% (avg_rpn_cls.avg, avg_rpn_bbox.avg, avg_rcnn_cls.avg, avg_rcnn_bbox.avg))	

				avg_rpn_bbox.clear()
				avg_rpn_cls.clear()
				avg_rcnn_cls.clear()
				avg_rcnn_bbox.clear()

				if args.use_tfboard:
					info = {
						'loss': loss_temp,
						'loss_rpn_cls': loss_rpn_cls,
						'loss_rpn_box': loss_rpn_box,
						'loss_rcnn_cls': loss_rcnn_cls,
						'loss_rcnn_box': loss_rcnn_box
					}
					logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)


				loss_temp = 0
				start = time.time()
		
		output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
		save_name = os.path.join(output_dir, name_to_save+'_curr.pth')
		save_checkpoint({
			'session': args.session,
			'epoch': epoch + 1,
			'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
			'optimizer': optimizer.state_dict(),
			'pooling_mode': cfg.POOLING_MODE,
			'class_agnostic': args.class_agnostic,
		}, save_name)
		print('save current model: {}'.format(save_name))
		
		if epoch >= 5:
			print('Finish epoch {}, now evaluate model performance...\r\n'.format(epoch))
			current_mAP = evaluator(copy.deepcopy(fasterRCNN.module), args, evl_rec=evl_rec)

			if evl_rec:
				print('In epoch {}, recall = {:.4f}, best recall = {:.4f}'.format(epoch, current_mAP, best_mAP))
			else:
				print('In epoch {}, mAP = {:.4f}, best mAP = {:.4f}'.format(epoch, current_mAP, best_mAP))
			
			if current_mAP > best_mAP:
				best_mAP = current_mAP
				save_name = os.path.join(output_dir, name_to_save+'_best.pth')
				save_checkpoint({
					'session': args.session,
					'epoch': epoch + 1,
					'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
					'optimizer': optimizer.state_dict(),
					'pooling_mode': cfg.POOLING_MODE,
					'class_agnostic': args.class_agnostic,
				}, save_name)
				print('save best model: {}'.format(save_name))

	if args.use_tfboard:
		logger.close()

def mm_trainer(model, student_model, dataloader, optimizer, args):
	# mimicking implementation
	# a module for training student network

	name_to_save = args.save_model[:-4] if args.save_model[-4:] == '.pth' else args.save_model
	CROP_WIDTH, CROP_HEIGHT = 50, 50

	evl_rec = args.evl_rec

	fasterRCNN = model
	fasterRCNN.eval()

	student_net = student_model
	student_net.train()

	best_mAP = 0.0
	best_recall = 0.0
	lr = args.lr

	# initilize the tensor holder here.
	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)

	# ship to cuda
	if args.cuda:
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if args.use_tfboard:
		from tensorboardX import SummaryWriter
		logger = SummaryWriter("logs")

	avg_mse_loss, avg_rpn_cls, avg_rpn_bbox = AverageMeter(), AverageMeter(), AverageMeter()
	#curr_recall = evaluator(copy.deepcopy(fasterRCNN.module), args, evl_rec=True)

	for epoch in range(args.start_epoch, args.max_epochs + 1):
		# setting to train mode
		student_net.train()
		loss_temp = 0
		start = time.time()
		# current_mAP = evaluator(copy.deepcopy(student_net.module), args, evl_rec = True)
		if epoch % (args.lr_decay_step + 1) == 0:
				adjust_learning_rate(optimizer, args.lr_decay_gamma)
				lr *= args.lr_decay_gamma

		data_iter = iter(dataloader)

		for step in range(len(dataloader)):
			data = next(data_iter)
			im_data.data.resize_(data[0].size()).copy_(data[0])
			im_info.data.resize_(data[1].size()).copy_(data[1])
			gt_boxes.data.resize_(data[2].size()).copy_(data[2])
			num_boxes.data.resize_(data[3].size()).copy_(data[3])

			#fasterRCNN.zero_grad()
			student_net.zero_grad()

			rois, _, _, \
			_, _, \
			_, _, \
			_, base_feat, pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

			std_rois, cls_prob, bbox_pred, \
			rpn_loss_cls, rpn_loss_box, \
			RCNN_loss_cls, RCNN_loss_bbox, \
			rois_label, std_base_feat, std_pooled_feat = student_net(im_data, im_info, gt_boxes, num_boxes)

			#rois, base_feat = torch.zeros(std_rois.shape).cuda(), torch.zeros(std_base_feat.shape).cuda()
			base_feat, pooled_feat = base_feat.detach(), pooled_feat.detach()

			if False:
				from model.roi_align.modules.roi_align import RoIAlignAvg
				RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
				pooled_feat = RCNN_roi_align(base_feat, std_rois.view(-1, 5)).detach()

				MSELoss = nn.MSELoss()
				mse_loss = MSELoss(std_pooled_feat, pooled_feat)
				#pdb.set_trace()
			else:
				std_feat, tch_feat = [], []
				for i in range(im_data.shape[0]):
					std_img_feat, tch_img_feat = [], []
					dy, dx = base_feat.shape[2], base_feat.shape[3]
					for j in range(std_rois.shape[1]):

						stride = 16 if args.net != 'inception' else 17
						x1, y1, x2, y2 = int(std_rois[i][j][1].item()/stride),\
										 int(std_rois[i][j][2].item()/stride),\
										 int(std_rois[i][j][3].item()/stride),\
										 int(std_rois[i][j][4].item()/stride)

						if x2>dx or x1>dx or y2>dy or y1>dy:
						 	pdb.set_trace()
						
						if y2>y1 and x2>x1:
							std_img_feat.append(std_base_feat[i][:, y1:y2+1, x1:x2+1])
							tch_img_feat.append(base_feat[i][:, y1:y2+1, x1:x2+1])
						
					std_feat.append(std_img_feat)
					tch_feat.append(tch_img_feat)

				MSELoss = nn.MSELoss()
				mse_loss = torch.zeros(1).cuda() if args.cuda else torch.zeros(1)
				for i in range(len(std_feat)):
					for j in range(len(std_feat[i])):
						if math.isnan(MSELoss(std_feat[i][j], tch_feat[i][j]).item()):
							pdb.set_trace()
						mse_loss += MSELoss(std_feat[i][j], tch_feat[i][j])
						# print(mse_loss, std_feat[i][j].shape, tch_feat[i][j].shape)
						# pdb.set_trace()
				mse_loss /= (std_rois.shape[0]*std_rois.shape[1])


			avg_mse_loss.update(mse_loss.mean().item())
			avg_rpn_cls.update(rpn_loss_cls.mean().item())
			avg_rpn_bbox.update(rpn_loss_box.mean().item())

			loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + mse_loss.mean() 
			
			# loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
			# 		 + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
			loss_temp += loss.item()

			# backward
			optimizer.zero_grad()
			loss.backward()

			# for key, value in dict(student_net.named_parameters()).items():
			# 	print(key, value.grad)
			# pdb.set_trace()

			# if args.mimic:
			# 	clip_gradient(student_net, 10.)
			optimizer.step()

			if step % args.disp_interval == 0:
				end = time.time()
				if step > 0:
					loss_temp /= (args.disp_interval + 1)

				if args.mGPUs:
					loss_rpn_cls = rpn_loss_cls.mean().item()
					loss_rpn_box = rpn_loss_box.mean().item()

					loss_rcnn_cls = RCNN_loss_cls.mean().item()
					loss_rcnn_box = RCNN_loss_bbox.mean().item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt
				else:
					loss_rpn_cls = rpn_loss_cls.item()
					loss_rpn_box = rpn_loss_box.item()

					loss_rcnn_cls = RCNN_loss_cls.item()
					loss_rcnn_box = RCNN_loss_bbox.item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt

				print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
																% (args.session, epoch, step, len(dataloader), loss_temp, lr))
				print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
				print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, MSE loss: %.4f" \
											% (avg_rpn_cls.avg, avg_rpn_bbox.avg, avg_mse_loss.avg))
				avg_mse_loss.clear()
				avg_rpn_bbox.clear()
				avg_rpn_cls.clear()
				# print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
				# 							% (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
				if args.use_tfboard:
					info = {
						'loss': loss_temp,
						'loss_rpn_cls': loss_rpn_cls,
						'loss_rpn_box': loss_rpn_box,
						'loss_rcnn_cls': loss_rcnn_cls,
						'loss_rcnn_box': loss_rcnn_box
					}
					logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)


				loss_temp = 0
				start = time.time()
		
		output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
		save_name = os.path.join(output_dir, name_to_save+'_curr.pth')
		save_checkpoint({
			'session': args.session,
			'epoch': epoch + 1,
			'model': student_net.module.state_dict() if args.mGPUs else student_net.state_dict(),
			'optimizer': optimizer.state_dict(),
			'pooling_mode': cfg.POOLING_MODE,
			'class_agnostic': args.class_agnostic,
		}, save_name)
		print('save current model: {}'.format(save_name))
		
		print('Finish epoch {}, now evaluate model performance...\r\n'.format(epoch))
		

		if evl_rec:
			curr_recall = evaluator(copy.deepcopy(student_net.module), args, evl_rec=True)
			# curr_recall = evaluator(student_net.module, args, evl_rec=True)
			print('In epoch {}, recall = {:.4f}'.format(epoch, curr_recall))
			print('best recal = {:.4f}'.format(best_recall))
			if curr_recall > best_recall:
				best_recall = curr_recall
				save_name = os.path.join(output_dir, name_to_save+'_best.pth')
				save_checkpoint({
					'session': args.session,
					'epoch': epoch + 1,
					'model': student_net.module.state_dict() if args.mGPUs else student_net.state_dict(),
					'optimizer': optimizer.state_dict(),
					'pooling_mode': cfg.POOLING_MODE,
					'class_agnostic': args.class_agnostic,
				}, save_name)
				print('save best model: {}'.format(save_name))

		else:
			current_mAP = evaluator(copy.deepcopy(student_net.module), args, evl_rec=False)
			print('In epoch {}, mAP = {:.4f}'.format(epoch, current_mAP))
		
			if current_mAP > best_mAP:
				best_mAP = current_mAP
				save_name = os.path.join(output_dir, name_to_save+'_best.pth')
				save_checkpoint({
					'session': args.session,
					'epoch': epoch + 1,
					'model': student_net.module.state_dict() if args.mGPUs else student_net.state_dict(),
					'optimizer': optimizer.state_dict(),
					'pooling_mode': cfg.POOLING_MODE,
					'class_agnostic': args.class_agnostic,
				}, save_name)
				print('save best model: {}'.format(save_name))

	if args.use_tfboard:
		logger.close()