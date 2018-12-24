import numpy as np 
import pdb

def evaluate_recall(det_bbox, i, imdb, thr=0.5):
	
	img_name = imdb.image_path_at(i)
	img_idx = img_name[-10:-4]
	gt_info = imdb._load_pascal_annotation(img_idx)
	gt_bbox = gt_info['boxes']

	TP, GT = 0, 0

	for i in range(len(gt_bbox)):
		if not gt_info['gt_ishard'][i]:
		#if True:
			GT += 1
			for j in range(len(det_bbox)):

				ixmin = np.maximum(gt_bbox[i][0], det_bbox[j][0])
				iymin = np.maximum(gt_bbox[i][1], det_bbox[j][1])
				ixmax = np.minimum(gt_bbox[i][2], det_bbox[j][2])
				iymax = np.minimum(gt_bbox[i][3], det_bbox[j][3])

				iw = np.maximum(ixmax - ixmin + 1., 0.)
				ih = np.maximum(iymax - iymin + 1., 0.)

				inters = iw * ih


				uni = ((det_bbox[j][2] - det_bbox[j][0] + 1.) * (det_bbox[j][3] - det_bbox[j][1] + 1.) +
						(gt_bbox[i][2] - gt_bbox[i][0] + 1.) *
						(gt_bbox[i][3] - gt_bbox[i][1] + 1.) - inters)

				overlaps = inters / uni

				if overlaps > thr:
					TP += 1
					break

	return TP, GT


def evaluate_final_recall(det_bbox, i, imdb, thr=0.5):
	
	img_name = imdb.image_path_at(i)
	img_idx = img_name[-10:-4]
	gt_info = imdb._load_pascal_annotation(img_idx)
	gt_bbox = gt_info['boxes']

	TP, GT = 0, 0

	for i in range(len(gt_bbox)):
		detected = False
		if not gt_info['gt_ishard'][i] and not detected:
			GT += 1
			for j in range(len(det_bbox)):
				
				for k in range(0, len(det_bbox[0]), 4):
					ixmin = np.maximum(gt_bbox[i][0], det_bbox[j][k:k+4][0])
					iymin = np.maximum(gt_bbox[i][1], det_bbox[j][k:k+4][1])
					ixmax = np.minimum(gt_bbox[i][2], det_bbox[j][k:k+4][2])
					iymax = np.minimum(gt_bbox[i][3], det_bbox[j][k:k+4][3])

					iw = np.maximum(ixmax - ixmin + 1., 0.)
					ih = np.maximum(iymax - iymin + 1., 0.)

					inters = iw * ih


					uni = ((det_bbox[j][k:k+4][2] - det_bbox[j][0] + 1.) * (det_bbox[j][k:k+4][3] - det_bbox[j][1] + 1.) +
							(gt_bbox[i][2] - gt_bbox[i][0] + 1.) *
							(gt_bbox[i][3] - gt_bbox[i][1] + 1.) - inters)

					overlaps = inters / uni

					#pdb.set_trace()
					if overlaps > thr:
						detected = True
						TP += 1
						break

				if detected:
					break

	return TP, GT