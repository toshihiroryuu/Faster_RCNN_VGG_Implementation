from __future__ import division
import random
import pprint
import sys
import time
import numpy as np

import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

from keras_frcnn.simple_parser import get_data
from keras_frcnn import vgg

sys.setrecursionlimit(40000)



train_path='/home/quest/udacity_driving_datasets/trainsample.txt'
print("Train path : {}".format(train_path))

n_rois=32              #region of interest
epochs=10
epoch_length = 100      #no of iterations

print("Number of Epochs : {}".format(epochs))
config_filename="config.pickle"
output_weight_path='./model_frcnn.hdf5'
#input_weight_path=''

C = config.Config()

C.model_path = output_weight_path
C.num_rois = int(n_rois)

C.network = 'vgg'
 
C.base_net_weights = vgg.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(train_path)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {i: j for i, j in class_mapping.items()}

print("Training images per class:")
pprint.pprint(classes_count)
print("Num classes (including bg) = {}".format(len(classes_count)))

config_output_filename = config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print("Config has been written to {}, and can be loaded when testing to ensure correct results".format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']     #train val split from all_imgs
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print("Num train samples {}".format(len(train_imgs)))
print("Num val samples {}".format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, vgg.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, vgg.get_img_output_length,K.image_dim_ordering(), mode='val')

 # image shape for tensorflow is (None, None, 3)

img_input = Input(shape=(None, None, 3))
roi_input = Input(shape=(None, 4))


shared_layers = vgg.nn_base(img_input, trainable=True)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = vgg.rpn(shared_layers, num_anchors)

classifier = vgg.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)


model_all = Model([img_input, roi_input], rpn[:2] + classifier)

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)

model_rpn.compile(	optimizer=optimizer, 
					loss=[losses.rpn_loss_cls(num_anchors),  # loss depends on number of anchors
					losses.rpn_loss_regr(num_anchors)])

model_classifier.compile(	optimizer=optimizer_classifier, 
							loss=[losses.class_loss_cls,losses.class_loss_regr(len(classes_count)-1)],    #loss depends on number of classes
  							metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

model_all.compile( optimizer='sgd', loss='mae')   # mae = mean absolute error


iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {i: j for i, j in class_mapping.items()}
print("Started training")

for epoch in range(epochs):

	progbar = generic_utils.Progbar(epoch_length)
	print("Epoch {}/{}".format(epoch + 1, epochs))

	while True:
		try:

			if len(rpn_accuracy_rpn_monitor) == epoch_length :
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print("Average number of overlapping bounding boxes from RPN = {} for {} previous iterations".format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print("RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.")

			X, Y, img_data = next(data_gen_train)

			loss_rpn = model_rpn.train_on_batch(X, Y)

			P_rpn = model_rpn.predict_on_batch(X)

			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []
			
			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			if C.num_rois > 1:
				if len(pos_samples) < C.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
				try:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples
			else:
				# in the extreme case where num_rois = 1, we pick a random pos or neg sample
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = random.choice(neg_samples)
				else:
					sel_samples = random.choice(pos_samples)

			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
									  ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

			iter_num += 1
			
			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				print("Mean number of bounding boxes from RPN overlapping ground truth boxes: {}".format(mean_overlapping_bboxes))
				print("Classifier accuracy for bounding boxes from RPN: {}".format(class_acc))
				print("Loss RPN classifier: {}".format(loss_rpn_cls))
				print("Loss RPN regression: {}".format(loss_rpn_regr))
				print("Loss Detector classifier: {}".format(loss_class_cls))
				print("Loss Detector regression: {}".format(loss_class_regr))
				print("Elapsed time: {}".format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					
					print("Total loss decreased from {} to {}, saving weights".format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(C.model_path)

				break

		except Exception as e:
			print("Exception: {}".format(e))
			continue

print("Training completed")
