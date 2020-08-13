# from PIL import Image
import sys
import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.pyplot as plt # plt 用于显示图片
import numpy as np
# import d2lzh1981 as d2l
import pandas as pd
import json
sys.path.append('/home/qinyuanze/code/center/CenterNet/data/')
VOC_BBOX_LABEL_NAMES = ('hat','person')

# 图片展示
def vis_img(img, ax=None):
	"""Visualize a color image.
		load the image_path to numpy.ndarray:
			img = mpimg.imread(img_path)
		Args:
			img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
				This is in RGB format and the range of its value is
				:math:`[0, 255]`.
			ax (matplotlib.axes.Axis): The visualization is displayed on this
				axis. If this is :obj:`None` (default), a new axis is created.
		Returns:
			~matploblib.axes.Axes:
			Returns the Axes object with the plot for further tweaking.
	"""
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
	ax.imshow(img.astype(np.uint8))
	return ax

def vis_bbox(img, bbox, label=None, score=None, ax=None):
	"""Visualize bounding boxes inside image.
	Args:
		img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
			This is in RGB format and the range of its value is
			:math:`[0, 255]`.
	bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
			:math:`R` is the number of bounding boxes in the image.
			Each element is organized
			by :math:`( x_{min},y_{min}, x_{max}, y_{max})` in the second axis.
		label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
			The values correspond to id for label names stored in
			:obj:`label_names`. This is optional.
		score (~numpy.ndarray): A float array of shape :math:`(R,)`.
			 Each value indicates how confident the prediction is.
			 This is optional.
		label_names (iterable of strings): Name of labels ordered according
			to label ids. If this is :obj:`None`, labels will be skipped.
		ax (matplotlib.axes.Axis): The visualization is displayed on this
			axis. If this is :obj:`None` (default), a new axis is created.
	Returns:
		~matploblib.axes.Axes:
		Returns the Axes object with the plot for further tweaking.
	"""

	label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']

	if label is not None and not len(bbox) == len(label):
		raise ValueError('The length of label must be same as that of bbox')
	if score is not None and not len(bbox) == len(score):
		raise ValueError('The length of score must be same as that of bbox')

	ax = vis_img(img, ax=ax)

	# If there is no bounding box to display, visualize the image and exit.
	if len(bbox) == 0:
		return ax

	for i, bb in enumerate(bbox):
		xy = (bb[0], bb[1])
		width = bb[2] - bb[0]
		height = bb[3] - bb[1]
		ax.add_patch(plt.Rectangle(
			xy, bb[2],bb[3], fill=False, edgecolor='red', linewidth=1))

		caption = list()

		# add label
		if label is not None and label_names is not None:
			lb = label[i]
			if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
				raise ValueError('No corresponding name is given')
			caption.append(label_names[lb])

		# add score
		if score is not None:
			sc = score[i]
			caption.append('{:.2f}'.format(sc))

		if len(caption) > 0:
			ax.text(bb[0],bb[1],': '.join(caption), style='italic',
					bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
	return ax

def read_json_result(json_file, i=2):
	with open(json_file, 'r') as f:
		content = f.read()
	a = json.loads(content)
	df = pd.DataFrame(a)
	image_1 = df[df.image_id == i]
	image_1_bbox = list(image_1.bbox)
	image_1_score = list(image_1.score)
	image_1_label = list(image_1.category_id)

	return image_1_bbox, image_1_score, image_1_label

def img_show(img_path):
	fig, ax = plt.subplots(1,1)
	img = mpimg.imread(img_path)
	ax.imshow(img.astype(np.uint8))
	plt.show()
	return

def img_annotate(img_path):
	fig, ax = plt.subplots(1, 1)
	img = mpimg.imread(img_path)
	json_file_path = r'/home/qinyuanze/code/center/CenterNet/exp/ctdet/coco_res18_test/results.json'

	image_1_bbox, image_1_score, image_1_label = read_json_result(json_file_path)
	ax = vis_bbox(img, image_1_bbox, ax=ax)
	plt.show()
	return

if __name__ == '__main__':
	img_path = '/home/qinyuanze/code/center/CenterNet/data/coco_helmet/val/images/000114.jpg'
	# img_show(img_path)
	img_annotate(img_path)