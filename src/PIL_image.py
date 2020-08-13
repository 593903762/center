from PIL import Image, ImageDraw,ImageFont
import numpy as np
import matplotlib.pyplot  as  plt
import matplotlib.image  as mpimg
import json
import pandas as pd
import os
import sys
sys.path.append(r'/usr/share/fonts')
sys.path.append(r'/home/qinyuanze/code/center/CenterNet')
import pickle

sys.path.append('/home/qinyuanze')

'''
数据说明：

对于coco_helmat数据集，groundtruth只有两个类，一个是person，一个是hat
对于hat：只要有人戴着帽子就进行标注
对于person：有人没有戴着帽子就会被标注
所以，对于同一个人，只会有hat和person任选其一
'''

def test_pil():
	img_path = '/home/qinyuanze/code/center/CenterNet/data/coco_helmet/val/images/000034.jpg'
	img = Image.open(img_path)
	# im_np = np.asarray(img)
	draw = ImageDraw.Draw(img)
	for i in range(2):
		for j in range(2):
			point = (100*i, 200*j)
			draw.text(point, str(point), fill=128)
	img.save('test.jpg')

class draw_result_json(object):
	def __init__(self, score, data_dir):
		# 设定的score阈值
		self.score = score

		self.data_dir = data_dir

		# img_data_dir
		self.val_img_dir = os.path.join(self.data_dir, 'val/images/')

		# result
		self.json_file_path = r'/home/qinyuanze/code/center/CenterNet/exp/ctdet/coco_res18_test/results.json'
		self.save_img_dir = os.path.join(self.data_dir,'annotation_img_show/result_json_show'+str(self.score))
		if not os.path.exists(self.save_img_dir):
			os.makedirs(self.save_img_dir)

		self.empty_img_dir = os.path.join(self.save_img_dir, 'empty.txt')

		# label
		self.val_anno = os.path.join(self.data_dir, 'annotations/instances_val.json')

		# image_id_to_image_name
		self.image_id_to_image_name_file = os.path.join(self.data_dir, 'annotation_img_show/image_id_to_image_name_file.txt')

	def image_id_to_image_name(self):
		if os.path.exists(self.image_id_to_image_name_file):
			with open(self.image_id_to_image_name_file, 'rb') as f:
				img_info = pickle.load(f)
				img_info = pd.DataFrame(img_info)
			return img_info
		with open(self.val_anno, 'r') as f:
			content = f.read()
			a = json.loads(content)
		img_info = a['images']
		img_info = pd.DataFrame(img_info)
		del img_info['width'], img_info['height']
		with open(self.image_id_to_image_name_file, 'wb') as f:
			pickle.dump(img_info, f)
			img_info = pd.DataFrame(img_info)
		return img_info

	def deal_draw_json_result(self):
		with open(self.json_file_path, 'r') as f:
			content = f.read()
		a = json.loads(content)
		df = pd.DataFrame(a)
		image_num = np.unique(df.image_id)
		img_info = self.image_id_to_image_name()

		for i in image_num:
			one_image_info = df[df.image_id == i]
			one_image_info = one_image_info[one_image_info.score > self.score]
			if list(one_image_info.index) == []:
				write_log =  "the index {0} image is none\n".format(i)
				print(write_log)
				with open(self.empty_img_dir, 'a', encoding='utf-8') as f:
					f.write(write_log)
				continue
			img_name = img_info[img_info.id == i].iat[0,1] # get file_name
			img_path = os.path.join(self.val_img_dir,img_name)

			self.draw_img_save(img_path, one_image_info)
		return

	def draw_img_save(self, img_path, one_image_info):
		'''
		在helmat数据集中，annotation数据标注格式分别为(x,y,width,height)
		在PIL.draw中，函数draw.rectangle([x1,y1,x2,y2],outline,width)
		在PIL中，遵循opencv坐标轴，即（x是从左到右，y是从上到下）增大
		:param img_path:图片路径
		:return:
		'''
		img = Image.open(img_path)
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype(r"/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", size=20)
		# font = ImageFont.truetype("arial.ttf", 20)
		for i in one_image_info.index :
			bb = one_image_info.loc[i,'bbox']
			width = bb[2]
			height = bb[3]
			bb[2] = width+bb[0]
			bb[3] = height+bb[1]
			if one_image_info.loc[i,'category_id'] == 1:
				draw.rectangle(bb, outline='#ff0000', width=3)
				draw.text((bb[0], bb[1]), 'hat', font=font, fill="#ff0000")
			if one_image_info.loc[i,'category_id'] == 2:
				draw.rectangle(bb, outline='#00ff00', width=3)
				draw.text((bb[0], bb[1]), 'person', font=font, fill="#00ff00")

		img_file_name = os.path.split(img_path)[-1]
		save_name = ''.join([str(one_image_info.iloc[0, 0]), '_', img_file_name])
		save_path = os.path.join(self.save_img_dir, save_name)
		img.save(save_path)
		return

class draw_annotation_img(object):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		self.train_img_dir = os.path.join(self.data_dir, 'train/images/')
		self.val_img_dir = os.path.join(self.data_dir, 'val/images/')
		self.train_anno = os.path.join(self.data_dir,'annotations/instances_train.json')
		self.val_anno = os.path.join(self.data_dir, 'annotations/instances_val.json')
		self.save_dir = os.path.join(self.data_dir, 'annotation_img_show/annotatino_val/')
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
	def read_annotation_json(self):
		with open(self.val_anno, 'r') as f:
			content = f.read()
		a = json.loads(content)

		# image info
		img_info = a['images']
		img_info = pd.DataFrame(img_info)
		del img_info['width'], img_info['height']

		# annotation info
		anno_info = a['annotations']
		anno_info = pd.DataFrame(anno_info)
		save_col = ['image_id', 'bbox', 'category_id']
		for i in anno_info.columns:
			if i not in save_col:
				del anno_info[i]
		return img_info,anno_info

	def draw_img_save(self):
		img_info, anno_info = self.read_annotation_json()
		val_image_num = np.unique(anno_info['image_id'])

		for i in val_image_num:
			img_file_name = img_info.iloc[i, 1]
			img_id = img_info.iloc[i, 0]
			one_img_info = anno_info.loc[lambda x: x.image_id==img_id, :]
			img_file_path = os.path.join(self.val_img_dir, img_file_name)
			self.draw_img(img_file_path, one_img_info)
		return

	def draw_img(self, img_path, one_img_info):
		img = Image.open(img_path)
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype(r"/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", size=20)

		for i in one_img_info.index:
			bb = one_img_info.loc[i,'bbox']
			width = bb[2]
			height = bb[3]
			bb[2] = width + bb[0]
			bb[3] = height + bb[1]
			if one_img_info.loc[i,'category_id'] == 1:
				draw.rectangle(bb, outline='#ff0000', width=3)
				draw.text((bb[0], bb[1]), 'hat', font=font, fill="#ff0000")
			elif one_img_info.loc[i,'category_id'] == 2:
				draw.rectangle(bb, outline='#00ff00', width=3)
				draw.text((bb[0], bb[1]), 'person', font=font, fill="#00ff00")
			else:
				draw.rectangle(bb, outline='#0000ff', width=3)
				draw.text((bb[0], bb[1]), 'person', font=font, fill="#0000ff")
		img_file_name = os.path.split(img_path)[-1]
		save_name = ''.join([str(one_img_info.iloc[0,0]),'_',img_file_name])
		save_path = os.path.join(self.save_dir,save_name )
		img.save(save_path)
		return


def img_show(img_path):
	fig, ax = plt.subplots(1,1)
	img = mpimg.imread(img_path)
	ax.imshow(img.astype(np.uint8))
	plt.show()
	return

def draw_rectangle_img(img_path, bb, info_txt_path):
	img = Image.open(img_path)
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype(r"/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", size=20)
	# font = ImageFont.truetype("arial.ttf", 20)
	# width = bb[2]
	# height = bb[3]
	# bb[2] = width+bb[0]
	# bb[3] = height+bb[1]
	with open(info_txt_path, 'r') as f:
		num = f.readline()
		bbs = []
		info = f.readline().split('\n')[0]
		while info:
			xmin = float(info.split(' ')[1])
			ymin = float(info.split(' ')[2])
			xmax = float(info.split(' ')[3])
			ymax = float(info.split(' ')[4])
			bb = [xmin, ymin, xmax, ymax]
			bbs.append(bb)
			info = f.readline().split('\n')[0]
	for bb in bbs:
		print(bb)
		draw.rectangle(tuple(bb), outline='#ff0000', width=3)
		# draw.text((bb[0], bb[1]), '', font=font, fill="#ff0000")
	img_file_name = os.path.split(img_path)[-1]
	
	save_path = os.path.join('./', img_file_name)
	print(save_path)
	img.save(save_path)


if __name__ == '__main__':
	# 绘制result.json里面的图片
	# img_path = '/home/qinyuanze/code/center/CenterNet/data/coco_helmet/annotation_img_show/result_json_show0.4'
	# show_result_ob = draw_result_json()
	# show_result_ob.draw_img(img_path)
	# img_show('test.jpg')

	# 绘制annotation中的图片
	# show_annot_ob = draw_annotation_img(data_dir = ''/home/qinyuanze/code/center/CenterNet/data/coco_helmet/'')
	# show_annot_ob.draw_img_save()

	# 绘制result.json中的图片
	# show_result_ob = draw_result_json(0.4, data_dir = '/home/qinyuanze/code/center/CenterNet/data/coco_helmet/')
	# show_result_ob.deal_draw_json_result()

	# img_show(os.path.join(img_path, '192_000660.jpg'))


	pass


