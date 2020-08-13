from PIL import Image, ImageDraw,ImageFont
import numpy as np
import matplotlib.pyplot  as  plt
import matplotlib.image  as mpimg
import json
import pandas as pd
import os
import sys
import  pickle
sys.path.append('/home/qinyuanze')
sys.path.append(r'/home/qinyuanze/code/center/CenterNet/')
'''
数据说明：
该脚本针对于ttfnet对coco数据集的test测试，将其得到的json结果绘制到原始coco图片当中

数据规模：
累计得到498774个bbox，category共有90个类

数据示例：
{'image_id': 397133,
 'bbox': [382.9284973144531,
  71.58695220947266,
  121.8326416015625,
  272.15364837646484],
 'score': 0.7958970069885254,
 'category_id': 1}
'''

class test_one_image(object):
	def __init__(self, img_dir, json_file_path, score):
		self.img_dir = img_dir
		self.save_path = r'/home/qinyuanze/save_path'
		if not os.path.exists(self.save_path):
			os.mkdir(self.save_path)
		# self.name = self.img_path.split(r'/')[-1]
		# self.img_id = int(self.name.split(r'.')[-2].lstrip('0'))

		self.json_path = json_file_path
		self.json_load_file = r'/home/qinyuanze/img_info.txt'

		self.score = score

	def read_json(self):
		if os.path.exists(self.json_load_file):
			with open(self.json_load_file, 'rb') as f:
				img_info = pickle.load(f)
				img_info = pd.DataFrame(img_info)
			return img_info

		with open(self.json_path, 'r') as f:
			content = f.read()
			a = json.loads(content)

		img_info = pd.DataFrame(a)
		# write into dumps
		with open(self.json_load_file, 'wb') as f:
			pickle.dump(a, f)
		return img_info

	def plt_all_img(self):
		dirs = os.listdir(self.img_dir)
		img_info = self.read_json()
		for ind,i in enumerate(dirs):
			img_path = os.path.join(self.img_dir, i)
			img_id = int(i.split(r'.')[-2].lstrip('0'))
			one_img_info = img_info[img_info.image_id == img_id]
			self.plt_img(one_img_info, img_path=img_path)
			print('plot the {} img'.format(ind))
		return

	def plt_img(self, one_img_info, img_path):
		img = Image.open(img_path)
		draw = ImageDraw.Draw(img)
		font = ImageFont.truetype(r"/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", size=20)
		for i in one_img_info.index:
			bb = one_img_info.loc[i, 'bbox']
			one_score = one_img_info.loc[i, 'score']
			if one_score < self.score:
				continue
			width = bb[2]
			height = bb[3]
			bb[2] = width + bb[0]
			bb[3] = height + bb[1]
			if one_img_info.loc[i, 'category_id'] == 1:
				draw.rectangle(bb, outline='#ff0000', width=3)
				draw.text((bb[0], bb[1]), '1', font=font, fill="#ff0000")
			elif one_img_info.loc[i, 'category_id'] == 2:
				draw.rectangle(bb, outline='#00ff00', width=3)
				draw.text((bb[0], bb[1]), '2', font=font, fill="#00ff00")
			else:
				draw.rectangle(bb, outline='#0000ff', width=3)
				draw.text((bb[0], bb[1]), '3', font=font, fill="#0000ff")

		img_name = img_path.split(r'/')[-1]
		save_path = os.path.join(self.save_path, img_name)
		img.save(save_path)

		'''
		def plt_all_img(self):
			dirs = os.listdir(self.img_dir)
			img_info = self.read_json()
			for ind, i in enumerate(dirs, start=1):
				img_path = os.path.join(self.img_dir, i)
				# img_id = int(i.split(r'.')[-2].lstrip('0'))
				img_id = ind
				one_img_info = img_info[img_info.image_id == img_id]
				self.plt_img(one_img_info, img_path=img_path)
				print("plot the {} image".format(ind))
			return

		def plt_img(self, one_img_info, img_path):
			img = Image.open(img_path)
			draw = ImageDraw.Draw(img)
			font = ImageFont.truetype(r"/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", size=20)
			for i in one_img_info.index:
				bb = one_img_info.loc[i, 'bbox']
				one_score = one_img_info.loc[i, 'score']
				if one_score < self.score:
					continue
				width = bb[2]
				height = bb[3]
				bb[2] = width + bb[0]
				bb[3] = height + bb[1]
				if one_img_info.loc[i, 'category_id'] == 1:
					draw.rectangle(bb, outline='#ff0000', width=3)
					draw.text((bb[0], bb[1]), '1', font=font, fill="#ff0000")
			# elif one_img_info.loc[i, 'category_id'] == 2:
			# 	draw.rectangle(bb, outline='#00ff00', width=3)
			# 	draw.text((bb[0], bb[1]), '2', font=font, fill="#00ff00")
			# else:
			# 	draw.rectangle(bb, outline='#0000ff', width=3)
			# 	draw.text((bb[0], bb[1]), '3', font=font, fill="#0000ff")

			img_name = img_path.split(r'/')[-1]
			save_path = os.path.join(self.save_path, img_name)
			img.save(save_path)
		'''
def img_show(img_path):
	fig, ax = plt.subplots(1,1)
	img = mpimg.imread(img_path)
	ax.imshow(img.astype(np.uint8))
	plt.show()
	return

if __name__ == '__main__':
	base_path = r'/home/qinyuanze/img&json'
	img_dirs = os.path.join(base_path, r'val2017')
	img_path = os.path.join(base_path, r'val2017/000000229747.jpg')
	json_file_path = os.path.join(base_path, r'tmp.pkl.bbox.json')
	test = test_one_image(img_dir = img_dirs,
	                      json_file_path=json_file_path,
	                      score=0.4)
	test.plt_all_img()

	# img_show(r'/home/qinyuanze/save_path/000000229747.jpg')
	# img_show(img_path)




