import torch
from torch import nn as nn
import numpy as np
import torch.nn.functional as F
import random
import os
import cv2
import math
import pandas as pd
from tqdm import trange

torch.set_default_tensor_type(torch.FloatTensor)

#训练路径
path = "train/TRAIN"
#设置batch_size为256——更改为32，由于内存不足
batch_size = 64
#设置当前batch的数
batch_num = 0
#类别数
#是否碰撞为一个二分类问题
num_of_class = 2
#简称：SA：SteeringAngel;PC:ProbabilityofCollision
#获取图像Steering值
df=pd.read_csv('train/data.csv',sep=" ")["Steering"]
#获取图像列表及当前获取到的索引值
picture_list = os.listdir(path)
now_picture = 0


#组成batch数据
def select_data(path,batch_size,picture_list,now_picture,batch_num):
	img_list = []
	tag_list = []
	if now_picture > len(picture_list)-batch_size:
		now_picture = 0
		batch_num += 1
	for i in range(batch_size):
		image = cv2.imread(path+"/"+picture_list[now_picture],cv2.IMREAD_GRAYSCALE)
		image = np.array([image], dtype='float32')
		image_tag = df[int(picture_list[now_picture].strip(".jpg"))]
		img_list.append(image)
		tag_list.append([image_tag])
		now_picture += 1

	if torch.cuda.is_available():
		return torch.tensor(img_list).cuda(),torch.tensor(tag_list).cuda(),batch_num,now_picture
	else:
		return torch.tensor(img_list),torch.tensor(tag_list),batch_num,now_picture


#构建模型的Architecture
class DroNet(nn.Module):
	def __init__(self):
		super(DroNet,self).__init__()
		#模型
		#第一层为一个5*5的卷积和3*3的最大池化，步长均为2，padding=1
		self.Layer1 = nn.Sequential(
			nn.Conv2d(1,32,kernel_size=5,stride=2,padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2),
			)
		self.ResBlock1 = nn.Sequential(
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			#原文代码中，这里只有一个要指定步长2，第二个不用
			nn.Conv2d(32,32,kernel_size=3,stride=2,padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),	
			)
		self.ResBlock2 = nn.Sequential(
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32,64,kernel_size=3,stride=2,padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
			)
		self.ResBlock3 = nn.Sequential(
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64,128,kernel_size=3,stride=2,padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
			)
		self.DropoutLayer = nn.Sequential(
			nn.Dropout(0.5),
			nn.ReLU(inplace=True),
			)
		self.Fc1 = nn.Sequential(
			nn.Linear(8192,1),
			)
		self.Fc2 = nn.Sequential(
			nn.Linear(8192,2),
			#gitHub公开源码中这里加了一个sigmoid，应该是将概率化为0-1的区间
			nn.Sigmoid(),
			)
		self.Shortcut1 = nn.Sequential(
			nn.Conv2d(32,32,kernel_size=1,stride=2,padding=1),
			)
		self.Shortcut2 = nn.Sequential(
			nn.Conv2d(32,64,kernel_size=1,stride=2,padding=1),
			)
		self.Shortcut3 = nn.Sequential(
			nn.Conv2d(64,128,kernel_size=1,stride=2,padding=1),
			)
	def forward(self,x):
		x = self.Layer1(x)
		x1 = self.ResBlock1(x)
		x2 = self.Shortcut1(x)
		x = x1 + x2
		x1 = self.ResBlock2(x)
		x2 = self.Shortcut2(x)
		x = x1 + x2
		x1 = self.ResBlock3(x)
		x2 = self.Shortcut3(x)
		x = x1 + x2
		x = self.DropoutLayer(x)
		x = x.view(x.size(0), -1)
		#先训练SA
		SA = self.Fc1(x)
		#PC = self.Fc2(x)
		return SA#,PC


if __name__ == '__main__':
	#训练
	model = DroNet()
	model.load_state_dict(torch.load("model120batch.tar"))
	model.eval()
	SAloss_fn = nn.MSELoss()
	PCloss_fn = nn.BCELoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	#使用cuda训练
	if torch.cuda.is_available():
		model = model.cuda()
		SAloss_fn = SAloss_fn.cuda()
		PCloss_fn = PCloss_fn.cuda()

	#更改迭代次数*8，由于batchsize减小为1/8，设置batch_num为30
	for it in trange(20988,ncols=60):
		x,SA,batch_num,now_picture = select_data(path,batch_size,picture_list,now_picture,batch_num)
		#SA_pred,PC_pred = model(x)
		#SAloss = SAloss_fn(SA_pred,SA)
		#PCloss = PCloss_fn(PC_pred,PC)
		#定义总损失Totloss_fn = SAloss + max(0,1-math.exp(-decay(epoch-epoch0)))*PCloss
		SA_pred = model(x)
		SAloss = SAloss_fn(SA_pred,SA)
		#print("batch " + str(batch_num),"iteration " +str(it),"loss "+str(SAloss.item()))
		optimizer.zero_grad()
		SAloss.backward()
		optimizer.step()
		if batch_num > 30:
			break

	torch.save(model.state_dict(), "model150batch.tar")
