from torch import nn
import torch
import numpy as np
from eventCameraResNet8_train import DroNet,select_data
import os
import pandas as pd
import bisect

#测试参数确定
batch_size = 1
batch_num = 0
path = "train/TEST"
model = DroNet()
model.load_state_dict(torch.load("model120batch.tar"))
model.eval()
now_picture = 0
picture_list = os.listdir(path)
df=pd.read_csv('train/data.csv',sep=" ")["Steering"]
error_max = 10
right = 0

if __name__ == '__main__':
	#开始测试
	if torch.cuda.is_available():
		model = model.cuda()

	for i in range(100000):
		if batch_num >= 1:
			break
		else:
			x,SA,batch_num,now_picture = select_data(path,batch_size,picture_list,now_picture,batch_num)
			SA_pred = model(x)
			loss = (SA_pred-SA).cpu().detach().numpy()
			if abs(loss)<= error_max:
				right += 1
			print("pircture test no. " + str(now_picture),"loss " + str(loss[0][0]))

	print(right/len(picture_list))