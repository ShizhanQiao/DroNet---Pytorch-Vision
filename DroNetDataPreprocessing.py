import pandas as pd
import cv2
#预处理：将每个图像变换成200*200的灰度图
if __name__ == '__main__':
	df=pd.read_csv('train/data.csv',sep=" ")
	picname = df["Name"]

	for i in picname:
		img = cv2.imread("train/training/"+i,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(200,200))
		cv2.imwrite("train/TRAIN/"+i+".jpg",img)
		if int(i.strip(".jpg")) % 1000 == 0:
			print("now processing" + i)
