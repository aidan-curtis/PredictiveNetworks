
import os
import imageio
import numpy as np
root = "./results/mnist_predrnn_pp"
folder = "2000"
total_loss = 0
total_loss_num = 0
for test_num in range(1, 11):
	path = root+"/"+folder+"/"+str(test_num)+"/"
	for image_num in range(11, 21):
		gt = imageio.imread(path+"gt"+str(image_num)+".png")
		pd = imageio.imread(path+"pd"+str(image_num)+".png")
		total_loss+=np.linalg.norm(gt-pd)
		total_loss_num += 1

print(float(total_loss)/float(total_loss_num))