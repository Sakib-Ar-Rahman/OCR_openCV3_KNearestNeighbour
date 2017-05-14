import numpy as np
import cv2 as cv2
import time
from matplotlib import pyplot as plt
import image_input_processing


# Training model with digits.png
def train_model_KNN(knn):
	img = cv2.imread('digits.png')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Now we split the image to 5000 cells, each 20x20 size
	cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

	# Make it into a Numpy array. It size will be (50,100,20,20)
	x = np.array(cells)

	# Now we prepare train_data and test_data.
	train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
	test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

	# Create labels for train and test data
	k = np.arange(10)
	train_labels = np.repeat(k,250)[:,np.newaxis]
	test_labels = train_labels.copy()

	# Initiate kNN, train the data, then test it with test data for k=1
	# knn = cv2.ml.KNearest_create()
	knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
	ret, result, neighbours, dist = knn.findNearest(test, k=1)
	print (cv2.ml.ROW_SAMPLE)

	# Now we check the accuracy of classification
	# For that, compare the result with test_labels and check which are wrong
	matches = (result==test_labels)
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/result.size
	print accuracy


	# save the data
	np.savez('knn_data.npz',train=train, train_labels=train_labels)



# ---------------------------- FEEDING WEBCAM IMAGE TO TEST ------------------------------
knn = cv2.ml.KNearest_create()
train_model_KNN(knn)
image_input_processing.main()
img = cv2.imread('resized_image.png')

# Now we prepare train_data and test_data.
test = img.reshape(-1,400).astype(np.float32) # Size = (2500,400)
(thresh, test2) = cv2.threshold(test, 128, 255, cv2.THRESH_BINARY)
# Initiate kNN, train the data, then test it with test data for k=1
ret, result, neighbours2, dist = knn.findNearest(test2, 10)


print("The neighbours: %s" %(neighbours2))
print ("The Number was: %s" %(str(ret)))
