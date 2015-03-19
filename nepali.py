# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import cv2
import scipy.io as sio
import pickle

# grab the MNIST dataset (if this is the first time you are running
# this script, this make take a minute -- the 55mb MNIST digit dataset
# will be downloaded)
print "[X] downloading data..."
dataset = datasets.fetch_mldata("MNIST Original")

# scale the data to the range [0, 1] and then construct the training
# and testing splits

(trainX, testX, trainY, testY) = train_test_split(
	dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)
print trainX.shape, trainY.shape
print type(trainY), trainY

# train the Deep Belief Network with 784 input units (the flattened,
# 28x28 grayscale image), 300 hidden units, 10 output units (one for
# each possible output classification, which are the digits 1-10)

try:
	with open('data.pkl', 'rb') as input:
	    dbn = pickle.load(input)

except:

	dbn = DBN(
		[trainX.shape[1], 900, 60],
		learn_rates = 0.3,
		learn_rate_decays = 0.9,
		epochs = 10,
		verbose = 1)
	dbn.fit(trainX, trainY)
	with open('data.pkl', 'wb') as output:
	    pickle.dump(dbn, output, pickle.HIGHEST_PROTOCOL)

# # compute the predictions for the test data and show a classification
# # report

preds = dbn.predict(testX)
print classification_report(testY, preds)
nepali = ["0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "aa",
    "i",
    "ii",
    "u",
    "uu",
    "ri",
    "ai",
    "aii",
    "o",
    "ou",
    "am",
    "a:",
    "ka",
    "kha",
    "ga",
    "gha",
    "nha",
    "cha",
    "chha",
    "ja",
    "jha",
    "ya",
    "ta",
    "tha",
    "da",
    "dha",
    "ara",
    "ta:",
    "tha:",
    "da:",
    "dha:",
    "na",
    "pa",
    "pha",
    "bha",
    "ma",
    "ye",
    "ra",
    "la",
    "wa",
    "sa",
    "kha",
    "sa",
    "sha-kha",
    "sha",
    "ha",
    "gya",
    "tra"
    ]


# # randomly select a few of the test instances
# for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
# 	# classify the digit
# 	pred = dbn.predict(np.atleast_2d(testX[i]))
 
# 	# reshape the feature vector to be a 28x28 pixel image, then change
# 	# the data type to be an unsigned 8-bit integer
# 	image = (testX[i] * 255).reshape((28, 28)).astype("uint8")
 
# 	# show the image and prediction
# 	print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
# 	cv2.imshow("Digit", image)
# 	cv2.waitKey(0) 



img = cv2.imread("./input.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inv = 255-gray
x_top = 0
y_top = 0
x_bottom = 0
y_bottom = 0


for x,row in enumerate(inv):
    for y,pix in enumerate(row):
        if pix>100:
            if x<x_top:
                x_top = x
            if x>x_bottom:
                x_bottom = x
            if y<y_top:
                y_top = y
            if y>y_bottom:
                y_bottom = y
img_croped = inv[x_top:x_bottom, y_top:y_bottom]
if img_croped.shape[0] > img_croped.shape[1]:
    size_max = img_croped.shape[0]
else:
    size_max = img_croped.shape[1]
padding = 3
size_max = size_max + 2*padding
blank_image = np.zeros((size_max,size_max), np.uint8)
height_offset = (size_max - img_croped.shape[0])/2
width_offset = (size_max - img_croped.shape[1])/2
blank_image[height_offset:height_offset + img_croped.shape[0],width_offset:width_offset + img_croped.shape[1]] = img_croped
final = cv2.resize(blank_image, (28, 28))
print final.shape
final_image = np.ravel(final)/255
pred = dbn.predict(np.atleast_2d(final_image))

print "The input image is ", nepali[int(pred[0])]
cv2.imshow('img',final)
cv2.waitKey(0)