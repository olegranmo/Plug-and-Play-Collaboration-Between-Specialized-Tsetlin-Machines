import numpy as np
from keras.datasets import cifar10
import cv2
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer

device = "GPU"
max_included_literals = 32
resolution = 8
factor = 1

(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
Y_train=Y_train.reshape(Y_train.shape[0])
Y_test=Y_test.reshape(Y_test.shape[0])

##################################
##### Histogram of Gradients #####
##################################

imageSize = 32  #The size of the original image - in pixels - assuming this is a square image
channels = 3    #The number of channels of the image. A RBG color image, has 3 channels
classes = 10    #The number of classes available for this dataset

winSize = imageSize
blockSize = 12
blockStride = 4
cellSize = 4
nbins = 18
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = True
hog = cv2.HOGDescriptor((winSize,winSize),(blockSize, blockSize),(blockStride,blockStride),(cellSize,cellSize),nbins,derivAperture, winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)

fd = hog.compute(X_train_org[0])
X_train_hog = np.empty((X_train_org.shape[0], fd.shape[0]), dtype=np.uint32)
for i in range(X_train_org.shape[0]):
    fd = hog.compute(X_train_org[i]) 
    X_train_hog[i] = fd >= 0.1

fd = hog.compute(X_test_org[0])
X_test_hog = np.empty((X_test_org.shape[0], fd.shape[0]), dtype=np.uint32)
for i in range(X_test_org.shape[0]):
    fd = hog.compute(X_test_org[i])
    X_test_hog[i] = fd >= 0.1

tm_hog = TMClassifier(
    number_of_clauses=2000*factor,
    T=50*factor,
    s=10.0,
    max_included_literals=max_included_literals,
    platform=device,
    weighted_clauses=False
)

#################################
##### Adaptive Thresholding #####
#################################

X_train_threshold = np.copy(X_train_org)
X_test_threshold = np.copy(X_test_org)

for i in range(X_train_threshold.shape[0]):
    for j in range(X_train_threshold.shape[3]):
        X_train_threshold[i,:,:,j] = cv2.adaptiveThreshold(X_train_org[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

for i in range(X_test_threshold.shape[0]):
    for j in range(X_test_threshold.shape[3]):
        X_test_threshold[i,:,:,j] = cv2.adaptiveThreshold(X_test_org[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

tm_threshold = TMClassifier(
    number_of_clauses=2000*factor,
    T=500*factor,
    s=10.0,
    max_included_literals=max_included_literals,
    platform=device,
    weighted_clauses=True,
    patch_dim=(10, 10)
)

##############################
##### Color Thermometers #####
##############################

X_train_thermometer = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution), dtype=np.uint8)
for z in range(resolution):
    X_train_thermometer[:,:,:,:,z] = X_train_org[:,:,:,:] >= (z+1)*255/(resolution+1)

X_test_thermometer = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution), dtype=np.uint8)
for z in range(resolution):
    X_test_thermometer[:,:,:,:,z] = X_test_org[:,:,:,:] >= (z+1)*255/(resolution+1)

X_train_thermometer = X_train_thermometer.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3*resolution))
X_test_thermometer = X_test_thermometer.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3*resolution))

tm_thermometer_3 = TMClassifier(
    number_of_clauses=2000*factor,
    T=1500*factor,
    s=2.5,
    max_included_literals=max_included_literals,
    platform=device,
    weighted_clauses=True,
    patch_dim=(3, 3),
)

tm_thermometer_4 = TMClassifier(
    number_of_clauses=2000*factor,
    T=1500*factor,
    s=2.5,
    max_included_literals=max_included_literals,
    platform=device,
    weighted_clauses=True,
    patch_dim=(4, 4),
)

############################
##### Training of Team #####
############################

for epoch in range(100):
    print("#%d" % (epoch+1), end=' ')
    tm_hog.fit(X_train_hog, Y_train)
    Y_test_hog, Y_test_scores_hog = tm_hog.predict(X_test_hog, return_class_sums=True)
    print("HoG: %.1f%%" % (100*(Y_test_hog == Y_test).mean()), end=' ')

    tm_threshold.fit(X_train_threshold, Y_train)
    Y_test_threshold, Y_test_scores_threshold = tm_threshold.predict(X_test_threshold, return_class_sums=True)
    print("Adaptive Thresholding: %.1f%%" % (100*(Y_test_threshold == Y_test).mean()), end=' ')

    tm_thermometer_3.fit(X_train_thermometer, Y_train)
    Y_test_thermometer_3, Y_test_scores_thermometer_3 = tm_thermometer_3.predict(X_test_thermometer, return_class_sums=True)
    print("3x3 Color Thermometers: %.1f%%" % (100*(Y_test_thermometer_3 == Y_test).mean()), end=' ')

    tm_thermometer_4.fit(X_train_thermometer, Y_train)
    Y_test_thermometer_4, Y_test_scores_thermometer_4 = tm_thermometer_4.predict(X_test_thermometer, return_class_sums=True)
    print("4x4 Color Thermometers: %.1f%%" % (100*(Y_test_thermometer_4 == Y_test).mean()), end=' ')

    ##### Team Decision #####

    votes = np.zeros(Y_test_scores_hog.shape, dtype=np.float32)
    for i in range(Y_test.shape[0]):
        votes[i] += 1.0*Y_test_scores_threshold[i]/(np.max(Y_test_scores_threshold) - np.min(Y_test_scores_threshold))
        votes[i] += 1.0*Y_test_scores_thermometer_3[i]/(np.max(Y_test_scores_thermometer_3) - np.min(Y_test_scores_thermometer_3))
        votes[i] += 1.0*Y_test_scores_thermometer_4[i]/(np.max(Y_test_scores_thermometer_4) - np.min(Y_test_scores_thermometer_4))
        votes[i] += 1.0*Y_test_scores_hog[i]/(np.max(Y_test_scores_hog) - np.min(Y_test_scores_hog))
    Y_test_team = votes.argmax(axis=1)

    print("Team: %.1f%%" % (100*(Y_test_team == Y_test).mean()))
    print()