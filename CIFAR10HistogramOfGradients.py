import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from keras.datasets import cifar10
import cv2
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from time import time

patch_size = 0

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=50, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=False, type=bool)
    parser.add_argument("--epochs", default=100, type=int)

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
    Y_train = Y_train
    Y_test = Y_test

    Y_train=Y_train.reshape(Y_train.shape[0])
    Y_test=Y_test.reshape(Y_test.shape[0])
    
    fd = hog.compute(X_train_org[0])
    X_train = np.empty((X_train_org.shape[0], fd.shape[0]), dtype=np.uint32)
    for i in range(X_train_org.shape[0]):
        fd = hog.compute(X_train_org[i]) 
        X_train[i] = fd >= 0.1

    fd = hog.compute(X_test_org[0])
    X_test = np.empty((X_test_org.shape[0], fd.shape[0]), dtype=np.uint32)
    for i in range(X_test_org.shape[0]):
        fd = hog.compute(X_test_org[i])
        X_test[i] = fd >= 0.1

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        type_i_ii_ratio=args.type_i_ii_ratio
    )

    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)
        stop_testing = time()

        result_test = 100*(Y_test_scores.argmax(axis=1) == Y_test).mean()

        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (epoch+1, result_test, stop_training-start_training, stop_testing-start_testing))
        
        np.savetxt("CIFAR10HistogramOfGradients_%d_%d_%d_%.2f_%d_%d_%d.txt" % (epoch+1, args.num_clauses, args.T, args.s, patch_size, args.max_included_literals, args.weighted_clauses), Y_test_scores, delimiter=',') 



