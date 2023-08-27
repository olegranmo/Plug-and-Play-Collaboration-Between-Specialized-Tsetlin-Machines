import logging
import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler
import numpy as np
from keras.datasets import cifar10
import cv2
from skimage.feature import hog
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer

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

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=50, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=False, type=bool)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--type_i_ii_ratio", default=1.0, type=float)

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
    Y_train = Y_train
    Y_test = Y_test

    Y_train=Y_train.reshape(Y_train.shape[0])
    Y_test=Y_test.reshape(Y_test.shape[0])
    
    fd = hog.compute(X_train_org[0])
    print(fd.shape)
    print(fd)
    X_train = np.empty((X_train_org.shape[0], fd.shape[0]), dtype=np.uint32)#dtype=np.float32)
    for i in range(X_train_org.shape[0]):
        fd = hog.compute(X_train_org[i]) 
        X_train[i] = fd >= 0.1

    fd = hog.compute(X_test_org[0])
    X_test = np.empty((X_test_org.shape[0], fd.shape[0]), dtype=np.uint32)#, dtype=np.float32)
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
        tm.fit(X_train, Y_train)

        Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)

        max_score = Y_test_scores.max(axis=1)
        max_score_index = Y_test_scores.argmax(axis=1)
        sorted_index = np.argsort(-1*max_score)

        correct = 0.0
        total = 0.0
        for i in sorted_index:
            if max_score_index[i] == Y_test[i]:
                correct += 1
            total += 1

            if total % 100 == 0:
                print("%d %.2f %.2f" % (max_score[i], total/sorted_index.shape[0], correct/total))

        np.savetxt("CIFAR10HistogramOfGradients_%d_%d_%.1f_%d_%d_%d.txt" % (epoch, args.num_clauses, args.T, args.s, patch_size, args.max_included_literals), Y_test_scores, delimiter=',') 



