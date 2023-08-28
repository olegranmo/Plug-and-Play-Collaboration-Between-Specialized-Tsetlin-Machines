import logging
import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler
import numpy as np
from keras.datasets import cifar10
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

resolution = 8

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=32000, type=int)
    parser.add_argument("--T", default=24000, type=int)
    parser.add_argument("--s", default=2.5, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="CPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--type_i_ii_ratio", default=1.0, type=float)
    parser.add_argument("--patch_size", default=3, type=int)
    parser.add_argument("--clause_drop_p", default=0.0, type=float)

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
    
    X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], resolution), dtype=np.uint8)
    for z in range(resolution):
        X_train[:,:,:,:,z] = X_train_org[:,:,:,:] >= (z+1)*255/(resolution+1)

    X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], resolution), dtype=np.uint8)
    for z in range(resolution):
        X_test[:,:,:,:,z] = X_test_org[:,:,:,:] >= (z+1)*255/(resolution+1)

    X_train = X_train.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3*resolution))
    X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3*resolution))

    Y_train=Y_train.reshape(Y_train.shape[0])
    Y_test=Y_test.reshape(Y_test.shape[0])
    
    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        type_i_ii_ratio=args.type_i_ii_ratio,
        patch_dim=(args.patch_size, args.patch_size),
        clause_drop_p=args.clause_drop_p
    )

    for epoch in range(args.epochs):
        tm.fit(X_train, Y_train)

        Y_test_scores = tm.score(X_test)

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

        np.savetxt("CIFAR10Thermometer_%d_%d_%d_%.1f_%d_%d_%d_%.2f.txt" % (epoch, args.num_clauses, args.T, args.s, args.patch_size, resolution, args.max_included_literals, args.clause_drop_p), Y_test_scores, delimiter=',') 



