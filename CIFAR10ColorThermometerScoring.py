import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from keras.datasets import cifar10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=1500, type=int)
    parser.add_argument("--s", default=2.5, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patch_size", default=3, type=int)
    parser.add_argument("--resolution", default=8, type=int)

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
    
    X_train = np.empty((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], X_train_org.shape[3], args.resolution), dtype=np.uint8)
    for z in range(args.resolution):
        X_train[:,:,:,:,z] = X_train_org[:,:,:,:] >= (z+1)*255/(args.resolution+1)

    X_test = np.empty((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], X_test_org.shape[3], args.resolution), dtype=np.uint8)
    for z in range(args.resolution):
        X_test[:,:,:,:,z] = X_test_org[:,:,:,:] >= (z+1)*255/(args.resolution+1)

    X_train = X_train.reshape((X_train_org.shape[0], X_train_org.shape[1], X_train_org.shape[2], 3*args.resolution))
    X_test = X_test.reshape((X_test_org.shape[0], X_test_org.shape[1], X_test_org.shape[2], 3*args.resolution))

    Y_train=Y_train.reshape(Y_train.shape[0])
    Y_test=Y_test.reshape(Y_test.shape[0])
    
    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        patch_dim=(args.patch_size, args.patch_size),
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

        np.savetxt("CIFAR10ColorThermometers_%d_%d_%d_%.1f_%d_%d_%d_%d.txt" % (epoch+1, args.num_clauses, args.T, args.s, args.patch_size, args.resolution, args.max_included_literals, args.weighted_clauses), Y_test_scores, delimiter=',') 



