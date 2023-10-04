import numpy as np
import keras
from keras.datasets import imdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imdb-num-words", default=5000, type=int)
parser.add_argument("--imdb-index-from", default=2, type=int)
parser.add_argument("--epoch", default=1, type=int)

args = parser.parse_args()

train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
train_x, train_y = train
test_x, test_y = test

Y_test = test_y.astype(np.uint32)

Y_test_scores_1 = np.loadtxt("class_sums/IMDBAnalyzer_%d_10000_8000_2.00_1_1_32_1_word_0.75_4899.txt" % (args.epoch), delimiter=',')
Y_test_scores_2 = np.loadtxt("class_sums/IMDBAnalyzer_%d_10000_8000_2.00_2_2_32_1_word_0.75_5000.txt" % (args.epoch), delimiter=',')
Y_test_scores_3 = np.loadtxt("class_sums/IMDBAnalyzer_%d_10000_8000_2.00_3_3_32_1_word_0.75_5000.txt" % (args.epoch), delimiter=',')
Y_test_scores_4 = np.loadtxt("class_sums/IMDBAnalyzer_%d_10000_8000_2.00_4_4_32_1_char_wb_0.75_3000.txt" % (args.epoch), delimiter=',')

votes = np.zeros(Y_test_scores_1.shape, dtype=np.float32)
for i in range(Y_test.shape[0]):
    votes[i] += 1.0*Y_test_scores_1[i]/(np.max(Y_test_scores_1) - np.min(Y_test_scores_1))
    votes[i] += 1.0*Y_test_scores_2[i]/(np.max(Y_test_scores_2) - np.min(Y_test_scores_2))

Y_test_predicted = votes.argmax(axis=1)

print("Team Accuracy: %.2f" % (100*(Y_test_predicted == Y_test).mean()))