import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def import_data():
    X = np.genfromtxt("train_X_svm.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_svm.csv", delimiter=',', dtype=np.float64)
    print(X.shape, Y.shape)
    return X, Y

def train_model(X, Y):
    # max = 0
    # count = 0
    # i = 0.1
    # c = 0
    # while i <= 5:
    #     pipe = make_pipeline(StandardScaler(), SVC(C=i))
    #     # train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=0)
    #     train_X = X[:700]
    #     test_X = X[700:]
    #     train_Y = Y[:700]
    #     test_Y = Y[700:]
    #     pipe.fit(train_X, train_Y)
    #     if count%10 == 0:
    #         print(count)
    #     if pipe.score(test_X, test_Y) > max:
    #         max = pipe.score(test_X, test_Y)
    #         c = i
    #     count += 1
    #     i += 0.1

    pipe = make_pipeline(StandardScaler(), SVC(C=1))
    pipe.fit(X, Y)
    return pipe

def save_model(W, file_name):
    saved_model = pickle.dump(W, open(file_name, "wb"))

if __name__ == "__main__":
    X, Y = import_data()
    W = train_model(X, Y)
    save_model(W, "WEIGHTS_FILE.csv")


