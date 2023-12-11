import time
import random
import numpy as np
import idx2numpy
import matplotlib.pyplot
from tqdm import tqdm


##################################
# load dataset

train_img: np.ndarray = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')   # (60k, 28, 28)
train_lab: np.ndarray = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')   # (60k,)
test_img: np.ndarray = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')   # (10k, 28, 28)
test_lab: np.ndarray = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')   # (10k,)

# num = 12
# print(train_lab[num])
# matplotlib.pyplot.imshow(train_img[num], cmap="Greys", interpolation="None")
# matplotlib.pyplot.show()

##################################


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(sigx: np.ndarray):
    return sigx * (1 - sigx)


def leaky_relu(x: np.ndarray):
    return np.maximum(x, 0.01 * x)


def leaky_relu_d(lrx: np.ndarray):
    return np.where(lrx > 0, 1, 0.01)


def softmax(x: np.ndarray):
    # x : 1 x 10
    s = np.exp(x)
    return s / s.sum()


def softmax_d(smx: np.ndarray):
    # x : 1 x 10
    # smx : 1 x 10
    def f(i, j):
        return (smx[0][i] * (1 - smx[0][i])) if i == j else (- smx[0][i] * smx[0][j])
    return np.array([[f(i, j) for j in range(10)] for i in range(10)])     # 10 x 10


def cross_entropy(pred, lab):
    # pred : 1 x 10
    # lab : 0 - 9
    return - np.log(pred[0][lab])


def cross_entropy_d(pred, lab):
    # pred : 1 x 10
    # lab : 0 - 9
    d = np.zeros((1, 10))
    d[0][lab] = -1. / pred[0][lab]
    return d    # 1 x 10


class mnist_net:
    def __init__(self, size_h1, size_h2, act, act_d):
        self.size = {
            'in': 28*28,        # input = 784
            'h1': size_h1,      # hidden 1
            'h2': size_h2,      # hidden 2
            'out': 10,          # output
        }
        self.w = [
            np.random.randn(self.size['in'], self.size['h1']) * np.sqrt(1./self.size['h1']),    # 784 x h1
            np.random.randn(self.size['h1'], self.size['h2']) * np.sqrt(1./self.size['h2']),    # h1 x h2
            np.random.randn(self.size['h2'], self.size['out']) * np.sqrt(1./self.size['out']),  # h2 x 10
        ]
        self.act = act
        self.act_d = act_d

    def forward(self, img):
        # img : 1 x 784
        p1 = img @ self.w[0]  # 1 x h1
        a1 = self.act(p1)      # 1 x h1

        p2 = a1 @ self.w[1]  # 1 x h2
        a2 = self.act(p2)     # 1 x h2

        p3 = a2 @ self.w[2]  # 1 x 10
        a3 = softmax(p3)     # 1 x 10

        return a3

    def backward(self, img, lab, lr):
        # img : 1 x 784
        # lab : 0 - 9

        # forward
        p1 = img @ self.w[0]  # 1 x h1
        a1 = self.act(p1)      # 1 x h1

        p2 = a1 @ self.w[1]  # 1 x h2
        a2 = self.act(p2)     # 1 x h2

        p3 = a2 @ self.w[2]  # 1 x 10
        a3 = softmax(p3)     # 1 x 10
        # loss = cross_entropy(a3, lab)

        # backward
        delta_w = {}
        err = cross_entropy_d(a3, lab) @ softmax_d(a3)  # 1 x 10, L / p3

        delta_w[2] = np.outer(a2, err)  # h2 x 10, L / w2
        err = err @ self.w[2].T     # 1 x h2, L / a2
        err = err * self.act_d(a2)   # 1 x h2, L / p2

        delta_w[1] = np.outer(a1, err)  # h1 x h2, L / w1
        err = err @ self.w[1].T     # 1 x h1, L / a1
        err = err * self.act_d(a1)   # 1 x h1, L / p1

        delta_w[0] = np.outer(img, err)  # 784 x h1, L / w0

        # update
        for i in range(3):
            self.w[i] -= delta_w[i] * lr

    def test(self):
        loss = 0
        right = 0
        for i in range(10_000):
            img = test_img[i].reshape(1, 784) / 255.
            lab = test_lab[i]

            p: np.ndarray = self.forward(img)

            loss += cross_entropy(p, lab)
            if np.argmax(p) == lab:
                right += 1
        print('----- test result -----')
        print('avg loss = ', loss / 10000.)
        print('accuracy = ', right / 100., '%')

    def train(self, lr, num):
        print('----- train -----')
        for _ in tqdm(range(num)):
            i = random.randrange(60_000)
            img = train_img[i].reshape(1, 784) / 255.
            lab = train_lab[i]

            self.backward(img, lab, lr)

##################################


net = mnist_net(256, 64, leaky_relu, leaky_relu_d)
net.test()
print('')

for i in range(70):
    print('epoch = ', i + 1)
    net.train(0.001 if i < 30 else 0.0001, 1000)
    net.test()
    print('')
