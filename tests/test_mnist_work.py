import unittest

import chainer
import chainer.links as L
import numpy as np
from chainer import training
from chainer.training import extensions
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from loss import Loss
from models.mlp import MLP
from models.cnn import CNN


class RandomModel(chainer.Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(RandomModel, self).__init__()
        with self.init_scope():
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        if self.l3.W is not None:
            w = np.random.rand(640).astype('float32')
            self.l3.W.data = w.reshape((10, 64))
        self.l3.b.data = np.random.rand(10).astype('float32')
        return self.l3(x)


class TestMnistWork(unittest.TestCase):
    """検証テスト
    小さいデータセットに対して学習を通し、モデルを検証する。
    """

    def test_mlp(self):
        """sklern.digits データセットで学習を流す - MLP
        """
        print("MLP Working check.")
        model = MLP()
        loss = Loss(model)
        classifier = L.Classifier(model, lossfun=loss)
        self.check_by_digits(classifier)

    def test_cnn(self):
        """sklern.digits データセットで学習を流す - CNN
        """
        print("CNN Working check.")
        model = CNN(filter_size=2)
        loss = Loss(model)
        classifier = L.Classifier(model, lossfun=loss)
        self.check_by_digits(classifier, cnn=True)

    def test_random(self):
        """ランダム
        """
        print("Random Working check.")
        model = RandomModel()
        loss = Loss(model)
        classifier = L.Classifier(model, lossfun=loss)
        self.check_by_digits(classifier)

    def check_by_digits(self, classifier, cnn=False):
        batchsize = 10

        digits = load_digits()
        digits.data = minmax_scale(digits.data).astype("float32")

        if cnn:
            digits.data = np.reshape(digits.data, (len(digits.data), 1, 8, 8))

        digits = list(zip(digits.data,
                          digits.target))

        train, test = train_test_split(digits)

        train_iter = chainer.iterators.SerialIterator(train, batchsize)
        test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                     repeat=False,
                                                     shuffle=False)

        # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(classifier)

        # Set up a trainer
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=-1)
        trainer = training.Trainer(updater, (10, 'epoch'))
        trainer.extend(extensions.Evaluator(
            test_iter, classifier, device=-1))

        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        print('')
        trainer.run()


if __name__ == "__main__":
    unittest.main()
