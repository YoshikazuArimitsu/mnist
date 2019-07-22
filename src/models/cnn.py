import chainer
import chainer.links as L
import chainer.functions as F


class CNN(chainer.Chain):
    """CNNモデルクラス
    """

    def __init__(self, n_out=10, filter_size=5):
        """.ctor

        Args:
            n_out (int, optional): 出力ラベル数. Defaults to 10.
            filter_size (int, optional): フィルタサイズ. Defaults to 5.
        """
        super(CNN, self).__init__()
        with self.init_scope():
            self.cn1 = L.Convolution2D(None, 32, filter_size)
            self.cn2 = L.Convolution2D(None, 64, filter_size)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        """Forward計算

        Args:
            x(array):   データ
        """
        h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 2)
        return self.l3(h2)
