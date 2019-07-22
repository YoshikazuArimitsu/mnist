import chainer
import chainer.links as L
import chainer.functions as F


class MLP(chainer.Chain):
    """MLPモデルクラス
    """

    def __init__(self, n_mid_units=100, n_out=10):
        """.ctor

        Args:
            n_mid_units (int, optional): 中間層のサイズ. Defaults to 100.
            n_out (int, optional): 出力ラベル数. Defaults to 10.
        """
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        """Forward計算

        Args:
            x(array):   データ
        """
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
