import chainer
from chainer.cuda import to_cpu


class Predictor(chainer.Chain):
    """推論クラス
    """

    def __init__(self, model):
        """.ctor

        Args:
            model: モデル
        """
        super(Predictor, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, x):
        """推論の実行

        Args:
            x (array): データ

        Returns:
            int: ラベル(0~9)
        """
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            y = self.model(x)
            y = y.array

        y = to_cpu(y)
        pred_label = y.argmax(axis=1)[0]
        return pred_label
