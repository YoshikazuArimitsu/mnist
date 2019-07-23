import chainer
from chainer.functions.loss import softmax_cross_entropy


class Loss(chainer.Chain):
    """Loss計算クラス
    """

    def __init__(self, model):
        """.ctor

        Args:
            model: モデル
        """
        super(Loss, self).__init__()
        with self.init_scope():
            self.model = model

    """
    def __call__(self, x, t):
        y = self.model(x)
        loss = F.mean_squared_error(x, t)
        return loss
    """

    def __call__(self, y, t):
        """Loss計算

        Args:
            y (chainer.Variable): 推論結果
            t (chainer.Variable): 正解ラベル

        Returns:
            chainer.Variable: Loss
        """
        loss = softmax_cross_entropy.softmax_cross_entropy(y, t)
        return loss
