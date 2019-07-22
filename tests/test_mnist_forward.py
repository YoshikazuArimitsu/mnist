import unittest

import numpy as np

from models.mlp import MLP
from models.cnn import CNN


class TestMnistForward(unittest.TestCase):
    """
    動作テスト。
    Modelについてそもそも入力から出力までエラーを吐かずに動くかどうかをチェックします。
    """

    def test_forward_MLP(self):
        """伝搬チェック MLP
        """
        data = self.create_test_data()
        model = MLP()
        model(data)

    def test_forward_CNN(self):
        """伝搬チェック CNN
        """
        data = self.create_test_data()
        data = np.reshape(data, (1, 1, 28, 28))
        model = CNN()
        model(data)

    def create_test_data(self):
        x = np.random.rand(784).astype(np.float32)
        x = np.reshape(x, (1, 784))
        return x


if __name__ == "__main__":
    unittest.main()
