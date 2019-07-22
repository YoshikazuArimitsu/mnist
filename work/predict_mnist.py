import argparse

import chainer
from models.mlp import MLP
from models.cnn import CNN
from predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--model', '-m', default='mlp',
                        help='Model(mlp/cnn)')
    args = parser.parse_args()

    model = CNN() if args.model == 'cnn' else MLP(args.unit, 10)
    chainer.serializers.load_npz('models/mnist.npz', model)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    predictor = Predictor(model)

    # Load the MNIST dataset
    if args.model == 'cnn':
        _, test = chainer.datasets.get_mnist(ndim=3)
    else:
        _, test = chainer.datasets.get_mnist()

    x = test[42][0]
    x = model.xp.asarray(x[None, ...])
    y = predictor(x)
    print(y)


if __name__ == '__main__':
    main()
