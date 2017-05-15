import numpy as np

from learning.ann import Network, Dense, Tanh


def build_net():
    return Network(784, [Dense(60), Tanh(), Dense(10)])


def pull_mnist():
    from csxdata import CData, roots
    frame = CData(roots["misc"] + "mnist.pkl.gz", None, None, 10000, fold=False)
    frame.transformation = "std"
    return frame


if __name__ == '__main__':
    model = build_net()
    mnist = pull_mnist()
    allx = len(mnist.learning)
    strl = len(str(allx))
    epochs = 30
    estr = len(str(epochs))

    for e in range(1, 30):
        bcosts = []
        cost, acc = model.evaluate(*mnist.table("testing"))
        print("Epoch: {:>{w}}/{} Acc: {:>6.2%} Cost: {:>6.4f}"
              .format(e, epochs, acc, cost, w=estr))
        batches = mnist.batchgen(20, "learning")
        for b, (X, Y) in enumerate(batches, start=1):
            cost = model.learn_batch(X, Y)
            bcosts.append(cost / Y.shape[0])
            print("\rDone: {:>{w}}/{} Cost: {:>6.4f}"
                  .format(b, allx//20, np.mean(bcosts), w=strl), end="")
        print()
