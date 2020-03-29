import statistics


def make_plot(scores, num_epoch):
    print("mean score", statistics.mean(scores))
    print("max score", max(scores))

    import matplotlib.pyplot as plt

    plt.plot(scores)
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.show()
    plt.ylim(-500, -50)
    plt.xlim(0, num_epoch)
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.plot(scores)
    plt.show()
    plt.ylim(-150, -50)
    plt.xlim(0, num_epoch)
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.plot(scores)
    plt.show()