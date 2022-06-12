import matplotlib.pyplot as plt

def livePlot(episode, data):
    if(episode == 0):
        plt.plot(episode, data[-1])
    else:
        plt.scatter(episode, data[-1])
    plt.draw()
    plt.show(block = False)
    plt.pause(0.00001)
