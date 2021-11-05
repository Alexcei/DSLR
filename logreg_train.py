
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

HOUSES = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def std_scaler(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def gradient_descent(x, y, epochs, lr, eps):
    theta = np.zeros((1, x.shape[1]))
    size = len(x)
    loss_prev = 1
    losses = []
    
    i = 0
    while i < epochs:
        p = sigmoid(theta @ x.T)
        grad = (p - y.T) @ x / size
        loss = np.sum(y.T * np.log(p) + (1 - y.T) * np.log(1 - p)) / -size
        losses.append(loss)
        theta -= lr * grad
        i += 1
        if abs(loss - loss_prev) < eps:
            break
        loss_prev = loss
        
    print(f'Calculation stopped on {i} epochs.')       
    return theta[0].tolist(), losses


def preproc(df):    
    labels = df['Hogwarts House']
    
    y = []
    for house in HOUSES:
        y.append(np.array(labels == house).astype(int))
        
    df = df.replace(np.nan, 0)
    df = df.iloc[:, 5:]
    x = std_scaler(df.values)
    
    return x, y


def train(x, y, epochs=1000, lr=0.1, eps=0.0001):
    history = []
    thetas = []
    for one_class in y:
        theta, losses = gradient_descent(x, one_class, epochs, lr, eps)
        history.append(losses)
        thetas.append(theta)
        
    plot(history)
    return thetas


def plot(history):
    plt.figure(figsize=(14, 8))
    for i, house in enumerate(history):
        sns.scatterplot(x=range(len(house)), y=house, label=HOUSES[i])
    plt.legend(loc = 'upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('plots/training.png')
    print('Plot saved to plots/training.png')


def main():
    args = sys.argv
    if len(args) == 2:
        try:
            df = pd.read_csv(args[1], index_col='Index')
            x, y = preproc(df)
            thetas = train(x, y)
            thetas = pd.DataFrame(thetas, columns=df.iloc[:, 5:].columns, index=HOUSES)
            thetas.to_csv('data/thetas.csv', index=False)
        except Exception as e:
            print("Can't train")
            print(e)
    else:
        exit('Input example: python logreg_train.py data/dataset_train.csv')


if __name__ == '__main__':
    main()