import pandas as pd
import numpy as np
import sys

HOUSES = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def std_scaler(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def predict(df, thetas):
    df = df.replace(np.nan, 0)
    df = df.iloc[:, 5:]
    
    x = std_scaler(df.values)
    result = []
    for w in thetas.values:
        sig = sigmoid(x.dot(w))
        result.append(sig)
        
    result = [HOUSES[i] for i in np.argmax(result, axis=0)]
    return result


def main():
    args = sys.argv
    if len(args) == 3:
        try:
            df = pd.read_csv(args[1], index_col='Index')
            thetas = pd.read_csv(args[2])

            pred = predict(df, thetas)
            pred = pd.DataFrame(pred, columns=['Hogwarts House'])
            pred.index.name = 'Index'
            pred.to_csv('data/houses.csv')
        except Exception as e:
            print("Can't predict")
            print(e)
    else:
        exit('Input example: python logreg_predict.py data/dataset_test.csv data/thetas.csv')


if __name__ == '__main__':
    main()