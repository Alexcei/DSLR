import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def get_grades(dataset, prep_dataset, house, topic):
    df = prep_dataset[dataset["Hogwarts House"] == house][topic]
    return df.dropna()


def histogram(df):
    prep_dataset = df.select_dtypes(include=['int64', 'float64'])

    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    kwargs = dict(bins=25, alpha=0.5)
    for i, col in enumerate(prep_dataset.columns):
        plt.subplot(5, 3, i + 1)
        plt.hist(get_grades(df, prep_dataset, "Gryffindor", col), label = 'Gry', color = 'r', **kwargs)
        plt.hist(get_grades(df, prep_dataset, "Ravenclaw", col), label = 'Rav', color = 'b', **kwargs)
        plt.hist(get_grades(df, prep_dataset, "Slytherin", col), label = 'Sly', color = 'g', **kwargs)
        plt.hist(get_grades(df, prep_dataset, "Hufflepuff", col), label = 'Huf', color = 'y', **kwargs)
        plt.legend(loc = 'upper right')
        plt.title(col)
    fig.tight_layout()
    plt.savefig('plots/histograms.png')
    print('The histograms are saved in "plots/histograms.png" in the current directory.')


def main():
    args = sys.argv
    if len(args) == 1:
        try:
            os.makedirs('plots', exist_ok=True)
            df = pd.read_csv('data/dataset_train.csv', index_col='Index')
            histogram(df)
        except Exception as e:
            print("This program uses 'data/dataset_train.csv'")
            print(e)
    else:
        exit('This program does not accept parameters')


if __name__ == '__main__':
    main()
