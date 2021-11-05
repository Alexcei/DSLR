import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


def preprocess(df):
    cols = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
    cols = ["Hogwarts House"] + cols
    return df[cols]


def pair_plot(df):
    prep_dataset = preprocess(df)
    sns.set(style="ticks", color_codes=True)
    sns.pairplot(prep_dataset, hue="Hogwarts House", markers=".", height=2)
    plt.savefig('plots/pair_plots.png')
    print('The pair_plots are saved in "plots/pair_plots.png" in the current directory.')


def main():
    args = sys.argv
    if len(args) == 1:
        try:
            os.makedirs('plots', exist_ok=True)
            df = pd.read_csv('data/dataset_train.csv', index_col='Index')
            pair_plot(df)
        except Exception as e:
            print("This program uses 'data/dataset_train.csv'")
            print(e)
    else:
        exit('This program does not accept parameters')


if __name__ == '__main__':
    main()