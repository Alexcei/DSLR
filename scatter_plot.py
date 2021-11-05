import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def get_grades(dataset, prep_dataset, house, topic):
    return prep_dataset[dataset["Hogwarts House"] == house][topic]


def scatter_plot(df):
    prep_dataset = df.select_dtypes(include=['int64', 'float64'])

    fig, axes = plt.subplots(13, 13, figsize=(60, 40))
    i = 1
    for col in prep_dataset.columns:
        for col2 in prep_dataset.columns:
            plt.subplot(13, 13, i)
            i += 1
            plt.scatter(get_grades(df, prep_dataset, "Gryffindor", col), 
                get_grades(df, prep_dataset, "Gryffindor", col2), label='Gry', color='r')
            plt.scatter(get_grades(df, prep_dataset, "Ravenclaw", col),
                get_grades(df, prep_dataset, "Ravenclaw", col2), label='Rav', color='b')
            plt.scatter(get_grades(df, prep_dataset, "Slytherin", col),
                get_grades(df, prep_dataset, "Slytherin", col2), label='Sly', color='g')
            plt.scatter(get_grades(df, prep_dataset, "Hufflepuff", col),
                get_grades(df, prep_dataset, "Hufflepuff", col2), label='Huf', color='y')
            plt.legend(loc='upper right')
            plt.title(col)
    fig.tight_layout()
    plt.savefig('plots/scatter_plot.png')
    print('The scatter_plot are saved in "plots/scatter_plot.png" in the current directory.')

def main():
    args = sys.argv
    if len(args) == 1:
        try:
            os.makedirs('plots', exist_ok=True)
            df = pd.read_csv('data/dataset_train.csv', index_col='Index')
            scatter_plot(df)
        except Exception as e:
            print("This program uses 'data/dataset_train.csv'")
            print(e)
    else:
        exit('This program does not accept parameters')


if __name__ == '__main__':
    main()
