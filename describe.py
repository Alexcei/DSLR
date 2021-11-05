import numpy as np
import pandas as pd
import math
import sys


def ft_count(arr):
    count = 0
    for val in arr:
        if not np.isnan(val):
            count += 1
            
    return count


def ft_sum(arr):
    sum_ = 0
    for val in arr:
        if not np.isnan(val):
            sum_ += val
            
    return sum_


def ft_mean(arr):
    count = ft_count(arr)
    if count == 0:
        return None
    sum_ = ft_sum(arr)

    return sum_ / count


def ft_min(arr):
    if ft_count(arr) == 0:
        return None
    min_ = arr[0]
    for val in arr:
        if not np.isnan(val) and val < min_:
            min_ = val
            
    return min_


def ft_max(arr):
    if ft_count(arr) == 0:
        return None
    max_ = arr[0]
    for val in arr:
        if not np.isnan(val) and val > max_:
            max_ = val
            
    return max_


def ft_std(arr):
    count = ft_count(arr) - 1
    if count < 1:
        return None
    
    mean_ = ft_mean(arr)
    sum_diff_sq = 0
    for val in arr:
        if not np.isnan(val):
            sum_diff_sq += (val - mean_)**2

    return (sum_diff_sq / count)**0.5


def ft_percent(arr, p):
    arr = [var for var in arr if not np.isnan(var)]
    count = ft_count(arr)
    if count == 0:
        return None
    
    arr.sort()
    k = (count - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    
    d0 = arr[int(f)] * (c - k)
    d1 = arr[int(c)] * (k - f)
    
    return d0 + d1


def describe(df):
    index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    df = df.select_dtypes(include=['int64', 'float64'])
    result = pd.DataFrame(columns=df.columns, index=index)
    
    for column in df.columns:
        result.loc['count', column] = ft_count(df[column].values)
        result.loc['mean', column] = ft_mean(df[column].values)
        result.loc['std', column] = ft_std(df[column].values)
        result.loc['min', column] = ft_min(df[column].values)
        result.loc['25%', column] = ft_percent(df[column].values, 0.25)
        result.loc['50%', column] = ft_percent(df[column].values, 0.5)
        result.loc['75%', column] = ft_percent(df[column].values, 0.75)
        result.loc['max', column] = ft_max(df[column].values)
        
    return result.astype('float64')


def main():
    args = sys.argv
    if len(args) == 2:
        try:
            df = pd.read_csv(args[1])
            print(describe(df))
        except Exception as e:
            print(f"Can't process the file '{args[1]}'")
            print(e)
    else:
        exit('Input example: python describe.py data/dataset_train.csv')


if __name__ == '__main__':
    main()
