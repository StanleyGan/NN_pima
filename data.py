import pandas as pd
import os

'''
Returns the original dataset as a dataframe
Input: current working directory which contains the folder "data"
'''
def original(currentPath):
    df = pd.read_csv(os.path.join(currentPath, "data", "pima-diabetes.csv"))

    return df

'''
Returns the imputed training dataset used for my R project, as a dataframe
Input: current working directory which contains the folder "data"
'''
def trainImputed(currentPath):
    df = pd.read_csv(os.path.join(currentPath, "data", "pima_train.csv"))
    df = df.drop(df.columns[0], axis=1)
    return df

'''
Returns the imputed test datasets used for my R project, as a list of dataframes
Input: current working directory which contains the folder "data"
'''
def testImputed(currentPath):
    df_list = list()

    for i in range(1,6):
        df = pd.read_csv(os.path.join(currentPath, "data", "pima_test{}.csv".format(i)))
        df = df.drop(df.columns[0], axis=1)
        df_list.append(df)

    return df_list

if __name__ == "__main__":
   print("")
