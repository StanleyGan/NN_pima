import pandas as pd
import os
import copy

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

'''
Returns scaled test sets
Input: scaler, sklearn.StandardScaler()
        test_list, a list of test sets. A list where each element is a numpy 2D array of
        form [X,y] where X is the input and y is the label
'''
def returnScaledTestList(scaler, test_list):
    test_split_scaled_list = copy.deepcopy(test_list)

    for idx, test_data in enumerate(test_split_scaled_list):
        # print(test_data)
        temp_scaled_test = scaler.transform(test_data[0])
        test_split_scaled_list[idx][0] = temp_scaled_test

    return test_split_scaled_list

if __name__ == "__main__":
   print("")
