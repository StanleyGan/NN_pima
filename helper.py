from model import MLP
import itertools
from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_validate(X, y, neurons, activations, dropout, inputDim, outputDim, lr, num_epochs, batch_size, seed, n_splits=5):
    skf=StratifiedKFold(n_splits=n_splits,random_state=seed)

    acc_list = list()
    cv_count = 0
    for train_index, val_index in skf.split(X,y[:,0]):
        print("Cross validation set: {}".format(cv_count+1))
        X_train = X[train_index, :]
        y_train = y[train_index, :]
        X_val = X[val_index]
        y_val = y[val_index]

        mlp = MLP()
        mlp.buildModel(neurons=neurons, activations=activations, dropout=dropout,
                        inputDim=inputDim, outputDim=outputDim)
        mlp.train(X=X_train, y=y_train, X_test=X_val, y_test=y_val, lr=lr,
                    num_epochs=num_epochs, batch_size=batch_size, seed=seed,
                    printResults=False, returnResults=False)

        results = mlp.predict(X_val)
        # print results["y_pred_cls"][:,0]
        accuracy = (results["y_pred_cls"][:,0] == y_val[:,0]).sum() / float(len(results["y_pred_cls"]))
        acc_list.append(accuracy)
        print("Accuracy: {}".format(accuracy))

        cv_count += 1

    return acc_list


def CV_pipeline(X, y,neurons, activations, dropout, lr, inputDim, outputDim, num_epochs=5000, batch_size=128, seed=1, n_splits=5):
    parameters_acc = dict()
    parameters_details = dict()
    track = 0

    for (neu, act, dro, lr_s) in itertools.izip(neurons, activations, dropout, lr):
        print("Parameter set {}:".format(track +1))
        acc_list = cross_validate(X, y, neurons=neu, activations=act, dropout=dro, lr=lr_s, inputDim=inputDim,
        outputDim=outputDim, num_epochs=num_epochs, batch_size=batch_size, seed=seed, n_splits=n_splits)

        temp_dict = dict()
        temp_dict["neurons"] = neu
        temp_dict["activations"] = act
        temp_dict["dropout"] = dro
        temp_dict["learning_rate"] = lr_s
        parameters_acc[track] = np.mean(acc_list)
        parameters_details[track] = temp_dict

        track += 1

        print("")

    best_acc = max(parameters_acc.values())
    best_param = [i for i in parameters_acc.keys() if parameters_acc[i] == best_acc]

    print("The best cross validation accuracy is: {}".format(best_acc))
    print("The parameters which achieve this accuracy are: ")
    for idx, par in enumerate(best_param):
        print("{0}: {1}".format(idx, parameters_details[par]))
