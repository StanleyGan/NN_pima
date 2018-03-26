from model import MLP
import itertools
from sklearn.model_selection import StratifiedKFold

def cross_validate(X, y, neurons, activations, dropout, inputDim, outputDim, lr, num_epochs, batch_size, seed, n_splits=5):
    skf=StratifiedKFold(n_splits=n_splits,random_state=seed)

    for train_index, val_index in skf.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

        mlp = MLP()
        mlp.buildModel(neurons=neurons, activations=activations, dropout=dropout,
                        inputDim=inputDim, outputDim=outputDim)
        mlp.train(X=X_train, y=y_train, X_test=X_val, y_test=y_val, lr=lr,
                    num_epochs=num_epochs, batch_size=batch_size, seed=seed,
                    printResults=False, returnResults=False)

        results = mlp.predict(X_val)
        accuracy = (results["y_pred_cls"][:,0] == y_val[:,0]).sum() / float(len(results["y_pred_cls"]))
        

def CV_pipeline(X, y,neurons, activations, dropout, lr, inputDim, outputDim, num_epochs=5000, batch_size=128, seed=1, n_splits=5):

    for (neu, act, dro, lr_s) in itertools.product(neurons, activations, dropout, lr):
        cross_validate(X, y, neurons=neu, activations=act, dropout=dro, lr=lr_s, seed, n_splits)
