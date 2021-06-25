import time
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, criterion, optimizer, num_epochs, X_train, y_train, X_val, y_val):

    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    for t in range(num_epochs):
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train)

        with torch.no_grad():
            ## TODO: MODEL EVAL
            y_val_pred = model(X_val)
            val_loss[t] = criterion(y_val_pred, y_val)

        print('Epoch {}, Train MSE {:.2f}, Val MSE {:.2f}'.format(t, loss.item(), val_loss[t]))
        train_loss[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))

    return y_train_pred, train_loss, val_loss
