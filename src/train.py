import time
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def train(model, criterion, optimizer, num_epochs, X_train, y_train, X_val, y_val, train_loader):

#     train_loss = []
#     val_loss = []
#     start_time = time.time()

#     for t in range(num_epochs):
#         model.train()
#         epoch_losses = []

#         hidden_train = None

#         for data in train_loader:

#             X_batch, y_batch = data[0].to(device), data[1].to(device)
#             # if hidden_train is not None:
#             #     print('forwarding x of shape {}, hidden of shape {}'.format(X_batch.shape, hidden_train.shape))
#             y_train_pred, hidden_train = model(X_batch, hidden_train)

#             h_0, c_0 = hidden_train
#             h_0.detach_(), c_0.detach_()
#             hidden_train = (h_0, c_0)

#             print('y train pred size: ', y_train_pred.shape)
#             print('y_batch size: ', y_batch.shape)
#             # TODO: change y_batch size to BSx21 instead of BSx1
#             loss = criterion(y_train_pred, y_batch)

#             epoch_losses.append(loss.item())

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         with torch.no_grad():
#             model.eval()
#             train_loss.append(np.mean(epoch_losses))
#             y_val_pred, _ = model(X_val, hidden_train)
#             val_loss.append(criterion(y_val_pred, y_val).item())
#             print('Epoch {}, Train MSE {:.3f}, Val MSE {:.3f}'.format(t, np.mean(epoch_losses), val_loss[-1]))


#     training_time = time.time() - start_time
#     print("Training time: {}".format(training_time))

#     return train_loss, val_loss

def train(model, criterion, optimizer, num_epochs, X_train, y_train, X_val, y_val, leftover, bptt, bs, T):

    train_loss = []
    val_loss = []
    start_time = time.time()

    for t in range(num_epochs):
        model.train()
        epoch_losses = []

        hidden_train = None

        j = np.random.randint(0, leftover)
        X_train_epoch = X_train[j:j + T*bs]
        X_train_epoch = X_train_epoch.reshape((bs, T, -1))

        y_train_epoch = y_train[j:j+T*bs]
        y_train_epoch = y_train_epoch.reshape((bs, T, -1))

        for i in range(0, T, bptt):

            x = X_train_epoch[:, i:i+bptt, :].to(device)
            y = y_train_epoch[:, i:i+bptt, :].to(device)
            y_train_pred, hidden_train = model(x, hidden_train)
            
            h_0, c_0 = hidden_train
            h_0.detach_(), c_0.detach_()
            hidden_train = (h_0, c_0)

            loss = criterion(y_train_pred, y)
            
            epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_hidden = None

            for k in range(0, len(X_train), bptt):
                x = X_train[k:k+bptt, :].unsqueeze(dim=0).to(device)
                _, val_hidden = model(x, val_hidden)

            train_loss.append(np.mean(epoch_losses))
            y_val_pred, _ = model(X_val.unsqueeze(dim=0), val_hidden)
            y_val_pred = y_val_pred.squeeze(dim=0)
            val_loss.append(criterion(y_val_pred, y_val).item())
            print('Epoch {}, Train MSE {:.3f}, Val MSE {:.3f}'.format(t, np.mean(epoch_losses), val_loss[-1]))


    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))

    return train_loss, val_loss