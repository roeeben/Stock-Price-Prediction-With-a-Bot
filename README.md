# Short Term Stock Price Prediction with a Simple Trading Bot
<h1 align="center">
  <img src="./assets/MoneyOverTime.gif">
</h1>
  <p align="center">
    <a href="mailto:orel.ts@campus.technion.ac.il">Orel Tsioni</a> •
    <a href="mailto:roeebs123@gmail.com">Roee Ben Shlomo</a>
  </p>

In this small project we're predicting the price of a stock 1 minute into the future using an LSTM architecture that feeds on parallel batches that overlap between different epochs in a randomized manner.
We proceed to use a simple bot that's given an initial stake of money to try and have a profit, in an environment where the Buy&Hold strategy won't suffice.
More info down below.

- [Short Term Stock Price Prediction with a Simple Trading Bot](#short-term-stock-price-prediction-with-a-simple-trading-bot)
  * [Data](#data)
  * [Architecture](#architecture)
  * [Optuna](#optuna)
  * [Bot](#bot)
  * [Files in the repository](#files-in-the-repository)
  * [Results](#results)
  * [Further Work](#further-work)
  * [DISCLAIMER](#disclaimer)


## Data
We got our hands on a dataset that involves a couple of months of the CGEN stock that's divided for single minutes, with a total of 17,000 minutes.
More specifically, we used the closing price and the volume of each minute.

Instead of feeding these raw features to the model, we used [FinTA](https://github.com/peerchemist/finta) in order to have more sophisticated features that are used by traders to try and manually predict stocks' future.

After having these features we divided the mentioned 17k minutes into chronologically ordered train, validation and test sets of ~12k, ~3.5k and ~1.5k respectively. 

We then splitted the training data to batches and each batch to have sequences that are `bptt` long. Every epoch the batches started from a random offset, which helped us to learn the connection between different batches along the training.

## Architecture
We used PyTorch to create a model with an LSTM layer which has a dropout, in addition to a fully connected layer.

The model was written in a general fashion: we set all of his layers size, aswell as the mentioned `bptt` and `batch_size` as variables and proceeded to use [Optuna](https://github.com/optuna/optuna) to optimize over the validation loss.

This loss, aswell as the training loss, was defined as the MSE between the LSTM's prediction of a minute and the closing price of the next minute, which is of course what we're trying to predict.

## Optuna

As mentioned, all of the following hyperparameters were found by Optuna and the analysis can be found in `Optuna Optimization - CGEN.ipynb` :

|File name    | Purpose     | Value |
|-------------|---------|------|
|`hidden_dim`| size of LSTM hidden dimension | 93 |
|`num_layers`| number of LSTM layers | 2 |
|`num_epochs`| number of epochs | 123 |
|`dropout_p`| probability of dropout | 0.12 |
|`lr`| learning rate | 0.004 |
|`bs`| batch size | 415 |
|`bptt`| length of sequence in an iteration (in minutes) | 16 |
|`Optimizer`| kind of gradient-based optimizer to use | Adam |

## Bot
After training the model and feeding the test data, we use a simple bot that tries to buy the stock before it surges and sell it before it gets down. That way, it theoretically can have a profit despite the fact that the stock eventually gets to a lower price than its starting price.

Every minute the bot checks the percentage of different between the actual price of the current minute and the prediction of of the next minute's price. If the percentage is greater than some predefined positive threshold it generally buys a stock, and if the percentage is lower than some predefined negative threshold it sells a stock. For more specifics we refer to `stock_bot.py` .




## Files in the repository

|File name         | Purpose |
|----------------------|------|
|`data_prep.py`| All of the data transformations (train/valid/test splits & feature engineering) |
|`model.py`| The LSTM model|
|`stock_bot.py`| Everything that's related to the bot, including to its definition and simulations |
|`train.py`| The training loop|
|`CGEN_original.pkl`| The actual test prices for the bot to use |
|`CGEN_predict.pkl`| Our prediction for the test prices for the bot to use |
|`Model Training - CGEN.ipynb`| A notebook which shows our data with the features, aswell as the training procedure and graphs |
|`Optuna Optimization - CGEN.ipynb`| A notebook which has the entire hyperparameters optimization using Optuna|



## Results

After training the model for the mentioned period, feeding the test data to get a prediction and then giving the actual & predicted prices to the bot, we got the following:

<p class="aligncenter">
<img src="./assets/CGEN test.png">
</p>

In green we see the points in which the model buys, and in red: sells. 
The bot has had 52% of successful trades, but we don't give this number too much thoughts since a trader can have a profit even with 10% successful trades, aslong as the losing trades don't lose as much as the winning trades gain.
Overall, it was given 5000$ at the start of the simulation and finished with a value of 5204$, i.e. it gained 200$, which the Buy&Holder has lost 54$.

Up top we have the animation of the bot running in action on the test data, and if we freeze the animation at the end:

<p class="aligncenter">
  <img src="./assets/Bot on Test.png">
</p>

We can see how the bot made its profit: from minute ~600 to ~1100 we have an increase of the stock price, which the bot manages to utilize, and after that the stock starts to decrease back. While the Buy&Hold strategy lose all the previously gained money in that period, it seems that our bot manages to detect the fall and sells everything, and thus hold on to its profits.

We are, however, aware that the bot wasn't tested on a long enough period, and this method is very much likely to fail in a more diverse setting. What we're showing is a success for this specific period (even though there weren't any tuning on that period, since it's the test set).


## Further Work

First, anyone who'd like is welcome to just run the `Model Training - CGEN.ipynb` to retrain the model, or run `stock_bot.py` to run the bot and/or edit it.

We most definitely have some improvements to our small project in mind, including but not limited to:
- Involving various stocks instead of a single one.
- Calculating many more features (using [FinTA](https://github.com/peerchemist/finta)) and have [Optuna](https://github.com/optuna/optuna) to choose which features to pick. 
- Making the bot more sophisticated, maybe not deterministic or actually train its hyperparameters on a validation set or another stock.
- Training for a longer period: we specifically cut a period in which the Buy&Hold strategy loses a bit because we wanted to compete it, but it doesn't have to be the case.


## References
* [A nice LSTM article](https://web.stanford.edu/class/cs379c/archive/2018/class_messages_listing/content/Artificial_Neural_Network_Technology_Tutorials/OlahLSTM-NEURAL-NETWORK-TUTORIAL-15.pdf) by Stanford.
* [FinTA](https://github.com/peerchemist/finta).
* [Optuna](https://github.com/optuna/optuna).


## DISCLAIMER

We'd like to further emphasize in addition to our comments above that our predictions and bot are nowhere near reliable and we'd highly advise not to try any of the provided here in a live or real setting.
