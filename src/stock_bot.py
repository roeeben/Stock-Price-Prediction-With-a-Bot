import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import sys
import pickle
import matplotlib.pyplot as plt


sys.path.append('../')


def buy_and_sell(actual, predicted, thresh, money=10000):
    number_of_stocks = (int)(money / actual[0])
    left = money - (int)(money / actual[0]) * actual[0] + actual[len(actual) - 1] * number_of_stocks

    number_of_stocks = 0

    buying_percentage_threshold = thresh
    selling_percentage_threshold = thresh

    decision_outcome = np.zeros(len(actual))
    buy_sell_points = np.zeros(len(actual))
    for i in range(len(actual) - 1):
        pct_change = 100 * ((predicted[i + 1]/actual[i]) - 1)
        buy_sell_points[i] = 0
        # print("actual {} pred {} pct {}".format(actual[i], predicted[i + 1], pct_change))
        num_of_stocks_prop = int(100 * abs(pct_change))
        # print("current balance is {} with {} stocks and stocks prop {} pct_change {} in iter {}".format(money,number_of_stocks, num_of_stocks_prop, pct_change, i))
        next_hold = money + number_of_stocks * actual[i + 1] # what amount should be in the next step in case we don't do anything
        if pct_change > buying_percentage_threshold:
            for j in range(num_of_stocks_prop, 0, -1):
                #Buying of stock
                if (money >= j * actual[i]):
                    money -= j * actual[i]
                    number_of_stocks += j
                    buy_sell_points[i] = 1
                    break
        elif pct_change < -selling_percentage_threshold:
            for j in range(num_of_stocks_prop, 0, -1):
                #Selling of stock
                if (number_of_stocks >= j):
                    money += j * actual[i]
                    number_of_stocks -= j
                    buy_sell_points[i] = -1
                    break
        next_act = money + number_of_stocks * actual[i + 1]
        decision_outcome[i + 1] = 100 *(next_act/next_hold) - 100 # perecent of amount of money lost or earned from action in compare to passive case
    money += number_of_stocks * actual[len(actual) - 1]
    #
    print(money) #Money if we traded
    print(left)  #Money if we just bought as much at the start and sold near the end (Buy and hold)

    return left ,money, pct_change, buy_sell_points, decision_outcome




with open('test_original.pkl', 'rb') as file:
    test_orig = pickle.load(file)
with open('test_predict.pkl', 'rb') as file:
    test_pred = pickle.load(file)

buy_and_hold, money, y_pct_change, buy_and_sell_dates, decision_outcome = buy_and_sell(test_orig.to_numpy().flatten(), test_pred.to_numpy().flatten(), 0.1)

# plot data prep
buy_value = np.zeros(len(test_orig.to_numpy()))
for idx,val in enumerate(test_orig.to_numpy()):
    if buy_and_sell_dates[idx]==1:
        buy_value[idx] = val
buy_idx = test_orig.index[buy_value>0]
buy_value = buy_value[buy_value>0]

sell_value = np.zeros(len(test_orig.to_numpy()))
for idx,val in enumerate(test_orig.to_numpy()):
    if buy_and_sell_dates[idx]==-1:
        sell_value[idx] = val

sell_idx = test_orig.index[sell_value>0]
sell_value = sell_value[sell_value>0]

## plots
plt.figure(1)
plt.plot(test_orig.index, test_orig.values, label="Original test data")  # plots the x and y
plt.plot(test_pred.index, test_pred.values, label="Predicted test data")
plt.scatter(buy_idx, buy_value*0.95, color='green', marker='o', label="Buying Points")
plt.scatter(sell_idx, sell_value*1.05,color='red', marker='o', label="Selling Points")

plt.grid(True)  # turns on axis grid
# plt.ylim(0)  # sets the y axis min to zero
# plt.xlim(0, 100)  # sets the y axis min to zero
plt.xticks(rotation=45, fontsize=10)  # rotates the x axis ticks 90 degress and font size 10
plt.title('Ford test data')  # prints the title on the top
plt.ylabel('Stock Price For Ford')  # labels y axis
plt.xlabel('Date')  # labels x axis
plt.legend()

plt.figure(2)
mask = decision_outcome != 0
col = np.where(decision_outcome<0,'r',np.where(decision_outcome > 0, 'g', "None"))
plt.axhline(y=0.0, color='k', linestyle='-')
plt.scatter(test_orig.index, decision_outcome, label="Desicion outcome, red is negative, green positive", color=col, marker='o')  # plots the x and y

plt.grid(True)  # turns on axis grid
plt.xticks(rotation=45, fontsize=10)  # rotates the x axis ticks 90 degress and font size 10
plt.title('Outcome of each buying and selling desicion')  # prints the title on the top
plt.ylabel('Percent of positive/negative impact')  # labels y axis
plt.xlabel('time')  # labels x axis
plt.legend()


plt.figure(3)
threshold = []
money = []
buy_and_hold = []
for i in range(0, 100):
    thresh = 0.001 * i
    b, m, y_pct_change, buy_and_sell_dates, d = buy_and_sell(test_orig.to_numpy().flatten(), test_pred.to_numpy().flatten(), thresh)
    threshold.insert(i, thresh)
    money.insert(i, m)
    buy_and_hold.insert(i, b)

plt.plot(threshold, money, label="Total amount of money for given threshhold")  # plots the x and y
plt.plot(threshold, buy_and_hold, label="Buy and hold amount ")  # plots the x and y

plt.grid(True)  # turns on axis grid
# plt.ylim(0)  # sets the y axis min to zero
# plt.xlim(0, 100)  # sets the y axis min to zero
plt.xticks(rotation=45, fontsize=10)  # rotates the x axis ticks 90 degress and font size 10
plt.title('Money earned after stock trading with bot for given threshold')  # prints the title on the top
plt.ylabel('Money')  # labels y axis
plt.xlabel('Threshold')  # labels x axis
plt.legend()

plt.show()