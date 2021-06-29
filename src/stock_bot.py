import pandas as pd
import numpy as np
import warnings
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib
print(matplotlib.matplotlib_fname())
warnings.filterwarnings('ignore')


sys.path.append('../')



def buy_and_sell(orig, pred, thresh_buy,thresh_sell=None, money=10000):
    """Stock bot routine that buy and sell stock holds through time

    Parameters:
    orig : original stock data array
    pred : predicted stock data array
    thresh_buy: percent threshold for buying stock
    thresh_sell: percent threshold for selling stock
    money: money the bot starts with for trading

    Returns:
    left: Buy and Hold amount for stock during the given period
    money_accum: total money the bot holds (with stock) for time
    pct_change: last pct change of the stock
    buy_sell_points: buying and selling points. -1 are selling points 1 are buying points and 0 are passive
    decision_outcome: percent of profit or loss against passive approach for the given descision
   """
    actual = orig
    predicted = pred
    number_of_stocks = (int)(money / actual[0])
    left = money - (int)(money / actual[0]) * actual[0] + actual[len(actual) - 1] * number_of_stocks # buy and hold amount can be calculated now

    number_of_stocks = 0 # we start with 0 stocks

    # its optional that threshold of buy and sell are automatically the same
    buying_percentage_threshold = thresh_buy
    if thresh_sell is None:
        selling_percentage_threshold = thresh_buy
    else:
        selling_percentage_threshold = thresh_sell

    money_accum = [money]
    decision_outcome = np.zeros(len(actual))
    buy_sell_points = np.zeros(len(actual))
    for i in range(len(actual) - 1): # we loop through all data set (all minutes)
        pct_change = 100 * ((predicted[i + 1]/actual[i]) - 1) #we predict the change of the stock value against current value
        buy_sell_points[i] = 0 # for start this point is neither buy or selling point
        num_of_stocks_prop = int(50 * abs(pct_change)) # this value is the initial number of stocks we intend to buy or sell and proportional to pct_change
        next_hold = money + number_of_stocks * actual[i + 1] # what amount should be in the next step in case we don't do anything

        if pct_change > buying_percentage_threshold:
            # we passed buying thresh
            for j in range(num_of_stocks_prop, 0, -1): # trying to buy from num_of_stocks_prop to 0
                if (money >= j * actual[i]):
                    money -= j * actual[i]
                    number_of_stocks += j
                    buy_sell_points[i] = 1 # buying point marked as 1 in this list
                    break
        elif pct_change < -selling_percentage_threshold:
            # we passed selling thresh
            for j in range(num_of_stocks_prop, 0, -1):
                if (number_of_stocks >= j):
                    money += j * actual[i]
                    number_of_stocks -= j
                    buy_sell_points[i] = -1 #selling point marked as -1 in this list
                    break
        next_act = money + number_of_stocks * actual[i + 1] # used to calculate the actual outcome of the bot action against passive approach
        money_accum.append(money + number_of_stocks * actual[i]) # appending money amount for this time index
        decision_outcome[i + 1] = 100 *(next_act/next_hold) - 100 # perecent of amount of money lost or earned from action in compare to passive case
    money += number_of_stocks * actual[len(actual) - 1]
    money_accum.append(money)

    return left, money_accum, pct_change, buy_sell_points, decision_outcome


def buying_and_selling_plt(orig, pred, threshold_buy, threshold_sell, money_start):
    """ Plots Prediction over original stock data over time with bot buying and selling points

    Parameters:
    orig : original stock data array
    pred : predicted stock data array
    thresh_buy: percent threshold for buying stock
    thresh_sell: percent threshold for selling stock
    money: money the bot starts with for trading

    Returns:
   """

    test_orig = orig['Price']
    test_pred = pred['Price']

    # using the bot
    buy_and_hold, money, y_pct_change, buy_and_sell_dates, decision_outcome = buy_and_sell(test_orig.to_numpy().flatten(), test_pred.to_numpy().flatten(), threshold_buy, threshold_sell, money_start)
    # plot data prep
    buy_value = np.zeros(len(test_orig.to_numpy()))
    for idx, val in enumerate(test_orig.to_numpy()):
        if buy_and_sell_dates[idx] == 1:
            buy_value[idx] = val
    buy_idx = test_orig.index[buy_value > 0] # buying scatter for where this vec( buy and sell dates) is 1
    buy_value = buy_value[buy_value > 0]

    sell_value = np.zeros(len(test_orig.to_numpy()))
    for idx, val in enumerate(test_orig.to_numpy()):
        if buy_and_sell_dates[idx] == -1:
            sell_value[idx] = val

    sell_idx = test_orig.index[sell_value > 0] # selling scatter for where this vec( buy and sell dates) is -1
    sell_value = sell_value[sell_value > 0]

    # buying and selling points
    ax = orig.plot(x='Date', y='Price', label="Original test data")
    pred.plot(x='Date', y='Price', label="Predicted test data", ax= ax)
    plt.scatter(buy_idx, buy_value * 0.95, color='green', marker='o', label="Buying Points")
    plt.scatter(sell_idx, sell_value * 1.05, color='red', marker='o', label="Selling Points")

    ax.text(0.3, 0.95, "Net Worth after trading:" + str("%.2f" % money[len(money) - 1]) + "$", size=20, transform=ax.transAxes, bbox=dict(facecolor='blue', alpha=0.5))
    ax.text(0.3, 0.90,"Buy and Hold amount:" + str("%.2f" % buy_and_hold) + "$", transform=ax.transAxes, size= 20, bbox=dict(facecolor='red', alpha=0.5))
    plt.grid(True)  # turns on axis grid
    # plt.ylim(0)  # sets the y axis min to zero
    # plt.xlim(0, 100)  # sets the y axis min to zero
    plt.xticks(rotation=45, fontsize=16)  # rotates the x axis ticks 90 degress and font size 10
    plt.title('CGEN test data', fontsize=18)  # prints the title on the top
    plt.ylabel('Stock Price For CGEN',fontsize=18)  # labels y axis
    plt.xlabel('Date')  # labels x axis
    plt.legend(fontsize=18)
    plt.show()

## includes animation
def money_over_time_plt(orig, pred, threshold_buy, threshold_sell, money_start):
    """ Plots animation of the bot money through time with stock value in another subplot

    Parameters:
    orig : original stock data array
    pred : predicted stock data array
    thresh_buy: percent threshold for buying stock
    thresh_sell: percent threshold for selling stock
    money: money the bot starts with for trading

    Returns:
   """

    test_orig = orig['Price']
    test_pred = pred['Price']
    buy_and_hold, money, y_pct_change, buy_and_sell_dates, decision_outcome = buy_and_sell(test_orig.to_numpy().flatten(), test_pred.to_numpy().flatten(), threshold_buy, threshold_sell, money_start)
    # money over time
    bot_df = pd.DataFrame()
    bot_df['Stock'] = list(test_orig.values)
    bot_df['Money'] = money[1:]
    bot_df['Date'] = orig['Date']

    fig, (ax1, ax2) = plt.subplots(2, 1)

    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for blank space between subplots
    hspace = 0.5  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


    ax1.set_title('Money over time',  fontsize=18)
    ax1.set_xlabel('Minutes', fontsize=18)
    ax1.set_ylabel('Net worth in $', fontsize=18)
    ax1_text = ax1.text(0.3, 0.5, '', transform=ax1.transAxes, size=20, bbox=dict(facecolor='red', alpha=0.3))

    ax2.set_title('Stock over time', fontsize=18)
    ax2.set_xlabel('Minutes', fontsize=18)
    ax2.set_ylabel('Stock Value in $', fontsize=18)
    ax2_text = ax2.text(0.3, -0.8, '', transform=ax1.transAxes, size=20, bbox=dict(facecolor='red', alpha=0.3))

    # updating each frame of money
    def update_money(i=int):
        ax1.legend(["net worth"],fontsize=18)
        plt.sca(ax1)
        ax1_text.set_text("Current Net worth:" + "%.2f" % bot_df['Money'][i])
        # ax1.scatter(buy_idx[:i], buy_value[:i] , color='green', marker='o', label="Buying Points")
        # ax1.scatter(sell_idx[:i], sell_value[:i] , color='red', marker='o', label="Selling Points")
        plt.xticks(rotation=45, fontsize=18)  # rotates the x axis ticks 90 degress and font size 18
        p = ax1.plot(bot_df['Date'].index[:i], bot_df['Money'][:i].values)  # note it only returns the dataset, up to the point i
        p[0].set_color('r')  # set the colour of each curve

    # updating each frame of stock value
    def update_stock(i=int):
        ax2.legend(["Stock Value"],fontsize=18)
        plt.sca(ax2)
        ax2_text.set_text("Current Stock Value:" + "%.2f" % bot_df['Stock'][i])
        plt.xticks(rotation=45, fontsize=18)  # rotates the x axis ticks 90 degress and font size 18
        p = ax2.plot(bot_df['Stock'][:i].index,  bot_df['Stock'][:i].values)  # note it only returns the dataset, up to the point i

        p[0].set_color('b')  # set the colour of each curve

    def update_all(i=int):
        if i == last_frame:
            i = len(bot_df) - 1
        update_money(i)
        update_stock(i)

    # for faster animation we show every 20 frame
    show_every_x_frams = 40
    frames = np.arange(0, len(bot_df)+ 1, show_every_x_frams)
    last_frame = frames[len(frames) - 1 ]
    animator = ani.FuncAnimation(fig, update_all, frames=frames , interval=40, repeat=False, blit=False)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    # plt.rcParams["animation.convert_path"] = "C:\ProgramFiles\IImageMagick-7.1.0-Q16-HDRI\convert"
    #
    # animator.save(filename="gif.gif" , writer='imagemagick')

    plt.show()


def outcome_of_transactions_plt(orig, pred, threshold_buy, threshold_sell, money_start):
    """ Plots outcome graph with green points for good decisions and red for bad ones (percent of earning or losing)

    Parameters:
    orig : original stock data array
    pred : predicted stock data array
    thresh_buy: percent threshold for buying stock
    thresh_sell: percent threshold for selling stock
    money: money the bot starts with for trading

    Returns:
   """
    test_orig = orig['Price']
    test_pred = pred['Price']
    new_df = pd.DataFrame()
    new_df['Date'] = orig['Date']

    buy_and_hold, money, y_pct_change, buy_and_sell_dates, decision_outcome = buy_and_sell(test_orig.to_numpy().flatten(), test_pred.to_numpy().flatten(), threshold_buy, threshold_sell, money_start)
    # outcome of buying and selling points

    count_good = sum(map(lambda x: x > 0, decision_outcome))
    count_bad = sum(map(lambda x: x < 0, decision_outcome))

    # deals_sum = sum(abs(decision_outcome))
    # good_sum = sum(decision_outcome[decision_outcome > 0])
    # bad_sum = sum(abs(decision_outcome[decision_outcome < 0]))

    good_deals_pct = 100 * count_good/(count_bad + count_good) # calculating good deal percent
    # good_deals_pct_weight = 100 * good_sum / deals_sum
    col = np.where(decision_outcome < 0, 'r', np.where(decision_outcome > 0, 'g', "None"))
    ax = plt.gca()
    ax.text(0.3, 0.5, "%.2f" % good_deals_pct + "% good deals" , transform=ax.transAxes, size=20, bbox=dict(facecolor='red', alpha=0.3))
    # ax.text(0.3, 0.6, "%.2f" % good_deals_pct_weight + "% good deals weighted", transform=ax.transAxes, size=20,bbox=dict(facecolor='red', alpha=0.3))
    plt.axhline(y=0.0, color='k', linestyle='-')
    plt.scatter(orig['Date'].index, decision_outcome, label="Desicion outcome, red is negative, green positive", color=col,
                marker='.')  # plots the x and y

    plt.grid(True)  # turns on axis grid
    plt.xticks(rotation=45, fontsize=10)  # rotates the x axis ticks 90 degress and font size 10
    plt.title('Outcome of each buying and selling Desicion', fontsize=15 )  # prints the title on the top
    plt.ylabel('Percent of positive/negative impact', fontsize=15)  # labels y axis
    plt.xlabel('Minutes',fontsize=15)  # labels x axis
    plt.legend(fontsize=10)
    plt.show()


def money_for_threshold(orig, pred, money_start):
    """ Plots Final holdings in function of the threshold (where threshold for buying and selling are equal)

    Parameters:
    orig : original stock data array
    pred : predicted stock data array
    thresh_buy: percent threshold for buying stock
    thresh_sell: percent threshold for selling stock
    money: money the bot starts with for trading

    Returns:
   """

    test_orig = orig['Price']
    test_pred = pred['Price']
    # money earned against threshold
    plt.figure()
    threshold = []
    money = []
    buy_and_hold = []
    for i in range(0, 1000):
        thresh = 0.001 * i
        b, m, y_pct_change, buy_and_sell_dates, d = buy_and_sell(test_orig.to_numpy().flatten(), test_pred.to_numpy().flatten(), thresh, thresh, money_start)
        threshold.insert(i, thresh)
        money.insert(i, m[len(m) - 1])
        buy_and_hold.insert(i, b)

    plt.plot(threshold, money, label="Finaly Net worth")  # plots the x and y
    plt.plot(threshold, buy_and_hold, label="Buy and hold amount")  # plots the x and y

    plt.grid(True)  # turns on axis grid
    # plt.ylim(0)  # sets the y axis min to zero
    # plt.xlim(0, 100)  # sets the y axis min to zero
    plt.xticks(rotation=45, fontsize=10)  # rotates the x axis ticks 90 degress and font size 10
    plt.title('Money VS Buying/Selling Threshold')  # prints the title on the top
    plt.ylabel('Money')  # labels y axis
    plt.xlabel('Threshold')  # labels x axis
    plt.legend()
    plt.show()


with open('../data/CGEN_original.pkl', 'rb') as file:
    CGEN_original = pickle.load(file)
with open('../data/CGEN_predict.pkl', 'rb') as file:
    CGEN_predict = pickle.load(file)

CGEN_original = CGEN_original[100:].reset_index()
CGEN_predict = CGEN_predict[100:].reset_index()
money_start = 5000
threshold_buy = 0.0015
threshold_sell = 0.0015
buying_and_selling_plt(CGEN_original, CGEN_predict, threshold_buy, threshold_sell, money_start)
outcome_of_transactions_plt(CGEN_original, CGEN_predict, threshold_buy, threshold_sell, money_start)
money_over_time_plt(CGEN_original, CGEN_predict, threshold_buy, threshold_sell, money_start)
money_for_threshold(CGEN_original, CGEN_predict, money_start)