""" This is the python file version on the manipulation and machine learning of the stock data. The Jupyter Notebook version is an easier-to-read version and is included in this repo."""

import bs4 as bs
import datetime as dt
# os to check for and create directories
import os
# use datetime to specify dates for Pandas datareader
import pandas_datareader.data as web
import pandas as pd
import pickle
import requests
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
# see distributions of classes both in dataset and in algorithm's predictions.
from collections import Counter
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

style.use('ggplot')

def save_nasdaq_tickers():

    # vist Wiki page, get response which has the source code
    # turn the .text attribute to soup using BeautifulSoup
    # (BS turns source code to a BS object that you can treat more like Python object)
    resp = requests.get('https://en.wikipedia.org/wiki/NASDAQ-100')
    soup = bs.BeautifulSoup(resp.text, 'lxml')

    # the specific solution to searching through the wiki table on that specific page
    # had to look into the sourcecode
    # we find in html the table has class="wikitable sortable"
    table = soup.find('table', {'class': 'wikitable sortable'})

    # for each row after header row, ticker is table data td, grab .text of it and append to tickers list
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text[:-1]
        tickers.append(ticker)

    # save the tickers list with pickle
    with open("nasdaqtickers.pickle","wb") as f:
        pickle.dump(tickers,f)

    print(tickers)

    return tickers

# handle for whether or not to reload the nasdaq list
# if we ask it to, program will re-pull nasdaq list
# else, use our pickle
def get_nasdaq_data_from_yahoo(reload_nasdaq=False):

    if reload_nasdaq:
        tickers = save_nasdaq_tickers()
    else:
        with open("nasdaqtickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    # we want to pull all the data from Yahoo for every sock and save it
    # first, create new directory with stock data per company
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    # pull the data
    start = dt.datetime(2017, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

# join stock datasets together
def compile_data():

    # pull previously made list of tickers, begin with an empty dataframe called main_df
    with open("nasdaqtickers.pickle","rb") as f:
        tickers = pickle.load(f)
    main_df = pd.DataFrame()

    # read in each stock's dataframe
    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        # we are only interested in the Adj Close data
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        # if there is nothing in main_df,  then start with current df, otherwise, use the Pandas join function
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        # output the count of the current ticker if it's evenly divisible by 10
        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('nasdaq_joined_closes.csv')

def visualize_data():
    df = pd.read_csv('nasdaq_joined_closes.csv')

    # build a correlation table from the nasdaq_joined_closes data, save into csv
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('nasdaq_corr.csv')

    # make a heatmap:

    # first, need actual data itself to graph
    # get numpy array of just the values (which are correlation numbers)
    data1 = df_corr.values

    # build the figure and axis
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    # create the heatmap with pcolor
    # use the RdYlGn colormap - gives red for negative correlations, green for positive correlations,
    # yellow for no-correlations
    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    # set x and y axis so we know which companies are which
    # create tick markers
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)

    # flip yaxis so that graph is easier to read (as there is more space between x's and y's
    # flip xaxis to be at top of graph (to make it more like a correlation table)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()

    # add company names to currently nameless ticks:
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)

    # rotate xtics, which are the actual tickers themselves, as they would normally be written out horizontally
    # tell color map that range is from -1 to +1
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    # plt.savefig("correlations.png", dpi = (300))
    plt.show()

# process data to help create the labels
# function takes one ticker as parameter
def process_data_for_labels(ticker):
    # each model is trained on a single company
    # next we need to know number of days into the future we need prices for (7 days)
    # read in data for close prices for all companies saved in past
    # get list of existing tickers
    # fill any missing with 0 for now
    # now we want to grab the % changed values for next 7 days
    hm_days = 7
    df = pd.read_csv('nasdaq_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    # create dataframe columns for specific ticker using string formatting to create custom names
    # get future values with .shift which shifts a column up or down
    # shift a negative amount which will take that column and would shift column up by i rows
    # gives future values i days in advanced which can calculate percentage change against
    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    # return tickers and dataframe
    # now we can have some feature sets that the algorithms can use to find relationships
    df.fillna(0, inplace=True)
    return tickers, df

# create label
# if the price rises more than 2% in the next 7 days - this is a buy
# if it drops more than 2% in next 7 days - this is a sell
# if doesn't do either or these - this is a hold
# this will be mapped to a Pandas DataFrame column
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):

    # take any ticker, create the dataset, create the target column (the label)
    # target column will have wither -1, 0 or  for each row
    # get the distribution
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))

    # clean up data
    # some data may be missing - replace with 0
    # some data may be infinite - convert to NaN, and drop the NaN
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # features metric should be the company's percent change that day instead of the day's price stocks
    # (companies will change in price before others)
    # convert stock prices to % changes
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # create features and labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    # X contains featuresets (daily & changes for every Nasdaq company)
    # y is the target (the label) - to map the featuresets to
    return X,y,df

def machine_learning_KNNC(ticker):
    X, y, df = extract_featuresets(ticker)

    # shuffle data, create training and testing samples
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

    # apply K Nearest Neighbors classifer
    clf = neighbors.KNeighborsClassifier()

    # # apply voting classifier
    # clf = VotingClassifier([('lsvc', svn.LinearSVC()),
    #                         ('knn', neighbors.KNeighborsClassifier()),
    #                         ('rfor', RandomForestClassifier())])

    # train classifier on data
    # take X data and fit to the y data, for each pairs of X's an y's
    clf.fit(X_train, y_train)

    # test
    # take some featuresets, X_test, make a prediction, see if it matches labels, y_test
    # return percentage accuracy in decimal form
    confidence = clf.score(X_test, y_test)

    # print accuracy
    # get predictions of X_testdata, output distribution using counter
    print('~ results with K nearest neighbours classifier for', ticker, '~')
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))
    print()
    print()

def machine_learning_VC(ticker):
    X, y, df = extract_featuresets(ticker)

    # shuffle data, create training and testing samples
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

    # apply voting classifier
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    # train classifier on data
    # take X data and fit to the y data, for each pairs of X's an y's
    clf.fit(X_train, y_train)

    # test
    # take some featuresets, X_test, make a prediction, see if it matches labels, y_test
    # return percentage accuracy in decimal form
    confidence = clf.score(X_test, y_test)

    # print accuracy
    # get predictions of X_testdata, output distribution using counter
    print('~ results with voting classifier for', ticker, '~')
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))
    print()
    print()

machine_learning_KNNC('GOOGL')

machine_learning_VC('GOOGL')