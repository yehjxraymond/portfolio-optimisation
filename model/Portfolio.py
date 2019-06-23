import numpy as np
import pandas as pd
import math
import datetime
from functools import reduce


def removeNonTradingDays(df):
    # FIXME This does not add trading days into the data set
    return df[df.index.dayofweek < 5]


def forwardFillPrices(df):
    return df.fillna(method="ffill")


def preprocessData(df):
    funcToApply = [removeNonTradingDays, forwardFillPrices]
    return reduce(lambda o, func: func(o), funcToApply, df)


def testRemoveNonTradingDays():
    data = pd.read_csv("./test/USDSGD.csv", index_col="Date", parse_dates=True)
    assert data.index.contains("2009-11-15")
    assert not removeNonTradingDays(data).index.contains("2009-11-15")


def testForwardFillPrices():
    data = pd.read_csv("./test/USDSGD.csv", index_col="Date", parse_dates=True)
    assert math.isnan(data.loc["2009-11-12"]["Open"])
    assert not math.isnan(forwardFillPrices(data).loc["2009-11-12"]["Open"])

class Portfolio:
    def __init__(self):
        self.assetNames = []
        self.assetDatas = []
        self.assetReturns = []
        self.assetReturnsDf = None

        self.exchange = {}

    def portfolioReturns(self, weights, interval=None):
        self.returnsDataframeExist()
        if interval == None:
            interval = self.commonInterval()
        fromDate, toDate = interval
        return np.sum(self.assetReturnsDf[fromDate:toDate].mean() * weights) * 252

    def portfolioVariance(self, weights, interval=None):
        self.returnsDataframeExist()
        if interval == None:
            interval = self.commonInterval()
        fromDate, toDate = interval
        return np.dot(
            weights, np.dot(self.assetReturnsDf[fromDate:toDate].cov() * 252, weights)
        )

    def portfolioPerformance(self, weights, rf=0, interval=None):
        rets = self.portfolioReturns(weights, interval)
        var = self.portfolioVariance(weights, interval)
        return {"returns": rets, "variance": var, "sharpe": (rets - rf) / var}

    def returnsDataframeExist(self):
        if not isinstance(self.assetReturnsDf, pd.DataFrame):
            self.generateReturnsDataframe()

    def generateReturnsDataframe(self):
        merged = reduce(
            lambda left, right: pd.merge(left, right, on=["Date"], how="outer"),
            self.assetReturns,
        )
        merged.columns = self.assetNames
        self.assetReturnsDf = merged

    def addAsset(self, file, name):
        data = pd.read_csv(
            file, index_col="Date", parse_dates=True, usecols=["Date", "Adj Close"]
        )
        preprocessedData = preprocessData(data)
        returns = np.log(preprocessedData / preprocessedData.shift(1))

        self.assetNames.append(name)
        self.assetDatas.append(preprocessedData)
        self.assetReturns.append(returns)

    def addExchangeRate(self, file, name, base=False):
        ex = pd.read_csv(
            file, index_col="Date", parse_dates=True, usecols=["Date", "Close"]
        )
        preprocessedData = preprocessData(ex)
        if base:
            preprocessedData["Close"] = 1 / preprocessedData["Close"]
        self.exchange[name] = preprocessedData

    def exchangeAdjustment(self, asset, currency):
        data = self.assetDatas[asset]
        # FIXME Need to account for exchange range being subset of data
        ex = self.exchange[currency].reindex(data.index, method="ffill")

        adjustedData = data.mul(ex["Close"], axis=0)
        returns = np.log(adjustedData / adjustedData.shift(1))

        self.assetDatas[asset] = adjustedData
        self.assetReturns[asset] = returns

    def commonInterval(self):
        latestStart = None
        earliestEnd = None
        for data in self.assetDatas:
            if latestStart == None or data.index[0] > latestStart:
                latestStart = data.index[0]
            if earliestEnd == None or data.index[-1] < earliestEnd:
                earliestEnd = data.index[-1]
        return (latestStart, earliestEnd)

    # def estimatePerformance(self):


def testAddAsset():
    p = Portfolio()
    p.addAsset("./test/Asset.csv", "Asset1")

    assert p.assetNames == ["Asset1"]
    assert p.assetDatas[0].iloc[0]["Adj Close"] == 15.996025
    assert p.assetReturns[0].iloc[1]["Adj Close"] == math.log(
        p.assetDatas[0].iloc[1]["Adj Close"] / p.assetDatas[0].iloc[0]["Adj Close"]
    )


def testAddExchangeQuoted():
    # base = False if portfolio currency is the quoted pair, ie SGD in USD/SGD
    p = Portfolio()
    p.addExchangeRate("./test/USDSGD.csv", "USD")

    assert (
        p.exchange["USD"].iloc[0]["Close"] == 1.2867
    ), "Closing price should not be inverse"
    assert not p.exchange["USD"].iloc[1]["Close"] == None, "NA values should be filled"


def testAddExchangeBase():
    # base = True if portfolio currency is the base pair, ie SGD in SGD/EUR
    p = Portfolio()
    p.addExchangeRate("./test/SGDEUR.csv", "EUR", True)

    assert (
        p.exchange["EUR"].iloc[0]["Close"] == 2.081078831266128
    ), "Closing price should be inverse"
    assert not p.exchange["EUR"].iloc[1]["Close"] == None, "NA values should be filled"


def testExchangeAdjustment():
    p = Portfolio()
    p.addAsset("./test/Asset.csv", "Asset1")
    p.addExchangeRate("./test/USDSGD.csv", "USD")
    p.exchangeAdjustment(0, "USD")

    assert p.assetDatas[0].iloc[0]["Adj Close"] == 20.864415208749996
    assert p.assetReturns[0].iloc[-1]["Adj Close"] == 2.2565439772970107e-05
    assert not p.assetDatas[0].isnull().values.any(), "NaN is introduced in data"
    assert (
        not p.assetReturns[0].iloc[1:-1].isnull().values.any()
    ), "NaN is introduced in returns"


def testCommonInterval():
    p = Portfolio()
    p.addAsset("./test/Asset.csv", "Asset1")
    p.addAsset("./test/Asset2.csv", "Asset2")
    fromDate, toDate = p.commonInterval()

    assert fromDate == datetime.datetime.fromisoformat("2009-11-25")
    assert toDate == datetime.datetime.fromisoformat("2009-12-09")


def testGenerateReturnsDataframe():
    p = Portfolio()
    p.addAsset("./test/Asset.csv", "Asset1")
    p.addAsset("./test/Asset2.csv", "Asset2")
    p.addAsset("./test/Asset2.csv", "Asset3")
    p.generateReturnsDataframe()

    assert all(
        [
            a == b
            for a, b in zip(p.assetReturnsDf.columns, ["Asset1", "Asset2", "Asset3"])
        ]
    )

def testPortfolioReturns():
    p = Portfolio()
    p.addAsset("./test/Asset.csv", "Asset1")
    p.addAsset("./test/Asset2.csv", "Asset2")
    assert p.portfolioReturns([0.5, 0.5]) == -0.2504711195035464


def testPortfolioVariance():
    p = Portfolio()
    p.addAsset("./test/Asset.csv", "Asset1")
    p.addAsset("./test/Asset2.csv", "Asset2")
    assert p.portfolioVariance([0.5, 0.5]) == 0.0037867597178967947


def testPortfolioPerformance():
    p = Portfolio()
    p.addAsset("./test/Asset.csv", "Asset1")
    p.addAsset("./test/Asset2.csv", "Asset2")
    perf = p.portfolioPerformance([0.5, 0.5])
    assert perf == {
        "returns": -0.2504711195035464,
        "variance": 0.0037867597178967947,
        "sharpe": -66.14391674226972,
    }
