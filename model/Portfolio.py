import numpy as np
import pandas as pd
import math
import datetime
from functools import reduce
import backtrader as bt
import backtrader.analyzers as btanalyzers

# TODO Portfolio plotter
# TODO Move backtester to it's own module
# TODO Value at risk calculations


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


class RebalanceStrategy(bt.Strategy):
    lastRebalanced = None
    addedOrders = []
    params = (
        ("rebalance", True),
        ("rebalancePeriod", 30),
        ("weights", []),
        ("assetNames", []),
    )

    def rebalance(self):
        self.lastRebalanced = self.datetime.date()
        for i in range(len(self.params.weights)):
            order = self.order_target_percent(
                data=self.params.assetNames[i], target=self.params.weights[i]
            )

            if order:
                self.addedOrders.append(
                    {
                        "date": self.datetime.date(),
                        "asset": self.params.assetNames[i],
                        "size": order.size,
                        "price": order.price,
                        "type": order.ordtype,
                    }
                )

    def next(self):
        if self.lastRebalanced == None:
            self.rebalance()
        elif (
            self.datetime.date() - self.lastRebalanced
        ).days > self.params.rebalancePeriod and self.params.rebalance:
            self.rebalance()
        # Access -1, because drawdown[0] will be calculated after "next"
        # print("DrawDown: %.2f" % self.stats.drawdown.drawdown[-1])
        # print("MaxDrawDown: %.2f" % self.stats.drawdown.maxdrawdown[-1])


class Portfolio:
    def __init__(self):
        self.assetNames = []
        self.assetDatas = []
        self.assetReturns = []
        self.assetReturnsDf = None
        self.exchange = {}

        self.rf = 0

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

    def portfolioPerformance(self, weights, interval=None):
        rets = self.portfolioReturns(weights, interval)
        var = self.portfolioVariance(weights, interval)
        return {"returns": rets, "variance": var, "sharpe": (rets - self.rf) / var}

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

    def backtest(
        self,
        weights=None,
        interval=None,
        rebalance=True,
        startValue=100000.0,
        rebalancePeriod=30,
    ):
        if interval == None:
            interval = self.commonInterval()

        fromdate, todate = interval
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(startValue)
        cerebro.addstrategy(
            RebalanceStrategy,
            rebalance=True,
            weights=weights,
            assetNames=self.assetNames,
            rebalancePeriod=rebalancePeriod,
        )
        cerebro.addobserver(bt.observers.DrawDown)
        cerebro.addanalyzer(
            btanalyzers.SharpeRatio, _name="sharpe", riskfreerate=self.rf
        )
        cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(btanalyzers.PeriodStats, _name="periodstats")
        for i in range(len(self.assetNames)):
            df = self.assetDatas[i].copy()
            df.columns = ["close"]
            df["open"] = self.assetDatas[i].shift(-1)
            df["high"] = self.assetDatas[i]
            df["low"] = self.assetDatas[i]
            data = bt.feeds.PandasData(dataname=df, fromdate=fromdate, todate=todate)
            cerebro.adddata(data, name=self.assetNames[i])
        results = cerebro.run()

        valueEnd = cerebro.broker.getvalue()
        periodStats = results[0].analyzers.periodstats.get_analysis()
        drawdownStats = results[0].analyzers.drawdown.get_analysis()
        sharpe = results[0].analyzers.sharpe.get_analysis()["sharperatio"]

        drawdown = drawdownStats["drawdown"]
        drawdownPeriod = drawdownStats["len"]
        moneydown = drawdownStats["moneydown"]
        maxDrawdown = drawdownStats["max"]["drawdown"]
        maxDrawdownPeriod = drawdownStats["max"]["len"]
        maxMoneydown = drawdownStats["max"]["moneydown"]

        averageReturns = periodStats["average"]
        standardDeviation = periodStats["stddev"]
        positiveYears = periodStats["positive"]
        negativeYears = periodStats["negative"]
        noChangeYears = periodStats["nochange"]
        bestYearReturns = periodStats["best"]
        worstYearReturns = periodStats["worst"]

        def plot():
            return cerebro.plot(volume=False)

        return (
            {
                "dateStart": interval[0],
                "dateEnd": interval[1],
                "days": (interval[1] - interval[0]).days,
                "valueStart": startValue,
                "valueEnd": valueEnd,
                "sharpe": sharpe,
                "drawdown": drawdown,
                "drawdownPeriod": drawdownPeriod,
                "moneydown": moneydown,
                "maxDrawdown": maxDrawdown,
                "maxDrawdownPeriod": maxDrawdownPeriod,
                "maxMoneydown": maxMoneydown,
                "averageReturns": averageReturns,
                "standardDeviation": standardDeviation,
                "positiveYears": positiveYears,
                "negativeYears": negativeYears,
                "noChangeYears": noChangeYears,
                "bestYearReturns": bestYearReturns,
                "worstYearReturns": worstYearReturns,
            },
            plot,
        )

