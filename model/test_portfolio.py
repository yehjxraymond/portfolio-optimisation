import math
import datetime
from Portfolio import Portfolio

# TODO Test if backtest if creating orders
def testBacktest():
    p = Portfolio()
    p.addAsset("../data/A35.csv", "Asset1")
    p.addAsset("../data/BAB.csv", "Asset2")

    results, plot = p.backtest([0.5, 0.5])
    
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
