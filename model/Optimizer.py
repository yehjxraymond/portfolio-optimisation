import scipy.optimize as sco
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold


def returns(data, weights):
    return np.sum(data.mean() * weights) * 252


def variance(data, weights):
    return np.dot(weights, np.dot(data.cov() * 252, weights))


def sharpe(data, weights, rf=0.0):
    rets = returns(data, weights)
    var = variance(data, weights)
    return (rets - rf) / var


def optimize(returnsDf, rf=0.0):
    numberOfAssets = returnsDf.shape[1]
    constraints = {"type": "eq", "fun": lambda x: np.sum(abs(x)) - 1}
    bounds = tuple((0, 1) for x in range(numberOfAssets))
    initial = np.array(numberOfAssets * [1.0 / numberOfAssets])
    options = sco.minimize(
        lambda x: -sharpe(returnsDf, x),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return options["x"]


class Optimizer:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.portfolio.returnsDataframeExist()

    def optimizeSharpe(self, interval=None):
        if interval == None:
            interval = self.portfolio.commonInterval()
        startDate, endDate = interval
        rf = self.portfolio.rf
        data = self.portfolio.assetReturnsDf.loc[startDate:endDate]
        return optimize(data, rf)

    def kfoldTs(self, folds=5):
        rf = self.portfolio.rf
        startDate, endDate = self.portfolio.commonInterval()
        returnsSubsetCommon = self.portfolio.assetReturnsDf.loc[startDate: endDate]
        tscv = TimeSeriesSplit(n_splits=folds)
        weightsList = []
        performanceList = []

        for trainIndex, testIndex in tscv.split(returnsSubsetCommon):
            trainSubset = returnsSubsetCommon.iloc[trainIndex]
            testSubset = returnsSubsetCommon.iloc[testIndex]
            weights = optimize(trainSubset, rf)
            weightsList.append(weights)

            performance = sharpe(testSubset, weights, rf)
            performanceList.append(performance)

        weightsDf = pd.DataFrame(weightsList)
        weightsAvg = list(np.mean(weightsDf))

        return (
            weightsAvg,
            {
                "sharpeRaw": performanceList,
                "sharpeAvg": np.mean(performanceList),
                "sharpeStd": np.std(performanceList),
                "weightsRaw": weightsList,
                "weightsStd": np.std(weightsDf),
            },
        )

    def kfold(self, folds=5):
        rf = self.portfolio.rf
        startDate, endDate = self.portfolio.commonInterval()
        returnsSubsetCommon = self.portfolio.assetReturnsDf.loc[startDate:endDate]
        kf = KFold(folds)
        weightsList = []
        performanceList = []

        for trainIndex, testIndex in kf.split(returnsSubsetCommon):
            trainSubset = returnsSubsetCommon.iloc[trainIndex]
            testSubset = returnsSubsetCommon.iloc[testIndex]
            weights = optimize(trainSubset, rf)
            weightsList.append(weights)

            performance = sharpe(testSubset, weights, rf)
            performanceList.append(performance)

        weightsDf = pd.DataFrame(weightsList)
        weightsAvg = list(np.mean(weightsDf))

        return (
            weightsAvg,
            {
                "sharpeRaw": performanceList,
                "sharpeAvg": np.mean(performanceList),
                "sharpeStd": np.std(performanceList),
                "weightsRaw": weightsList,
                "weightsStd": np.std(weightsDf),
            },
        )
