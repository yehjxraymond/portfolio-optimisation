from Portfolio import Portfolio

import logging
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
import scipy.optimize as sco

# logging.basicConfig(level=logging.INFO)


def randomWeight(length):
    w = np.random.random(length)
    w /= np.sum(w)
    return w


class OptimisationResults:
    def __init__(self, names, **kwargs):
        self.rf = kwargs.get("rf", 0)
        self.results = pd.DataFrame(columns=["Sharpe", "Returns", "Std"] + names)

    def addData(self, data):
        logging.info(data)
        self.results = self.results.append(data, ignore_index=True)

    def plotEfficientFrontier(self):
        # Show all points
        plt.scatter(
            self.results["Std"],
            self.results["Returns"],
            c=self.results.index,
            marker="o",
        )
        plt.grid(True)
        plt.xlabel("Expected Standard Deviation")
        plt.ylabel("Expected Returns")
        plt.colorbar(label="Index")

        # Show risk-free rate
        plt.axhline(self.rf, label="rf", color="r", linestyle="--")

        # Show Tangent portfolio to point of highest sharpe
        highestSharpe = self.optimisedWeights()
        x = np.linspace(0, self.results["Std"].max())
        y = highestSharpe["Sharpe"] * x + self.rf
        plt.plot(
            x,
            y,
            label=f'y = {highestSharpe["Sharpe"]:.4f}x + {self.rf}',
            color="r",
            linestyle="--",
        )

        plt.legend()
        plt.show()

    def plotConvergence(self):
        # Show maximum sharpe at each iteration
        plt.subplot(221)
        plt.plot(self.results.cummax()["Sharpe"])
        plt.title("Max Sharpe")

        # Show difference between sharpe
        plt.subplot(222)
        plt.plot(self.results.diff().abs()["Sharpe"])
        plt.title("Difference Between Sharpe")

        # Select subset of top 10% of portfolio
        numOfAssets = self.results.shape[1] - 3
        numOfTopAllocations = max(10, math.ceil(0.05 * self.results.shape[0]))
        topAllocations = self.results.sort_values(by=["Sharpe"]).iloc[
            -numOfTopAllocations:, 3:
        ]

        # Show distribution of asset allocation for each asset for stability test
        plt.subplot(212)
        violinData = list(
            map(lambda x: topAllocations[x].values, topAllocations.columns)
        )
        plt.violinplot(violinData, showmeans=True)
        plt.xticks(range(1, 1 + numOfAssets), labels=topAllocations.columns)
        plt.title("Allocation for each asset for top portfolios")
        plt.show()

    def optimisedWeights(self):
        return self.results.iloc[self.results["Sharpe"].idxmax()]


def expectedPortfolioRet(returns, weight):
    return np.sum(returns.mean() * weight) * 252


def expectedPortfolioVar(returns, weight):
    return np.dot(weight, np.dot(returns.cov() * 252, weight))


def expectedSharpeRatio(returns, weight, rf=0):
    return (expectedPortfolioRet(returns, weight) - rf) / expectedPortfolioVar(
        returns, weight
    )


# Bayesian Optimizer for slow backtests
def bayesianOptimizer(portfolio, interval=None, **kwargs):
    # Options
    noSimulations = kwargs.get("sims", 20)

    optResults = OptimisationResults(portfolio.assetNames, rf=portfolio.rf)

    def black_box_function(**kwargs):
        weights = [v for v in kwargs.values()]
        if np.sum(weights) == 0:
            return 0
        normalisedWeights = np.array(weights) / np.sum(weights)
        normalisedWeightsDict = dict(zip(kwargs.keys(), normalisedWeights))
        results, _ = portfolio.backtest(weights=normalisedWeights, interval=interval)
        optResults.addData(
            {
                **normalisedWeightsDict,
                "Sharpe": results["sharpe"],
                "Returns": results["averageReturns"],
                "Std": results["standardDeviation"],
            }
        )
        return results["sharpe"]

    pbounds = {i: (0, 1) for i in portfolio.assetNames}
    optimizer = BayesianOptimization(
        f=black_box_function, pbounds=pbounds, random_state=1
    )
    optimizer.maximize(init_points=1, n_iter=noSimulations)
    return optResults


# Monte Carlo Optimizer using statistical model of the returns distribution
def monteCarloProxyOptimizer(portfolio, interval=None, **kwargs):
    # Options
    noSimulations = kwargs.get("sims", 1000)

    # portfolio.generateReturnsDataframe()
    optResults = OptimisationResults(portfolio.assetNames, rf=portfolio.rf)

    for _ in range(noSimulations):
        randWeights = randomWeight(len(portfolio.assetNames))
        results = dict(zip(portfolio.assetNames, randWeights))

        ret = expectedPortfolioRet(portfolio.assetReturnsDf, randWeights)
        var = expectedPortfolioVar(portfolio.assetReturnsDf, randWeights)
        std = math.sqrt(var)
        sharpe = (ret - portfolio.rf) / std

        results["Sharpe"] = sharpe
        results["Std"] = std
        results["Returns"] = ret

        optResults.addData(results)

    return optResults


# Monte Carlo Optimiser generates random weight of the assets and run full backtest on the weights
def monteCarloOptimizer(portfolio, interval=None, **kwargs):
    # Options
    noSimulations = kwargs.get("sims", 50)

    optResults = OptimisationResults(portfolio.assetNames, rf=portfolio.rf)

    for _ in range(noSimulations):
        randWeights = randomWeight(len(portfolio.assetNames))
        results = dict(zip(portfolio.assetNames, randWeights))
        (backtestRes, _) = portfolio.backtest(weights=randWeights, interval=interval)

        results["Sharpe"] = backtestRes["sharpe"]
        results["Std"] = backtestRes["standardDeviation"]
        results["Returns"] = backtestRes["averageReturns"]

        optResults.addData(results)

    return optResults


# Sequential Least Squares Programming optimizer using the statistical model for the assets
def slsqpOptimizer(portfolio, interval=None, **kwargs):
    optResults = OptimisationResults(portfolio.assetNames, rf=portfolio.rf)

    def black_box_function(weights):
        results = dict(zip(portfolio.assetNames, weights))
        performance = portfolio.portfolioPerformance(weights)

        results["Sharpe"] = performance["sharpe"]
        results["Std"] = math.sqrt(performance["variance"])
        results["Returns"] = performance["returns"]
        optResults.addData(results)
        return -performance["sharpe"]

    numberOfAssets = len(portfolio.assetNames)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for x in range(numberOfAssets))
    initial = np.array(numberOfAssets * [1.0 / numberOfAssets])
    sco.minimize(
        black_box_function,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return optResults


def testFn():
    p = Portfolio()
    p.addAsset("../data/A35.csv", "A35")
    p.addAsset("../data/BAB.csv", "BAB")
    p.addAsset("../data/IWDA.csv", "IWDA")
    p.generateReturnsDataframe()
    p.rf = 0.02
    results = monteCarloProxyOptimizer(p, sims=1000)
    # results.plotEfficientFrontier()
    results.plotConvergence()
