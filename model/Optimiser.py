import scipy.optimize as sco
import Portfolio as Portfolio
import numpy as np

class Optimizer:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def minSharpe(self, weights):
        return -self.portfolio.portfolioReturns(weights)

    def optimize(self):
        numberOfAssets = len(self.portfolio.assetNames)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for x in range(numberOfAssets))
        initial = np.array(numberOfAssets * [1.0 / numberOfAssets])
        options = sco.minimize(
            self.minSharpe,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        print(options)

def testPortfolioReturns():
    p = Portfolio.Portfolio()
    p.addAsset("./test/Asset.csv", "Asset1")
    p.addAsset("./test/Asset2.csv", "Asset2")
    o = Optimizer(p)
    # print(o.minSharpe([0, 1]))
    o.optimize()
