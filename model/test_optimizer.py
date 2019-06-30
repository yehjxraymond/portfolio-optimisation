from Portfolio import Portfolio
from Optimizer import Optimizer

p = Portfolio()
p.addAsset("../data/A35.csv", "Asset1")
p.addAsset("../data/BAB.csv", "Asset2")
p.addAsset("../data/IWDA.csv", "Asset3")

def testOptimizeSharpe():
    optimizer = Optimizer(p)
    weights = optimizer.optimizeSharpe()
    assert weights[0] == 0.10091446157346126
    assert weights[1] == 0.6462589503813203
    assert weights[2] == 0.2528265880452185

def testKfold():
    optimizer = Optimizer(p)
    weights, tests = optimizer.kfold(5)

    assert weights[0] == 0.10243975243355911
    assert weights[1] == 0.6461617173968426
    assert weights[2] == 0.2513985301695983

    assert tests["sharpeStd"] == 10.323176570539907
    assert tests["sharpeAvg"] == 24.638854539592632
    assert len(tests["weightsStd"]) == 3
    assert len(tests["sharpeRaw"]) == 5
    assert len(tests["weightsRaw"]) == 5

def testKfoldTs():
    optimizer = Optimizer(p)
    weights, tests = optimizer.kfoldTs(5)

    assert weights[0] == 0.08369825368561112
    assert weights[1] == 0.6594406892589346
    assert weights[2] == 0.25686105705545426

    assert tests["sharpeStd"] == 7.0303562877671535
    assert tests["sharpeAvg"] == 23.89759452999132
    assert len(tests["weightsStd"]) == 3
    assert len(tests["sharpeRaw"]) == 5
    assert len(tests["weightsRaw"]) == 5