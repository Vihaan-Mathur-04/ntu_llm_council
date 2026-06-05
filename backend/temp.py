import pandas as pd

expr = pd.read_csv(
    "/Users/vihaan_mathur/ntu_project/llm_council/pbmc.data.reduced.csv",
    index_col=0
)

gt = pd.read_csv(
    "/Users/vihaan_mathur/ntu_project/llm_council/pbmc.ground.truth.annotations.csv"
)

print("Expression shape:", expr.shape)
print("Ground truth rows:", len(gt))

print(expr.shape)
print(expr.iloc[:5, :5])
print(expr.columns[:5])