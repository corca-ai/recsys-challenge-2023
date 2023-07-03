from ensemble import ensemble

files = ["lgb32.csv", "lgb27.csv", "lgb15.csv", "lgb.csv"]

ensemble(
    files,
    output_path="lgbm_ensemble.csv",
    weight=[0.35, 0.25, 0.25, 0.2],
    method="sigmoid",
)
