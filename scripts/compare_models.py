from xad.compare import compare_models, flag_all_reference, summarise
from xad.data import load_skab

df = load_skab("data/raw/SKAB")
results = compare_models(df)
results.to_csv("docs/compare_if_vs_eif.csv", index=False)

means, eif_wins = summarise(results)
means.loc["flag-all"] = flag_all_reference(df)
means.to_csv("docs/compare_if_vs_eif_summary.csv")
n_exp = results["experiment"].nunique()
print("== mean over experiments (flag-all = flag every evaluated row) ==")
print(means.round(3).to_string())
print(f"\nEIF beats IF on F1 in {eif_wins} of {n_exp} experiments")
