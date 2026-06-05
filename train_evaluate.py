"""
train_evaluate.py  -  Reproduce paper results: 5-seed evaluation
=================================================================
Usage:
    python scripts/train_evaluate.py --dataset D1 --csv_path data/D1.csv --target Performance_Index
"""

import argparse, os, json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pipeline        import LearnSenseBoostVerify
from src.preprocessing   import make_splits, SEEDS
from baselines.baselines import get_tree_baselines
from evaluation.metrics  import compute_all_metrics, paired_ttest_effect_size


PROPOSED_DEFAULTS = {
    "tgpsf_kwargs":   {"delta": 0.5, "lambda_reg": 0.01, "epsilon": 0.10},
    "rgeabte_kwargs": {"T": 50, "d_max": 3, "gamma": 0.5, "xi": 0.3},
    "ccsgpr_kwargs":  {
        "d_h": 64, "k": 0.5, "tau_r": 0.6, "delta": 0.10,
        "xi": 0.01, "eta": 0.001, "epochs": 200, "patience": 15,
    },
}


def evaluate_seed(seed_idx, X_tr, X_te, y_tr, y_te):
    results = {}

    # Tree baselines
    for name, clf in get_tree_baselines(random_state=SEEDS[seed_idx]).items():
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        m = compute_all_metrics(y_te, preds)
        m["train_sec"] = round(time.time() - t0, 3)
        results[name] = m

    # Proposed pipeline
    rng_seed = SEEDS[seed_idx]
    model = LearnSenseBoostVerify(
        tgpsf_kwargs   = PROPOSED_DEFAULTS["tgpsf_kwargs"],
        rgeabte_kwargs = dict(**PROPOSED_DEFAULTS["rgeabte_kwargs"], random_state=rng_seed),
        ccsgpr_kwargs  = dict(**PROPOSED_DEFAULTS["ccsgpr_kwargs"],  random_state=rng_seed),
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_sec = round(time.time() - t0, 3)

    # Stage-2 only
    s2_preds = model.predict_stage2(X_te)
    results["RGEAB-TE_Stage2"] = compute_all_metrics(y_te, s2_preds)

    # Full-coverage CCSG-PR (fair comparator, 100% coverage)
    fc_preds, _ = model.predict_full_coverage(X_te)
    m_fc = compute_all_metrics(y_te, fc_preds)
    m_fc["train_sec"] = train_sec
    results["CCSG-PR_Full_Coverage"] = m_fc

    # Post-rejection CCSG-PR
    labels, omega, accepted = model.predict(X_te)
    m_pr = compute_all_metrics(y_te, labels, accepted_mask=accepted)
    m_pr["train_sec"] = train_sec
    results["CCSG-PR_Post_Rejection"] = m_pr

    results["_feature_reduction"] = {
        "n_original":    len(model.tgpsf.similarity_scores_),
        "n_retained":    len(model.tgpsf.selected_features_),
        "reduction_pct": round(model.tgpsf.reduction_pct(), 1),
    }
    return results


def aggregate(per_seed_results):
    all_methods = [k for k in per_seed_results[0] if not k.startswith("_")]
    agg = {}
    for method in all_methods:
        vals_per_metric = {}
        for seed_res in per_seed_results:
            for metric, val in seed_res[method].items():
                if isinstance(val, float):
                    vals_per_metric.setdefault(metric, []).append(val)
        agg[method] = {
            m: {"mean": round(float(np.mean(v)), 4),
                "std":  round(float(np.std(v, ddof=1)), 4)}
            for m, v in vals_per_metric.items()
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--target",   required=True)
    ap.add_argument("--out_dir",  default="results")
    ap.add_argument("--sep",      default=",")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df  = pd.read_csv(args.csv_path, sep=args.sep)
    X   = df.drop(columns=[args.target])
    y   = LabelEncoder().fit_transform(df[args.target].astype(str))

    print("Dataset " + args.dataset + ": " + str(X.shape[0]) + " samples, " +
          str(X.shape[1]) + " features")

    splits           = make_splits(X, y)
    per_seed_results = []

    for i, (Xtr, Xte, ytr, yte) in enumerate(splits):
        print("  Seed " + str(SEEDS[i]) + " ...", end=" ", flush=True)
        res = evaluate_seed(i, Xtr, Xte, ytr, yte)
        per_seed_results.append(res)
        fc  = res["CCSG-PR_Full_Coverage"]
        pr  = res["CCSG-PR_Post_Rejection"]
        print("Full-cov=" + str(round(fc["accuracy"], 4)) +
              "  Post-rej=" + str(round(pr.get("post_rej_accuracy", float("nan")), 4)) +
              "  Cov="      + str(round(pr.get("coverage", 1.0), 2)))

    agg  = aggregate(per_seed_results)
    path = os.path.join(args.out_dir, args.dataset + "_results.json")
    with open(path, "w") as f:
        json.dump({"dataset": args.dataset, "seeds": SEEDS, "results": agg}, f, indent=2)
    print("Results saved to " + path)

    # Print summary
    print("")
    header = "{:<32}  {:>10}  {:>9}  {:>9}".format("Method", "Acc mean", "Acc std", "F1 mean")
    print(header)
    print("-" * 65)
    for method, metrics in agg.items():
        if "accuracy" in metrics:
            row = "{:<32}  {:>10.4f}  {:>9.4f}  {:>9.4f}".format(
                method,
                metrics["accuracy"]["mean"],
                metrics["accuracy"]["std"],
                metrics.get("f1", {}).get("mean", float("nan")),
            )
            print(row)


if __name__ == "__main__":
    main()
