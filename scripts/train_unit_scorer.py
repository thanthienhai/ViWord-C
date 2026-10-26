"""Train the unit scorer of direction B (viword.unit_scorer): a cross-encoder over (question, clause) pairs.

  python scripts/train_unit_scorer.py --labels data/labels/vibid_train_utility.jsonl --label utility \
      --out runs/unit_utility_s0 --seed 0
  python scripts/train_unit_scorer.py ... --label gold --out runs/unit_gold_s0 --seed 0         # H4 control
  python scripts/train_unit_scorer.py ... --exclude-heldout-articles --out runs/unit_utility_unseen_s0
                                      # without the train questions on a dev/test gold article (ablation)

The train/dev split of the label file is by gold article and fixed (seed 0), independent of --seed.
"""
import argparse

from viword.data import read_jsonl
from viword.unit_scorer import LABELS, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--label", default="utility", choices=list(LABELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3",
                    help="a multilingual reranker to start from (an MLM checkpoint such as xlm-roberta-base learned "
                         "erratically on the smoke labels, a reranker steadily)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--groups-per-step", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--tau", type=float, default=0.5, help="temperature of the utility targets (nats per token)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-share", type=float, default=0.1)
    ap.add_argument("--exclude-heldout-articles", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N label rows (smoke tests)")
    args = ap.parse_args()

    rows = read_jsonl(args.labels)[:args.limit]
    if args.exclude_heldout_articles:
        rows = [r for r in rows if not r.get("heldout_article")]
    meta = train(rows, args.out, label=args.label, model_name=args.model_name, epochs=args.epochs, lr=args.lr,
                 groups_per_step=args.groups_per_step, max_length=args.max_length, tau=args.tau, seed=args.seed,
                 dev_share=args.dev_share)
    print(f"{args.out}: {meta['n_train_groups']} train / {meta['n_dev_groups']} dev questions, "
          f"best dev loss {min(h['loss'] for h in meta['history']):.4f}")


if __name__ == "__main__":
    main()
