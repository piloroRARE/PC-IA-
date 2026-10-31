#!/usr/bin/env python3
"""
Ablation multi-graines pour BioText
===================================
Compare les variantes (sans Hebbian, sans sommeil, sans sparse, connectivité
figée…) sur plusieurs graines et affiche moyenne ± écart-type de la loss
d'entraînement ET de la loss de validation (texte jamais vu).

Une seule graine ne permet pas de conclure : les écarts de la v2
(0.675 / 0.763 / 0.848) pouvaient venir du hasard.

Usage :
    python ablation.py                      # 3 graines, 300 étapes
    python ablation.py --seeds 5 --steps 300
    python ablation.py --cpu --corpus mon_texte.txt
"""

import sys
import argparse
import statistics
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS = {
    "complet":              {},
    "sans hebbien/STDP":    {"use_hebbian": False},
    "sans sommeil":         {"use_sleep": False},
    "sans sparse":          {"use_sparse": False},
    "connectivité figée":   {"prune_frac_max": 0.0, "grow_step": 0},
    "topk_min=2":           {"topk_min": 2},
}


def load_bt():
    path = HERE / "biotext_light.py"
    spec = importlib.util.spec_from_file_location("biotext_light", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["biotext_light"] = module
    spec.loader.exec_module(module)
    return module


def run_one(bt, overrides, seed, steps, text):
    """Entraîne une variante, restaure la config ensuite. Retourne (loss_train, loss_val)."""
    cfg = bt.cfg
    changes = {**overrides, "seed": seed, "train_steps": steps}
    saved = {k: (k in vars(cfg), getattr(cfg, k)) for k in changes}
    try:
        for k, v in changes.items():
            setattr(cfg, k, v)
        model = bt.train(text=text, verbose=False)
        h = model.history
        if not h["loss"]:
            return float("nan"), float("nan")
        return h["loss"][-1], h["val_loss"][-1]
    finally:
        for k, (had_instance_attr, old) in saved.items():
            if had_instance_attr:
                setattr(cfg, k, old)
            else:
                delattr(cfg, k)      # retombe sur la valeur de classe


def fmt(values):
    clean = [v for v in values if v == v]          # écarte les NaN
    if not clean:
        return "      n/a      "
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return f"{statistics.mean(clean):.3f} ± {sd:.3f}"


def main():
    p = argparse.ArgumentParser(description="Ablation multi-graines BioText")
    p.add_argument("--seeds", type=int, default=3, help="Nombre de graines (défaut: 3)")
    p.add_argument("--steps", type=int, default=300, help="Étapes par run (défaut: 300)")
    p.add_argument("--corpus", type=str, default=None, help="Fichier texte UTF-8 (défaut: corpus d'exemple)")
    p.add_argument("--cpu", action="store_true", help="Force le mode CPU")
    args = p.parse_args()

    bt = load_bt()
    if args.cpu:
        bt.cfg.device = "cpu"
        bt.cfg.mixed_precision = False
        bt.cfg.pin_memory = False

    text = Path(args.corpus).read_text(encoding="utf-8", errors="replace") if args.corpus else None
    seeds = [42 + i for i in range(args.seeds)]
    n_runs = len(VARIANTS) * len(seeds)
    print(f"\nAblation : {len(VARIANTS)} variantes × {len(seeds)} graines = {n_runs} runs de {args.steps} étapes\n")

    results = {}
    done = 0
    for name, overrides in VARIANTS.items():
        tr, va = [], []
        for seed in seeds:
            done += 1
            print(f"  [{done}/{n_runs}] {name} (graine {seed})…", flush=True)
            t, v = run_one(bt, overrides, seed, args.steps, text)
            tr.append(t)
            va.append(v)
        results[name] = (tr, va)

    print("\n" + "=" * 66)
    print(f"{'Variante':24} {'Loss train':>18} {'Loss validation':>20}")
    print("-" * 66)
    for name, (tr, va) in results.items():
        print(f"{name:24} {fmt(tr):>18} {fmt(va):>20}")
    print("=" * 66)

    ref_va = results["complet"][1]
    print("\nÉcart de validation vs « complet » (positif = la variante est PIRE) :")
    for name, (_, va) in results.items():
        if name == "complet":
            continue
        diffs = [v - r for v, r in zip(va, ref_va) if v == v and r == r]
        if diffs:
            worse = sum(d > 0 for d in diffs)
            print(f"  {name:24} Δ moyen = {statistics.mean(diffs):+.3f}  "
                  f"(pire sur {worse}/{len(diffs)} graines)")
    print("\nLecture : un effet n'est crédible que s'il est cohérent sur toutes les graines "
          "et plus grand que l'écart-type.\n")


if __name__ == "__main__":
    main()
