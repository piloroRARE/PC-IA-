#!/usr/bin/env python3
"""
BioText Transformer — Script de démarrage ONE-LINER (v3)
========================================================
Lance tout : hardware check → entraînement → génération → sauvegarde

Usage:
    python start.py

Ou avec arguments :
    python start.py --steps 500 --d-model 128 --no-analysis
    python start.py --no-hebbian --no-sleep          # ablations
    python start.py --resume --steps 200             # reprend depuis les poids sauvés
    python start.py --corpus mon_texte.txt           # entraîne sur ton propre texte
    python start.py --prompt "La plasticité" --gen-len 200
"""

import sys
import argparse
import importlib.util
from pathlib import Path

# Répertoire du projet (fonctionne quel que soit le répertoire courant)
HERE = Path(__file__).resolve().parent


def _load_module(name):
    """Importe un module du projet par son chemin absolu."""
    path = HERE / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_analysis():
    """Lance hardware_check.py."""
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : ANALYSE HARDWARE")
    print("=" * 70)
    hw_module = _load_module("hardware_check")
    hw_module.main()


def apply_overrides(bt, args):
    """Applique les options CLI à la config AVANT la construction du modèle."""
    cfg = bt.cfg
    overrides = {
        "d_model":     args.d_model,
        "n_layers":    args.n_layers,
        "n_heads":     args.n_heads,
        "d_ff":        args.d_ff,
        "batch_size":  args.batch_size,
        "train_steps": args.steps,
        "lr":          args.lr,
        "max_edges":   args.max_edges,
        "seed":        args.seed,
        "val_frac":    args.val_frac,
    }
    for key, val in overrides.items():
        if val is not None:
            print(f"  ! Config override : {key}={val}")
            setattr(cfg, key, val)
    if args.no_hebbian:
        print("  ! Ablation : Hebbian/STDP désactivé")
        cfg.use_hebbian = False
    if args.no_sleep:
        print("  ! Ablation : phase sommeil désactivée")
        cfg.use_sleep = False
    if args.no_sparse:
        print("  ! Ablation : sparse coding désactivé")
        cfg.use_sparse = False
    if args.cpu:
        print("  ! Forçage CPU")
        cfg.device = "cpu"
        cfg.mixed_precision = False
        cfg.pin_memory = False
    # d_model doit être divisible par n_heads (sinon l'attention plante)
    bt.fix_heads(cfg, verbose=True)
    print()


def run_training(args):
    """Lance biotext_light.py avec la config personnalisée."""
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE")
    print("=" * 70 + "\n")

    bt = _load_module("biotext_light")
    apply_overrides(bt, args)

    text = None
    if args.corpus:
        text = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
        print(f"  Corpus : {args.corpus} ({len(text):,} caractères)\n")

    if args.resume:
        # Reprise : on charge les poids (l'architecture est relue depuis le
        # checkpoint) puis on continue l'entraînement.
        save_path = Path(args.save_path)
        if save_path.exists():
            print(f"  ↻ Reprise depuis {save_path}")
            model = bt.FastBioTextTransformer.load(save_path)
            model = bt.train(text=text, model=model)
            return model, bt
        print(f"  ⚠ {save_path} introuvable, entraînement depuis zéro")

    model = bt.train(text=text)
    return model, bt


def main():
    parser = argparse.ArgumentParser(
        description="🧠 BioText Transformer — Démarrage rapide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python start.py                              # Full pipeline (check + train)
  python start.py --no-analysis                # Train seulement
  python start.py --steps 500 --d-model 128    # Config perso
  python start.py --corpus mon_texte.txt       # Ton propre corpus (10 % final = validation)
  python start.py --no-hebbian --no-sleep      # Ablations
  python start.py --resume --steps 200         # Continue depuis les poids sauvés
        """)

    parser.add_argument("--no-analysis", action="store_true", help="Saute l'analyse hardware")
    parser.add_argument("--steps", type=int, default=None, help="Étapes d'entraînement (défaut: 300)")
    parser.add_argument("--d-model", type=int, default=None, help="Dimension du modèle (défaut: 64)")
    parser.add_argument("--n-layers", type=int, default=None, help="Nombre de couches (défaut: 2)")
    parser.add_argument("--n-heads", type=int, default=None, help="Nombre de têtes (défaut: 2)")
    parser.add_argument("--d-ff", type=int, default=None, help="Dimension FFN (défaut: 128)")
    parser.add_argument("--batch-size", type=int, default=None, help="Taille de batch (défaut: 8)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (défaut: 3e-3)")
    parser.add_argument("--max-edges", type=int, default=None, help="Budget d'arêtes par couche (défaut: 256)")
    parser.add_argument("--seed", type=int, default=None, help="Graine aléatoire (défaut: 42)")
    parser.add_argument("--corpus", type=str, default=None, help="Fichier texte UTF-8 d'entraînement")
    parser.add_argument("--val-frac", type=float, default=None,
                        help="Part finale du corpus tenue à part pour la validation (défaut: 0.1)")
    parser.add_argument("--cpu", action="store_true", help="Force le mode CPU")
    parser.add_argument("--no-hebbian", action="store_true", help="Ablation : sans Hebbian/STDP")
    parser.add_argument("--no-sleep", action="store_true", help="Ablation : sans phase sommeil")
    parser.add_argument("--no-sparse", action="store_true", help="Ablation : sans sparse coding")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder les poids")
    parser.add_argument("--resume", action="store_true", help="Reprend depuis les poids sauvegardés")
    parser.add_argument("--save-path", type=str, default=str(HERE / "biotext_light_weights.pt"),
                        help="Chemin des poids (défaut: ./biotext_light_weights.pt dans le dossier du projet)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt de génération supplémentaire")
    parser.add_argument("--gen-len", type=int, default=150, help="Longueur de génération pour --prompt")

    args = parser.parse_args()

    # Bannière
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║       🧠  BioText Transformer — Démarrage Rapide           ║")
    print("║                                                            ║")
    print("║   Transformer + Plasticité Bio-inspirée                    ║")
    print("║   • Hebbian + STDP (stabilisés, appliqués après l'optim.)  ║")
    print("║   • Connectivité dynamique (pruning/growing homéostatique) ║")
    print("║   • Sparse coding (top-k)                                  ║")
    print("║   • Phase sommeil (replay + consolidation)                 ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Étape 1 : Hardware analysis
    if not args.no_analysis:
        try:
            run_analysis()
        except FileNotFoundError:
            print("⚠ hardware_check.py non trouvé, on saute l'analyse")
        except Exception as e:  # l'analyse ne doit jamais bloquer l'entraînement
            print(f"⚠ Analyse hardware échouée ({type(e).__name__}: {e}), on continue")

    # Étape 2 : Entraînement
    saved_path = None
    try:
        model, bt = run_training(args)

        # Génération supplémentaire à la demande
        if args.prompt:
            print("\n" + "=" * 70)
            print("GÉNÉRATION PERSONNALISÉE")
            print("=" * 70)
            print(f"\n  Prompt : {args.prompt!r}")
            print("  Greedy   : " + model.generate(args.prompt, length=args.gen_len, greedy=True))
            print("  Sampling : " + model.generate(args.prompt, length=args.gen_len))

        # Étape 3 : Sauvegarde
        if not args.no_save:
            print("\n" + "=" * 70)
            print("ÉTAPE 3 : SAUVEGARDE")
            print("=" * 70)
            saved_path = model.save(args.save_path)
            print(f"\n  ✓ Poids sauvegardés : {saved_path.resolve()}")
            print(f"  Taille : {saved_path.stat().st_size / 1e6:.2f} Mo")

    except FileNotFoundError as e:
        print(f"❌ Fichier introuvable : {e}")
        sys.exit(1)
    except Exception as e:
        print("\n❌ Erreur lors de l'entraînement :")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Résumé final (ne liste que ce qui existe vraiment)
    generated = []
    if saved_path is not None:
        generated.append(f"    • {saved_path.name}  (poids du modèle + composante hebbienne + config)")
    if (HERE / "config_example.py").exists():
        generated.append("    • config_example.py  (exemple de config personnalisée)")
    generated_txt = "\n".join(generated) if generated else "    (aucun)"
    guide = "    5. Consulte GUIDE_UTILISATION.md pour explorer\n" if (HERE / "GUIDE_UTILISATION.md").exists() else ""

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"""
  ✓ Pipeline terminé avec succès !

  Prochaines étapes :
    1. Relance avec --steps 1000 pour une meilleure qualité
    2. Augmente --d-model 128 --n-layers 4 pour plus de capacité (GPU ≥ 10 Go)
    3. --resume pour continuer l'entraînement depuis les poids sauvegardés
    4. --corpus fichier.txt pour entraîner sur ton propre texte ;
       ablation.py pour comparer les variantes sur plusieurs graines
{guide}
  Fichiers générés :
{generated_txt}

  Questions ? Lance :
    python start.py --help
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
