"""
Hardware Checker — Analyse ta config PC et recommande des settings BioText (v3)
===============================================================================
Détecte GPU / CPU / RAM, lit (optionnellement) un CSV de composants,
fait un mini-benchmark et recommande une configuration + une commande prête à lancer.

Usage :
    python hardware_check.py
    python hardware_check.py --csv mes_composants.csv
    python hardware_check.py --write-config      # écrit config_example.py (opt-in)
"""

import csv
import os
import sys
import time
import argparse
import platform
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────
# CSV composants (optionnel)
# ─────────────────────────────────────────────
def parse_price_csv(csv_path):
    """Parse un CSV Composant / Modèle / Prix (colonnes en français)."""
    components = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                comp = row.get("Composant")
                if comp and comp != "TOTAL":
                    components[comp] = {
                        "model": row.get("Modèle", ""),
                        "price": row.get("Prix", ""),
                    }
    except Exception as e:
        print(f"  ⚠ Erreur lecture CSV : {e}")
    return components


# ─────────────────────────────────────────────
# Détection matérielle
# ─────────────────────────────────────────────
def _cpu_model():
    """Nom du CPU, multi-plateforme, sans dépendance externe."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Windows":
            name = os.environ.get("PROCESSOR_IDENTIFIER")
            if name:
                return name
        elif system == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except Exception:
        pass
    return platform.processor() or None


def detect_hardware():
    """Détecte GPU, CPU, RAM."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cpu_model": _cpu_model(),
        "ram_gb": None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_mem_gb": None,
        "gpu_cc": None,
        "bf16": False,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
    }

    # RAM (psutil optionnel)
    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / 1e9
    except Exception:
        try:  # fallback Linux
            info["ram_gb"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except Exception:
            pass

    # GPU
    if info["gpu_available"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = props.total_memory / 1e9
        info["gpu_cc"] = f"{props.major}.{props.minor}"
        info["bf16"] = torch.cuda.is_bf16_supported()

    return info


# ─────────────────────────────────────────────
# Mini-benchmark
# ─────────────────────────────────────────────
def benchmark(hw, n_iter=20):
    """Mesure un matmul 1024×1024 : ordre de grandeur de la vitesse.
    TF32 est activé sur GPU pour refléter les réglages réels de l'entraînement
    (setup_backend de biotext_light.py)."""
    if hw["gpu_available"]:
        torch.backends.cuda.matmul.allow_tf32 = True

    results = {}
    for device in (["cuda"] if hw["gpu_available"] else []) + ["cpu"]:
        try:
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            for _ in range(3):  # warm-up
                (a @ b).sum().item()
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                c = a @ b
            if device == "cuda":
                torch.cuda.synchronize()
            else:
                c.sum().item()
            dt = (time.perf_counter() - t0) / n_iter
            results[device] = dt * 1000  # ms
        except Exception as e:
            results[device] = None
            print(f"  ⚠ Benchmark {device} échoué : {e}")
    return results


# ─────────────────────────────────────────────
# Recommandations
# ─────────────────────────────────────────────
def suggest_command(hw):
    """Commande start.py adaptée à la machine détectée."""
    if hw["gpu_available"]:
        g = hw["gpu_mem_gb"] or 0
        if g >= 10:
            return ("python start.py --d-model 128 --n-layers 4 --n-heads 4 --d-ff 256 "
                    "--batch-size 32 --max-edges 512 --steps 1000")
        if g >= 6:
            return ("python start.py --d-model 96 --n-layers 3 --n-heads 3 --d-ff 192 "
                    "--batch-size 16 --steps 600")
        return "python start.py --d-model 64 --n-layers 2 --batch-size 8 --steps 300"
    return "python start.py"


def recommend_config(hw, components, bench):
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║ 🖥  ANALYSE HARDWARE & RECOMMANDATIONS BioText             ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # ── Config détectée ─────────────────────────────────────────────
    print("📋 CONFIGURATION DÉTECTÉE :")
    print(f"  OS           : {hw['os']}")
    print(f"  CPU          : {hw['cpu_count']} threads | {hw['cpu_model'] or '(détection échouée)'}")
    print(f"  Threads torch: {hw['torch_threads']}")
    if hw["ram_gb"]:
        print(f"  RAM          : {hw['ram_gb']:.1f} Go")
    print(f"  GPU          : {'OUI' if hw['gpu_available'] else 'NON'}")
    if hw["gpu_available"]:
        print(f"    Modèle     : {hw['gpu_name']}")
        print(f"    Mémoire    : {hw['gpu_mem_gb']:.1f} Go")
        print(f"    Compute    : {hw['gpu_cc']} | bf16 : {'oui' if hw['bf16'] else 'non'}")
    print(f"  PyTorch      : {hw['pytorch_version']} (Python {hw['python_version']})")

    # ── Benchmark ────────────────────────────────────────────────────
    if bench:
        print("\n⏱  BENCHMARK (matmul 1024×1024 fp32) :")
        for dev, ms in bench.items():
            if ms is not None:
                print(f"  {dev.upper():5} : {ms:7.2f} ms / matmul")
        if bench.get("cuda") and bench.get("cpu"):
            print(f"  → GPU ≈ {bench['cpu'] / bench['cuda']:.0f}× plus rapide que le CPU")

    # ── Composants CSV ───────────────────────────────────────────────
    if components:
        print("\n📦 COMPOSANTS (du CSV) :")
        for comp, data in components.items():
            print(f"  {comp:15} : {data['model']}   (prix : {data['price']})")

    # ── Recommandations ─────────────────────────────────────────────
    print("\n🎯 RECOMMANDATIONS BioText :")
    if hw["gpu_available"]:
        gpu_mem = hw["gpu_mem_gb"]
        if gpu_mem >= 10:
            print(f"\n  ✓ GPU de {gpu_mem:.1f} Go → Configuration OPTIMALE")
            print("    d_model      : 128-256")
            print("    n_layers     : 4-6")
            print("    batch_size   : 32-64")
            print("    d_ff         : 256-512")
            print("    max_edges    : 512")
            print("    mixed_prec.  : True (fp16 + GradScaler, TF32 activé)")
            print("    → Le modèle par défaut (88k params) s'entraîne en quelques secondes ;")
            print("      n'hésite pas à le grossir.")
        elif gpu_mem >= 6:
            print(f"\n  ✓ GPU de {gpu_mem:.1f} Go → Configuration BONNE")
            print("    d_model      : 64-128")
            print("    n_layers     : 2-4")
            print("    batch_size   : 16-32")
            print("    mixed_prec.  : True")
        else:
            print(f"\n  ⚠ GPU de {gpu_mem:.1f} Go → Configuration MINIMALE")
            print("    d_model      : 32-64")
            print("    n_layers     : 1-2")
            print("    batch_size   : 8")
            print("    mixed_prec.  : True (obligatoire)")
    else:
        print("\n  ℹ Pas de GPU détecté → Mode CPU")
        print(f"    torch utilise {hw['torch_threads']} threads")
        print("    d_model      : 64 (défaut)")
        print("    n_layers     : 2")
        print("    batch_size   : 8 (×2 accumulation)")
        print("    → ordre de grandeur : ~65 ms/étape sur 2 threads ; 300 étapes ≈ 20 s,")
        print("      bien moins avec davantage de cœurs")

    if hw["cpu_count"]:
        print(f"\n  CPU {hw['cpu_count']} threads :")
        if hw["cpu_count"] >= 8:
            print("    ✓ Bon pour les opérations CPU (Hebbian, replay, préparation des batches)")
        else:
            print("    ⚠ Peu de threads : garde le modèle petit en mode CPU")

    print("\n👉 COMMANDE SUGGÉRÉE POUR TA MACHINE :")
    print(f"     {suggest_command(hw)}")

    # ── Profils rapides ─────────────────────────────────────────────
    print("\n⚡ PROFILS PRÉ-CONFIGURÉS :")
    print("\n  1️⃣ ULTRA-RAPIDE (prototype) :")
    print("     python start.py --d-model 32 --n-layers 1 --batch-size 8 --steps 200")
    print("\n  2️⃣ ÉQUILIBRÉ (défaut) :")
    print("     python start.py")
    print("\n  3️⃣ ROBUSTE (GPU ≥ 10 Go) :")
    print("     python start.py --d-model 128 --n-layers 4 --n-heads 4 --d-ff 256 "
          "--batch-size 32 --max-edges 512 --steps 1000")
    print("\n  4️⃣ TON PROPRE TEXTE :")
    print("     python start.py --corpus mon_texte.txt --steps 1000")

    # ── Notes ───────────────────────────────────────────────────────
    print("\n💡 NOTES :")
    print("  • Résultats attendus : loss démarre à ≈ 5.5 (= log 256) et descend sous 1.0 en 300 étapes")
    print("  • La loss de VALIDATION (texte tenu à part) reste nettement plus haute : c'est normal,")
    print("    elle mesure la généralisation ; l'écart avec la loss d'entraînement = mémorisation")
    print("  • Les arêtes dynamiques sont élaguées / recréées à chaque phase sommeil (toutes les 30 étapes)")
    print("  • Mixed Precision (AMP fp16) + TF32 = 30-50 % plus rapide sur RTX 30xx")
    print("  • Ablations : --no-hebbian, --no-sleep, --no-sparse (ou ablation.py pour le multi-graines)")
    print("\n")


# ─────────────────────────────────────────────
# Fichier config exemple (opt-in : --write-config)
# ─────────────────────────────────────────────
CONFIG_EXAMPLE = '''# Exemple de configuration personnalisée BioText (v3)
# Copie les valeurs voulues dans la classe Config de biotext_light.py,
# ou passe-les en ligne de commande via start.py (voir --help).

class Config:
    # Device : détecté automatiquement ("cuda" si dispo, sinon "cpu")

    # Modèle
    d_model  = 64      # 128 sur GPU >= 10 Go
    n_heads  = 2       # doit diviser d_model (corrigé automatiquement sinon)
    n_layers = 2       # 4 sur GPU >= 10 Go
    d_ff     = 128

    # Plasticité hebbienne (stabilisée : composante H bornée à 5 % de ||W||)
    use_hebbian    = True
    eta_hebb       = 2e-4
    eta_stdp       = 1e-4
    hebb_max_ratio = 0.05

    # Connectivité dynamique
    max_edges      = 256
    tau_prune      = 0.15   # sur le score normalisé (score / max)
    prune_frac_max = 0.25   # au plus 25 % d'arêtes élaguées par sommeil
    min_edges      = 32

    # Sparse coding
    topk_ratio = 0.03
    topk_min   = 4

    # Sommeil
    sleep_every = 30
    sleep_steps = 10

    # Entraînement
    batch_size  = 8
    grad_accum  = 2
    lr          = 3e-3
    train_steps = 300
    val_frac    = 0.1   # part finale d'un corpus perso tenue à part (validation)

    # Sauvegarde (chemin ancré sur le dossier du projet)
    save_path = "./biotext_light_weights.pt"
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyse hardware pour BioText")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV composants (colonnes : Composant, Modèle, Prix)")
    parser.add_argument("--no-bench", action="store_true", help="Saute le mini-benchmark")
    parser.add_argument("--write-config", action="store_true",
                        help="Écrit config_example.py dans le dossier du projet")
    args = parser.parse_args(argv if argv is not None else [])

    hw = detect_hardware()

    # CSV : argument explicite, sinon ./Composant-Modle-Prix.csv s'il existe
    csv_path = Path(args.csv) if args.csv else HERE / "Composant-Modle-Prix.csv"
    components = parse_price_csv(csv_path) if csv_path.exists() else {}

    bench = {} if args.no_bench else benchmark(hw)
    recommend_config(hw, components, bench)

    if args.write_config:
        out = HERE / "config_example.py"
        try:
            out.write_text(CONFIG_EXAMPLE, encoding="utf-8")
            print(f"📄 Config exemple écrite : {out}")
        except OSError as e:
            print(f"⚠ Impossible d'écrire {out} : {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
