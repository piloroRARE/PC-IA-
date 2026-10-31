"""
BioText Transformer — Version LÉGÈRE & RAPIDE pour PC Gamer (v3)
================================================================
Optimisé pour :
  • RTX 3060 (12 Go VRAM) — AMP fp16 + TF32 + attention fusionnée (SDPA)
  • Tous les cœurs CPU en mode CPU
  • Ratio vitesse/plasticité équilibré

Mécanismes bio :
  ✓ Plasticité des poids (Hebbian + STDP avec traces temporelles)
  ✓ Plasticité structurelle (pruning/growing d'arêtes, budget fixe)
  ✓ Sparse coding (top-k exact)
  ✓ Phase sommeil (replay + consolidation + élagage)

Usage :
    python biotext_light.py

Corrections v3 (par rapport à la v2) :
  1. Hebbian/STDP appliqué APRÈS optimizer.step() (et non pendant le forward) :
     avant, les poids étaient modifiés en place entre le forward et le backward,
     donc le gradient était calculé avec des poids légèrement différents.
  2. Checkpoint : la config sauvegardée contient maintenant les overrides CLI,
     et load() retrouve l'architecture à partir des formes des tenseurs →
     `--resume` fonctionne même si on ne répète pas --d-model, etc.
  3. print_config() affichait les valeurs par défaut de la classe et ignorait
     les overrides (c'était un classmethod) → méthode d'instance.
  4. prune_and_grow : la médiane d'importance des nouvelles arêtes est calculée
     AVANT d'activer les nouveaux slots (elle était biaisée vers 0).
  5. Mode CPU forcé (--cpu) : les threads CPU sont maintenant réglés selon
     cfg.device et non cfg.cuda_avail.
  6. n_heads corrigé automatiquement si d_model n'est pas divisible (fix_heads).
  7. Top-k exact (scatter des indices) : plus de dépassement de k sur égalités.
  8. Chemin de sauvegarde ancré sur le fichier (plus sur le répertoire courant).
  9. torch.load(weights_only=True) : plus sûr (pas de désérialisation pickle).

Améliorations v3 :
  • Jeu de validation tenu à part (phrases inédites pour le corpus d'exemple,
    10 % finaux pour un corpus perso) : la loss de validation mesure la
    généralisation, pas la mémorisation.
  • Le sommeil affiche la loss de validation avant/après (la loss « replay »
    est mesurée sur des données déjà apprises : elle baisse mécaniquement).
  • Corpus personnalisé : train(text=...) (CLI : start.py --corpus fichier.txt).
"""

import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────
# 0. CONFIG — AJUSTÉE POUR PC GAMER
# ─────────────────────────────────────────────
class Config:
    # ── DÉTECTION HARDWARE ──────────────────────────────────
    cuda_avail    = torch.cuda.is_available()
    gpu_name      = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    # Budget mémoire GPU (RTX 3060 = 12 Go, on vise 60-70 %)
    if cuda_avail:
        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb  = props.total_memory / 1e9
        max_gpu_mem = gpu_mem_gb * 0.65
    else:
        max_gpu_mem = 2.0  # fallback CPU (RAM)

    device = "cuda" if cuda_avail else "cpu"

    # Threads CPU : tous les cœurs disponibles
    cpu_threads = max(1, os.cpu_count() or 1)

    # ── ARCHITECTURE RÉDUITE ──────────────────────────────
    vocab_size    = 256          # octets UTF-8
    seq_len       = 64           # court = rapide

    d_model       = 64
    n_heads       = 2            # 64 / 2 têtes = 32 par tête
    n_layers      = 2
    d_ff          = 128
    dropout       = 0.05

    # ── PLASTICITÉ ────────────────────────────────────────
    # Structurelle (connectivité dynamique)
    max_edges     = 256          # budget maximal d'arêtes par couche (taille fixe)
    tau_alloc     = 0.6          # occupation initiale du budget (60 % → 153 arêtes)
    tau_prune     = 0.15         # seuil d'élagage sur le score NORMALISÉ (score / max)
    prune_frac_max= 0.25         # on n'élague jamais plus de 25 % des arêtes par sommeil
    min_edges     = 32           # plancher : jamais moins de 32 arêtes actives
    grow_step     = 8            # arêtes d'exploration créées à chaque sommeil
    edge_proj_init= 0.02         # écart-type initial du poids d'une arête neuve
    gate_lr_mult  = 5.0          # lr des logits de gates = 5 × lr
    importance_ema= 0.9          # lissage EMA du score d'importance

    # Poids (Hebbian/STDP)
    use_hebbian   = True         # False = ablation
    eta_hebb      = 2e-4         # pas hebbien (sur activations standardisées)
    eta_stdp      = 1e-4         # pas STDP
    stdp_tau      = 15.0         # constante de temps des traces (en tokens)
    decay_w       = 0.99         # oubli synaptique de H (appliqué à chaque pas d'optimiseur)
    hebb_max_ratio= 0.05         # ||H||_F ≤ 5 % de ||W||_F (garde-fou anti-explosion)

    # Sparse coding
    use_sparse    = True         # False = ablation
    topk_ratio    = 0.03         # 3 % d'activations conservées…
    topk_min      = 4            # …avec un plancher (3 % de 64 = 1 seule feature sinon)

    # ── PHASE SOMMEIL ──────────────────────────────────────
    use_sleep     = True         # False = ablation
    replay_cap    = 512          # taille du replay buffer (séquences)
    sleep_every   = 30           # consolidation toutes les 30 étapes
    sleep_steps   = 10           # étapes de replay par sommeil
    sleep_lr_mult = 0.5          # lr réduit pendant le sommeil

    # ── ENTRAÎNEMENT ───────────────────────────────────────
    lr            = 3e-3
    weight_decay  = 0.01
    batch_size    = 8
    grad_accum    = 2            # accumulation de gradients (batch effectif = 16)
    train_steps   = 300
    seed          = 42
    val_frac      = 0.1          # part finale du corpus perso tenue à part (0 = pas de validation)

    # ── GÉNÉRATION ─────────────────────────────────────────
    gen_len       = 120
    gen_temperature = 0.7
    gen_topk      = 8
    gen_rep_penalty = 1.3        # pénalise les caractères répétés récemment

    # ── OPTIMISATIONS GPU ──────────────────────────────────
    mixed_precision = cuda_avail  # AMP fp16 si GPU
    amp_dtype       = torch.float16
    pin_memory      = cuda_avail

    # ── SAUVEGARDE ─────────────────────────────────────────
    save_path     = str(HERE / "biotext_light_weights.pt")

    def print_config(self):
        """Méthode d'INSTANCE : reflète les overrides posés sur cfg (v2 : classmethod → ignorait les overrides)."""
        c = self
        print("╔════════════════════════════════════════════════════════╗")
        print("║ BioText Transformer — CONFIG PC GAMER (v3)             ║")
        print("╠════════════════════════════════════════════════════════╣")
        print(f"║ Device         : {c.device.upper():37} ║")
        print(f"║ GPU            : {c.gpu_name[:37]:37} ║")
        if c.device == "cuda":
            print(f"║ GPU Mem Budget : {c.max_gpu_mem:.1f} Go (max){' ':24} ║")
        else:
            print(f"║ Threads CPU    : {c.cpu_threads:<37} ║")
        arch = f"d_model={c.d_model}, n_layers={c.n_layers}, d_ff={c.d_ff}"
        print(f"║ Modèle         : {arch:37} ║")
        print(f"║ Têtes attention: {c.n_heads:<37} ║")
        bs = f"{c.batch_size} (×{c.grad_accum} accum = eff. {c.batch_size * c.grad_accum})"
        print(f"║ Batch size     : {bs:37} ║")
        print(f"║ Mixed Precision: {str(c.mixed_precision):37} ║")
        print(f"║ Hebbian/STDP   : {str(c.use_hebbian):37} ║")
        print(f"║ Sparse coding  : {str(c.use_sparse) + f' (top-k {c.topk_ratio:.0%}, min {c.topk_min})':37} ║")
        print(f"║ Sommeil        : {str(c.use_sleep) + f' (toutes les {c.sleep_every} étapes)':37} ║")
        print("╚════════════════════════════════════════════════════════╝")


cfg = Config()

# Paramètres qui déterminent la forme des tenseurs du modèle
ARCH_KEYS = ("vocab_size", "seq_len", "d_model", "n_heads", "n_layers", "d_ff", "max_edges")


def fix_heads(c=None, verbose=False):
    """Garantit d_model divisible par n_heads (~32 dims par tête). Retourne True si modifié."""
    c = c or cfg
    if c.d_model % c.n_heads == 0:
        return False
    new_heads = max(1, c.d_model // 32)
    while c.d_model % new_heads:
        new_heads -= 1
    if verbose:
        print(f"  ! n_heads ajusté {c.n_heads} → {new_heads} (d_model={c.d_model} doit être divisible)")
    c.n_heads = new_heads
    return True


def setup_backend():
    """Réglages globaux PyTorch : threads CPU, TF32 sur Ampere, seed."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.device == "cuda":
        # TF32 : matmuls ~2× plus rapides sur RTX 30xx sans perte notable
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(cfg.cpu_threads)


# ─────────────────────────────────────────────
# 1. TOP-K SPARSE GATE
# ─────────────────────────────────────────────
class TopKGate(nn.Module):
    """Ne conserve que les k activations les plus fortes (en valeur absolue)
    par token — analogue du sparse coding cortical (1-4 % de neurones actifs).
    v3 : masque construit à partir des indices du top-k → exactement k actives
    (la comparaison `>= seuil` en gardait davantage en cas d'égalités)."""
    def __init__(self, k_ratio=0.03, k_min=4):
        super().__init__()
        self.k_ratio = k_ratio
        self.k_min   = k_min
        self.last_active_frac = 0.0   # mesure de sparsité (pour les stats)

    def forward(self, x):
        if not cfg.use_sparse:
            return x
        dim = x.shape[-1]
        k = min(dim, max(self.k_min, math.ceil(dim * self.k_ratio)))
        idx = torch.topk(x.abs(), k, dim=-1).indices
        mask = torch.zeros_like(x).scatter_(-1, idx, 1.0)
        self.last_active_frac = k / dim
        return x * mask


# ─────────────────────────────────────────────
# 2. PLASTICITÉ STRUCTURELLE (MD-DCM)
# ─────────────────────────────────────────────
class FastDynamicConnectivity(nn.Module):
    """Arêtes dynamiques entre features du flux résiduel.

    Tous les tenseurs ont une taille FIXE `max_edges` et un masque booléen
    `active`. Élaguer = désactiver un slot ; faire pousser = réactiver un slot
    avec une nouvelle paire (src, dst). Les nn.Parameter ne sont jamais
    recréés et l'optimiseur reste cohérent.

    Score d'importance : s_e = g_e × |corr(src, dst)|  (Pearson sur le batch),
    lissé par EMA puis NORMALISÉ par le max des arêtes actives.
    """
    def __init__(self, d_model, max_edges=256):
        super().__init__()
        self.d_model   = d_model
        self.max_edges = max_edges

        n_init = max(cfg.min_edges, int(max_edges * cfg.tau_alloc))
        n_init = min(n_init, max_edges)
        src, dst = self._sample_edges(max_edges)
        active = torch.zeros(max_edges, dtype=torch.bool)
        active[:n_init] = True

        self.register_buffer("edge_src", src)
        self.register_buffer("edge_dst", dst)
        self.register_buffer("active", active)
        self.register_buffer("importance", torch.zeros(max_edges))
        self.register_buffer("age", torch.zeros(max_edges, dtype=torch.long))  # nb de mises à jour

        # logits > 0 → gate initiale ≈ 0.73 (une arête neuve est « ouverte »)
        self.logits    = nn.Parameter(torch.full((max_edges,), 1.0))
        self.edge_proj = nn.Parameter(torch.randn(max_edges) * cfg.edge_proj_init)

    # ---- utilitaires ------------------------------------------------------
    def _sample_edges(self, n, device="cpu"):
        """Tire n paires (src, dst) sans boucle sur soi-même."""
        src = torch.randint(0, self.d_model, (n,), device=device)
        dst = torch.randint(0, self.d_model, (n,), device=device)
        same = src == dst
        dst[same] = (dst[same] + 1) % self.d_model
        return src, dst

    @property
    def gates(self):
        return torch.sigmoid(self.logits)

    @property
    def n_active(self):
        return int(self.active.sum().item())

    def active_gates(self):
        return self.gates[self.active]

    # ---- forward ----------------------------------------------------------
    def forward(self, x):
        if not self.active.any():
            return x
        eff = self.gates * self.edge_proj * self.active.to(x.dtype)  # [E]
        contrib = x[..., self.edge_src] * eff                        # [B, T, E]
        delta = torch.zeros_like(x).index_add(-1, self.edge_dst, contrib)
        return x + delta

    # ---- importance -------------------------------------------------------
    @torch.no_grad()
    def update_importance(self, x):
        """EMA de s_e = g_e × |corr_Pearson(x[src], x[dst])| sur le batch."""
        if not self.active.any():
            return
        flat = x.reshape(-1, x.shape[-1]).float()          # [N, D]
        flat = flat - flat.mean(dim=0, keepdim=True)
        std  = flat.std(dim=0, keepdim=True) + 1e-6
        z    = flat / std
        corr = (z[:, self.edge_src] * z[:, self.edge_dst]).mean(dim=0).abs()  # ∈ [0, 1]
        score = self.gates * corr
        a = cfg.importance_ema
        self.importance = torch.where(
            self.active, a * self.importance + (1 - a) * score, self.importance
        )
        self.age += self.active.long()

    # ---- pruning / growing ------------------------------------------------
    @torch.no_grad()
    def prune_and_grow(self, optimizer=None):
        """Élague les arêtes faibles (score normalisé < tau_prune), puis fait
        pousser de nouvelles arêtes. Retourne (n_pruned, n_grown)."""
        n_act = self.n_active
        if n_act == 0:
            n_pruned = 0
        else:
            imp = self.importance.clone()
            imp[~self.active] = float("inf")            # les inactives ne comptent pas
            max_imp = self.importance[self.active].max().clamp_min(1e-8)
            norm_score = imp / max_imp                  # ∈ [0, 1] pour les actives

            # Candidates : score faible ET arête assez « vieille » pour être jugée
            mature = self.age >= 5
            cand = (norm_score < cfg.tau_prune) & self.active & mature

            # Garde-fous : max 25 % élaguées, jamais sous min_edges
            max_prune = min(int(cfg.prune_frac_max * n_act), max(0, n_act - cfg.min_edges))
            n_pruned = min(int(cand.sum().item()), max_prune)
            if n_pruned > 0:
                cand_idx = cand.nonzero(as_tuple=True)[0]
                order = norm_score[cand_idx].argsort()      # les plus faibles d'abord
                kill = cand_idx[order[:n_pruned]]
                self.active[kill] = False
                self.importance[kill] = 0.0
                self.age[kill] = 0

        # Médiane de référence calculée AVANT d'activer les nouveaux slots
        # (v2 : les slots neufs, d'importance 0, faussaient la médiane vers le bas)
        if self.active.any():
            med = self.importance[self.active].median()
        else:
            med = torch.zeros((), device=self.importance.device)

        # Croissance homéostatique : grow_step arêtes d'exploration + la moitié
        # des élaguées (jamais au-delà du budget max_edges).
        free = (~self.active).nonzero(as_tuple=True)[0]
        n_grow = min(len(free), cfg.grow_step + n_pruned // 2)
        if n_grow > 0:
            slots = free[torch.randperm(len(free), device=free.device)[:n_grow]]
            src, dst = self._sample_edges(n_grow, device=self.edge_src.device)
            self.edge_src[slots] = src
            self.edge_dst[slots] = dst
            self.active[slots] = True
            self.age[slots] = 0
            self.importance[slots] = med       # évite l'élagage immédiat
            # Nouvelle synapse : paramètres réinitialisés
            self.logits.data[slots] = 1.0
            self.edge_proj.data[slots] = torch.randn(n_grow, device=self.edge_proj.device) * cfg.edge_proj_init
            # État Adam des slots recyclés remis à zéro
            if optimizer is not None:
                for p in (self.logits, self.edge_proj):
                    st = optimizer.state.get(p)
                    if st:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in st:
                                st[key][slots] = 0.0
        return n_pruned, n_grow


# ─────────────────────────────────────────────
# 3. HEBBIAN / STDP (STABILISÉ, APPLIQUÉ APRÈS L'OPTIMISEUR)
# ─────────────────────────────────────────────
class FastHebbianSTDP:
    """Plasticité locale des poids, appliquée aux projections Q, K, V et à la FFN.

    Hebbian : Δw = η_h · corr(post, pre)          (activations standardisées)
    STDP    : Δw = η_s · (post·trace_pre − trace_post·pre)
              traces exponentielles causales le long des positions :
              pré AVANT post → renforcement (LTP), post AVANT pré → affaiblissement (LTD)

    Stabilité : la contribution hebbienne est stockée séparément dans H
    (W_effectif = W_gradient + H). H décroît (oubli synaptique) et sa norme
    est plafonnée à `max_ratio` × ||W||.

    v3 : `step()` ne fait que CALCULER et mettre en attente Δw pendant le
    forward ; `apply()` (appelé après optimizer.step()) met à jour H et les
    poids. Les poids ne changent donc plus entre forward et backward.
    """
    def __init__(self, eta_hebb=2e-4, eta_stdp=1e-4, tau=15.0, decay=0.99, max_ratio=0.1):
        self.eta_hebb  = eta_hebb
        self.eta_stdp  = eta_stdp
        self.tau       = tau
        self.decay     = decay
        self.max_ratio = max_ratio
        self.enabled   = True
        self.H         = {}       # nom → composante hebbienne accumulée
        self._pending  = {}       # nom → [poids, Δw cumulé, nb de micro-batches]
        self._kernels  = {}       # cache des noyaux temporels par (T, device)
        self.last_update_norm = {}

    def _kernel(self, T, device):
        """Noyau causal K[t, s] = exp(-(t-s)/tau) pour s < t (trace du passé)."""
        key = (T, str(device))
        if key not in self._kernels:
            t = torch.arange(T, device=device, dtype=torch.float32)
            diff = t[:, None] - t[None, :]
            K = torch.exp(-diff / self.tau) * (diff > 0).float()
            self._kernels[key] = K
        return self._kernels[key]

    @staticmethod
    def _standardize(a):
        """Standardise chaque feature sur les échantillons (B·T)."""
        a = a.float()
        flat = a.reshape(-1, a.shape[-1])
        mu = flat.mean(dim=0)
        sd = flat.std(dim=0) + 1e-5
        return ((a - mu) / sd)

    @torch.no_grad()
    def step(self, name, weight, pre_act, post_act):
        """Calcule Δw (Hebbian + STDP) et le met en attente. Ne touche PAS aux poids."""
        if not (self.enabled and cfg.use_hebbian):
            return
        B, T, _ = pre_act.shape
        N = B * T
        pre  = self._standardize(pre_act.detach())     # [B, T, d_in]
        post = self._standardize(post_act.detach())    # [B, T, d_out]

        # Hebbian : matrice de corrélation post × pre  → [d_out, d_in]
        dw_hebb = self.eta_hebb * torch.einsum("btp,btq->pq", post, pre) / N

        # STDP : traces temporelles causales
        K = self._kernel(T, pre.device)                # [T, T]
        pre_trace  = torch.einsum("ts,bsq->btq", K, pre)
        post_trace = torch.einsum("ts,bsp->btp", K, post)
        ltp = torch.einsum("btp,btq->pq", post, pre_trace)   # pré avant post
        ltd = torch.einsum("btp,btq->pq", post_trace, pre)   # post avant pré
        dw_stdp = self.eta_stdp * (ltp - ltd) / N

        dw = dw_hebb + dw_stdp
        # Clip par entrée : les traces peuvent amplifier ~tau fois, et les
        # features creuses (top-k) donnent des z-scores très grands.
        lim = 3.0 * (self.eta_hebb + self.eta_stdp)
        dw = dw.clamp_(-lim, lim).to(weight.dtype)

        pend = self._pending.get(name)
        if pend is None:
            self._pending[name] = [weight, dw, 1]
        else:
            pend[1] = pend[1] + dw
            pend[2] += 1
        self.last_update_norm[name] = dw.norm()        # tenseur (pas de sync GPU)

    @torch.no_grad()
    def apply(self):
        """À appeler APRÈS optimizer.step() : met à jour H (oubli + plafond) et W_eff = W_grad + H."""
        for name, (weight, dw_sum, n) in self._pending.items():
            dw = dw_sum / n
            H_old = self.H.get(name)
            if H_old is None:
                H_old = torch.zeros_like(weight)
            H_new = self.decay * H_old + dw

            # Plafond : ||H|| ≤ max_ratio × ||W_grad||  (W_grad = W − H_old)
            w_grad = weight.data - H_old
            max_norm = self.max_ratio * w_grad.norm()
            h_norm = H_new.norm()
            if h_norm > max_norm:
                H_new = H_new * (max_norm / (h_norm + 1e-12))

            weight.data.add_(H_new - H_old)
            self.H[name] = H_new
        self._pending.clear()

    def state_dict(self):
        return {k: v.cpu() for k, v in self.H.items()}

    def load_state_dict(self, sd, device="cpu"):
        self.H = {k: v.to(device) for k, v in sd.items()}
        self._pending.clear()


# ─────────────────────────────────────────────
# 4. ATTENTION OPTIMISÉE (CAUSALE)
# ─────────────────────────────────────────────
class FastBioAttention(nn.Module):
    def __init__(self, d_model, n_heads, plasticity, layer_id):
        super().__init__()
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_head     = d_model // n_heads
        self.layer_id   = layer_id
        self.plasticity = plasticity

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.gate = TopKGate(cfg.topk_ratio, cfg.topk_min)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Plasticité locale (calcul seulement ; application après l'optimiseur)
        if self.training:
            pfx = f"layer{self.layer_id}.attn"
            self.plasticity.step(f"{pfx}.q", self.W_q.weight, x, Q)
            self.plasticity.step(f"{pfx}.k", self.W_k.weight, x, K)
            self.plasticity.step(f"{pfx}.v", self.W_v.weight, x, V)

        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention causale fusionnée : chaque position ne voit que le passé.
        out = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True,
            dropout_p=cfg.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.gate(out)              # sparse coding
        out = self.W_o(out)
        return self.drop(out)


# ─────────────────────────────────────────────
# 5. FFN LÉGER (avec plasticité)
# ─────────────────────────────────────────────
class FastBioFFN(nn.Module):
    def __init__(self, d_model, d_ff, plasticity, layer_id):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff, bias=False)
        self.fc2  = nn.Linear(d_ff, d_model, bias=False)
        self.gate = TopKGate(cfg.topk_ratio, cfg.topk_min)
        self.drop = nn.Dropout(cfg.dropout)
        self.plasticity = plasticity
        self.layer_id   = layer_id

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.gate(h)                  # sparse coding sur la couche cachée
        out = self.fc2(h)
        if self.training:
            pfx = f"layer{self.layer_id}.ffn"
            self.plasticity.step(f"{pfx}.fc1", self.fc1.weight, x, h)
            self.plasticity.step(f"{pfx}.fc2", self.fc2.weight, h, out)
        return self.drop(out)


# ─────────────────────────────────────────────
# 6. COUCHE BIO COMPLÈTE
# ─────────────────────────────────────────────
class FastBioLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, plasticity, layer_id):
        super().__init__()
        self.attn = FastBioAttention(d_model, n_heads, plasticity, layer_id)
        self.ffn  = FastBioFFN(d_model, d_ff, plasticity, layer_id)
        self.dcm  = FastDynamicConnectivity(d_model, cfg.max_edges)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # pré-LayerNorm (plus stable)
        x = x + self.ffn(self.ln2(x))
        x = self.dcm(x)                   # arêtes dynamiques inter-features
        if self.training:
            self.dcm.update_importance(x.detach())
        return x


# ─────────────────────────────────────────────
# 7. MODÈLE COMPLET
# ─────────────────────────────────────────────
class FastBioTextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        fix_heads(cfg, verbose=True)       # d_model divisible par n_heads, sinon l'assert plante
        self.plasticity = FastHebbianSTDP(
            eta_hebb=cfg.eta_hebb, eta_stdp=cfg.eta_stdp,
            tau=cfg.stdp_tau, decay=cfg.decay_w, max_ratio=cfg.hebb_max_ratio,
        )
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.d_model)
        self.layers  = nn.ModuleList([
            FastBioLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, self.plasticity, i)
            for i in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight       # poids liés

        # Init GPT-style N(0, 0.02²) → loss initial ≈ log(256) ≈ 5.55
        nn.init.normal_(self.embed.weight,   std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.replay = deque(maxlen=cfg.replay_cap)
        self.history = {}

    def forward(self, ids):
        B, T = ids.shape
        pos  = torch.arange(T, device=ids.device).unsqueeze(0)
        x    = self.embed(ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        return self.head(x)

    # ---- pertes -----------------------------------------------------------
    @staticmethod
    def lm_loss(logits, ids):
        """Cross-entropy next-token : logits[t] prédit ids[t+1]."""
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1),
        )

    @torch.no_grad()
    def eval_loss(self, ids_list, n=16):
        """Loss sur n fenêtres FIXES (mêmes à chaque appel) d'une liste d'octets.
        Sert à la validation : données jamais vues à l'entraînement."""
        L = cfg.seq_len
        if ids_list is None or len(ids_list) < L + 1:
            return float("nan")
        was_training = self.training
        self.eval()
        rng = random.Random(0)
        starts = [rng.randint(0, len(ids_list) - L) for _ in range(n)]
        ids = torch.tensor([ids_list[s: s + L] for s in starts], dtype=torch.long, device=cfg.device)
        loss = self.lm_loss(self(ids), ids).item()
        if was_training:
            self.train()
        return loss

    # ---- replay -----------------------------------------------------------
    def push_to_replay(self, ids_batch):
        for seq in ids_batch:
            self.replay.append(seq.detach().cpu())

    @torch.no_grad()
    def eval_replay_loss(self, n=32):
        """Loss moyen sur un échantillon fixe du replay. ATTENTION : ce sont des
        données d'entraînement ; cette valeur baisse mécaniquement après des pas
        de gradient dessus. Pour la généralisation, voir eval_loss(val_ids)."""
        if len(self.replay) == 0:
            return float("nan")
        was_training = self.training
        self.eval()
        rng = random.Random(0)
        sample = rng.sample(list(self.replay), min(n, len(self.replay)))
        ids = torch.stack(sample).to(cfg.device)
        loss = self.lm_loss(self(ids), ids).item()
        if was_training:
            self.train()
        return loss

    # ---- sommeil ----------------------------------------------------------
    def sleep(self, optimizer, scaler=None, val_ids=None, verbose=True):
        """Phase sommeil : replay + consolidation + élagage/croissance."""
        if not cfg.use_sleep or len(self.replay) < cfg.batch_size:
            return
        log = print if verbose else (lambda *a, **k: None)
        log("  ◌ Phase sommeil (consolidation)...")
        loss_before = self.eval_replay_loss()
        has_val = val_ids is not None and len(val_ids) >= cfg.seq_len + 1
        val_before = self.eval_loss(val_ids) if has_val else None

        # lr réduit pendant le sommeil (consolidation douce)
        base_lrs = [g["lr"] for g in optimizer.param_groups]
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * cfg.sleep_lr_mult

        self.train()
        for _ in range(cfg.sleep_steps):
            batch = random.sample(list(self.replay), min(cfg.batch_size, len(self.replay)))
            ids   = torch.stack(batch).to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=cfg.amp_dtype):
                    loss = self.lm_loss(self(ids), ids)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self.lm_loss(self(ids), ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
            self.plasticity.apply()          # Hebbian/STDP après l'optimiseur
        optimizer.zero_grad(set_to_none=True)

        for g, lr0 in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr0

        # Plasticité structurelle
        for i, layer in enumerate(self.layers):
            n_before = layer.dcm.n_active
            n_pruned, n_grown = layer.dcm.prune_and_grow(optimizer)
            log(f"    Couche {i} : {n_before} arêtes → élaguées {n_pruned}, "
                f"créées {n_grown} → {layer.dcm.n_active} actives")

        loss_after = self.eval_replay_loss()
        log(f"    Loss replay (donnée apprise)  : {loss_before:.4f} → {loss_after:.4f} "
            f"({'↓' if loss_after < loss_before else '↑'})")
        if has_val:
            val_after = self.eval_loss(val_ids)
            log(f"    Loss validation (tenue à part) : {val_before:.4f} → {val_after:.4f} "
                f"({'↓' if val_after < val_before else '↑'})")

    # ---- génération -------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt, length=None, temperature=None, top_k=None,
                 rep_penalty=None, greedy=False):
        """Génère du texte octet par octet (fenêtre glissante seq_len)."""
        length      = length or cfg.gen_len
        temperature = cfg.gen_temperature if temperature is None else temperature
        top_k       = cfg.gen_topk if top_k is None else top_k
        rep_penalty = cfg.gen_rep_penalty if rep_penalty is None else rep_penalty

        self.eval()
        ids = torch.tensor([text_to_ids(prompt)], device=cfg.device)
        out_ids = ids[0].tolist()
        for _ in range(length):
            ctx = ids[:, -cfg.seq_len:]
            logits = self(ctx)[0, -1].float()
            # Pénalité de répétition sur les 8 derniers octets (anti "uuuuu")
            if rep_penalty > 1.0:
                recent = torch.tensor(out_ids[-8:], device=logits.device)
                logits[recent] = torch.where(
                    logits[recent] > 0, logits[recent] / rep_penalty, logits[recent] * rep_penalty
                )
            if greedy or temperature <= 0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.numel()))
                    logits[logits < v[-1]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1))
            out_ids.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=cfg.device)], dim=1)
        return ids_to_text(out_ids)

    # ---- sauvegarde / chargement -----------------------------------------
    def save(self, path=None):
        path = Path(path or cfg.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # v3 : classe + instance → les overrides CLI (posés sur cfg) sont sauvegardés
        merged = {**vars(Config), **vars(cfg)}
        torch.save({
            "model": self.state_dict(),
            "hebbian_H": self.plasticity.state_dict(),
            "config": {k: v for k, v in merged.items()
                       if not k.startswith("_") and isinstance(v, (int, float, str, bool))},
        }, path)
        return path

    @staticmethod
    def _infer_arch(sd):
        """Retrouve l'architecture à partir des FORMES des tenseurs (robuste,
        y compris pour les checkpoints v2 dont la config était incomplète)."""
        vocab, d_model = sd["embed.weight"].shape
        return {
            "vocab_size": int(vocab),
            "d_model":    int(d_model),
            "seq_len":    int(sd["pos_emb.weight"].shape[0]),
            "n_layers":   1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers.")),
            "d_ff":       int(sd["layers.0.ffn.fc1.weight"].shape[0]),
            "max_edges":  int(sd["layers.0.dcm.logits"].shape[0]),
        }

    @classmethod
    def load(cls, path=None, device=None):
        device = device or cfg.device
        ckpt = torch.load(Path(path or cfg.save_path), map_location=device, weights_only=True)
        arch = cls._infer_arch(ckpt["model"])
        saved = ckpt.get("config", {})
        # n_heads n'est pas déductible des formes : on prend celui de la config sauvée
        if "n_heads" in saved and arch["d_model"] % int(saved["n_heads"]) == 0:
            arch["n_heads"] = int(saved["n_heads"])
        for k, v in arch.items():
            if getattr(cfg, k) != v:
                print(f"  ↻ {k} : checkpoint = {v} (config courante : {getattr(cfg, k)}) → checkpoint appliqué")
                setattr(cfg, k, v)
        model = cls().to(device)
        model.load_state_dict(ckpt["model"])
        model.plasticity.load_state_dict(ckpt.get("hebbian_H", {}), device)
        return model


# ─────────────────────────────────────────────
# 8. DONNÉES — Texte français custom
# ─────────────────────────────────────────────
SAMPLE_SENTENCES = [
    "Le cerveau contient environ quatre-vingt-six milliards de neurones.",
    "Chaque neurone est connecté à d'autres par des synapses dynamiques.",
    "La plasticité synaptique permet l'apprentissage continu.",
    "Les règles hebbienne et STDP gouvernent le changement synaptique.",
    "Durant le sommeil, la consolidation mémoire rejoue les expériences.",
    "L'architecture du cerveau est parcellisée en couches hiérarchiques.",
    "Les spikes sont les signaux événementiels du système nerveux.",
    "Le sparse coding économise l'énergie cérébrale et booste la performance.",
    "La plasticité structurelle crée et élimine les connexions synaptiques.",
    "Un PC gamer moderne simule ces phénomènes à petite échelle efficacement.",
]


def load_sample_text():
    """Corpus d'exemple complet (10 phrases × 15). Conservé pour compatibilité."""
    return (" ".join(SAMPLE_SENTENCES) + " ") * 15


def text_to_ids(text):
    """Encodage octet par octet (UTF-8) → vocab 256, réversible (accents OK)."""
    return list(text.encode("utf-8"))


def ids_to_text(ids):
    return bytes(ids).decode("utf-8", errors="replace")


def prepare_data(text=None, val_frac=None):
    """Retourne (train_ids, val_ids). val_ids = None si pas de validation possible.

    • Corpus d'exemple : 8 phrases répétées pour l'entraînement ; les 2 dernières
      phrases, jamais vues, servent de validation (mesure la généralisation).
    • Corpus perso : les `val_frac` derniers pour cent du texte sont tenus à part.
    """
    val_frac = cfg.val_frac if val_frac is None else val_frac
    if text is None:
        train_text = (" ".join(SAMPLE_SENTENCES[:-2]) + " ") * 15
        val_text   = " ".join(SAMPLE_SENTENCES[-2:])
        train_ids  = text_to_ids(train_text)
        val_ids    = text_to_ids(val_text)
    else:
        cut = int(len(text) * (1 - val_frac))
        train_ids = text_to_ids(text[:cut])
        val_ids   = text_to_ids(text[cut:])
        if val_frac <= 0 or len(val_ids) < cfg.seq_len + 1:
            train_ids, val_ids = text_to_ids(text), None   # trop court : tout en entraînement
    if val_ids is not None and len(val_ids) < cfg.seq_len + 1:
        val_ids = None
    if len(train_ids) < cfg.seq_len + 2:
        raise ValueError(f"Corpus trop court : {len(train_ids)} octets (minimum {cfg.seq_len + 2}).")
    return train_ids, val_ids


def make_batch(ids_list, batch_size, seq_len, device):
    n = len(ids_list) - seq_len - 1
    starts = [random.randint(0, max(0, n)) for _ in range(batch_size)]
    batch  = torch.tensor([ids_list[s: s + seq_len] for s in starts], dtype=torch.long)
    if cfg.pin_memory:
        batch = batch.pin_memory()
    return batch.to(device, non_blocking=True)


# ─────────────────────────────────────────────
# 9. BOUCLE D'ENTRAÎNEMENT OPTIMISÉE
# ─────────────────────────────────────────────
def train(text=None, verbose=True, model=None):
    """Boucle d'entraînement. `model` permet de reprendre depuis des poids chargés.
    `text` : corpus perso (str) ; None = corpus d'exemple.
    Retourne le modèle ; model.history contient loss d'entraînement et de validation."""
    setup_backend()
    fix_heads(cfg, verbose=verbose)
    if verbose:
        cfg.print_config()

    fresh = model is None
    if fresh:
        model = FastBioTextTransformer()
    model = model.to(cfg.device)

    # Deux groupes de paramètres : les logits de connectivité (gates) reçoivent
    # un lr plus élevé et pas de weight decay.
    gate_params  = [layer.dcm.logits for layer in model.layers]
    gate_ids     = {id(p) for p in gate_params}
    other_params = [p for p in model.parameters() if id(p) not in gate_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {"params": gate_params,  "lr": cfg.lr * cfg.gate_lr_mult, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.mixed_precision else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_ids, val_ids = prepare_data(text)
    if verbose:
        print(f"\n  Paramètres : {n_params:,}")
        val_info = f", validation : {len(val_ids):,} octets tenus à part" if val_ids else ", pas de validation"
        print(f"  Entraînement : {len(train_ids):,} octets{val_info}")
        n_edges = [layer.dcm.n_active for layer in model.layers]
        print(f"  Arêtes initiales par couche : {n_edges}")
        print(f"  Loss de référence (aléatoire) : log({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}\n")

    history = {"step": [], "loss": [], "val_loss": [], "edges": []}
    running = []
    t0 = time.time()

    for step in range(1, cfg.train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        # Accumulation de gradients : grad_accum micro-batches par étape
        for _ in range(cfg.grad_accum):
            ids = make_batch(train_ids, cfg.batch_size, cfg.seq_len, cfg.device)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=cfg.amp_dtype):
                    loss = model.lm_loss(model(ids), ids) / cfg.grad_accum
                scaler.scale(loss).backward()
            else:
                loss = model.lm_loss(model(ids), ids) / cfg.grad_accum
                loss.backward()
            step_loss += loss.item()
            model.push_to_replay(ids)

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.plasticity.apply()     # Hebbian/STDP : après l'optimiseur (v3)

        running.append(step_loss)
        if step == 1 and verbose:
            hint = f" (attendu ≈ log({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f})" if fresh else " (reprise)"
            print(f"  Étape   1/{cfg.train_steps} | Loss={step_loss:.4f}{hint}")

        if step % 10 == 0:
            mean_loss = sum(running) / len(running)
            running = []
            val_loss = model.eval_loss(val_ids) if val_ids else float("nan")
            edges = [layer.dcm.n_active for layer in model.layers]
            gates = [f"{layer.dcm.active_gates().mean().item():.2f}" for layer in model.layers]
            history["step"].append(step)
            history["loss"].append(mean_loss)
            history["val_loss"].append(val_loss)
            history["edges"].append(edges)
            if verbose:
                val_txt = f" | Val={val_loss:.4f}" if val_ids else ""
                print(
                    f"  Étape {step:3d}/{cfg.train_steps} | "
                    f"Loss={mean_loss:.4f}{val_txt} | "
                    f"Arêtes={edges} | gate moy.={gates} | "
                    f"{(time.time() - t0) / step * 1000:.0f} ms/étape"
                )

        if cfg.use_sleep and step % cfg.sleep_every == 0:
            if verbose:
                print(f"\n  ─────── Sommeil (étape {step}) ───────")
            model.sleep(optimizer, scaler, val_ids=val_ids, verbose=verbose)
            if verbose:
                print()

    elapsed = time.time() - t0
    if verbose:
        print("\n" + "=" * 60)
        print(f"  ✓ Entraînement terminé en {elapsed:.1f} s "
              f"({elapsed / max(1, cfg.train_steps) * 1000:.0f} ms/étape)")
        print("=" * 60)

    # ── Génération ──────────────────────────────────────────
    if verbose:
        prompt = "Le cerveau"
        print("\n  Génération (greedy + pénalité de répétition) :")
        print("  " + model.generate(prompt, greedy=True).replace("\n", " "))
        print(f"\n  Génération (sampling T={cfg.gen_temperature}, top-k={cfg.gen_topk}) :")
        print("  " + model.generate(prompt).replace("\n", " "))
        print()

        print("  Statistiques finales :")
        for i, layer in enumerate(model.layers):
            dcm = layer.dcm
            g = dcm.active_gates()
            print(f"    Couche {i} | arêtes actives={dcm.n_active}/{cfg.max_edges} | "
                  f"gates>0.5={(g > 0.5).sum().item()} | "
                  f"gates min/moy/max={g.min().item():.2f}/{g.mean().item():.2f}/{g.max().item():.2f} | "
                  f"sparsité attn={layer.attn.gate.last_active_frac:.1%} "
                  f"ffn={layer.ffn.gate.last_active_frac:.1%}")
        if model.plasticity.H:
            hn = {k: f"{v.norm().item():.3f}" for k, v in model.plasticity.H.items()}
            print(f"    ||H|| hebbien par matrice : {hn}")
        if history["val_loss"] and val_ids:
            print(f"\n  Loss finale : train={history['loss'][-1]:.4f} | validation={history['val_loss'][-1]:.4f}")
            print("  (la validation porte sur du texte jamais vu : l'écart = mémorisation)")

    model.history = history
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n")
    model = train()
    print("\n  ✓ Modèle prêt ! Tu peux le sauver ou l'affiner.\n")

    path = model.save(cfg.save_path)
    size_mb = path.stat().st_size / 1e6
    print(f"  Poids sauvegardés dans : {path.resolve()} ({size_mb:.2f} Mo)\n")
