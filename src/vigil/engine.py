"""Closed-loop rollouts of the VIGIL network: training losses, validation statistics and
evaluation records; simulated polling trajectories for model-independent policies."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from vigil import policies
from vigil.data.dataset import CAFData
from vigil.models.vigil_net import VIGILNet
from vigil.sim import retrieval as R

SIGMA_OBS = 0.0125  # ~0.001 m3/m3 sensor resolution in normalised units


def context(data: CAFData, t: torch.Tensor) -> torch.Tensor:
    """Exogenous context per node [B,N,Fw+Fs+Fc] for hours t [B]."""
    B, N = t.shape[0], data.N
    w = data.w[t][:, None, :].expand(B, N, -1)
    s = data.s[None].expand(B, -1, -1)
    return torch.cat([w, s, data.crop[t]], dim=-1)


def ctx_dim(data: CAFData) -> int:
    return data.w.shape[1] + data.s.shape[1] + data.crop.shape[2]


MEM_DIM = 4 * 5 + 3


def beta_nll(mu, sigma, y, beta: float = 0.5):
    """beta-NLL of Seitzer et al. (2022); NaN targets are ignored. Returns (sum, count)."""
    m = torch.isfinite(y)
    yy = torch.nan_to_num(y)
    nll = 0.5 * ((yy - mu) ** 2 / sigma ** 2) + torch.log(sigma)
    if beta > 0:
        nll = nll * (sigma.detach() ** (2 * beta))
    nll = nll * m
    return nll.sum(), m.sum()


def mse_loss(mu, y):
    m = torch.isfinite(y)
    return (((torch.nan_to_num(y) - mu) ** 2) * m).sum(), m.sum()


def bce(logit, e):
    m = torch.isfinite(e)
    l = F.binary_cross_entropy_with_logits(logit, torch.nan_to_num(e), reduction="none") * m
    return l.sum(), m.sum()


@dataclass
class PolicySpec:
    """Per-row policy assignment for a batch of rollouts."""
    names: list            # one of 'voi', 'rr', 'random', 'ars', 'oracle' per row
    k: torch.Tensor        # [B] polls per hour
    eta: torch.Tensor      # [B] urgent/strategic weight for 'voi'
    max_age: float = 168.0
    tiebreak: torch.Tensor | None = None
    gen: torch.Generator | None = None
    kappa: torch.Tensor | float = 0.5   # shrinkage of logger-specific uncertainty growth (VoI)
    delta: torch.Tensor | None = None  # [D] event threshold in normalised units (VoI urgent term)


def choose(spec: PolicySpec, m: R.Memory, prior, truth: torch.Tensor) -> torch.Tensor:
    B, N = m.node_age.shape
    score = torch.zeros(B, N)
    names = np.array(spec.names)
    if (names == "rr").any():
        tb = spec.tiebreak if spec.tiebreak is not None else torch.zeros(B, N)
        score = torch.where(torch.as_tensor(names == "rr")[:, None], policies.rr_score(m, tb), score)
    if (names == "random").any():
        score = torch.where(torch.as_tensor(names == "random")[:, None], policies.random_score(m, spec.gen), score)
    if (names == "ars").any():
        score = torch.where(torch.as_tensor(names == "ars")[:, None], policies.ars_score(m), score)
    # 'voi' (proposed): Whittle index of the value of information, mixing strategic (look-ahead
    # uncertainty accumulated since the last poll, shrunk by kappa) and urgent (expected undetected
    # wetting increments) terms with weight eta. Ablations: 'voi_greedy' (no index),
    # 'voi_head' (urgent term from the event head p(1-p)), 'voi_myopic' (current-value variance).
    voi_rows = None
    for name in ("voi", "voi_greedy", "voi_head", "voi_myopic"):
        if not (names == name).any():
            continue
        if prior is None:
            sc = policies.random_score(m, spec.gen)
        else:
            mu, sg, logit, nu, sl = prior
            if name == "voi_myopic":
                # ablation: classic uncertainty sampling on the current-value variance (no look-ahead
                # head, no floor), which Proposition 1 predicts is attracted by transient sensor noise
                g_s = policies.state_growth(sg, torch.zeros_like(m.floor_l), m.avail, spec.kappa, m.node_age)
            else:
                g_s = policies.state_growth(sl, m.floor_l, m.avail, spec.kappa, m.node_age)
            if name == "voi_head":
                pe = torch.sigmoid(logit)
                g_u = (m.avail * (pe * (1 - pe) - m.floor_e).clamp(min=0.0)).sum(-1)
            else:
                g_u = policies.urgent_mass(mu, sg, m.last_x, spec.delta, m.avail)
            eta = torch.as_tensor(spec.eta, dtype=g_s.dtype)
            eta = eta[:, None] if eta.dim() == 1 else eta
            g = (1 - eta) * g_s / g_s.mean(1, keepdim=True).clamp(min=1e-12) \
                + eta * g_u / g_u.mean(1, keepdim=True).clamp(min=1e-12)
            if name == "voi_greedy":
                sc = g
            else:
                rows = torch.as_tensor(names == name)[:, None]
                m.cum_g = torch.where(rows, m.cum_g + g, m.cum_g)
                voi_rows = rows if voi_rows is None else (voi_rows | rows)
                sc = policies.whittle_index(g, m)
        score = torch.where(torch.as_tensor(names == name)[:, None], sc, score)
    if (names == "oracle").any():
        est = prior[0] if prior is not None else m.last_x
        score = torch.where(torch.as_tensor(names == "oracle")[:, None], policies.oracle_score(est, truth), score)
    poll = R.select_top_k(score, spec.k, m.node_age, spec.max_age)
    if voi_rows is not None:
        m.cum_g = torch.where(poll & voi_rows, torch.zeros_like(m.cum_g), m.cum_g)
    return poll


def simulate_inputs(data: CAFData, t_start: torch.Tensor, steps: int, warm: int, spec: PolicySpec):
    """Memory trajectories under a model-independent policy (rr / random / ars).

    Returns dict of tensors shaped [B, warm+steps, N, ...]: 'inp' (context + post-poll memory
    features), 'last_x', 'obs', 'x' (truth), 'ev' (event truth), 'poll'.
    """
    B, N = t_start.shape[0], data.N
    t_init = t_start - warm - 1
    m = R.init_memory(data.x[t_init], data.tmp[t_init])
    out = {k: [] for k in ["inp", "last_x", "obs", "x", "ev", "poll"]}
    for s in range(warm + steps):
        t = t_start - warm + s
        R.advance(m, data.prcp_mm[t], data.vpd_kpa[t])
        x_t = data.x[t]
        poll = choose(spec, m, None, x_t)
        obs = R.observe(m, poll, x_t, data.tmp[t])
        out["inp"].append(torch.cat([context(data, t), R.memory_features(m)], -1))
        out["last_x"].append(m.last_x)
        out["obs"].append(obs)
        out["x"].append(x_t)
        out["ev"].append(data.ev[t])
        out["poll"].append(poll)
    return {k: torch.stack(v, 1) for k, v in out.items()}


def input_stats(data: CAFData, n_windows: int = 64, steps: int = 168, warm: int = 72, seed: int = 12345):
    """Per-feature mean/std of the per-node inputs (context + memory) over simulated training windows
    (random and round-robin polling at mixed budgets). Used to standardise network inputs."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = warm + 26, data.split["train"][1] - steps - 1
    t0 = torch.randint(lo, hi, (n_windows,), generator=g)
    names = ["random" if i % 2 else "rr" for i in range(n_windows)]
    k = torch.as_tensor([[1, 2, 3, 4, 6, 8][i % 6] for i in range(n_windows)])
    spec = PolicySpec(names=names, k=k, eta=torch.full((n_windows,), 0.5),
                      tiebreak=torch.rand(n_windows, data.N, generator=g) * 1e-3, gen=g)
    with torch.no_grad():
        sim = simulate_inputs(data, t0, steps, warm, spec)
    f = sim["inp"].reshape(-1, sim["inp"].shape[-1])
    mean, std = f.mean(0), f.std(0)
    std = torch.where(std < 1e-3, torch.ones_like(std), std)   # binary/constant features stay unscaled
    return mean, std


@dataclass
class Record:
    """Per-step outputs kept during evaluation (numpy, float32/16)."""
    mu: list = field(default_factory=list)
    sigma: list = field(default_factory=list)
    p: list = field(default_factory=list)
    nu: list = field(default_factory=list)
    poll: list = field(default_factory=list)

    def stack(self) -> dict:
        return {"mu": np.stack(self.mu, 1), "sigma": np.stack(self.sigma, 1),
                "p": np.stack(self.p, 1), "nu": np.stack(self.nu, 1), "poll": np.stack(self.poll, 1)}


RISE_WINDOW = 24


def rise(last_x: torch.Tensor, hist: torch.Tensor, delta_norm: torch.Tensor) -> torch.Tensor:
    """Rise of the latest retrieved value above the trailing minimum of the network's own estimates,
    in units of the event threshold (hist: [B,N,D,W])."""
    mn = torch.minimum(hist.min(-1).values, last_x)
    return (last_x - mn) / delta_norm


def rollout(model: VIGILNet, data: CAFData, t_start: torch.Tensor, length: int, warm: int,
            spec: PolicySpec, train: bool = True, beta: float = 0.5, w_event: float = 1.0,
            hetero: bool = True, record: bool = False, grad: bool | None = None, w_look: float = 1.0):
    """Run ``warm`` hours without gradients (memory and state burn-in) then ``length`` hours.

    ``train``: compute losses and statistics (returns (loss, stats)); ``grad``: build the autograd
    graph (defaults to ``train``). With ``record`` a Record of evaluation outputs is returned.
    Besides the training losses, stats contain fixed (beta-independent) validation criteria:
    plain Gaussian NLL and MSE of the prior and of the posterior (un-polled channels).
    """
    grad = train if grad is None else grad
    if spec.delta is None:
        spec.delta = torch.as_tensor(data.delta_norm)
    B, N, D = t_start.shape[0], data.N, data.D
    t_init = t_start - warm - 1
    m = R.init_memory(data.x[t_init], data.tmp[t_init])
    h = model.init_state(B, N)
    sup = model.supports()
    prev_post = None
    dn = torch.as_tensor(data.delta_norm)
    hist = m.last_x[..., None].repeat(1, 1, 1, RISE_WINDOW)
    fresh = torch.isfinite(data.x[t_init])
    tot = {"nll_prior": 0.0, "nll_post": 0.0, "bce": 0.0, "pnll_prior": 0.0, "pnll_post": 0.0,
           "mse_prior": 0.0, "mse_post": 0.0, "look": 0.0}
    loss = torch.zeros(())
    cnt = {"prior": 0, "post": 0, "ev": 0}
    rec = Record() if record else None
    for s in range(warm + length):
        t = t_start - warm + s
        grad_on = grad and s >= warm
        loss_on = train and s >= warm
        if s == warm and grad:
            h = h.detach()
            sup = model.supports()
        with torch.set_grad_enabled(grad_on):
            ctx = context(data, t)
            R.advance(m, data.prcp_mm[t], data.vpd_kpa[t])
            x_t, tmp_t, e_t, u_t = data.x[t], data.tmp[t], data.ev[t], data.u[t]
            if model.predict_update:
                h_m, prior = model.prior(h, ctx, R.memory_features(m), m.last_x, rise(m.last_x, hist, dn), sup)
                prior_for_policy = tuple(v.detach() for v in prior)
            else:
                prior_for_policy = prev_post
            if prior_for_policy is not None:
                # residual-uncertainty floor: the (same) prior head one hour after a channel was polled
                pe = torch.sigmoid(prior_for_policy[2])
                m.floor_l = torch.where(fresh, prior_for_policy[4] ** 2, m.floor_l)
                m.floor_e = torch.where(fresh, pe * (1 - pe), m.floor_e)
            poll = choose(spec, m, prior_for_policy, x_t)
            obs = R.observe(m, poll, x_t, tmp_t)
            fresh = obs > 0
            r_post = rise(m.last_x, hist, dn)
            if model.predict_update:
                h, post = model.update(h_m, prior, obs, m.last_x * obs, m.last_tmp * obs, m.last_x, r_post, sup)
            else:
                h, post = model.prior(h, ctx, R.memory_features(m), m.last_x, r_post, sup)
            prev_post = tuple(v.detach() for v in post)
            mu_now = torch.where(obs > 0, m.last_x, post[0]).detach()
            hist = torch.cat([hist[..., 1:], mu_now[..., None]], -1)
            if loss_on:
                unobs_truth = torch.where(obs > 0, torch.full_like(x_t, float("nan")), x_t)
                if model.predict_update:
                    if hetero:
                        l1, n1 = beta_nll(prior[0], prior[1], x_t, beta)
                    else:
                        l1, n1 = mse_loss(prior[0], x_t)
                    loss = loss + l1 / max(int(n1), 1)
                    tot["nll_prior"] += float(l1.detach())
                    cnt["prior"] += int(n1)
                    with torch.no_grad():
                        tot["pnll_prior"] += float(beta_nll(prior[0], prior[1], x_t, 0.0)[0])
                        tot["mse_prior"] += float(mse_loss(prior[0], x_t)[0])
                if hetero:
                    l2, n2 = beta_nll(post[0], post[1], unobs_truth, beta)
                else:
                    l2, n2 = mse_loss(post[0], unobs_truth)
                loss = loss + l2 / max(int(n2), 1)
                tot["nll_post"] += float(l2.detach())
                cnt["post"] += int(n2)
                with torch.no_grad():
                    tot["pnll_post"] += float(beta_nll(post[0], post[1], unobs_truth, 0.0)[0])
                    tot["mse_post"] += float(mse_loss(post[0], unobs_truth)[0])
                if w_look > 0:
                    # look-ahead heads: mean level of the next L hours and its (reducible) uncertainty
                    if hetero:
                        l4, n4 = beta_nll(post[3], post[4], u_t, beta)
                        if model.predict_update:
                            l4p, _ = beta_nll(prior[3], prior[4], u_t, beta)
                            l4 = l4 + l4p
                    else:
                        l4, n4 = mse_loss(post[3], u_t)
                    loss = loss + w_look * l4 / max(int(n4), 1)
                    tot["look"] += float(l4.detach())
                if w_event > 0:
                    l3, n3 = bce(post[2], e_t)
                    l3p = torch.zeros(())
                    if model.predict_update:
                        l3p, _ = bce(prior[2], e_t)
                    loss = loss + w_event * (l3 + l3p) / max(int(n3), 1)
                    tot["bce"] += float(l3.detach())
                    cnt["ev"] += int(n3)
            if record and s >= warm:
                mu = torch.where(obs > 0, m.last_x, post[0])
                sg = torch.where(obs > 0, torch.full_like(post[1], SIGMA_OBS), post[1])
                rec.mu.append(mu.detach().numpy().astype(np.float32))
                rec.sigma.append(sg.detach().numpy().astype(np.float32))
                rec.p.append(torch.sigmoid(post[2]).detach().numpy().astype(np.float16))
                rec.nu.append(post[3].detach().numpy().astype(np.float32))
                rec.poll.append(poll.numpy())
    if record:
        return rec
    pairs = [("nll_prior", "prior"), ("nll_post", "post"), ("bce", "ev"), ("pnll_prior", "prior"),
             ("pnll_post", "post"), ("mse_prior", "prior"), ("mse_post", "post"), ("look", "prior")]
    stats = {k: tot[k] / max(cnt[c], 1) for k, c in pairs}
    return loss / max(length, 1), stats


def selection_score(stats: dict, mcfg: dict) -> float:
    """Model-selection criterion on validation windows: MSE of the posterior field-state estimate on
    un-polled channels (the same criterion as for every baseline)."""
    return stats["mse_post"]
