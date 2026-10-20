"""Adaptive firing for variable-length sparse codec.

Computes a firing mask over the fixed-rate latent sequence h_t (after fc_prior, before
quantization); only fired positions are passed to the decoder as quantized tokens (sparse
decoding). Tokens at non-fired positions are never stored anywhere: the decoder reconstructs
the waveform from the fired tokens plus relative positions (durations) only.

Firing modes (mode):
- "adaptive":    adaptive firing driven by the firing signal (per-sequence normalized sum of the
            components listed in `signals`). Signal components: hnet (cosine dissimilarity of
            adjacent frames) /
            hnet_ms (multi-scale variant of hnet over wider frame offsets) / flux (log-mel
            flux) / distill (distilled decoder reconstruction difficulty)
- "stride": firing at a fixed interval (fixed-rate baseline)

Firing decision (decision):
- "clock" (default): deterministic, gradient-free budget allocation
  (water-filling over the signal). With head_in_budget=True (the default)
  the firing count is exactly round(target_ratio * T) per sequence
  (Python round: half to even), so no rate loss is needed.
- "level": the soft probability p_t = sigmoid((s_t - tau_q) / (scale_q * temp)) is converted to a
  hard mask via straight-through during training. tau_q is a per-sequence quantile of
  the signal.
The first frame always fires (it has no context).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.normalization import ChannelRMSNorm

_MEL_CACHE: dict = {}


def mel_flux(wav: torch.Tensor, sr: int = 16000, hop: int = 320, n_mels: int = 80):
    """Input waveform (B, T_samples) -> log-mel flux (B, T_frames), for the "flux" firing signal.

    Aligned to the 50 Hz latent grid (hop = sr / latent_rate). Computed in fp32 (unaffected by
    autocast). Notably, it is computed deterministically from the raw waveform, independent of
    the encoder latents.
    """
    import torchaudio

    key = (sr, hop, n_mels, wav.device.type, wav.device.index)
    if key not in _MEL_CACHE:
        _MEL_CACHE[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=hop, n_mels=n_mels
        ).to(wav.device)
    with torch.no_grad(), torch.autocast(wav.device.type, enabled=False):
        m = torch.log(_MEL_CACHE[key](wav.float()) + 1e-5)  # (B, n_mels, T')
        fl = torch.zeros(m.shape[0], m.shape[2], device=wav.device)
        fl[:, 1:] = (m[:, :, 1:] - m[:, :, :-1]).abs().mean(dim=1)
        # Remove the artifact where reflect padding (center=True) spuriously inflates the flux
        # over the last ~n_fft/2 samples.
        fl[:, -2:] = 0.0
    return fl


class DifficultyHead(nn.Module):
    """Small causal conv head predicting per-frame decoder reconstruction difficulty
    (distillation student).

    The head uses the current and preceding frames with left-padded convolutions.
    The caller detaches its input to prevent this loss from updating the encoder.
    """

    def __init__(self, dim: int, hidden: int = 256, kernel: int = 3, layers: int = 3):
        super().__init__()
        self.in_proj = nn.Conv1d(dim, hidden, 1)
        self.convs = nn.ModuleList(
            nn.ModuleDict(
                {
                    "conv": nn.Conv1d(hidden, hidden, kernel, dilation=2**i),
                    "norm": ChannelRMSNorm(hidden),
                }
            )
            for i in range(layers)
        )
        self.kernel = kernel
        self.out_proj = nn.Conv1d(hidden, 1, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(h)
        for blk in self.convs:
            pad = blk["conv"].dilation[0] * (self.kernel - 1)
            y = blk["conv"](F.pad(x, (pad, 0)))
            y = blk["norm"](y)
            x = x + F.silu(y)
        return F.softplus(self.out_proj(x)).squeeze(1)  # (B, T) ≥ 0


class FiringModule(nn.Module):
    """Computes the firing mask and composes the decoder input for sparse decoding."""

    def __init__(
        self,
        dim: int,
        mode: str = "adaptive",
        target_ratio: float = 0.25,
        temp: float = 1.0,
        decision: str = "clock",
        signals=None,
        head_in_budget: bool = True,
        max_gap: int = 64,
        hnet_ms_offsets=((1, 0), (1, 1), (2, 1)),
        grid_compose: str = "empty",
    ):
        super().__init__()
        assert mode in ("adaptive", "stride")
        assert decision in ("level", "clock")
        assert grid_compose in ("empty", "hold")
        self.grid_compose = grid_compose
        self.mode = mode
        self.target_ratio = float(target_ratio)
        self.temp = float(temp)
        self.decision = decision
        # Whether the forced first firing is counted inside the clock budget.
        # True: K - 1 threshold firings over t >= 1 plus the first frame give
        # exactly K = round(target_ratio * T) tokens per utterance.
        # False allocates the full budget over all frames (~K + 1 tokens with
        # the forced first firing).
        self.head_in_budget = bool(head_in_budget)
        self.max_gap = int(max_gap)

        # List of firing signals (each component is >= 0; components are per-sequence
        # mean-normalized and summed). "stride" uses no signals. For "adaptive" the default
        # when unspecified is ["hnet_ms", "flux"]
        if signals is None:
            signals = [] if mode == "stride" else ["hnet_ms", "flux"]
        self.signals = [str(x) for x in signals]
        allowed = {"hnet", "hnet_ms", "flux", "distill"}
        assert set(self.signals) <= allowed, f"unknown signals: {self.signals}"
        assert mode == "stride" or len(self.signals) > 0, (
            "signals is empty (only allowed for mode='stride')"
        )
        self.hnet_ms_offsets = [tuple(int(v) for v in o) for o in hnet_ms_offsets]

        # distill: distillation head for decoder reconstruction difficulty (the teacher is
        # computed on the training-loop side)
        self.difficulty = None
        if "distill" in self.signals:
            self.difficulty = DifficultyHead(dim)

        # Sparse-decoding side: placeholder for non-fired positions and relative-position embedding
        self.empty_emb = nn.Parameter(torch.zeros(dim))
        self.relpos_emb = nn.Embedding(self.max_gap + 1, dim)
        nn.init.normal_(self.relpos_emb.weight, std=0.02)

    def _norm(self, c: torch.Tensor) -> torch.Tensor:
        """Per-sequence mean normalization of one signal component (components are assumed
        non-negative)."""
        return c / c.mean(dim=1, keepdim=True).clamp_min(1e-8)

    def _sig_hnet(self, h: torch.Tensor, past: int, future: int) -> torch.Tensor:
        """δ_{past,future}(t) = 1 − cos(h_{t−past}, h_{t+future}); zero at the sequence edges."""
        B, D, T = h.shape
        a = F.pad(h, (past, 0))[:, :, :T]  # h_{t−past}
        b = F.pad(h, (0, future))[:, :, future:]  # h_{t+future}
        s = 1.0 - F.cosine_similarity(a, b, dim=1)
        if past > 0:
            s[:, :past] = 0.0
        if future > 0:
            s[:, -future:] = 0.0
        return s.clamp_min(0)

    def forward(self, h: torch.Tensor, flux: torch.Tensor | None = None) -> dict:
        """Allocate event starts on the latent frame grid.

        Args:
            h: Pre-quantization latents shaped (B, D, T).
            flux: Input mel flux shaped (B, T_flux), required when signals
                includes "flux". Compute it with mel_flux before calling.

        Returns:
            mask:      (B, T) hard {0,1}; carries a straight-through soft gradient only in
                       training with decision="level", otherwise no gradient
            mask_hard: (B, T) bool (no gradient)
            p:         (B, T) soft firing probability (identical to hard in stride mode and
                       for decision="clock")
            s:         (B, T) firing signal (detached; for visualization)
            d:         (B, T) difficulty-head output (only with the distill component;
                       carries gradient, consumed by distill_loss)
            rate:      scalar (measured hard firing rate)
        """
        B, D, T = h.shape
        device = h.device
        d = None

        if self.mode == "stride":
            k = max(1, round(1.0 / self.target_ratio))
            fire = torch.zeros(B, T, device=device, dtype=h.dtype)
            fire[:, ::k] = 1.0
            p = fire
            mask = fire
        else:
            comps = []
            for name in self.signals:
                if name == "hnet":
                    comps.append(self._norm(self._sig_hnet(h, 1, 0)))
                elif name == "hnet_ms":
                    # Multi-scale similarity: captures slow transitions (e.g. glides) with wider spans
                    ms = sum(self._norm(self._sig_hnet(h, a, b)) for a, b in self.hnet_ms_offsets)
                    comps.append(self._norm(ms))
                elif name == "flux":
                    assert flux is not None, (
                        "the flux argument is required when signals includes 'flux'"
                    )
                    fl = flux[:, :T]
                    if fl.shape[1] < T:
                        fl = F.pad(fl, (0, T - fl.shape[1]))
                    comps.append(self._norm(fl.to(h.dtype).clamp_min(0)))
                elif name == "distill":
                    d = self.difficulty(h.detach())
                    # The component passed to decide is detached: this prevents the head from
                    # acquiring a second objective via reconstruction/rate gradients through
                    # the level-mode straight-through path. The head is trained only by
                    # distill_loss (on the raw value in fire["d"])
                    comps.append(self._norm(d).detach())
            s = sum(comps)

            p, hard = self.decide(s, h.dtype)
            # The first frame always fires
            hard = hard.clone()
            hard[:, 0] = 1.0
            if self.training and self.decision == "level":
                # straight-through: forward ≈ hard (off by ~±1e-7 due to floating-point
                # non-associativity, harmless as an ST approximation), backward = ∂p.
                # clock carries no mask gradient (the encoder learns only through
                # reconstruction gradients of the fired content)
                mask = hard + p - p.detach()
            else:
                # Eval/inference uses strictly hard {0,1} — guarantees zero information
                # leakage into non-fired positions
                mask = hard

        mask_hard = mask.detach() > 0.5
        mask_hard[:, 0] = True
        rate = mask_hard.float().mean()

        if (
            self.mode != "stride"
            and getattr(self, "difficulty", None) is not None
            and d is not None
        ):
            return {
                "mask": mask,
                "mask_hard": mask_hard,
                "p": p,
                "s": s.detach(),
                "d": d,  # for the distillation loss (carries gradient)
                "rate": rate,
            }
        return {
            "mask": mask,
            "mask_hard": mask_hard,
            "p": p,
            "s": s.detach()
            if self.mode != "stride"
            else p.detach(),  # firing signal for visualization
            "rate": rate,
        }

    def distill_loss(self, fire: dict, frame_err: torch.Tensor) -> torch.Tensor:
        """Distillation loss for the difficulty head. Teacher = per-frame reconstruction error
        (assumed already detached).

        frame_err: (B, T') — log-mel error computed by the training loop. Its length can differ
        from the latent T by a few frames, so the two are aligned with min. Regresses the
        softplus-activated d onto the per-sequence mean-normalized teacher. Gradients flow only
        into the difficulty head.
        """
        d = fire.get("d")
        if d is None:
            return frame_err.new_zeros(())
        T = min(d.shape[1], frame_err.shape[1])
        t = frame_err[:, :T].detach().float()
        t = t / t.mean(dim=1, keepdim=True).clamp_min(1e-8)
        return F.mse_loss(d[:, :T].float(), t)

    def _waterfill_delta(self, sf: torch.Tensor, K: int, T: int) -> torch.Tensor:
        """Allocate a budget of K firings over the signal sf (B, T) by water-filling.

        Bounded increments sum to K where feasible. Near-zero signals use
        uniform increments.
        """
        ssum = sf.sum(dim=1, keepdim=True)
        uniform = torch.full_like(sf, K / T)
        if K > 0:
            capt = torch.ones_like(sf)
            lo = torch.zeros(sf.shape[0], 1, device=sf.device)
            hi = torch.full_like(lo, 1.0)
            for _ in range(60):
                need = torch.minimum(hi * sf, capt).sum(1, keepdim=True) < K
                if not need.any():
                    break
                hi = torch.where(need, hi * 2.0, hi)
            for _ in range(40):
                mid = (lo + hi) / 2
                f = torch.minimum(mid * sf, capt).sum(1, keepdim=True)
                lo = torch.where(f < K, mid, lo)
                hi = torch.where(f < K, hi, mid)
            delta = torch.minimum(hi * sf, capt)
        else:
            delta = torch.zeros_like(sf)
        return torch.where(ssum > 1e-8, delta, uniform)

    def decide(self, s: torch.Tensor, dtype=torch.float32):
        """Firing signal s (B, T) -> (p, hard). The decision rule is self.decision:

        - "level": quantile threshold on the raw level of s (tends to fire in
          bursts while a spike persists)
        - "clock": deterministic budget allocation. Water-filling converts the signal
          into bounded per-frame increments; with head_in_budget=True the first frame
          plus K - 1 half-integer-threshold firings over t >= 1 give exactly
          K = round(target_ratio * T) firings per sequence (head_in_budget=False
          allocates the full budget over all frames by integer crossings, ~K + 1
          with the forced first firing). A spike yields a single firing and low-signal regions still
          fire periodically through accumulation (coverage guarantee)
        clock is a deterministic decision without gradient.
        """
        B, T = s.shape
        if self.decision == "clock":
            with torch.no_grad():
                sf = s.float().clamp_min(0)
                # Python round() rounds half to even (e.g. T=50 gives K=12 at ratio 0.25).
                K = max(1, min(round(self.target_ratio * T), T))
                if not self.head_in_budget:
                    # Budget K over all T frames, firing wherever the cumulative
                    # sum crosses an integer; with the forced first firing the
                    # effective count is ~K + 1.
                    delta = self._waterfill_delta(sf, K, T)
                    c = delta.cumsum(dim=1)
                    c_prev = F.pad(c, (1, 0))[:, :-1]
                    hard = (c.floor() > c_prev.floor()).to(dtype)
                    return hard, hard
                # The forced first firing is part of the budget: allocate K - 1 over
                # t >= 1 and fire at the first index whose cumulative sum reaches each
                # half-integer threshold j - 0.5 (j = 1..K-1). The count is exact by
                # construction: there are K - 1 thresholds; delta <= 1 keeps adjacent
                # thresholds (spacing 1) on distinct frames except for exact ties (a
                # saturated increment delta = 1 whose cumulative sum lands exactly on
                # a threshold), which the monotone repair below resolves; and the 0.5
                # margin between the largest threshold and sum(delta) ~ K - 1 makes
                # every threshold reachable despite floating-point rounding of the
                # cumulative sum.
                Kb = K - 1
                head = torch.ones(sf.shape[0], 1, device=sf.device)
                if Kb > 0 and T > 1:
                    delta = self._waterfill_delta(sf[:, 1:], Kb, T - 1)
                    c = delta.cumsum(dim=1)
                    th = (torch.arange(1, Kb + 1, device=c.device, dtype=c.dtype) - 0.5).expand(
                        c.shape[0], Kb
                    )
                    pos = torch.searchsorted(c.contiguous(), th.contiguous())
                    # Monotone repair: force strictly increasing positions (a no-op
                    # except on exact ties), then cap from the end so the last
                    # position stays in range. A strictly increasing assignment
                    # always exists since K - 1 <= T - 1.
                    jj = torch.arange(Kb, device=c.device)
                    pos = torch.cummax(pos - jj, dim=1).values + jj
                    pos = torch.minimum(pos, c.shape[1] - Kb + jj)
                    tail = torch.zeros_like(sf[:, 1:])
                    tail.scatter_(1, pos, 1.0)
                else:
                    tail = torch.zeros_like(sf[:, 1:])
                hard = torch.cat([head, tail], dim=1).to(dtype)
            return hard, hard
        # "level"
        sf = s.float()
        tau_q = torch.quantile(sf.detach(), 1.0 - self.target_ratio, dim=1, keepdim=True)
        scale_q = sf.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
        p = torch.sigmoid((sf - tau_q) / (scale_q * self.temp)).to(dtype)
        hard = (p > 0.5).to(dtype)
        return p, hard

    def grid_distances(self, mask_hard: torch.Tensor):
        """Per-frame distance to the most recent firing and segment id (integer arithmetic,
        deterministic).

        Returns:
            dist: (B, T) long — t − position of the most recent firing (0 at fired positions,
                  clamped to max_gap)
            sid:  (B, T) long — 0-based id of the segment the frame belongs to (firing order)
        """
        B, T = mask_hard.shape
        idx = torch.arange(T, device=mask_hard.device).unsqueeze(0).expand(B, T)
        fired_idx = torch.where(mask_hard, idx, torch.zeros_like(idx))
        last_fire = torch.cummax(fired_idx, dim=1).values
        dist = (idx - last_fire).clamp(0, self.max_gap)
        sid = mask_hard.long().cumsum(dim=1) - 1
        return dist, sid

    def compose(self, q: torch.Tensor, fire: dict) -> torch.Tensor:
        """Compose the decoder input from the quantized latents q (B, D, T).

        Fired positions: q_t + relpos(0). Non-fired positions depend on grid_compose:
        - "empty" (default): empty_emb + relpos(distance to the most recent firing)
        - "hold": hold the fired token q of the containing segment
          + relpos(distance)
        Neither variant ever uses q at non-fired positions themselves (hold reads the q of
        the most recent fired position). The coding constraint of sparse decoding — the only
        information is the fired tokens plus relative positions — is identical for both.
        """
        B, D, T = q.shape
        mask = fire["mask"]  # (B, T), forward hard / backward soft
        mask_hard = fire["mask_hard"]  # (B, T) bool

        dist, _ = self.grid_distances(mask_hard)
        rel = self.relpos_emb(dist).transpose(1, 2).to(q.dtype)  # (B, D, T)

        m = mask.unsqueeze(1).to(q.dtype)  # (B, 1, T)
        if self.grid_compose == "hold":
            idx = torch.arange(T, device=q.device).unsqueeze(0).expand(B, T)
            fired_idx = torch.where(mask_hard, idx, torch.zeros_like(idx))
            last_fire = torch.cummax(fired_idx, dim=1).values  # (B, T)
            q_hold = q.gather(2, last_fire.unsqueeze(1).expand(B, D, T))
            return m * q + (1.0 - m) * q_hold + rel
        empty = self.empty_emb.view(1, D, 1).to(q.dtype)
        return m * q + (1.0 - m) * empty + rel

    def compose_stream(
        self,
        slots: torch.Tensor,
        durs: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Streaming version of compose: incrementally composes grid frames from (token, dur)
        chunks.

        dur form: as soon as token i (with dur d_i) arrives, all frames of segment i =
        [pos_i, pos_i + d_i) (the fired frame plus d_i−1 non-fired frames) are emitted as
        final. No lookahead for the next token and no special case for the trailing
        segment are needed. Returns values identical to the batch compose.

        Args:
            slots: (N, D) — quantized slot vectors (in firing order). N=0 is allowed
            durs: (N,) long — length of each segment (the last segment extends to the end of
                the grid)
            state: inter-chunk state (not needed for composition in the dur form; passthrough
                kept for interface compatibility)

        Returns:
            frames: (1, T_chunk, D)
            state: updated state
        """
        if state is None:
            state = {}
        rel = self.relpos_emb.weight  # (max_gap+1, D)
        outs = []
        for i in range(slots.shape[0]):
            d = int(durs[i])
            assert d >= 1, f"dur must be >= 1 (got {d})"
            outs.append(slots[i] + rel[0])
            for k in range(1, d):
                kk = min(k, self.max_gap)
                if self.grid_compose == "hold":
                    outs.append(slots[i] + rel[kk])
                else:
                    outs.append(self.empty_emb + rel[kk])

        if outs:
            frames = torch.stack(outs, dim=0).unsqueeze(0)
        else:
            D = self.relpos_emb.weight.shape[1]
            frames = torch.zeros(1, 0, D, device=rel.device, dtype=rel.dtype)
        return frames, state

    def segments(self, h: torch.Tensor, fire: dict) -> dict:
        """Compute the segment structure from the firing mask (shared preprocessing for
        aggregation).

        segment i = [pos_i, pos_{i+1}) (opening convention). Since the first frame always
        fires, every frame belongs to exactly one segment.

        Returns:
            sid:        (B, T) long — segment id of each frame
            dist:       (B, T) long — distance to the most recent firing (clamped to max_gap)
            anchor_pos: (B, N) long — firing position of each slot (0 for pad slots)
            dur:        (B, N) long — length of each segment (0 for pad slots)
            gaps:       (B, N) long — distance to the previous firing (0 for the first slot
                        and pad slots)
            slot_mask:  (B, N) bool — valid slots
            mean:       (B, D, N) — per-segment mean (aggregated in float32, cast to h.dtype)
            anchors:    (B, D, N) — h at the anchor positions
        """
        B, D, T = h.shape
        mask = fire["mask_hard"]
        device = h.device
        dist, sid = self.grid_distances(mask)
        n = mask.sum(dim=1)  # (B,)
        N = int(n.max().item())

        # Per-segment mean (scatter-aggregated in float32 -> safe under bf16)
        idx_d = sid.unsqueeze(1).expand(B, D, T)
        sums = torch.zeros(B, D, N, device=device, dtype=torch.float32)
        sums.scatter_add_(2, idx_d, h.float())
        cnt = torch.zeros(B, 1, N, device=device, dtype=torch.float32)
        cnt.scatter_add_(2, sid.unsqueeze(1), torch.ones(B, 1, T, device=device))
        mean = (sums / cnt.clamp_min(1.0)).to(h.dtype)

        # Anchor positions: write fired frame t into slot sid(t) (non-fired writes into
        # slot 0 carry the value 0, which is slot 0's correct anchor, so they are harmless)
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        anchor_pos = torch.zeros(B, N, dtype=torch.long, device=device)
        anchor_pos.scatter_(
            1,
            torch.where(mask, sid, torch.zeros_like(sid)),
            torch.where(mask, idx, torch.zeros_like(idx)),
        )
        slot_mask = torch.arange(N, device=device).unsqueeze(0) < n.unsqueeze(1)
        dur = cnt.squeeze(1).long() * slot_mask
        gaps = anchor_pos - F.pad(anchor_pos, (1, 0))[:, :-1]
        gaps[:, 0] = 0
        gaps = gaps.clamp_min(0) * slot_mask

        anchors = h.gather(2, anchor_pos.unsqueeze(1).expand(B, D, N))
        return {
            "sid": sid,
            "dist": dist,
            "anchor_pos": anchor_pos,
            "dur": dur,
            "gaps": gaps,
            "slot_mask": slot_mask,
            "mean": mean,
            "anchors": anchors,
        }

    @torch.no_grad()
    def extract_codes(self, vq_code: torch.Tensor, fire: dict):
        """Extract the per-utterance (token, dur) sequence for a downstream LM.

        vq_code: (B, Q, T) (Q=1 for a single stage, Q=num_quantizers for RVQ).
        returns: list of (tokens, durs) — tokens is (N,) for Q=1, (Q, N) for Q>1.
        dur_i = length of segment i = [pos_i, pos_{i+1}) (the last segment extends to the end
        of the grid). Σdurs = T. Assumes the first frame always fires (pos[0] == 0) and raises
        on violation (dur serialization cannot represent a leading offset, so it must not
        silently shift the timeline).
        """
        out = []
        mask_hard = fire["mask_hard"]
        T = int(vq_code.shape[2])
        for b in range(mask_hard.shape[0]):
            pos = mask_hard[b].nonzero(as_tuple=True)[0]
            tokens = vq_code[b, :, pos]  # (Q, N)
            if vq_code.shape[1] == 1:
                tokens = tokens[0]  # backward-compatible shape: (N,)
            if pos.numel() > 0 and int(pos[0]) != 0:
                raise ValueError(
                    f"first firing at frame {int(pos[0])} != 0 "
                    "(dur serialization requires pos[0] == 0)"
                )
            durs = torch.diff(pos, append=pos.new_tensor([T]))
            out.append((tokens, durs))
        return out

    @torch.no_grad()
    def extract_codes_slots(self, vq_code_slots: torch.Tensor, seg: dict):
        """(token, dur) extraction for the aggregation mode.

        vq_code_slots: (B, Q, N) — quantization indices of the slot sequence (firing order).
        returns: list of (tokens, durs) — tokens has the same shape as in extract_codes.
        durs is the dur from segments() (= segment lengths, Σdurs = T).
        """
        out = []
        for b in range(vq_code_slots.shape[0]):
            n = int(seg["slot_mask"][b].sum())
            tokens = vq_code_slots[b, :, :n]  # (Q, n)
            if vq_code_slots.shape[1] == 1:
                tokens = tokens[0]  # backward-compatible shape: (n,)
            if n > 0 and int(seg["anchor_pos"][b, 0]) != 0:
                raise ValueError(
                    f"first firing at frame {int(seg['anchor_pos'][b, 0])} != 0 "
                    "(dur serialization requires pos[0] == 0)"
                )
            out.append((tokens, seg["dur"][b, :n]))
        return out


class SegmentAggregator(nn.Module):
    """Pre-quantization interval compression: interval aggregation of segments anchored at firing
    positions.

    g_i = anchors_i + proj([anchor ; pool(segment) ; emb_dur(len)])
    pool is mode="mean" (segment mean) / "attn" (attention pooling with the anchor as query).
    The output g (B, D, N) is passed to the residual quantizer as-is (i.e. a token becomes
    an interval summary rather than a point sample).
    """

    def __init__(self, dim: int, mode: str = "mean", max_gap: int = 64):
        super().__init__()
        assert mode == "mean"
        self.mode = mode
        self.max_gap = int(max_gap)
        self.emb_dur = nn.Embedding(self.max_gap + 1, dim)
        nn.init.normal_(self.emb_dur.weight, std=0.02)
        self.proj = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor, seg: dict) -> torch.Tensor:
        """Return the aggregated vectors g (B, D, N) from h (B, D, T) and the output of segments()."""
        pool = seg["mean"]
        dur_e = self.emb_dur(seg["dur"].clamp(0, self.max_gap)).transpose(1, 2)
        feat = torch.cat([seg["anchors"], pool, dur_e.to(h.dtype)], dim=1)  # (B, 3D, N)
        g = seg["anchors"] + self.proj(feat.transpose(1, 2)).transpose(1, 2)
        return g * seg["slot_mask"].unsqueeze(1).to(h.dtype)
