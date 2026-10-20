import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.functional as F_audio
import wandb
from omegaconf import OmegaConf, open_dict
from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn

from rubatocodec.firing import FiringModule, SegmentAggregator
from rubatocodec.modules.decoder import CodecDecoderVocos
from rubatocodec.modules.encoder import CodecEncoder
from rubatocodec.modules.semantic import SemanticDecoder, SemanticEncoder, pool_semantic_features
from training.criterions import (
    GANLoss,
    MultiResolutionMelSpectrogramLoss,
    PerceptualLoss,
)
from training.module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from training.schedulers import build_warmup_decay_scheduler


def strip_secrets(cfg):
    """Return a copy of the training config without dataset access tokens."""
    cfg = copy.deepcopy(cfg)
    for split in ("train", "val"):
        for spec in OmegaConf.select(cfg, f"dataset.{split}.hf_datasets") or []:
            with open_dict(spec):
                spec.pop("token", None)
    return cfg


class CodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.construct_criteria()
        # Checkpoints keep the config (without secrets); train.py logs it to wandb.
        self.save_hyperparameters({"cfg": strip_secrets(cfg)}, logger=False)
        # Freeze the encoder side (CodecEnc/SemanticEncoder/fc_prior) to remove
        # non-stationarity of firing positions (the input h of the firing signal s
        # becomes fixed per utterance)
        if self.cfg.train.get("freeze_encoder", False):
            for module in (self.CodecEnc, self.SemanticEncoder_module, self.fc_prior):
                for p in module.parameters():
                    p.requires_grad = False
            print("freeze_encoder: froze CodecEnc / SemanticEncoder_module / fc_prior")
        self.automatic_optimization = False
        self._val_pesq_scores = []
        self._val_stoi_scores = []

    def construct_model(self):
        enccfg = self.cfg.model.codec_encoder

        self.CodecEnc = CodecEncoder(
            ngf=enccfg.ngf,
            up_ratios=enccfg.up_ratios,
            dilations=enccfg.dilations,
            hidden_dim=enccfg["hidden_dim"],
            activation=enccfg.get("activation", "snake_beta"),
        )

        deccfg = self.cfg.model.codec_decoder

        self.generator = CodecDecoderVocos(
            hidden_dim=deccfg.hidden_dim,
            depth=deccfg.depth,
            heads=deccfg.heads,
            pos_emb_dim=deccfg.pos_emb_dim,
            hop_length=self.CodecEnc.hop_length,
            vq_dim=deccfg.vq_dim,
            upsampler_factors=deccfg.get("upsampler_factors", None),
            upsampler_kernel_sizes=deccfg.get("upsampler_kernel_sizes", None),
            output_n_fft=deccfg.get("output_n_fft", None),
            output_hop_length=deccfg.get("output_hop_length", None),
            activation=deccfg.get("activation", "mixed"),
            upsampler_activation=deccfg.get("upsampler_activation", None),
            fsq_levels=deccfg.get("fsq_levels", None),
            fsq_num_quantizers=deccfg.get("fsq_num_quantizers", 1),
            dropout=deccfg.get("dropout", 0.1),
            mlp_expansion=deccfg.get("mlp_expansion", 4),
            attn_dropout=deccfg.get("attn_dropout", 0.0),
            padding_mode=deccfg.get("conv_padding_mode", "symmetric"),
            resnet_kernel_size=deccfg.get("resnet_kernel_size", 3),
            embed_kernel_size=deccfg.get("embed_kernel_size", 7),
            quantizer_cfg=deccfg.get("quantizer", None),
            attn_causal=deccfg.get("attn_causal", True),
        )

        mpdcfg = self.cfg.model.mpd
        self.discriminator = HiFiGANMultiPeriodDiscriminator(
            periods=mpdcfg.periods,
            max_downsample_channels=mpdcfg.max_downsample_channels,
            channels=mpdcfg.channels,
            channel_increasing_factor=mpdcfg.channel_increasing_factor,
        )

        mstftcfg = self.cfg.model.mstft
        self.spec_discriminator = SpecDiscriminator(
            stft_params=mstftcfg.stft_params,
            in_channels=mstftcfg.in_channels,
            out_channels=mstftcfg.out_channels,
            kernel_sizes=mstftcfg.kernel_sizes,
            channels=mstftcfg.channels,
            max_downsample_channels=mstftcfg.max_downsample_channels,
            downsample_scales=mstftcfg.downsample_scales,
            use_weight_norm=mstftcfg.use_weight_norm,
        )

        # Load semantic model (frozen, for semantic loss target)
        semantic_model_name = getattr(self.cfg.train, "semantic_model", "facebook/w2v-bert-2.0")
        self.semantic_layer_index = getattr(self.cfg.train, "semantic_layer_index", 16)

        from transformers import Wav2Vec2BertModel

        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            semantic_model_name, output_hidden_states=True
        )

        self.semantic_model.eval()
        self.semantic_model.requires_grad_(False)

        semantic_dim = self.semantic_model.config.hidden_size
        self.semantic_dim = semantic_dim

        self.fc_prior = nn.Linear(semantic_dim + enccfg["hidden_dim"], deccfg.vq_dim)
        self.fc_post_a = nn.Linear(deccfg.vq_dim, deccfg.hidden_dim)
        self.fc_post_s = nn.Linear(deccfg.vq_dim, semantic_dim)

        self.SemanticDecoder_module = SemanticDecoder(semantic_dim, semantic_dim, semantic_dim)
        self.SemanticEncoder_module = SemanticEncoder(semantic_dim, semantic_dim, semantic_dim)

        # Adaptive firing (variable-frame-rate sparse codec). Enabled via model.firing.enabled
        self.firing = None
        self.firing_agg = None
        fcfg = self.cfg.model.get("firing", None)
        if fcfg is not None and fcfg.get("enabled", False):
            max_gap = fcfg.get("max_gap", 64)
            self.firing = FiringModule(
                dim=deccfg.vq_dim,
                mode=fcfg.get("mode", "adaptive"),
                target_ratio=fcfg.get("target_ratio", 0.25),
                temp=fcfg.get("temp", 1.0),
                decision=fcfg.get("decision", "clock"),
                signals=fcfg.get("signals", None),
                head_in_budget=fcfg.get("head_in_budget", True),
                hnet_ms_offsets=fcfg.get("hnet_ms_offsets", ((1, 0), (1, 1), (2, 1))),
                max_gap=max_gap,
                grid_compose=fcfg.get("grid_compose", "empty"),
            )
            # Pre-quantization segment aggregation (shared by both decoder modes).
            # none = point selection
            agg_mode = fcfg.get("aggregate", "none")
            if agg_mode != "none":
                self.firing_agg = SegmentAggregator(
                    dim=deccfg.vq_dim,
                    mode=agg_mode,
                    max_gap=max_gap,
                )
            # Mel transform for the distillation teacher (per-frame log-mel error).
            # Choose a hop that matches the latent rate (sr/hop) (50 Hz: 44100/50=882).
            # The attribute name starts with firing_ so it passes the init_ckpt
            # missing-key check (firing prefix)
            if "distill" in self.firing.signals:
                out_sr = getattr(self.cfg.dataset.audio, "output_sr", self.cfg.dataset.audio.sr)
                latent_rate = self.cfg.dataset.audio.sr / self.CodecEnc.hop_length
                self.firing_distill_mel = torchaudio.transforms.MelSpectrogram(
                    sample_rate=out_sr,
                    n_fft=2048,
                    hop_length=int(round(out_sr / latent_rate)),
                    n_mels=80,
                )
        # Perceptual loss model (only loaded when phase 3 is used)
        schedule = getattr(self.cfg.train, "phase_schedule", None)
        if schedule is not None and schedule.phase3_ratio > 0:
            model_name = getattr(self.cfg.train, "perceptual_model", "facebook/wav2vec2-xls-r-300m")
            layers = getattr(self.cfg.train, "perceptual_layers", None)
            if layers is not None:
                layers = list(layers)
            self.perceptual_loss_fn = PerceptualLoss(model_name=model_name, layers=layers)
            output_sr = getattr(self.cfg.dataset.audio, "output_sr", self.cfg.dataset.audio.sr)
            input_sr = self.cfg.dataset.audio.sr
            if output_sr != input_sr:
                self.perceptual_resample = torchaudio.transforms.Resample(output_sr, input_sr)
            else:
                self.perceptual_resample = None

        # torch.compile sub-modules for training speedup
        if getattr(self.cfg.train, "compile", False):
            self.CodecEnc = torch.compile(self.CodecEnc)
            self.generator.backbone = torch.compile(self.generator.backbone)
            self.generator.head = torch.compile(self.generator.head)
            if self.generator.upsampler is not None:
                self.generator.upsampler = torch.compile(self.generator.upsampler)
            self.discriminator = torch.compile(self.discriminator)
            self.spec_discriminator = torch.compile(self.spec_discriminator)

    def construct_criteria(self):
        cfg = self.cfg.train
        self.criteria = nn.ModuleDict()
        output_sr = getattr(self.cfg.dataset.audio, "output_sr", self.cfg.dataset.audio.sr)
        if cfg.use_mel_loss:
            mel_params = cfg.get("mel_loss_params", {})
            self.criteria["mel_loss"] = MultiResolutionMelSpectrogramLoss(
                sample_rate=output_sr,
                n_mels=list(mel_params.get("n_mels", [5, 10, 20, 40, 80, 160, 320])),
                window_lengths=list(
                    mel_params.get("window_lengths", [32, 64, 128, 256, 512, 1024, 2048])
                ),
            )
        if cfg.use_feat_match_loss:
            self.criteria["fm_loss"] = nn.L1Loss()
        self.criteria["gan_loss"] = GANLoss()

    def _get_current_phase(self):
        """Determine current training phase based on global_step and phase ratios.

        Phase 1: mel + semantic + VQ only (no discriminator)
        Phase 2: + adversarial + discriminator + feature matching
        Phase 3: + perceptual loss

        Returns:
            int: 1, 2, or 3
        """
        schedule = getattr(self.cfg.train, "phase_schedule", None)
        if schedule is None:
            return 2  # Backward compat: no phase schedule means full GAN training

        max_steps = self.trainer.max_steps
        total = schedule.phase1_ratio + schedule.phase2_ratio + schedule.phase3_ratio
        if total <= 0:
            return 2

        r1 = schedule.phase1_ratio / total
        r2 = schedule.phase2_ratio / total

        phase1_end = int(r1 * max_steps)
        phase2_end = phase1_end + int(r2 * max_steps)

        step = self.global_step
        if step < phase1_end:
            return 1
        elif step < phase2_end:
            return 2
        else:
            return 3

    def forward(self, batch):
        wav = batch["wav"]
        feats = batch["feats"]

        vq_emb = self.CodecEnc(wav.unsqueeze(1))
        vq_emb = vq_emb.transpose(1, 2)

        with torch.no_grad():
            semantic_target = self.semantic_model(feats[:, 0, :, :])
            semantic_target = semantic_target.hidden_states[self.semantic_layer_index]
            semantic_target = semantic_target.detach()

        semantic_target = semantic_target.transpose(1, 2)
        semantic_target = pool_semantic_features(semantic_target, self.CodecEnc.hop_length)
        # Upsample semantic features if frame rate is significantly lower than codec encoder
        if semantic_target.shape[2] < vq_emb.shape[2] * 0.75:
            semantic_target = F.interpolate(
                semantic_target, size=vq_emb.shape[2], mode="linear", align_corners=False
            )
        semantic_target_processed = self.SemanticEncoder_module(semantic_target)
        # Align sequence lengths (different models have slightly different frame rates)
        min_seq = min(semantic_target_processed.shape[2], vq_emb.shape[2])
        semantic_target_processed = semantic_target_processed[:, :, :min_seq]
        vq_emb = vq_emb[:, :, :min_seq]
        semantic_target = semantic_target[:, :, :min_seq]
        vq_emb = torch.cat([semantic_target_processed, vq_emb], dim=1)
        vq_emb = self.fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

        # Adaptive firing: sparse decoding where only fired positions carry information.
        #   aggregate=none: point selection (quantize the latent at each fired position as-is)
        #   aggregate=mean: segment aggregation before quantization
        fire = None
        if self.firing is not None:
            # When signals include flux, compute mel flux from the input waveform (16k) and pass it
            aux_flux = None
            if "flux" in self.firing.signals:
                from rubatocodec.firing import mel_flux

                aux_flux = mel_flux(
                    batch["wav"], sr=self.cfg.dataset.audio.sr, hop=self.CodecEnc.hop_length
                )
            fire = self.firing(vq_emb, flux=aux_flux)

        # With quantizer=rvq the generator returns an aux dict (commit_loss, cb1_quantized).
        # Distill the first quantizer layer; single-stage FSQ uses dec_base directly.
        if fire is not None and self.firing_agg is not None:
            seg = self.firing.segments(vq_emb, fire)
            g = self.firing_agg(vq_emb, seg)  # (B, D, N) segment summaries
            # slot_mask: exclude pad slots (zero vectors) from the rvq VQ layers'
            # commitment/codebook losses (ignored in fsq mode)
            q_slots, vq_code, vq_aux = self.generator(g, vq=True, vq_mask=seg["slot_mask"])
            q_cb1_slots = vq_aux["cb1_quantized"] if isinstance(vq_aux, dict) else None
            B_, Dv, N_ = q_slots.shape
            T_ = seg["sid"].shape[1]
            apos = torch.where(
                seg["slot_mask"], seg["anchor_pos"], torch.full_like(seg["anchor_pos"], T_)
            )

            def _scatter_compose(slots):
                grid_q = torch.zeros(B_, Dv, T_ + 1, device=slots.device, dtype=slots.dtype)
                grid_q.scatter_(2, apos.unsqueeze(1).expand(B_, Dv, N_), slots)
                return self.firing.compose(grid_q[:, :, :T_], fire)

            dec_base = _scatter_compose(q_slots)
            dec_base_sem = _scatter_compose(q_cb1_slots) if q_cb1_slots is not None else dec_base
        elif fire is not None:
            vq_post_emb, vq_code, vq_aux = self.generator(vq_emb, vq=True)
            dec_base = self.firing.compose(vq_post_emb, fire)
            dec_base_sem = (
                self.firing.compose(vq_aux["cb1_quantized"], fire)
                if isinstance(vq_aux, dict)
                else dec_base
            )
        else:
            vq_post_emb, vq_code, vq_aux = self.generator(vq_emb, vq=True)
            dec_base = vq_post_emb
            dec_base_sem = vq_aux["cb1_quantized"] if isinstance(vq_aux, dict) else dec_base

        vq_loss = vq_aux["commit_loss"] if isinstance(vq_aux, dict) else vq_aux
        vq_mean_depth = vq_aux.get("mean_depth") if isinstance(vq_aux, dict) else None

        semantic_recon = self.fc_post_s(dec_base_sem.transpose(1, 2)).transpose(1, 2)
        semantic_recon = self.SemanticDecoder_module(semantic_recon)

        # Supervise semantic features at event starts.
        if fire is not None and self.cfg.train.get("sem_loss_fired_only", False):
            w = fire["mask_hard"][:, : semantic_recon.shape[2]].unsqueeze(1).float()
            sem_loss = ((semantic_recon - semantic_target).pow(2) * w).sum() / (
                w.sum() * semantic_recon.shape[1]
            ).clamp_min(1.0)
        else:
            sem_loss = F.mse_loss(semantic_recon, semantic_target)

        y_, _ = self.generator(self.fc_post_a(dec_base.transpose(1, 2)), vq=False)

        # Use 44.1kHz GT for loss computation if available
        wav_high = batch.get("wav_high", wav)
        y = wav_high.unsqueeze(1)

        # Trim to match lengths (small rounding differences possible)
        min_len = min(y.shape[-1], y_.shape[-1])
        y = y[..., :min_len]
        y_ = y_[..., :min_len]

        output = {
            "gt_wav": y,
            "gen_wav": y_,
            "vq_loss": vq_loss,
            "vq_code": vq_code,
            "semantic_recon_loss": sem_loss,
        }
        if vq_mean_depth is not None:
            output["vq_mean_depth"] = vq_mean_depth
        if fire is not None:
            output["fire_rate"] = fire["rate"]
            # Distillation: teacher = per-frame log-mel reconstruction error (both
            # detached; gradients flow only into the difficulty head)
            if fire.get("d") is not None:
                # Compute in fp32 with autocast disabled (the MelScale matmul would
                # otherwise run in bf16). With center=True the teacher carries a constant
                # -0.5 frame offset plus +/-1.16 frame smoothing from the n_fft window
                # (acceptable for a difficulty teacher)
                with torch.no_grad(), torch.autocast(y.device.type, enabled=False):
                    mel_gt = torch.log(self.firing_distill_mel(y.squeeze(1).float()) + 1e-5)
                    mel_gen = torch.log(self.firing_distill_mel(y_.squeeze(1).float()) + 1e-5)
                    frame_err = (mel_gt - mel_gen).abs().mean(dim=1)  # (B, T')
                output["distill_loss"] = self.firing.distill_loss(fire, frame_err)

        # Perceptual loss (only computed during training in phase 3)
        if self.training and hasattr(self, "perceptual_loss_fn") and self._get_current_phase() >= 3:
            gen_for_perceptual = y_.squeeze(1)
            if self.perceptual_resample is not None:
                gen_for_perceptual = self.perceptual_resample(gen_for_perceptual)
            gt_16k = wav
            min_len_16k = min(gen_for_perceptual.shape[-1], gt_16k.shape[-1])
            output["perceptual_loss"] = self.perceptual_loss_fn(
                gen_for_perceptual[..., :min_len_16k],
                gt_16k[..., :min_len_16k],
            )

        return output

    def compute_disc_loss(self, batch, output):
        y, y_ = output["gt_wav"], output["gen_wav"]
        y_ = y_.detach()
        p = self.discriminator(y)
        p_ = self.discriminator(y_)

        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.criteria["gan_loss"].disc_loss(p[i][-1], p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        sd_p = self.spec_discriminator(y)
        sd_p_ = self.spec_discriminator(y_)

        for i in range(len(sd_p)):
            real_loss, fake_loss = self.criteria["gan_loss"].disc_loss(sd_p[i][-1], sd_p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss

        output = {
            "real_loss": real_loss,
            "fake_loss": fake_loss,
            "disc_loss": disc_loss,
        }
        return output

    def compute_gen_loss(self, batch, output):
        y, y_ = output["gt_wav"], output["gen_wav"]
        vq_loss, _vq_code = output["vq_loss"], output["vq_code"]
        semantic_recon_loss = output["semantic_recon_loss"]
        gen_loss = 0.0
        output_dict = {}
        cfg = self.cfg.train
        phase = self._get_current_phase()

        # === All phases: mel loss, VQ loss, semantic loss ===

        # Mel spectrogram loss
        if cfg.use_mel_loss:
            mel_loss = self.criteria["mel_loss"](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            output_dict["mel_loss"] = mel_loss

        # VQ loss (rvq: sum of per-layer commit losses (Q,). Always None in fsq mode)
        if vq_loss is not None:
            vq_loss = vq_loss.sum() if torch.is_tensor(vq_loss) else sum(vq_loss)
            gen_loss += vq_loss * cfg.lambdas.get("lambda_commit_loss", 1.0)
            output_dict["vq_loss"] = vq_loss

        # rvq monitoring metrics: per-layer codebook perplexity and per-sample dropout
        # depth (no gradients needed). With DAC-style dropout all layers hold indices
        # during training
        if _vq_code is not None and _vq_code.ndim == 3 and _vq_code.shape[1] > 1:
            with torch.no_grad():
                # _vq_code is (B, Q, N). permute is required to put the layer axis Q
                # into columns (reshape(-1, Q) would mix indices across layers)
                code_flat = _vq_code.permute(0, 2, 1).reshape(-1, _vq_code.shape[1])
                perps = []
                for qi in range(code_flat.shape[1]):
                    idx = code_flat[:, qi]
                    idx = idx[idx >= 0]  # exclude pad slots (-1)
                    if idx.numel() == 0:
                        continue
                    counts = torch.bincount(
                        idx, minlength=self.generator.quantizer.codebook_size
                    ).float()
                    p = counts / counts.sum()
                    ent = -(p[p > 0] * p[p > 0].log()).sum()
                    perps.append(ent.exp())
                    if qi == 0:
                        output_dict["vq_perplexity_cb1"] = ent.exp()
                if perps:
                    output_dict["vq_perplexity_mean"] = torch.stack(perps).mean()
        if "vq_mean_depth" in output:
            output_dict["vq_mean_depth"] = output["vq_mean_depth"]

        # Semantic reconstruction loss
        output_dict["semantic_recon_loss"] = semantic_recon_loss
        gen_loss += semantic_recon_loss * cfg.lambdas.lambda_semantic_loss

        # Firing statistics and reconstruction-difficulty distillation
        if "fire_rate" in output:
            output_dict["fire_rate"] = output["fire_rate"]
        if "distill_loss" in output:
            distill_loss = output["distill_loss"]
            gen_loss += distill_loss * cfg.lambdas.get("lambda_distill_loss", 1.0)
            output_dict["distill_loss"] = distill_loss

        # === Phase 2+: GAN losses (adversarial + feature matching) ===
        if phase >= 2:
            self.set_discriminator_gradients(False)

            # GAN adversarial loss
            p_ = self.discriminator(y_)
            adv_loss_list = []
            for i in range(len(p_)):
                adv_loss_list.append(self.criteria["gan_loss"].gen_loss(p_[i][-1]))
            sd_p_ = self.spec_discriminator(y_)
            for i in range(len(sd_p_)):
                adv_loss_list.append(self.criteria["gan_loss"].gen_loss(sd_p_[i][-1]))
            adv_loss = sum(adv_loss_list)
            gen_loss += adv_loss * cfg.lambdas.lambda_adv
            output_dict["adv_loss"] = adv_loss

            # Feature Matching loss
            if cfg.use_feat_match_loss:
                fm_loss = 0.0
                with torch.no_grad():
                    p = self.discriminator(y)
                for i in range(len(p_)):
                    for j in range(len(p_[i]) - 1):
                        fm_loss += self.criteria["fm_loss"](p_[i][j], p[i][j].detach())
                gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
                output_dict["fm_loss"] = fm_loss
                spec_fm_loss = 0.0
                with torch.no_grad():
                    sd_p = self.spec_discriminator(y)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.criteria["fm_loss"](sd_p_[i][j], sd_p[i][j].detach())
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                output_dict["spec_fm_loss"] = spec_fm_loss

            self.set_discriminator_gradients(True)

        # === Phase 3: Perceptual loss ===
        if phase >= 3 and "perceptual_loss" in output:
            perceptual_loss = output["perceptual_loss"]
            gen_loss += perceptual_loss * cfg.lambdas.lambda_perceptual_loss
            output_dict["perceptual_loss"] = perceptual_loss

        output_dict["gen_loss"] = gen_loss
        return output_dict

    def training_step(self, batch, batch_idx):
        output = self(batch)

        gen_opt, disc_opt = self.optimizers()
        gen_sche, disc_sche = self.lr_schedulers()

        phase = self._get_current_phase()

        # Log current training phase
        self.log(
            "train/phase",
            float(phase),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=self.cfg.dataset.train.batch_size,
        )

        # Train discriminator (phase 2+ only)
        if phase >= 2:
            disc_losses = self.compute_disc_loss(batch, output)
            disc_loss = disc_losses["disc_loss"]
            disc_opt.zero_grad()
            self.manual_backward(disc_loss)
            self.clip_gradients(
                disc_opt,
                gradient_clip_val=self.cfg.train.disc_grad_clip,
                gradient_clip_algorithm="norm",
            )
            disc_opt.step()
            disc_sche.step()

            self.log_dict(
                disc_losses,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=self.cfg.dataset.train.batch_size,
                sync_dist=True,
            )

        # Train generator (all phases)
        gen_losses = self.compute_gen_loss(batch, output)
        gen_loss = gen_losses["gen_loss"]
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(
            gen_opt, gradient_clip_val=self.cfg.train.gen_grad_clip, gradient_clip_algorithm="norm"
        )
        gen_opt.step()
        gen_sche.step()

        # In manual optimization, each optimizer.step() increments global_step.
        # Correct so that 1 training_step = 1 global_step regardless of phase.
        if phase >= 2:
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed -= 1

        self.log_dict(
            gen_losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.cfg.dataset.train.batch_size,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        y = output["gt_wav"]
        y_ = output["gen_wav"]

        # Compute validation mel loss
        if "mel_loss" in self.criteria:
            mel_loss = self.criteria["mel_loss"](y_.squeeze(1), y.squeeze(1))
            self.log(
                "val/mel_loss", mel_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
            )

        # SI-SDR (GPU tensor computation)
        ref = y.squeeze(1)  # [B, T]
        est = y_.squeeze(1)  # [B, T]
        dot = torch.sum(ref * est, dim=-1, keepdim=True)
        s_ref_energy = torch.sum(ref**2, dim=-1, keepdim=True).clamp(min=1e-8)
        proj = dot * ref / s_ref_energy
        noise = est - proj
        si_sdr = 10 * torch.log10(
            torch.sum(proj**2, dim=-1).clamp(min=1e-8) / torch.sum(noise**2, dim=-1).clamp(min=1e-8)
        )
        self.log(
            "val/si_sdr", si_sdr.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # PESQ / STOI (CPU numpy computation, requires 16kHz)
        output_sr = getattr(self.cfg.dataset.audio, "output_sr", self.cfg.dataset.audio.sr)
        eval_sr = 16000
        gt_cpu = y.squeeze(1).cpu().float()  # [B, T]
        gen_cpu = y_.squeeze(1).cpu().float()  # [B, T]
        if output_sr != eval_sr:
            gt_cpu = F_audio.resample(gt_cpu, output_sr, eval_sr)
            gen_cpu = F_audio.resample(gen_cpu, output_sr, eval_sr)
        for i in range(gt_cpu.size(0)):
            ref_np = gt_cpu[i].numpy()
            est_np = gen_cpu[i].numpy()
            min_len = min(len(ref_np), len(est_np))
            ref_np = ref_np[:min_len]
            est_np = est_np[:min_len]
            try:
                self._val_pesq_scores.append(pesq_fn(eval_sr, ref_np, est_np, "wb"))
            except Exception:
                pass
            try:
                self._val_stoi_scores.append(stoi_fn(ref_np, est_np, eval_sr, extended=False))
            except Exception:
                pass

        # Log audio samples for the first batch
        max_samples = 4
        if batch_idx == 0 and self.logger and isinstance(self.logger, pl.loggers.WandbLogger):
            audios = []
            for i in range(min(max_samples, y.size(0))):
                gt_wav = y[i].squeeze(0).cpu().float()
                gen_wav = y_[i].squeeze(0).cpu().float()
                audios.append(
                    wandb.Audio(gt_wav.numpy(), sample_rate=output_sr, caption=f"GT #{i}")
                )
                audios.append(
                    wandb.Audio(gen_wav.numpy(), sample_rate=output_sr, caption=f"Recon #{i}")
                )
            self.logger.experiment.log(
                {"val/audio_samples": audios, "global_step": self.global_step}
            )

        return {}

    def on_validation_epoch_end(self):
        if self._val_pesq_scores:
            avg_pesq = float(np.mean(self._val_pesq_scores))
            self.log("val/pesq", avg_pesq, prog_bar=True, logger=True)
        if self._val_stoi_scores:
            avg_stoi = float(np.mean(self._val_stoi_scores))
            self.log("val/stoi", avg_stoi, prog_bar=True, logger=True)
        self._val_pesq_scores.clear()
        self._val_stoi_scores.clear()

    def on_save_checkpoint(self, checkpoint):
        """Strip _orig_mod. prefix from compiled module keys for checkpoint compatibility."""
        state_dict = checkpoint["state_dict"]
        checkpoint["state_dict"] = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

    def on_load_checkpoint(self, checkpoint):
        """Add _orig_mod. prefix when loading non-compiled checkpoints into compiled model."""
        if not getattr(self.cfg.train, "compile", False):
            return
        state_dict = checkpoint["state_dict"]
        compiled_prefixes = [
            "CodecEnc.",
            "generator.backbone.",
            "generator.head.",
            "generator.upsampler.",
            "discriminator.",
            "spec_discriminator.",
        ]
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if "._orig_mod." not in key:
                for prefix in compiled_prefixes:
                    if key.startswith(prefix):
                        new_key = prefix[:-1] + "._orig_mod." + key[len(prefix) :]
                        break
            new_state_dict[new_key] = value
        checkpoint["state_dict"] = new_state_dict

    def configure_optimizers(self):
        from itertools import chain

        disc_params = chain(self.discriminator.parameters(), self.spec_discriminator.parameters())

        gen_params = chain(
            self.CodecEnc.parameters(),
            self.generator.parameters(),
            self.fc_prior.parameters(),
            self.fc_post_a.parameters(),
            self.fc_post_s.parameters(),
            self.SemanticDecoder_module.parameters(),
            self.SemanticEncoder_module.parameters(),
            *([self.firing.parameters()] if self.firing is not None else []),
            *([self.firing_agg.parameters()] if self.firing_agg is not None else []),
        )

        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        gen_sche = build_warmup_decay_scheduler(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = build_warmup_decay_scheduler(disc_opt, **self.cfg.train.disc_schedule_params)

        return [gen_opt, disc_opt], [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.discriminator.parameters():
            p.requires_grad = flag

        for p in self.spec_discriminator.parameters():
            p.requires_grad = flag
