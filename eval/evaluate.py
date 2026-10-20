"""Evaluate reconstructed audio against the originals with multiple metrics.

Usage:
    python -m eval.evaluate \
        --input-dir eval_output/model_name \
        --metrics utmosv2,dnsmos,pesq,stoi,dwer,speaker_sim_wavlm \
        --device cuda

Input: ``<input-dir>/metadata.json`` with a ``samples`` list. Each sample has
``idx``, ``original`` and ``reconstructed`` (audio paths relative to the input
directory) and an optional reference ``text`` (needed only for ``ref_asr``).
``examples/reconstruct_dataset.py`` writes this layout.

Metrics (``--metrics``: comma-separated, or ``all``):
    utmosv2, utmos, dnsmos  Non-intrusive quality of the reconstruction.
    pesq, stoi              Intrusive quality against the original, at 16 kHz.
    mel_distance            Mel-spectrogram distance at ``--mel-sample-rate``.
    dwer, dcer              ASR error of the reconstruction, using the
                            original's ASR transcript as the reference.
    ref_asr                 ASR error against the reference ``text``:
                            raw CER (``cer``) plus normalized WER/CER
                            (``wer_norm``, ``cer_norm``).
    speaker_sim_wavlm       Speaker similarity with WavLM-Large SV (the SIM
                            reported in docs/evaluation.md). Requires
                            ``--speaker-wavlm-ckpt`` and ``--unispeech-dir``;
                            skipped with a warning otherwise.
    speaker_sim             Speaker similarity with SpeechBrain ECAPA
                            (``--speaker-model``).

ASR uses Whisper (``--asr-model``, ``--asr-language``) or Paraformer
(``--asr-model paraformer-zh``). Whisper inputs longer than 30 seconds use
long-form generation. Recognition summaries report micro error rates alongside
the utterance mean/std; rates are fractions and are not capped.

Output: ``results.json`` (or ``--output-file``) with ``summary`` (per-metric
mean/std/count, ignoring NaN) and ``per_sample`` scores, including
transcripts and edit counts for recognition metrics. See docs/evaluation.md
for the benchmark settings.
"""

import json
import math
import os
from argparse import ArgumentParser

ALL_METRICS = [
    "utmosv2",
    "utmos",
    "dnsmos",
    "pesq",
    "stoi",
    "ref_asr",
    "dwer",
    "dcer",
    "speaker_sim",
    "speaker_sim_wavlm",
    "mel_distance",
]


def aggregate_metric(values, key):
    """Compute mean and std for a metric, ignoring NaN values."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not valid:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    mean = sum(valid) / len(valid)
    if len(valid) > 1:
        std = (sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)) ** 0.5
    else:
        std = 0.0
    return {"mean": mean, "std": std, "count": len(valid)}


def run_pesq_stoi(original_paths, reconstructed_paths, **kwargs):
    """Run PESQ and STOI metrics."""
    from eval.metrics.pesq_stoi import compute_pesq_stoi

    print("Computing PESQ and STOI...")
    return compute_pesq_stoi(original_paths, reconstructed_paths)


def run_mel_distance(original_paths, reconstructed_paths, **kwargs):
    """Run mel spectrogram distance metric."""
    from eval.metrics.mel_distance import compute_mel_distance

    device = kwargs.get("device", "cpu")
    sample_rate = kwargs.get("mel_sample_rate", 16000)
    print(f"Computing mel distance (sr={sample_rate})...")
    return compute_mel_distance(
        original_paths, reconstructed_paths, sample_rate=sample_rate, device=device
    )


def run_utmosv2(reconstructed_paths, **kwargs):
    """Run UTMOSv2 metric."""
    from eval.metrics.utmos import compute_utmosv2

    device = kwargs.get("device", "cpu")
    print("Computing UTMOSv2...")
    return compute_utmosv2(reconstructed_paths, device=device)


def run_utmos(reconstructed_paths, **kwargs):
    """Run UTMOS metric."""
    from eval.metrics.utmos import compute_utmos

    device = kwargs.get("device", "cpu")
    print("Computing UTMOS...")
    return compute_utmos(reconstructed_paths, device=device)


def run_dnsmos(reconstructed_paths, **kwargs):
    """Run DNSMOS metric."""
    from eval.metrics.dnsmos import compute_dnsmos

    device = kwargs.get("device", "cpu")
    print("Computing DNSMOS...")
    return compute_dnsmos(reconstructed_paths, device=device)


def run_speaker_sim(original_paths, reconstructed_paths, **kwargs):
    """Run speaker similarity metric."""
    from eval.metrics.speaker import compute_speaker_similarity

    device = kwargs.get("device", "cpu")
    model_source = kwargs.get("speaker_model", "speechbrain/spkrec-ecapa-voxceleb")
    print(f"Computing speaker similarity ({model_source})...")
    return compute_speaker_similarity(
        original_paths,
        reconstructed_paths,
        model_source=model_source,
        device=device,
        cache_dir=kwargs.get("spk_cache_dir"),
    )


def run_speaker_sim_wavlm(original_paths, reconstructed_paths, **kwargs):
    """Run speaker similarity metric with the WavLM-Large SV embedder."""
    from eval.metrics.speaker_wavlm import compute_speaker_similarity_wavlm

    device = kwargs.get("device", "cpu")
    checkpoint_path = kwargs.get("speaker_wavlm_ckpt")
    unispeech_dir = kwargs.get("unispeech_dir")
    print(f"Computing speaker similarity (WavLM-SV, ckpt={checkpoint_path})...")
    return compute_speaker_similarity_wavlm(
        original_paths,
        reconstructed_paths,
        checkpoint_path=checkpoint_path,
        unispeech_dir=unispeech_dir,
        device=device,
        cache_dir=kwargs.get("spk_cache_dir"),
    )


def run_recognition(reconstructed_paths, reference_texts, original_paths=None, **kwargs):
    """Run reference-text and optional original-ASR comparisons."""
    from eval.metrics.asr import compute_recognition

    return compute_recognition(
        reconstructed_paths,
        reference_texts,
        original_paths=original_paths,
        model_name=kwargs.get("asr_model", "openai/whisper-large-v3"),
        language=kwargs.get("asr_language"),
        device=kwargs.get("device", "cpu"),
        batch_size=kwargs.get("asr_batch_size", 32),
        cache_dir=kwargs.get("asr_cache_dir"),
        revision=kwargs.get("asr_revision"),
    )


def main():
    parser = ArgumentParser(description="Evaluate reconstructed audio with multiple metrics")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing metadata.json (samples with 'original'/'reconstructed' "
        "relative audio paths and optional 'text')",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSON path (default: {input-dir}/results.json)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated metrics or 'all' (utmos: UTMOS; utmosv2: UTMOSv2)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--asr-model",
        type=str,
        default="openai/whisper-large-v3",
        help="ASR model for recognition metrics",
    )
    parser.add_argument(
        "--asr-language", type=str, default=None, help="Language for Whisper (e.g., ja, en)"
    )
    parser.add_argument("--asr-batch-size", type=int, default=32)
    parser.add_argument("--asr-revision", default=None, help="ASR model revision")
    parser.add_argument(
        "--asr-cache-dir", default=None, help="Shared directory for original-audio ASR transcripts"
    )
    parser.add_argument(
        "--speaker-model",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="Speaker model",
    )
    parser.add_argument(
        "--speaker-wavlm-ckpt",
        type=str,
        default=None,
        help="Path to UniSpeech wavlm_large_finetune.pth for speaker_sim_wavlm "
        "(the metric is skipped if unset)",
    )
    parser.add_argument(
        "--unispeech-dir",
        type=str,
        default=None,
        help="Path to a local checkout of https://github.com/microsoft/UniSpeech "
        "(required for speaker_sim_wavlm; the metric is skipped if unset)",
    )
    parser.add_argument(
        "--spk-cache-dir",
        type=str,
        default=None,
        help="Directory for speaker-embedding reuse across runs (keyed by audio file "
        "content; one subdirectory per speaker model)",
    )
    parser.add_argument(
        "--mel-sample-rate", type=int, default=16000, help="Sample rate for mel distance"
    )
    args = parser.parse_args()

    # Load metadata
    metadata_path = os.path.join(args.input_dir, "metadata.json")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    samples = metadata["samples"]
    original_paths = [os.path.join(args.input_dir, s["original"]) for s in samples]
    reconstructed_paths = [os.path.join(args.input_dir, s["reconstructed"]) for s in samples]
    reference_texts = [s.get("text", "") for s in samples]

    # Parse metrics
    if args.metrics == "all":
        metrics_to_run = list(ALL_METRICS)
    else:
        metrics_to_run = [m.strip() for m in args.metrics.split(",")]
        for m in metrics_to_run:
            if m not in ALL_METRICS:
                print(f"Warning: Unknown metric '{m}', skipping. Available: {ALL_METRICS}")
        metrics_to_run = [m for m in metrics_to_run if m in ALL_METRICS]

    kwargs = {
        "device": args.device,
        "asr_model": args.asr_model,
        "asr_language": args.asr_language,
        "asr_batch_size": args.asr_batch_size,
        "asr_cache_dir": args.asr_cache_dir,
        "asr_revision": args.asr_revision,
        "speaker_model": args.speaker_model,
        "speaker_wavlm_ckpt": args.speaker_wavlm_ckpt,
        "unispeech_dir": args.unispeech_dir,
        "spk_cache_dir": args.spk_cache_dir,
        "mel_sample_rate": args.mel_sample_rate,
    }

    # Initialize per-sample results
    per_sample = [{} for _ in samples]
    summary = {}

    # Run metrics sequentially (one model at a time to save memory)
    # PESQ and STOI are computed together
    if "pesq" in metrics_to_run or "stoi" in metrics_to_run:
        results = run_pesq_stoi(original_paths, reconstructed_paths, **kwargs)
        if "pesq" in metrics_to_run:
            for i, r in enumerate(results):
                per_sample[i]["pesq"] = r["pesq"]
                per_sample[i]["pesq_nb"] = r.get("pesq_nb")
            summary["pesq"] = aggregate_metric([r["pesq"] for r in results], "pesq")
            summary["pesq_nb"] = aggregate_metric([r.get("pesq_nb") for r in results], "pesq_nb")
            print(f"  PESQ: mean={summary['pesq']['mean']:.4f}, std={summary['pesq']['std']:.4f}")
            print(f"  PESQ-NB: mean={summary['pesq_nb']['mean']:.4f}")
        if "stoi" in metrics_to_run:
            for i, r in enumerate(results):
                per_sample[i]["stoi"] = r["stoi"]
            summary["stoi"] = aggregate_metric([r["stoi"] for r in results], "stoi")
            print(f"  STOI: mean={summary['stoi']['mean']:.4f}, std={summary['stoi']['std']:.4f}")

    if "mel_distance" in metrics_to_run:
        results = run_mel_distance(original_paths, reconstructed_paths, **kwargs)
        for i, r in enumerate(results):
            per_sample[i]["mel_distance"] = r["mel_distance"]
        summary["mel_distance"] = aggregate_metric(
            [r["mel_distance"] for r in results], "mel_distance"
        )
        print(
            f"  Mel Distance: mean={summary['mel_distance']['mean']:.4f}, std={summary['mel_distance']['std']:.4f}"
        )

    if "utmosv2" in metrics_to_run:
        results = run_utmosv2(reconstructed_paths, **kwargs)
        for i, r in enumerate(results):
            per_sample[i]["utmosv2"] = r["utmosv2"]
        summary["utmosv2"] = aggregate_metric([r["utmosv2"] for r in results], "utmosv2")
        print(
            f"  UTMOSv2: mean={summary['utmosv2']['mean']:.4f}, std={summary['utmosv2']['std']:.4f}"
        )

    if "utmos" in metrics_to_run:
        results = run_utmos(reconstructed_paths, **kwargs)
        for i, r in enumerate(results):
            per_sample[i]["utmos"] = r["utmos"]
        summary["utmos"] = aggregate_metric([r["utmos"] for r in results], "utmos")
        print(f"  UTMOS: mean={summary['utmos']['mean']:.4f}, std={summary['utmos']['std']:.4f}")

    if "dnsmos" in metrics_to_run:
        results = run_dnsmos(reconstructed_paths, **kwargs)
        for i, r in enumerate(results):
            per_sample[i]["dnsmos_p808"] = r["p808_mos"]
            per_sample[i]["dnsmos_sig"] = r["sig"]
            per_sample[i]["dnsmos_bak"] = r["bak"]
            per_sample[i]["dnsmos_ovr"] = r["ovr"]
        summary["dnsmos_p808"] = aggregate_metric([r["p808_mos"] for r in results], "p808_mos")
        summary["dnsmos_sig"] = aggregate_metric([r["sig"] for r in results], "sig")
        summary["dnsmos_bak"] = aggregate_metric([r["bak"] for r in results], "bak")
        summary["dnsmos_ovr"] = aggregate_metric([r["ovr"] for r in results], "ovr")
        print(f"  DNSMOS p808: mean={summary['dnsmos_p808']['mean']:.4f}")
        print(f"  DNSMOS ovr:  mean={summary['dnsmos_ovr']['mean']:.4f}")

    if "speaker_sim" in metrics_to_run:
        results = run_speaker_sim(original_paths, reconstructed_paths, **kwargs)
        for i, r in enumerate(results):
            per_sample[i]["speaker_similarity"] = r["speaker_similarity"]
        summary["speaker_similarity"] = aggregate_metric(
            [r["speaker_similarity"] for r in results], "speaker_similarity"
        )
        print(
            f"  Speaker Sim: mean={summary['speaker_similarity']['mean']:.4f}, std={summary['speaker_similarity']['std']:.4f}"
        )

    if "speaker_sim_wavlm" in metrics_to_run:
        if not args.speaker_wavlm_ckpt or not os.path.exists(args.speaker_wavlm_ckpt):
            print(
                "Warning: skipping speaker_sim_wavlm "
                f"(checkpoint not found: {args.speaker_wavlm_ckpt}; "
                "pass --speaker-wavlm-ckpt to enable)"
            )
        elif not args.unispeech_dir or not os.path.isdir(args.unispeech_dir):
            print(
                "Warning: skipping speaker_sim_wavlm "
                f"(UniSpeech checkout not found: {args.unispeech_dir}; "
                "pass --unispeech-dir to enable)"
            )
        else:
            results = run_speaker_sim_wavlm(original_paths, reconstructed_paths, **kwargs)
            for i, r in enumerate(results):
                per_sample[i]["speaker_similarity_wavlm"] = r["speaker_similarity_wavlm"]
            summary["speaker_similarity_wavlm"] = aggregate_metric(
                [r["speaker_similarity_wavlm"] for r in results], "speaker_similarity_wavlm"
            )
            print(
                f"  Speaker Sim (WavLM-SV): mean={summary['speaker_similarity_wavlm']['mean']:.4f}, "
                f"std={summary['speaker_similarity_wavlm']['std']:.4f}"
            )

    if {"ref_asr", "dwer", "dcer"}.intersection(metrics_to_run):
        from eval.metrics.asr import aggregate_recognition

        differential = bool({"dwer", "dcer"}.intersection(metrics_to_run))
        results = run_recognition(
            reconstructed_paths,
            reference_texts,
            original_paths=original_paths if differential else None,
            **kwargs,
        )
        keys = ["cer", "wer_norm", "cer_norm"] if "ref_asr" in metrics_to_run else []
        keys += [k for k in ("dwer", "dcer") if k in metrics_to_run]
        for sample, result in zip(per_sample, results, strict=True):
            for key in ("hypothesis", "reference", "original_hypothesis"):
                if key in result:
                    sample[key] = result[key]
            for key in keys:
                sample[key] = result[key]
                sample[key + "_counts"] = result[key + "_counts"]
        for key in keys:
            summary[key] = aggregate_recognition(results, key)

    # Build output
    output = {
        "config": {
            **metadata.get("config", {}),
            "metrics_computed": metrics_to_run,
            "eval_args": {
                "asr_model": args.asr_model,
                "asr_language": args.asr_language,
                "asr_batch_size": args.asr_batch_size,
                "asr_cache_dir": args.asr_cache_dir,
                "asr_revision": args.asr_revision,
                "speaker_model": args.speaker_model,
                "speaker_wavlm_ckpt": args.speaker_wavlm_ckpt,
                "unispeech_dir": args.unispeech_dir,
                "spk_cache_dir": args.spk_cache_dir,
                "mel_sample_rate": args.mel_sample_rate,
            },
        },
        "summary": summary,
        "per_sample": [{"idx": samples[i]["idx"], **per_sample[i]} for i in range(len(samples))],
    }

    # Save results
    output_file = args.output_file or os.path.join(args.input_dir, "results.json")
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for metric_name, stats in summary.items():
        if "micro" in stats:
            print(
                f"  {metric_name:25s}: micro={stats['micro']:.4f}, "
                f"mean={stats['mean']:.4f} [n={stats['count']}]"
            )
            continue
        print(
            f"  {metric_name:25s}: {stats['mean']:.4f} (+/- {stats['std']:.4f})  [n={stats['count']}]"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
