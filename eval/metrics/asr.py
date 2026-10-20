"""Speech recognition, normalized error rates, and original-audio caching."""

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

EVAL_SR = 16000


def text_normalizer(model_name, language, revision=None):
    """Return the normalization used for both reference and hypothesis."""
    if model_name.startswith("paraformer"):
        import string

        import zhconv
        from zhon.hanzi import punctuation

        def normalize(text):
            text = zhconv.convert(text, "zh-cn").strip()
            for symbol in punctuation + string.punctuation:
                if symbol != "'":
                    text = text.replace(symbol, "")
            return text.replace(" ", "")

        return normalize
    from transformers import WhisperTokenizer

    tokenizer = WhisperTokenizer.from_pretrained(model_name, revision=revision)
    if language == "en":
        return tokenizer.normalize
    return tokenizer.basic_normalize


def edit_counts(reference, hypothesis, *, characters=False):
    """Return edit counts, including insertions for an empty reference."""
    import jiwer

    if not reference.strip():
        return {
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hypothesis) if characters else len(hypothesis.split()),
            "reference_length": 0,
        }
    result = (jiwer.process_characters if characters else jiwer.process_words)(
        reference, hypothesis
    )
    return {
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "reference_length": result.hits + result.substitutions + result.deletions,
    }


def error_rate(counts):
    if counts is None or counts["reference_length"] == 0:
        return float("nan")
    return (
        sum(counts[k] for k in ("substitutions", "deletions", "insertions"))
        / counts["reference_length"]
    )


def score_transcripts(reference, hypothesis, normalize, language, *, differential=False):
    """Score normalized transcripts; reference=None denotes unavailable text."""
    word_key, char_key = ("dwer", "dcer") if differential else ("wer_norm", "cer_norm")
    result = {}
    ref, hyp = normalize(reference or ""), normalize(hypothesis)
    for key, characters in ((word_key, False), (char_key, True)):
        # Chinese word segmentation is not part of the evaluation protocol.
        valid = reference is not None and not (language == "zh" and not characters)
        r, h = (
            (ref.replace(" ", ""), hyp.replace(" ", ""))
            if characters and language != "en"
            else (ref, hyp)
        )
        c = edit_counts(r, h, characters=characters) if valid else None
        result[key] = error_rate(c)
        result[key + "_counts"] = c
    if not differential:
        c = (
            edit_counts(reference.strip(), hypothesis, characters=True)
            if reference is not None
            else None
        )
        result.update(cer=error_rate(c), cer_counts=c)
    return result


def aggregate_recognition(rows, key):
    """Report utterance mean/std and corpus micro error rate (fractions)."""
    valid = [
        r
        for r in rows
        if r.get(key + "_counts") is not None and r[key + "_counts"]["reference_length"] > 0
    ]
    values = [error_rate(r[key + "_counts"]) for r in valid]
    mean = sum(values) / len(values) if values else float("nan")
    std = (
        (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        if len(values) > 1
        else (0.0 if values else float("nan"))
    )
    totals = {
        k: sum(r[key + "_counts"][k] for r in valid)
        for k in ("substitutions", "deletions", "insertions", "reference_length")
    }
    return dict(
        mean=mean,
        std=std,
        count=len(valid),
        micro=error_rate(totals),
        undefined_count=len(rows) - len(valid),
        **totals,
    )


def transcribe_audio(
    audio_paths,
    model_name="openai/whisper-large-v3",
    language=None,
    device="cpu",
    batch_size=32,
    revision=None,
):
    """Transcribe audio with Whisper or Paraformer.

    Whisper inputs longer than 30 seconds use long-form (timestamped) generation.
    """
    if not audio_paths:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if model_name.startswith("paraformer"):
        from funasr import AutoModel

        kwargs = {"model": model_name, "device": device}
        if revision is not None:
            kwargs["model_revision"] = revision
        model = AutoModel(**kwargs)
        return [
            model.generate(input=str(p), batch_size_s=300)[0]["text"].strip() for p in audio_paths
        ]

    import soundfile as sf
    import torch
    import torchaudio.functional as F_audio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_name, revision=revision)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name, revision=revision, dtype=torch.float32
        )
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.float32
        )
    model = model.to(device).eval()
    generate_kwargs = {"language": language, "task": "transcribe"} if language is not None else {}
    hypotheses = []
    try:
        for start in range(0, len(audio_paths), batch_size):
            wavs = []
            for path in audio_paths[start : start + batch_size]:
                wav, sr = sf.read(path, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=-1)
                if sr != EVAL_SR:
                    wav = F_audio.resample(torch.from_numpy(wav), sr, EVAL_SR).numpy()
                wavs.append(wav)
            batch_hypotheses = [None] * len(wavs)
            for long_form in (False, True):
                indices = [i for i, wav in enumerate(wavs)
                           if (len(wav) > 30 * EVAL_SR) == long_form]
                if not indices:
                    continue
                group = [wavs[i] for i in indices]
                if long_form:
                    inputs = processor(
                        group, sampling_rate=EVAL_SR, return_tensors="pt",
                        truncation=False, padding="longest", return_attention_mask=True,
                    )
                    features = inputs.input_features.to(device)
                    extra = {
                        "attention_mask": inputs.attention_mask.to(device),
                        "return_timestamps": True,
                    }
                else:
                    features = processor(
                        group, sampling_rate=EVAL_SR, return_tensors="pt"
                    ).input_features.to(device)
                    extra = {}
                with torch.inference_mode():
                    tokens = model.generate(features, **generate_kwargs, **extra)
                decoded = processor.batch_decode(tokens, skip_special_tokens=True)
                for i, text in zip(indices, decoded, strict=True):
                    batch_hypotheses[i] = text.strip()
            hypotheses.extend(batch_hypotheses)
            print(f"  ASR: {len(hypotheses)}/{len(audio_paths)}", flush=True)
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return hypotheses


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_settings(model_name, language, device, batch_size, revision):
    versions = {}
    for package in ("torch", "torchaudio", "transformers", "funasr"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "model": model_name,
        "revision": revision,
        "language": language,
        "device": str(device),
        "batch_size": batch_size,
        "sample_rate": EVAL_SR,
        "whisper_dtype": "float32",
        "implementation_sha256": _file_hash(__file__),
        "versions": versions,
    }


def _store_cache(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, encoding="utf-8"
        ) as stream:
            tmp = stream.name
            json.dump(value, stream, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def compute_recognition(
    reconstructed_paths,
    reference_texts,
    *,
    original_paths=None,
    model_name="openai/whisper-large-v3",
    language=None,
    device="cpu",
    batch_size=32,
    cache_dir=None,
    revision=None,
):
    """Recognize reconstructions and optional originals with shared ASR settings."""
    if len(reconstructed_paths) != len(reference_texts):
        raise ValueError("Audio and reference lengths differ")
    if original_paths is not None and len(original_paths) != len(reconstructed_paths):
        raise ValueError("Original and reconstructed lengths differ")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    settings = _cache_settings(model_name, language, device, batch_size, revision)
    originals, misses, entries = {}, {}, {}
    keys = []
    for path in original_paths or []:
        identity = {"audio_sha256": _file_hash(path), "settings": settings}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        keys.append(key)
        if key in entries:
            continue
        entries[key] = identity
        cached = Path(cache_dir) / f"{key}.json" if cache_dir else None
        if cached and cached.exists():
            try:
                value = json.loads(cached.read_text(encoding="utf-8"))
                if value.get("identity") == identity and isinstance(value.get("hypothesis"), str):
                    originals[key] = value["hypothesis"]
                    continue
            except (ValueError, OSError):
                pass
        misses[key] = str(path)
    paths = list(misses.values()) + list(reconstructed_paths)
    hypotheses = transcribe_audio(paths, model_name, language, device, batch_size, revision)
    if len(hypotheses) != len(paths):
        raise RuntimeError("ASR returned an unexpected number of transcripts")
    for key, hyp in zip(misses, hypotheses[: len(misses)], strict=True):
        originals[key] = hyp
        if cache_dir:
            _store_cache(
                Path(cache_dir) / f"{key}.json", {"identity": entries[key], "hypothesis": hyp}
            )
    normalize = text_normalizer(model_name, language, revision)
    score_language = "zh" if model_name.startswith("paraformer") else language
    rows = []
    for i, (ref, hyp) in enumerate(zip(reference_texts, hypotheses[len(misses) :], strict=True)):
        ref = ref.strip() if ref and ref.strip() else None
        row = {"hypothesis": hyp, "reference": ref or ""}
        row.update(score_transcripts(ref, hyp, normalize, score_language))
        if original_paths is not None:
            row["original_hypothesis"] = originals[keys[i]]
            row.update(
                score_transcripts(
                    originals[keys[i]], hyp, normalize, score_language, differential=True
                )
            )
        rows.append(row)
    return rows


def compute_cer(
    audio_paths,
    reference_texts,
    model_name="openai/whisper-large-v3",
    language=None,
    device="cpu",
    batch_size=32,
):
    """Compute reference-text recognition metrics with Whisper."""
    return compute_recognition(
        audio_paths,
        reference_texts,
        model_name=model_name,
        language=language,
        device=device,
        batch_size=batch_size,
    )


def compute_cer_paraformer(audio_paths, reference_texts, device="cpu"):
    """Compute Chinese CER with Paraformer and simplified-character normalization."""
    return compute_recognition(
        audio_paths, reference_texts, model_name="paraformer-zh", language="zh", device=device
    )
