import os
import json
import tarfile
import urllib.request
from pathlib import Path

import runpod
import pandas as pd
import torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, ClapModel, ClapProcessor

BASE_DIR = Path(os.getenv("TOPAUDIO_BASE_DIR", "/runpod-volume/topaudio_ai"))
SNIPPETS_DIR = BASE_DIR / "caption_snippets_30s"
CATALOG_CSV = BASE_DIR / "catalog_caption_snippets.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_JSONL = RESULTS_DIR / "qwen2_audio_captions.jsonl"

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")
CLAP_MODEL_ID = os.getenv("CLAP_MODEL_ID", "laion/clap-htsat-unfused")

CLAP_RESULTS_JSONL = RESULTS_DIR / "clap_audio_labels.jsonl"

model = None
processor = None

clap_model = None
clap_processor = None

PROMPT = """
You are a professional music supervisor and music dataset labeling expert.

Analyze the provided instrumental music audio clip. Use only what you hear in the audio.

Return strict JSON only. Do not add explanations, markdown, or extra text.

JSON schema:
{
  "genre": "",
  "subgenre": "",
  "mood": [],
  "energy": "",
  "tempo_feel": "",
  "main_instruments": [],
  "rhythm_description": "",
  "vocal_status": "instrumental / vocal / vocal_chops / unknown",
  "production_style": "",
  "commercial_use_cases": [],
  "short_caption": "",
  "training_caption": "",
  "confidence": "low / medium / high"
}

Rules:
- Be conservative.
- Do not mention artist names.
- Do not mention copyrighted references.
- If unsure, use "unknown" or lower confidence.
- training_caption must be one clean sentence for AI music model training.
- Focus on genre, mood, instruments, arrangement, rhythm, and commercial production use.
"""


def load_model():
    global model, processor

    if model is not None and processor is not None:
        return

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model.eval()


def already_done():
    done = set()

    if RESULTS_JSONL.exists():
        with RESULTS_JSONL.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    stem = json.loads(line).get("stem")
                    if stem:
                        done.add(str(stem))
                except Exception:
                    pass

    return done


def download_file(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return str(dest)


def safe_extract_tar_gz(archive_path, dest_dir):
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if not str(member_path).startswith(str(dest_dir)):
                raise RuntimeError(f"Unsafe tar path: {member.name}")
        tar.extractall(path=dest_dir)

    return str(dest_dir)


def analyze_audio(audio_path):
    """
    Correct Qwen2-Audio path:
    - load local wav into waveform with librosa
    - pass waveform array to processor via audios=[audio]
    - decode only generated tokens, not prompt+answer
    """
    load_model()

    audio_path = str(audio_path)
    target_sr = processor.feature_extractor.sampling_rate

    audio, _ = librosa.load(
        audio_path,
        sr=target_sr,
        mono=True
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "local_audio"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=text,
        audio=[audio],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False
        )

    # Decode only the new assistant output, not the prompt.
    prompt_len = inputs["input_ids"].size(1)
    generated_ids = generated_ids[:, prompt_len:]

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return output


def action_health(job_input):
    return {
        "ok": True,
        "base_dir": str(BASE_DIR),
        "snippets_exists": SNIPPETS_DIR.exists(),
        "catalog_exists": CATALOG_CSV.exists(),
        "results_jsonl": str(RESULTS_JSONL),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "model_id": MODEL_ID
    }


def action_download_data(job_input):
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    catalog_url = job_input.get("catalog_url")
    snippets_url = job_input.get("snippets_url")

    downloaded = {}

    if catalog_url:
        downloaded["catalog"] = download_file(catalog_url, CATALOG_CSV)

    if snippets_url:
        archive_path = BASE_DIR / "caption_snippets_30s.tar.gz"
        downloaded["snippets_archive"] = download_file(snippets_url, archive_path)
        safe_extract_tar_gz(archive_path, BASE_DIR)
        downloaded["snippets_dir"] = str(SNIPPETS_DIR)

    snippets_count = len(list(SNIPPETS_DIR.glob("*.wav"))) if SNIPPETS_DIR.exists() else 0

    return {
        "ok": True,
        "downloaded": downloaded,
        "base_dir": str(BASE_DIR),
        "snippets_count": snippets_count,
        "catalog_exists": CATALOG_CSV.exists()
    }


def action_caption_batch(job_input):
    load_model()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    limit = int(job_input.get("limit", 5))
    only_stems = job_input.get("stems", None)

    if not CATALOG_CSV.exists():
        return {
            "ok": False,
            "error": f"Catalog not found: {CATALOG_CSV}"
        }

    df = pd.read_csv(CATALOG_CSV)
    done = already_done()

    if only_stems:
        only_stems = [str(s) for s in only_stems]
        df = df[df["stem"].astype(str).isin(only_stems)]

    processed = []
    errors = []

    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            stem = str(row["stem"])

            if stem in done:
                continue

            audio_path = SNIPPETS_DIR / f"{stem}_middle30.wav"

            if not audio_path.exists():
                errors.append({
                    "stem": stem,
                    "error": f"missing audio {audio_path}"
                })
                continue

            try:
                raw = analyze_audio(audio_path)
                record = {
                    "stem": stem,
                    "source_title": str(row.get("source_title", "")),
                    "bpm": str(row.get("bpm", "")),
                    "key": str(row.get("key", "")),
                    "audio_path": str(audio_path),
                    "qwen2_raw": raw,
                    "error": ""
                }
            except Exception as e:
                record = {
                    "stem": stem,
                    "source_title": str(row.get("source_title", "")),
                    "bpm": str(row.get("bpm", "")),
                    "key": str(row.get("key", "")),
                    "audio_path": str(audio_path),
                    "qwen2_raw": "",
                    "error": str(e)
                }
                errors.append({
                    "stem": stem,
                    "error": str(e)
                })

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            processed.append(stem)

            if len(processed) >= limit:
                break

    return {
        "ok": True,
        "processed_count": len(processed),
        "processed": processed,
        "errors": errors[:10],
        "results_jsonl": str(RESULTS_JSONL)
    }


def action_status(job_input):
    done = already_done()

    snippets_count = len(list(SNIPPETS_DIR.glob("*.wav"))) if SNIPPETS_DIR.exists() else 0

    catalog_rows = 0
    if CATALOG_CSV.exists():
        try:
            catalog_rows = len(pd.read_csv(CATALOG_CSV))
        except Exception:
            pass

    return {
        "ok": True,
        "base_dir": str(BASE_DIR),
        "snippets_count": snippets_count,
        "catalog_rows": catalog_rows,
        "captions_done": len(done),
        "results_jsonl": str(RESULTS_JSONL)
    }


def action_read_results(job_input):
    limit = int(job_input.get("limit", 5))

    if not RESULTS_JSONL.exists():
        return {
            "ok": False,
            "error": f"Results file not found: {RESULTS_JSONL}"
        }

    records = []

    with RESULTS_JSONL.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except Exception as e:
            records.append({
                "parse_error": str(e),
                "raw_line": line[:500]
            })

    return {
        "ok": True,
        "results_jsonl": str(RESULTS_JSONL),
        "total_lines": len(lines),
        "records": records
    }


def action_clear_results(job_input):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()

    return {
        "ok": True,
        "message": "Results cleared",
        "results_jsonl": str(RESULTS_JSONL)
    }



# -----------------------------
# CLAP zero-shot music labeling
# -----------------------------

GENRE_LABELS = [
    "corporate motivational background music",
    "upbeat commercial pop background music",
    "positive advertising background music",
    "inspiring business presentation music",
    "electronic technology background music",
    "modern electronic background music",
    "edm pop background music",
    "synthwave retro electronic music",
    "cinematic emotional background music",
    "cinematic orchestral background music",
    "epic orchestral trailer music",
    "ambient cinematic background music",
    "peaceful ambient background music",
    "emotional piano background music",
    "acoustic folk background music",
    "happy acoustic guitar background music",
    "funk groove background music",
    "rock energetic background music",
    "hip hop urban background music",
    "luxury lounge background music"
]

MOOD_LABELS = [
    "happy and positive",
    "uplifting and inspiring",
    "motivational and confident",
    "optimistic and bright",
    "playful and fun",
    "warm and friendly",
    "calm and peaceful",
    "dreamy and atmospheric",
    "emotional and touching",
    "romantic and tender",
    "sad and melancholic",
    "dramatic and cinematic",
    "dark and tense",
    "mysterious and deep",
    "energetic and powerful",
    "epic and heroic",
    "relaxed and meditative",
    "modern and stylish"
]

INSTRUMENT_LABELS = [
    "piano",
    "soft piano",
    "acoustic guitar",
    "electric guitar",
    "strings",
    "full orchestra",
    "cinematic drums",
    "electronic drums",
    "synthesizer",
    "synth pads",
    "pluck synth",
    "bass",
    "sub bass",
    "percussion",
    "claps",
    "bells",
    "brass",
    "choir",
    "violin",
    "cello"
]

USE_CASE_LABELS = [
    "advertising music", "corporate video music", "business presentation music",
    "youtube background music", "travel vlog music", "lifestyle video music",
    "technology promo music", "fashion video music", "family video music",
    "cinematic trailer music", "documentary music", "film score music",
    "game music", "meditation background music", "sports promo music",
    "children animation music", "luxury brand music"
]


def load_clap_model():
    global clap_model, clap_processor

    if clap_model is not None and clap_processor is not None:
        return

    clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    # Keep CLAP in float32. fp16 causes dtype mismatch in batch norm on some CUDA builds.
    clap_model = ClapModel.from_pretrained(
        CLAP_MODEL_ID,
        torch_dtype=torch.float32
    )

    if torch.cuda.is_available():
        clap_model = clap_model.to("cuda")

    clap_model.eval()


def clap_rank(audio_path, labels, top_k=5):
    load_clap_model()

    target_sr = clap_processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)

    texts = [f"This is {label}." for label in labels]

    inputs = clap_processor(
        text=texts,
        audio=[audio],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = clap_model(**inputs)
        logits = outputs.logits_per_audio[0]
        probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()

    ranked = sorted(
        [{"label": labels[i], "score": float(probs[i])} for i in range(len(labels))],
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked[:top_k]


def create_training_caption(genre_top, mood_top, instr_top, use_top, row):
    # Use CLAP as a ranking signal, but keep captions conservative and non-contradictory.

    genre = genre_top[0]["label"] if genre_top else "instrumental background music"

    # Normalize overly broad genre labels for training captions.
    genre_map = {
        "positive advertising background music": "commercial background music",
        "upbeat commercial pop background music": "upbeat commercial pop background music",
        "corporate motivational background music": "corporate motivational background music",
        "inspiring business presentation music": "corporate inspirational background music",
        "modern electronic background music": "modern electronic background music",
        "edm pop background music": "electronic pop background music",
        "cinematic emotional background music": "cinematic emotional background music",
        "epic orchestral trailer music": "epic cinematic orchestral music",
        "ambient cinematic background music": "ambient cinematic background music",
        "emotional piano background music": "emotional piano background music",
        "acoustic folk background music": "acoustic folk background music",
        "rock energetic background music": "energetic rock background music",
        "funk groove background music": "funk groove background music",
    }
    genre = genre_map.get(genre, genre)

    raw_moods = [x["label"] for x in mood_top if x.get("score", 0) >= 0.08]

    # Avoid contradictory mood sets.
    positive = [
        "happy and positive",
        "uplifting and inspiring",
        "motivational and confident",
        "optimistic and bright",
        "playful and fun",
        "warm and friendly",
        "energetic and powerful",
        "epic and heroic",
    ]
    emotional_dark = [
        "sad and melancholic",
        "dramatic and cinematic",
        "dark and tense",
        "mysterious and deep",
        "emotional and touching",
        "romantic and tender",
    ]
    calm = [
        "calm and peaceful",
        "dreamy and atmospheric",
        "relaxed and meditative",
    ]

    def first_group(moods):
        pos_hits = [m for m in moods if m in positive]
        dark_hits = [m for m in moods if m in emotional_dark]
        calm_hits = [m for m in moods if m in calm]

        groups = [
            ("positive", pos_hits),
            ("emotional_dark", dark_hits),
            ("calm", calm_hits),
        ]
        groups = [g for g in groups if g[1]]

        if not groups:
            return moods[:2]

        # choose group with highest first occurrence in original ranking
        best = min(groups, key=lambda g: moods.index(g[1][0]))
        return best[1][:2]

    moods = first_group(raw_moods)
    if not moods and mood_top:
        moods = [mood_top[0]["label"]]

    raw_instruments = [x["label"] for x in instr_top if x.get("score", 0) >= 0.05]

    # Remove cinematic drums from non-cinematic/non-epic genres to reduce false positives.
    if not any(w in genre for w in ["cinematic", "epic", "orchestral", "trailer"]):
        raw_instruments = [i for i in raw_instruments if i != "cinematic drums"]

    # Avoid soft piano + piano duplicates.
    cleaned_instruments = []
    for inst in raw_instruments:
        if inst == "soft piano" and "piano" in cleaned_instruments:
            continue
        if inst == "piano" and "soft piano" in cleaned_instruments:
            cleaned_instruments = [x for x in cleaned_instruments if x != "soft piano"]
        if inst not in cleaned_instruments:
            cleaned_instruments.append(inst)

    instruments = cleaned_instruments[:4]

    raw_uses = [x["label"] for x in use_top if x.get("score", 0) >= 0.08]
    uses = raw_uses[:2]

    bpm = str(row.get("bpm", "")).strip()
    key = str(row.get("key", "")).strip()

    parts = [f"Instrumental {genre}"]

    if bpm:
        parts.append(f"around {bpm} BPM")
    if key:
        parts.append(f"in {key}")
    if instruments:
        parts.append("featuring " + ", ".join(instruments))
    if moods:
        parts.append("with a " + ", ".join(moods) + " mood")
    if uses:
        parts.append("suitable for " + ", ".join(uses))

    return ", ".join(parts) + "."

def already_done_clap():
    done = set()

    if CLAP_RESULTS_JSONL.exists():
        with CLAP_RESULTS_JSONL.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    stem = json.loads(line).get("stem")
                    if stem:
                        done.add(str(stem))
                except Exception:
                    pass

    return done


def action_clap_batch(job_input):
    load_clap_model()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    limit = int(job_input.get("limit", 5))
    only_stems = job_input.get("stems", None)

    if not CATALOG_CSV.exists():
        return {
            "ok": False,
            "error": f"Catalog not found: {CATALOG_CSV}"
        }

    df = pd.read_csv(CATALOG_CSV)
    done = already_done_clap()

    if only_stems:
        only_stems = [str(s) for s in only_stems]
        df = df[df["stem"].astype(str).isin(only_stems)]

    processed = []
    errors = []

    with CLAP_RESULTS_JSONL.open("a", encoding="utf-8") as f:
        for _, row in df.iterrows():
            stem = str(row["stem"])

            if stem in done:
                continue

            audio_path = SNIPPETS_DIR / f"{stem}_middle30.wav"

            if not audio_path.exists():
                errors.append({
                    "stem": stem,
                    "error": f"missing audio {audio_path}"
                })
                continue

            try:
                genre_top = clap_rank(audio_path, GENRE_LABELS, top_k=5)
                mood_top = clap_rank(audio_path, MOOD_LABELS, top_k=7)
                instr_top = clap_rank(audio_path, INSTRUMENT_LABELS, top_k=7)
                use_top = clap_rank(audio_path, USE_CASE_LABELS, top_k=5)

                training_caption = create_training_caption(
                    genre_top, mood_top, instr_top, use_top, row
                )

                record = {
                    "stem": stem,
                    "source_title": str(row.get("source_title", "")),
                    "bpm": str(row.get("bpm", "")),
                    "key": str(row.get("key", "")),
                    "audio_path": str(audio_path),
                    "genre_top": genre_top,
                    "mood_top": mood_top,
                    "instruments_top": instr_top,
                    "use_case_top": use_top,
                    "training_caption": training_caption,
                    "error": ""
                }
            except Exception as e:
                record = {
                    "stem": stem,
                    "source_title": str(row.get("source_title", "")),
                    "bpm": str(row.get("bpm", "")),
                    "key": str(row.get("key", "")),
                    "audio_path": str(audio_path),
                    "genre_top": [],
                    "mood_top": [],
                    "instruments_top": [],
                    "use_case_top": [],
                    "training_caption": "",
                    "error": str(e)
                }
                errors.append({
                    "stem": stem,
                    "error": str(e)
                })

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            processed.append(stem)

            if len(processed) >= limit:
                break

    return {
        "ok": True,
        "processed_count": len(processed),
        "processed": processed,
        "errors": errors[:10],
        "results_jsonl": str(CLAP_RESULTS_JSONL)
    }


def action_read_clap_results(job_input):
    limit = int(job_input.get("limit", 5))

    if not CLAP_RESULTS_JSONL.exists():
        return {
            "ok": False,
            "error": f"CLAP results file not found: {CLAP_RESULTS_JSONL}"
        }

    records = []

    with CLAP_RESULTS_JSONL.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except Exception as e:
            records.append({
                "parse_error": str(e),
                "raw_line": line[:500]
            })

    return {
        "ok": True,
        "results_jsonl": str(CLAP_RESULTS_JSONL),
        "total_lines": len(lines),
        "records": records
    }


def action_clear_clap_results(job_input):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if CLAP_RESULTS_JSONL.exists():
        CLAP_RESULTS_JSONL.unlink()

    return {
        "ok": True,
        "message": "CLAP results cleared",
        "results_jsonl": str(CLAP_RESULTS_JSONL)
    }


def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "health")

    if action == "health":
        return action_health(job_input)

    if action == "download_data":
        return action_download_data(job_input)

    if action == "caption_batch":
        return action_caption_batch(job_input)

    if action == "status":
        return action_status(job_input)

    if action == "read_results":
        return action_read_results(job_input)

    if action == "clear_results":
        return action_clear_results(job_input)

    if action == "clap_batch":
        return action_clap_batch(job_input)

    if action == "read_clap_results":
        return action_read_clap_results(job_input)

    if action == "clear_clap_results":
        return action_clear_clap_results(job_input)

    return {
        "ok": False,
        "error": f"Unknown action: {action}"
    }


runpod.serverless.start({"handler": handler})
