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
    "cinematic music", "corporate music", "pop music", "electronic music",
    "ambient music", "orchestral music", "classical music", "rock music",
    "acoustic music", "folk music", "hip hop music", "trap music",
    "techno music", "house music", "edm music", "synthwave music",
    "funk music", "jazz music", "lounge music", "piano music",
    "trailer music", "epic music", "children music", "holiday music"
]

MOOD_LABELS = [
    "happy", "uplifting", "inspiring", "motivational", "positive",
    "optimistic", "playful", "fun", "bright", "warm", "calm",
    "peaceful", "dreamy", "emotional", "romantic", "sad",
    "dramatic", "dark", "tense", "mysterious", "deep", "energetic",
    "powerful", "epic", "relaxed", "meditative", "melancholic"
]

INSTRUMENT_LABELS = [
    "piano", "electric guitar", "acoustic guitar", "ukulele", "strings",
    "orchestra", "synthesizer", "synth pads", "bass guitar", "sub bass",
    "drums", "electronic drums", "percussion", "claps", "bells",
    "glockenspiel", "choir", "brass", "woodwinds", "violin",
    "cello", "pluck synth", "whistle", "vocal chops"
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
    clap_model = ClapModel.from_pretrained(
        CLAP_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
    genre = genre_top[0]["label"] if genre_top else "instrumental music"
    moods = [x["label"] for x in mood_top[:3]]
    instruments = [x["label"] for x in instr_top[:5]]
    uses = [x["label"] for x in use_top[:3]]

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
