import os
import json
import tarfile
import urllib.request
from pathlib import Path

import runpod
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

BASE_DIR = Path(os.getenv("TOPAUDIO_BASE_DIR", "/runpod-volume/topaudio_ai"))
SNIPPETS_DIR = BASE_DIR / "caption_snippets_30s"
CATALOG_CSV = BASE_DIR / "catalog_caption_snippets.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_JSONL = RESULTS_DIR / "qwen2_audio_captions.jsonl"

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")

model = None
processor = None

PROMPT = """
You are a professional music supervisor and dataset labeling expert.

Analyze this instrumental music audio clip. Do not use any external metadata. Only use what you hear.

Return strict JSON with these fields:
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
- training_caption should be one clean sentence for AI music model training.
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
    load_model()

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
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
        audios=[str(audio_path)],
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

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output

def action_health(job_input):
    return {
        "ok": True,
        "base_dir": str(BASE_DIR),
        "snippets_exists": SNIPPETS_DIR.exists(),
        "catalog_exists": CATALOG_CSV.exists(),
        "results_jsonl": str(RESULTS_JSONL),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
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
        return {"ok": False, "error": f"Catalog not found: {CATALOG_CSV}"}

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
                errors.append({"stem": stem, "error": f"missing audio {audio_path}"})
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
                errors.append({"stem": stem, "error": str(e)})

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
        "snippets_count": snippets_count,
        "catalog_rows": catalog_rows,
        "captions_done": len(done),
        "results_jsonl": str(RESULTS_JSONL)
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

    return {
        "ok": False,
        "error": f"Unknown action: {action}"
    }

runpod.serverless.start({"handler": handler})
