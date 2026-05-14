#!/usr/bin/env python3
"""Batch transcribe audio files using faster-whisper large-v3.

Usage:
  python transcribe_batch.py                     # process all pending audio
  python transcribe_batch.py --model medium       # use smaller model (faster, less accurate)
  python transcribe_batch.py --vids Xxz5uh0L1mE  # specific videos only
"""

import argparse
import json
import sys
from pathlib import Path

AUDIO_DIR = Path(__file__).parent / "audio"
TRANSCRIPT_DIR = Path(__file__).parent / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def get_pending_audio() -> list[Path]:
    """Return audio files that don't have a transcript yet."""
    done = {f.stem for f in TRANSCRIPT_DIR.iterdir() if f.suffix == ".txt"}
    pending = []
    for f in sorted(AUDIO_DIR.glob("*.mp3")):
        vid = f.stem
        if vid in done:
            print(f"  SKIP {vid} (transcript exists)")
        else:
            pending.append(f)
    return pending


def transcribe(audio_path: Path, model_name: str = "large-v3") -> dict:
    """Run faster-whisper on one audio file, return segments."""
    from faster_whisper import WhisperModel

    vid = audio_path.stem
    print(f"\n{'='*60}")
    print(f"Transcribing {vid}...")
    print(f"  File: {audio_path.name} ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), language="zh", beam_size=5, vad_filter=True)

    print(f"  Detected language: {info.language} (p={info.language_probability:.2f})")
    print(f"  Duration: {info.duration:.0f}s")

    result = []
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        if len(result) % 50 == 0:
            print(f"  ... {len(result)} segments so far")

    print(f"  Done! {len(result)} segments transcribed.")
    return {"video_id": vid, "segments": result, "duration_s": info.duration}


def save_transcript(vid: str, data: dict):
    """Save as plain text transcript (readable) + JSON (for later processing)."""
    txt_path = TRANSCRIPT_DIR / f"{vid}.txt"
    json_path = TRANSCRIPT_DIR / f"{vid}.json"

    # Plain text version
    lines = []
    for seg in data["segments"]:
        ts = f"[{int(seg['start']//60):02d}:{int(seg['start']%60):02d}]"
        lines.append(f"{ts} {seg['text']}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # JSON version
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    size_kb = txt_path.stat().st_size / 1024
    print(f"  Saved: {txt_path.name} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe with faster-whisper")
    parser.add_argument("--model", default="large-v3", choices=["large-v3", "medium", "small"])
    parser.add_argument("--vids", nargs="*", help="Specific video IDs to transcribe")
    args = parser.parse_args()

    if args.vids:
        pending = [AUDIO_DIR / f"{v}.mp3" for v in args.vids]
        pending = [p for p in pending if p.exists()]
    else:
        pending = get_pending_audio()

    if not pending:
        print("No audio files to transcribe.")
        return

    total_seconds = 0
    for i, audio_path in enumerate(pending, 1):
        vid = audio_path.stem
        print(f"\n[{i}/{len(pending)}] Processing {vid}...")
        try:
            data = transcribe(audio_path, args.model)
            save_transcript(vid, data)
            total_seconds += data["duration_s"]
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    hours = total_seconds / 3600
    print(f"\n{'='*60}")
    print(f"Batch complete! Processed {len(pending)} videos ({hours:.1f}h of audio)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
