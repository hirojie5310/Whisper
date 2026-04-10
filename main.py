from pathlib import Path
from typing import Any, cast
import shutil
import whisper
import csv
import re
import tempfile

import numpy as np
import librosa
from pydub import AudioSegment

# ========= 設定 =========
folder_path = Path(r"S01110G_B3")
output_csv = folder_path / "S01110G_B3_all_transcribed.csv"
file_head = "G"  # ファイル名の頭部分（例: "04"）

MODEL_NAME = "medium"  # tiny, base, small, medium, large
DEVICE = "cuda"  # "cuda" or "cpu"

# 対象ファイル名: G_40_system_5th.mp3 ～ G_82_system_5th.mp3
START_NO = 1
END_NO = 23

# ---- 音量谷ベース分割設定 ----
SR = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 256
LOW_ENERGY_PERCENTILE = 35
MIN_BOUNDARY_GAP_MS = 250
MIN_CHUNK_MS = 500
MAX_CHUNK_MS = 1800
KEEP_MARGIN_MS = 100

# よくある誤認識の補正辞書
JA_FIX_DICT = {
    "およきする": "予期する",
    "人工": "人口",
}

# 明らかなノイズ行
NOISE_PHRASES = {
    ("thank you for", "watching!"),
    ("thank", "you."),
    ("thank you", "for watching!"),
}

NOISE_JAPANESE_LINES = {
    (
        "この文章は 日本語の文章の中で 日本語の文章の中で 日本語の文章の中で "
        "日本語の文章の中で 日本語の文章の中で 日本語の文章の中で 日本語の文章の中で "
        "日本語の文章の中で 日本語の文章の中で 日本語の文章の中で 日本語の文章の中で "
        "日本語の文章の中で 日本語の文章の中で 日本語の文章の中で 日本語の文章の中で "
        "日本語の文章の中で 日本語の文章の中で 日本語の文章の中で 日本語の文章の中で "
        "日本語の文章の中で 日本語の文章の中で 日本語の文章の"
    ),
    "この文は 日本語での文字の使用方法です",
}

# ========= 共通正規表現 =========
EN_JA_PATTERN = re.compile(r"^([A-Za-z][A-Za-z' -]*)\s+(.+)$")


def is_english_only(text: str) -> bool:
    text = text.strip()
    return bool(text) and bool(re.fullmatch(r"[A-Za-z' -.!?]+", text))


def is_japanese_like(text: str) -> bool:
    text = text.strip()
    return bool(text) and bool(re.search(r"[ぁ-んァ-ン一-龥]", text))


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("、", " ").replace("。", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_english(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(".", "").replace("!", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noise_japanese_line(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    return normalized in NOISE_JAPANESE_LINES


def fix_particle_wo_for_verbs(eng: str, jap: str) -> str:
    if not jap.startswith("お"):
        return jap

    # 動詞訳の可能性が高い場合のみ、先頭「お」を助詞「を」に補正
    looks_like_verb = (
        eng.startswith("to ")
        or "する" in jap
        or jap.endswith(("る", "う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む"))
    )
    if looks_like_verb:
        return "を" + jap[1:]
    return jap


def normalize_japanese(text: str) -> str:
    text = text.strip()

    # 末尾の英語を削除
    text = re.sub(r"\s+[A-Za-z' -]+$", "", text).strip()

    # 補正辞書
    text = JA_FIX_DICT.get(text, text)

    return text


def is_noise_pair(eng: str, jap: str) -> bool:
    eng_n = eng.strip().lower()
    jap_n = jap.strip().lower()

    if (eng_n, jap_n) in NOISE_PHRASES:
        return True

    # 和訳が日本語でない場合はノイズ候補
    if not is_japanese_like(jap):
        return True

    # 英単語側が長いフレーズならノイズ候補
    # ただし将来熟語を扱うなら、この条件は緩めてもよい
    if len(eng_n.split()) >= 3:
        return True

    return False


def extract_rows_from_segments(
    segments: list[dict[str, Any]],
    source_file: str,
) -> list[tuple[str, int, str, str]]:

    raw_texts: list[str] = []
    for seg in segments:
        text = clean_text(str(seg["text"]))
        if text:
            raw_texts.append(text)

    rows: list[tuple[str, int, str, str]] = []
    index_in_file = 1
    i = 0
    prev_eng_in_rows = None

    while i < len(raw_texts):
        text = raw_texts[i]

        # 1) 「英単語 + 和訳」
        m = EN_JA_PATTERN.match(text)
        if m:
            eng = normalize_english(m.group(1))
            jap = normalize_japanese(m.group(2).strip())
            jap = fix_particle_wo_for_verbs(eng, jap)

            # 例: knowledge / Nation 国 -> nation / 国
            lead = EN_JA_PATTERN.match(jap)
            if prev_eng_in_rows is not None and eng == prev_eng_in_rows and lead:
                next_eng = normalize_english(lead.group(1))
                next_jap = normalize_japanese(lead.group(2).strip())
                next_jap = fix_particle_wo_for_verbs(next_eng, next_jap)

                if next_eng != eng and is_japanese_like(next_jap):
                    eng = next_eng
                    jap = next_jap

            if jap and not is_english_only(jap) and not is_noise_japanese_line(jap):
                if not is_noise_pair(eng, jap):
                    rows.append((source_file, index_in_file, eng, jap))
                    prev_eng_in_rows = eng
                    index_in_file += 1
                i += 1
                continue

        # 2) 「英単語だけ」 + 次が「日本語だけ」
        if is_english_only(text):
            eng = normalize_english(text)

            if i + 1 < len(raw_texts):
                next_text = raw_texts[i + 1]

                if is_japanese_like(next_text) and not is_english_only(next_text):
                    jap = normalize_japanese(next_text)
                    jap = fix_particle_wo_for_verbs(eng, jap)
                    if (
                        jap
                        and not is_noise_japanese_line(jap)
                        and not is_noise_pair(eng, jap)
                    ):
                        rows.append((source_file, index_in_file, eng, jap))
                        prev_eng_in_rows = eng
                        index_in_file += 1
                        i += 2
                        continue

                # 次が「同じ英単語 + 和訳」
                m2 = EN_JA_PATTERN.match(next_text)
                if m2:
                    next_eng = normalize_english(m2.group(1))
                    next_jap = normalize_japanese(m2.group(2).strip())
                    next_jap = fix_particle_wo_for_verbs(next_eng, next_jap)

                    if eng == next_eng and next_jap and not is_english_only(next_jap):
                        if not is_noise_japanese_line(next_jap) and not is_noise_pair(
                            eng, next_jap
                        ):
                            rows.append((source_file, index_in_file, eng, next_jap))
                            prev_eng_in_rows = eng
                            index_in_file += 1
                        i += 2
                        continue

        i += 1

    # 近接重複除去
    deduped: list[tuple[str, int, str, str]] = []
    prev_eng = None
    prev_jap = None

    for _, _, eng, jap in rows:
        if eng == prev_eng and jap == prev_jap:
            continue
        deduped.append((source_file, len(deduped) + 1, eng, jap))
        prev_eng = eng
        prev_jap = jap

    return deduped


def build_target_files(folder: Path, start_no: int, end_no: int) -> list[Path]:
    files: list[Path] = []
    for i in range(start_no, end_no + 1):
        file_name = f"{file_head}_{i:02d}_system_5th.mp3"
        files.append(folder / file_name)
    return files


def transcribe_audio(model: Any, audio_file: Path) -> list[dict[str, Any]]:
    result = model.transcribe(
        str(audio_file),
        fp16=False,
        temperature=0,
        condition_on_previous_text=False,
        word_timestamps=True,
        initial_prompt=(
            "This audio is a vocabulary list. "
            "An English word is followed by its Japanese translation. "
            "Sometimes the English word is repeated. "
            "Transcribe carefully and keep word boundaries clear."
        ),
    )
    return cast(list[dict[str, Any]], result["segments"])


def ms_to_frame(ms: int, sr: int, hop_length: int) -> int:
    return max(1, int((ms / 1000) * sr / hop_length))


def detect_low_energy_boundaries(audio_path: Path) -> list[int]:
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    sr = int(sr)

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    if len(rms) == 0:
        return []

    window = 5
    if len(rms) >= window:
        kernel = np.ones(window) / window
        rms_smooth = np.convolve(rms, kernel, mode="same")
    else:
        rms_smooth = rms

    threshold = np.percentile(rms_smooth, LOW_ENERGY_PERCENTILE)

    candidate_frames = []
    for i in range(1, len(rms_smooth) - 1):
        if (
            rms_smooth[i] <= threshold
            and rms_smooth[i] <= rms_smooth[i - 1]
            and rms_smooth[i] <= rms_smooth[i + 1]
        ):
            candidate_frames.append(i)

    min_gap_frames = ms_to_frame(MIN_BOUNDARY_GAP_MS, sr, HOP_LENGTH)
    filtered_frames = []
    last_frame = -(10**9)

    for f in candidate_frames:
        if f - last_frame >= min_gap_frames:
            filtered_frames.append(f)
            last_frame = f

    boundaries_ms = [
        int(librosa.frames_to_time(f, sr=sr, hop_length=HOP_LENGTH) * 1000)
        for f in filtered_frames
    ]
    return boundaries_ms


def build_chunks_by_boundaries(audio_path: Path) -> list[AudioSegment]:
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    boundaries = detect_low_energy_boundaries(audio_path)
    print(f"谷候補数: {len(boundaries)}")

    points = [0] + boundaries + [total_ms]

    chunks: list[AudioSegment] = []
    current_start = points[0]

    for p in points[1:]:
        if p - current_start < MIN_CHUNK_MS:
            continue

        end = p
        if end - current_start > MAX_CHUNK_MS:
            split_point = current_start + MAX_CHUNK_MS
            start2 = max(0, current_start - KEEP_MARGIN_MS)
            end2 = min(total_ms, split_point + KEEP_MARGIN_MS)
            chunk = audio[start2:end2]
            if len(chunk) > 0:
                chunks.append(chunk)
            current_start = split_point
            continue

        start2 = max(0, current_start - KEEP_MARGIN_MS)
        end2 = min(total_ms, end + KEEP_MARGIN_MS)
        chunk = audio[start2:end2]
        if len(chunk) > 0:
            chunks.append(chunk)

        current_start = end

    if total_ms - current_start >= 250:
        start2 = max(0, current_start - KEEP_MARGIN_MS)
        end2 = total_ms
        chunk = audio[start2:end2]
        if len(chunk) > 0:
            chunks.append(chunk)

    return chunks


def transcribe_chunks(model: Any, audio_path: Path) -> list[dict[str, Any]]:
    chunks = build_chunks_by_boundaries(audio_path)
    print(f"分割チャンク数: {len(chunks)}")

    merged_segments: list[dict[str, Any]] = []
    if not chunks:
        return merged_segments

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, chunk in enumerate(chunks, start=1):
            chunk_path = Path(tmpdir) / f"chunk_{i:03d}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_segments = transcribe_audio(model, chunk_path)
            merged_segments.extend(chunk_segments)

    return merged_segments


print("ffmpeg =", shutil.which("ffmpeg"))
print("folder exists =", folder_path.exists())
print("folder path =", folder_path)

if not folder_path.exists():
    raise FileNotFoundError(f"フォルダが見つかりません: {folder_path}")

target_files = build_target_files(folder_path, START_NO, END_NO)

missing_files = [p.name for p in target_files if not p.exists()]
if missing_files:
    print("\n[warning] 以下のファイルが見つかりません:")
    for name in missing_files:
        print(" -", name)

existing_files = [p for p in target_files if p.exists()]
if not existing_files:
    raise FileNotFoundError("対象のmp3ファイルが1件も見つかりませんでした。")

print(f"\n対象ファイル数: {len(existing_files)}")

model = whisper.load_model(MODEL_NAME, device=DEVICE)

all_rows: list[tuple[int, str, int, str, str]] = []
global_index = 1

for audio_path in existing_files:
    print(f"\n=== Processing: {audio_path.name} ===")

    segments = transcribe_chunks(model, audio_path)
    file_rows = extract_rows_from_segments(segments, audio_path.name)

    if len(file_rows) == 0:
        print("抽出件数が0件のため、分割なしで再解析します。")
        segments = transcribe_audio(model, audio_path)
        file_rows = extract_rows_from_segments(segments, audio_path.name)

    for source_file, index_in_file, eng, jap in file_rows:
        all_rows.append((global_index, source_file, index_in_file, eng, jap))
        global_index += 1

    print(f"抽出件数: {len(file_rows)}")

with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "source_file", "index_in_file", "英単語", "和訳"])
    writer.writerows(all_rows)

print("\n--- first 20 rows ---")
for row in all_rows[:20]:
    print(row)

print(f"\n総件数: {len(all_rows)}")
print(f"CSV saved: {output_csv}")
