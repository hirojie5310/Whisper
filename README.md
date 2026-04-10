# Whisper
# 🎧 English Vocabulary Audio → CSV Converter

英単語＋和訳を読み上げた音声（mp3）から、
**英単語と和訳のペアを抽出しCSV化するツール**です。

Whisper + 音声分割 + 後処理により、単語帳形式の音声を高精度に構造化します。

---

## ✨ 特徴

* 🎤 OpenAI Whisper による高精度音声認識
* ✂️ 音量の谷（相対的な静音）で音声を自動分割
* 🔁 英語の繰り返し・ノイズ（例: "thank you for watching!"）を除去
* 🔤 英単語を自動で小文字化
* 🧹 和訳の不要な英語混入を除去
* 📁 複数mp3ファイルを一括処理
* 📄 1つのCSVにまとめて出力

---

## 📂 入力形式

以下のような音声を想定しています：

```
follow → に続く → follow（繰り返し）
consider → 考慮する → consider
...
```

---

## 📤 出力形式

```csv
index,source_file,index_in_file,英単語,和訳
1,02_02_system_5th.mp3,1,government,政府
2,02_02_system_5th.mp3,2,knowledge,知識
3,02_02_system_5th.mp3,3,nation,国
...
```

---

## 🛠️ 環境構築

### 1. Python

Python 3.9以上を推奨

---

### 2. 必要ライブラリ

```bash
pip install openai-whisper
pip install torch
pip install pydub
pip install librosa
pip install soundfile
pip install numpy
```

---

### 3. ffmpeg のインストール（必須）

Whisper / pydubで使用します。

#### Windows

1. https://ffmpeg.org/download.html からダウンロード
2. `C:\ffmpeg\bin` を環境変数 `Path` に追加

確認：

```bash
ffmpeg -version
```

---

## 🚀 使い方

### 1. 音声ファイル配置

```
S011102_B3/
 ├── 02_02_system_5th.mp3
 ├── 02_03_system_5th.mp3
 ├── ...
```

---

### 2. 設定

```python
folder_path = Path("S011102_B3")
START_NO = 2
END_NO = 81
```

---

### 3. 実行

```bash
python main.py
```

---

### 4. 出力

```
S011102_B3_all_transcribed.csv
```

---

## ⚙️ チューニング

### 音声分割の精度調整

```python
LOW_ENERGY_PERCENTILE = 35
MIN_BOUNDARY_GAP_MS = 250
MAX_CHUNK_MS = 1800
```

| パラメータ                 | 効果                  |
| --------------------- | ------------------- |
| LOW_ENERGY_PERCENTILE | 小さいほど分割減 / 大きいほど分割増 |
| MIN_BOUNDARY_GAP_MS   | 分割間隔の最小値            |
| MAX_CHUNK_MS          | チャンク最大長             |

---

### ノイズ除去ルール

以下のようなノイズを自動除去：

* thank you for watching!
* thank you
* 英語のみの和訳
* 長い英語フレーズ

---

## ⚠️ 注意点

* 完全自動では100%精度は出ません
* 和訳の誤認識は辞書補正が必要な場合あり
* GPU推奨（CPUだとかなり遅い）

---

## 🔧 改善アイデア（今後）

* 📚 英単語辞書によるスペル補正
* 🇯🇵 和訳辞書による自動修正
* 📊 Excel出力フォーマット拡張
* ⚡ 並列処理による高速化
* 🌐 Web UI化

---

## 📄 ライセンス

MIT License

---

## 🙌 作者メモ

このツールは「英単語音声 → CSV化」を効率化するために作成しました。
音声の特性に合わせて、分割・抽出ロジックをかなりチューニングしています。

---

## ⭐ お願い

役に立ったら Star ⭐ いただけると嬉しいです！
