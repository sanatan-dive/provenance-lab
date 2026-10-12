# Setup guide

How to get Provenance Lab running on your own computer. There are two paths:

- **Path A, run the app (about 15 minutes):** download our trained models and the demo samples, then start the web app.
- **Path B, rebuild everything from scratch (several hours):** download every dataset and retrain every model yourself.

Do Path A first even if you plan to do Path B. It confirms your setup works.

## What you need

| | |
|---|---|
| OS | macOS or Linux. On Windows, use WSL2 (Ubuntu); the commands below are written for a bash shell. |
| Python | 3.12 (`python3.12 --version`) |
| Tools | `git`, `curl` |
| RAM | 8 GB is enough for Path A, 16 GB for Path B |
| Disk | Path A: about 6 GB (Python packages + models). Path B: about 25 GB more for datasets. |
| GPU | Optional. The code uses an NVIDIA GPU (CUDA) or an Apple Silicon GPU (MPS) when it finds one, otherwise the CPU. |

## Path A: run the app

### 1. Get the code

```bash
git clone https://github.com/sanatan-dive/provenance-lab.git
cd provenance-lab/code
```

All later commands run inside `provenance-lab/code`.

### 2. Create the Python environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

This downloads about 3 GB (PyTorch is most of it). With an NVIDIA GPU on Linux, the default PyTorch wheel already includes CUDA.

### 3. Download the trained models and demo samples

They are in the [v1.0 release](https://github.com/sanatan-dive/provenance-lab/releases/tag/v1.0) because they are too large for git.

```bash
R=https://github.com/sanatan-dive/provenance-lab/releases/download/v1.0
curl -L -o results/text/text_bert_diverse_12k.pt     $R/text_bert_diverse_12k.pt
curl -L -o results/code/code_codebert_supcon_12k.pt  $R/code_codebert_supcon_12k.pt
curl -L -o results/image/image_genimage.pt           $R/image_genimage.pt
curl -L -o demo-data.tar.gz                          $R/demo-data.tar.gz
tar -xzf demo-data.tar.gz && rm demo-data.tar.gz
```

| File | Size | What it is |
|---|---|---|
| `text_bert_diverse_12k.pt` | 440 MB | essay detector (BERT + forensic features) |
| `code_codebert_supcon_12k.pt` | 500 MB | code detector (CodeBERT + syntax-tree features) |
| `image_genimage.pt` | 89 MB | image detector (ViT-Small + FFT branch) |
| `demo-data.tar.gz` | 400 MB | held-out test samples behind the "Human-made" / "AI-made" buttons, unpacked into `data/` |

The voice model (`results/audio/audio_w2v_modern.pkl`) is small and is already in the repo.
To check the downloads, compare `shasum -a 256 <file>` with `SHA256SUMS.txt` on the release page.

### 4. Start the app

```bash
.venv/bin/python server.py
```

Open http://127.0.0.1:8000. The first request for each kind of content takes a little longer because it downloads the pretrained encoders from Hugging Face (BERT, CodeBERT, ViT-Small, wav2vec 2.0, DistilGPT-2; about 2 GB in total, cached in `~/.cache/huggingface`).

You can paste text, drop in a file, or press "Human-made" / "AI-made" to load a real held-out test sample.

### 5. Check that it works (optional)

Leave the server running and in a second terminal, from `provenance-lab/code`:

```bash
.venv/bin/python tools/e2e_api.py 5      # sends test samples of every kind to the API; prints accuracy, writes results/e2e_api.json
.venv/bin/playwright install chromium
.venv/bin/python tools/e2e_ui.py         # headless browser test of every tab; screenshots in results/screens/
```

You can also run the detector from the command line:

```bash
.venv/bin/python detect.py path/to/file.jpg
.venv/bin/python detect.py --text "an essay ..."
```

## Path B: download every dataset and retrain

All datasets are public and come from Hugging Face. Each one downloads itself into `code/data/` the first time a training script needs it. The only exception is GenImage, which you download once by hand (step 1).

| Content | Dataset | Hugging Face source | Size |
|---|---|---|---|
| Essays | HC3 (human vs ChatGPT answers) | [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | small |
| Essays | MAGE (27 LLMs, 10 domains) | [yaful/MAGE](https://huggingface.co/datasets/yaful/MAGE) | ~0.5 GB |
| Code | CoDET-M4, Python subset | [DaniilOr/CoDET-M4](https://huggingface.co/datasets/DaniilOr/CoDET-M4) | ~0.5 GB |
| Images | Tiny-GenImage (7 generators + ImageNet) | [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) | ~8 GB |
| Images | CIFAKE (benchmark only) | [dragonintelligence/CIFAKE-image-dataset](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset) | small |
| Voice | ASVspoof 2019 LA | [SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA) | ~2.5 GB |
| Voice | Modern TTS (ElevenLabs, Polly, Hume...) | [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection) | ~1 GB |

Each dataset has its own license (see its Hugging Face page). GenImage is for non-commercial research use only.

Every command below runs from `provenance-lab/code`. `tools/memrun.py` is a memory watchdog: it stops a job that goes above `--limit-gb` or when the machine runs low on memory. Run one job at a time. Each job prints its test metrics at the end and writes them to `results/<content>/`.

### 1. Download GenImage

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('TheKernel01/Tiny-GenImage', repo_type='dataset', revision='refs/convert/parquet', local_dir='data/genimage/hf', max_workers=4)"
```

### 2. Train the four detectors the app uses

```bash
M=".venv/bin/python tools/memrun.py --limit-gb 3.5 -- .venv/bin/python"

# Essays: HC3 + 12,000 MAGE texts (OPT family held out), then set the threshold on MAGE validation
N_MAGE=12000 $M -m text.train_diverse
N_MAGE=12000 $M -m text.calibrate

# Code: CodeBERT + syntax-tree features on 12,000 CoDET-M4 Python files
$M -m code_det.train --n-train 12000 --n-test 1000 --epochs 3 --name code_codebert_supcon_12k

# Images: ViT-Small + FFT on Tiny-GenImage
$M -m image.genimage

# Voice: ASVspoof cache + spectrogram CNN, then wav2vec 2.0 + logistic regression, then add modern TTS
$M -m audio.train --epochs 15
$M -m audio.w2v
$M -m audio.modern
```

These write the same files you downloaded in Path A (`results/text/text_bert_diverse_12k.pt`, `results/code/code_codebert_supcon_12k.pt`, `results/image/image_genimage.pt`, `results/audio/audio_w2v_modern.pkl`) and they overwrite them.

### 3. Run the other experiments in the report (optional)

```bash
./run_all.sh                 # HC3/CIFAKE/ASVspoof benchmark models and the ablations (no contrastive loss, no FFT branch)
./run_final.sh               # public detectors, MAGE out-of-distribution, unseen-generator code runs
.venv/bin/python baselines.py
$M -m image.genimage --holdout Midjourney          # unseen image generator
$M -m provenance.robustness                        # watermark robustness under edits
.venv/bin/python summarize.py                      # results/summary.md: one table of every run
```

## Rebuild the report (optional)

This needs Node.js 20+ and LibreOffice (`soffice`). The script reads its numbers from `code/results/`, so rebuild the report after retraining.

```bash
cd Report/build && npm install && ../../code/.venv/bin/python make_figures.py && node build_report.js
cd .. && soffice --headless --convert-to pdf Project_Report.docx
```

The slides (`Presentation/Provenance_Lab.pptx` and `.pdf`) come prebuilt. Their build script, `Presentation/build/build_deck.js`, needs a theme helper that is not in this repo.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: results/.../*.pt` | Step 3 of Path A was skipped or a download failed. Re-run the `curl` line for that file. |
| `FileNotFoundError: data/...` when pressing a demo button | `demo-data.tar.gz` was not unpacked inside `code/`. |
| `Address already in use` on port 8000 | Another server is running. Stop it, or change the port in the last line of `server.py`. |
| A job gets killed by `memrun.py` | Your machine ran low on memory. Close other apps, or raise `--limit-gb` if you have more RAM. |
| Slow first request | The pretrained encoders are downloading from Hugging Face. This only happens once. |
