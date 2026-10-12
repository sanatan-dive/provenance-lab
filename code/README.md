# Provenance Lab

Detects whether an essay, a Python file, an image or a voice recording was made by a person or by a generative AI model.

Two layers:

1. **Provenance**: checks for verifiable signals. These are C2PA content credentials, AudioSeal audio watermarks, and text watermarks (green-list and SynthID-Text). If one is present, that is proof.
2. **Forensic detector**: if no signal is found, a trained deep model for that kind of content decides. It combines a pretrained encoder with hand-crafted forensic features and is trained with cross-entropy plus supervised contrastive loss.

| Content | Dataset | Encoder | Forensic features |
|---|---|---|---|
| Essays | HC3 | BERT | perplexity, burstiness, stylometry |
| Code | CoDET-M4 (Python) | CodeBERT | syntax-tree structure, identifier and comment style |
| Images | CIFAKE | ViT-Small + FFT-spectrum CNN | radial power spectrum, noise residual |
| Voice | ASVspoof 2019 LA | frozen wav2vec 2.0 + linear head (log-mel CNN as comparison) | MFCC, spectral shape, phase / group delay (CNN) |

## Results (held-out test sets)

| Content | Test set | Ours | Best public detector |
|---|---|---|---|
| Essays | MAGE (27 LLMs, unseen domains) | **85.3%** | 69.0% |
| Essays | HC3 | 97.2% | 92.8% |
| Code | CoDET-M4 (Python) | **97.6%** | none widely available |
| Images | GenImage (7 generators, 224 px) | **90.7%** | 62.5% |
| Voice | Modern TTS (ElevenLabs, Polly, Hume...) | **94.3%** | 78.0% |
| Voice | ASVspoof 2019 (unseen speakers) | 95.3% | 72.2% |

Deployed models: essays `text_bert_diverse_12k` (HC3 + MAGE, threshold calibrated on MAGE validation),
code `code_codebert_supcon_12k`, images `image_genimage`, voice `audio_w2v_modern.pkl`.
Contrastive loss: no measurable gain at this scale (every with/without difference is inside the 95% CI).
Full table of every run: `results/summary.md`. Report: `../Report/Project_Report.pdf`.

## Run the web app

```bash
.venv/bin/python server.py      # http://127.0.0.1:8000
```

You can drop in any file or paste text. The "Human-made" and "AI-made" buttons load real held-out test samples and show whether the model got them right.

From the command line:

```bash
.venv/bin/python detect.py path/to/file.jpg
.venv/bin/python detect.py --text "an essay ..."
```

## Train

```bash
./run_all.sh          # main models, then the ablations (no contrastive loss, no frequency branch)
./run_all.sh main     # main models only
.venv/bin/python summarize.py   # results/summary.md
```

Every job runs through `tools/memrun.py`, which kills it if it goes above 3 GB or the machine runs low on memory. Training holds out 10% of each training set for validation, stops early on validation loss, and restores the best epoch. It saves ROC curves, confusion matrices and learning curves to `results/<modality>/`.

Other experiments (each runs alone, under the memory watchdog):

```bash
.venv/bin/python baselines.py                    # simple baselines on the same splits
.venv/bin/python tools/memrun.py --limit-gb 3.5 -- .venv/bin/python market.py        # public detectors
.venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python ood_text.py        # MAGE out-of-distribution
.venv/bin/python tools/memrun.py --limit-gb 3.5 -- .venv/bin/python -m text.train_diverse   # diverse-data essay model
.venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python -m text.calibrate  # threshold on MAGE validation
./run_final.sh                                    # includes the unseen-generator (no-GPT) code runs
.venv/bin/python tools/e2e_api.py 5               # end-to-end API test (server running)
.venv/bin/python tools/e2e_ui.py                  # headless browser test + screenshots
```

The watermark robustness experiment:

```bash
.venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python -m provenance.robustness
```

## Layout

```
common/       shared detector head, SupCon loss, training loop, metrics
text/ code_det/ image/ audio/   one pipeline per modality: data, features, model, train, predict
provenance/   C2PA, AudioSeal, text watermarks, robustness experiment
detect.py     two-layer decision
server.py     web API;  web/ is the front end
samples.py    held-out test samples for the demo
```

## Notes on data hygiene

- **HC3:** the train/test split is by question, and tokenisation spacing in the human answers (`it 's`) is removed so it can't be used as a shortcut.
- **ASVspoof:** the split is by speaker, and leading/trailing silence is trimmed, because bona fide clips are known to carry more silence.
- **All subsets** are class-balanced and built with a fixed seed.
