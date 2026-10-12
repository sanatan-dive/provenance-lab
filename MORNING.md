# Provenance Lab: read this before class

## 1. Start the demo (30 seconds)

```bash
cd ~/"Documents/College/cs70x/CS756 Project-I/code"
.venv/bin/python server.py
```

Then open **http://127.0.0.1:8000**. Choose a tab, click **AI-made** or **Human-made** to load a real test sample, then click **Examine**. The report shows the verdict and whether the model was right.

- **Slides:** `Presentation/Provenance_Lab.pptx` (PDF next to it). **Report:** `Report/Project_Report.pdf` (Word version next to it).
- **Title slide:** Sanatan Sharma (CO23355) and Ryanveer Singh (CO23353), mentor Prof. Varun Gupta.

## 2. The 60-second pitch

> "AI now writes essays and code, makes images and clones voices. We built **Provenance Lab**, one system for all four.
>
> **Layer 1** checks for proof: C2PA content credentials and invisible watermarks like AudioSeal and SynthID-Text.
>
> If there's no proof, **Layer 2** runs a trained deep model: BERT for essays, CodeBERT for code, a Vision Transformer with a frequency branch for images, and wav2vec 2.0 for voice. Every verdict comes with evidence: highlighted sentences, an image heatmap or the spectrogram.
>
> On current AI content we reach 85% on essays from 27 AI models, 91% on images from 7 modern generators and 94% on modern voice clones such as ElevenLabs, while the best public detectors reach 69%, 63% and 78%. Code is at 97.6%.
>
> We also tested honestly: every detector drops on AI models it hasn't seen, so we train on diverse, current data and the system says 'uncertain' rather than accuse someone."

## 3. Key numbers

| | Ours | Best simple method | Best public detector |
|---|---|---|---|
| Essays, MAGE (27 AI models) | **85.3%** | 63.2% (perplexity) | 69.0% |
| Essays, HC3 | 97.2% | 98.4% (HC3 is easy) | 92.8% |
| Code | **97.6%** | 93.1% | none widely available |
| Images, GenImage (7 generators) | **90.7%** | - | 62.5% |
| Voice, modern TTS | **94.3%** | - | 78.0% |
| Voice, ASVspoof 2019 | 95.3% | 76.0% | 72.2% |

**Unseen generator:** with no GPT code in training, the code detector still catches 78–85% of GPT code.

**Watermarks:** 100% detected when untouched, and 0% after a speed change or added noise.

## 4. Likely questions

**"Can an AI detector be 100% accurate?"**
No, and we show it. Benchmark scores overstate reality: every detector drops on unseen generators. That's why we measure the gap, keep an "uncertain" band (30–70%), and check provenance first.

**"How do you know it isn't overfitting?"**
- 10% validation hold-out with early stopping.
- Learning curves for every model.
- Train–test gaps of 0.3–3.8 points.
- Leakage-free splits: by question for essays, by speaker for voice.
- Shortcuts found and removed: tokenisation spacing and literal \\n escapes in HC3, silence in ASVspoof, JPEG-vs-PNG in GenImage.

**"Did contrastive learning help?"**
Honestly, not measurably at this scale. Every difference is inside the 95% confidence interval. We report that as a finding. More generators and larger batches are future work.

**"How did voice get to 96%?"**
Our first model, a CNN trained from scratch on spectrograms, reached 80.7%. Switching to frozen wav2vec 2.0 features (a speech model pretrained on 960 hours of audio) with a linear head gave 96.3% on the same unseen-speaker test set. That's a lesson in itself: pretrained representations beat small models trained from scratch. Attack A10 is still hard (55% caught).

**"What about GPTZero or Turnitin?"**
They're closed and paid, so we cite independent studies: Weber-Wulff et al. 2023 found 14 tools, Turnitin included, "neither accurate nor reliable", and RAID (ACL 2024) found detectors "easily fooled". The open detectors we could run are compared on our exact test sets.

**"Why train on small subsets?"**
So the whole pipeline trains on one laptop overnight. These are baseline results, and full-scale training needs more compute.

## 5. Where everything is

- `code/README.md`: how to run every experiment
- `code/results/summary.md`: every training run in one table
- `code/results/screens/`: screenshots of the working app
- `code/results/e2e_api.json`: end-to-end test log (20 real samples, 0 errors)
