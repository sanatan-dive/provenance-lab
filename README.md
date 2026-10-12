# Provenance Lab: Deep Learning for Detecting AI-Generated Content

CS756 Project-I. Sanatan Sharma (CO23355) and Ryanveer Singh (CO23353). Mentor: Prof. Varun Gupta.

A local web app that checks whether an essay, a Python file, an image or a voice recording was made by a person or by a generative AI model. It first looks for provenance signals (C2PA, AudioSeal, text watermarks) and falls back to trained detectors (BERT, CodeBERT, ViT + FFT, wav2vec 2.0).

| Folder | Contents |
|---|---|
| `code/` | Training, evaluation and the web app. Start with [`code/README.md`](code/README.md). |
| `Report/` | Project report (`Project_Report.pdf`) and the scripts that build it. |
| `Presentation/` | Slides (`Provenance_Lab.pdf`) and the script that builds them. |

**To set it up on your own computer, follow [SETUP.md](SETUP.md).** It covers the environment, downloading the trained models and demo samples from the [v1.0 release](https://github.com/sanatan-dive/provenance-lab/releases/tag/v1.0), downloading every dataset, and retraining every model.
