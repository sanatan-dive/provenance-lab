// Builds Report/Project_Report.docx: a full project report with diagrams, charts and screenshots,
// every number read from code/results.   cd Report/build && python make_figures.py && node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, ImageRun, LevelFormat, BorderStyle, Footer, PageNumber, TabStopType, PageBreak, TableLayoutType,
} = require("docx");

const ROOT = path.resolve(__dirname, "../..");
const RES = path.join(ROOT, "code/results");
const FIG = path.join(ROOT, "Report/figures");
const OUT = path.join(ROOT, "Report/Project_Report.docx");
const J = (rel) => JSON.parse(fs.readFileSync(path.join(RES, rel), "utf8"));
const pct = (x) => `${(x * 100).toFixed(1)}%`;

// ---------- results ----------
const textM = J("text/text_bert_supcon_metrics.json");
const textCE = J("text/text_bert_ce_metrics.json");
const div = J("ood/mage_diverse_12k.json").calibrated;
const code = J("code/code_codebert_supcon_12k_metrics.json");
const code4 = J("code/code_codebert_supcon_metrics.json"), codeCE = J("code/code_codebert_ce_metrics.json");
const codePG = J("code/code_codebert_supcon_12k_per_generator.json");
const hoS = J("code/code_holdout_gpt_supcon_per_generator.json"), hoC = J("code/code_holdout_gpt_ce_per_generator.json");
const img = J("image/image_genimage_metrics.json");
const imgCif = J("image/image_vit_fft_supcon_metrics.json"), imgCifCE = J("image/image_vit_fft_ce_metrics.json"), imgVit = J("image/image_vit_only_supcon_metrics.json");
const imgGen = J("image/image_genimage_per_generator.json"), imgHo = J("image/image_genimage_nomidjourney_per_generator.json");
const audCNN = J("audio/audio_cnn_supcon_metrics.json"), audCE = J("audio/audio_cnn_ce_metrics.json"), audW2v = J("audio/audio_w2v_lr_metrics.json");
const vm = J("ood/voice_modern.json");
const base = J("baselines/baselines.json");
const market = J("market/market.json");
const ood = J("ood/mage.json");
const rob = J("provenance/robustness.json");
const best = (mod, set) => Math.max(...Object.values(market[mod]).filter((r) => r[set]).map((r) => r[set].accuracy));
const bestBase = (k) => Math.max(...Object.values(base[k]).map((v) => v.accuracy));
const ci = (a, b, n) => 1.96 * Math.sqrt((a * (1 - a) + b * (1 - b)) / n) * 100;
const diff = (a, b, n) => { const d = (b - a) * 100, c = ci(a, b, n); return `${d >= 0 ? "+" : ""}${d.toFixed(1)} ± ${c.toFixed(1)}`; };
const hc3Only = ood.models[Object.keys(ood.models)[0]];

// ---------- style ----------
const INK = "13202A", SOFT = "4A5A66", UV = "5A3FD0", LINE = "CDD6DB", WASH = "ECE8FB", PANEL = "F3F5F7";
const FONT = "Calibri";
const W = 9026; // A4 text width (DXA) with 1-inch margins
const run = (x) => (typeof x === "string" ? new TextRun(x) : x);
const p = (t, o = {}) => new Paragraph({ spacing: { after: 140, line: 300 }, alignment: AlignmentType.JUSTIFIED, ...o, children: (Array.isArray(t) ? t : [t]).map(run) });
const b = (t) => new TextRun({ text: t, bold: true });
const i_ = (t) => new TextRun({ text: t, italics: true });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(t)] });
const h1np = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 70, line: 288 }, children: (Array.isArray(t) ? t : [t]).map(run) });
const num = (t) => new Paragraph({ numbering: { reference: "num", level: 0 }, spacing: { after: 70, line: 288 }, children: (Array.isArray(t) ? t : [t]).map(run) });
const bd = { style: BorderStyle.SINGLE, size: 4, color: LINE };
const nob = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const cell = (t, w, o = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  margins: { top: 70, bottom: 70, left: 110, right: 110 },
  borders: { top: bd, bottom: bd, left: bd, right: bd },
  shading: o.head ? { type: ShadingType.CLEAR, fill: INK } : o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined,
  children: [new Paragraph({ alignment: o.right ? AlignmentType.RIGHT : AlignmentType.LEFT,
    children: [new TextRun({ text: String(t), bold: o.head || o.bold, color: o.head ? "FFFFFF" : INK, size: 19 })] })],
});
const table = (widths, header, rows, rightFrom = 99) => new Table({
  layout: TableLayoutType.FIXED,
  width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: widths,
  rows: [new TableRow({ tableHeader: true, children: header.map((h, k) => cell(h, widths[k], { head: true, right: k >= rightFrom })) })]
    .concat(rows.map((r, ri) => new TableRow({ children: r.map((c, k) => cell(c, widths[k], { right: k >= rightFrom, fill: ri % 2 ? PANEL : undefined })) }))),
});
let figN = 0, tabN = 0;
const figCap = (t) => p([new TextRun({ text: `Figure ${++figN}. `, bold: true, color: SOFT, size: 18 }), new TextRun({ text: t, color: SOFT, size: 18 })], { alignment: AlignmentType.CENTER, spacing: { before: 80, after: 260 } });
const tabCap = (t) => p([new TextRun({ text: `Table ${++tabN}. `, bold: true, color: SOFT, size: 18 }), new TextRun({ text: t, color: SOFT, size: 18 })], { alignment: AlignmentType.LEFT, spacing: { before: 80, after: 240 } });

function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), buf };
}
function image(file, widthPx = 600) {
  const { w, h, buf } = pngSize(file);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })] });
}
function imageGrid(files, labels, widthPx = 290) {
  const cw = Math.floor(W / 2);
  const rows = [];
  for (let k = 0; k < files.length; k += 2) {
    rows.push(new TableRow({ children: [0, 1].map((d) => {
      const f = files[k + d];
      const kids = f && fs.existsSync(f) ? [new Paragraph({ alignment: AlignmentType.CENTER, children: [(() => {
        const { w, h, buf } = pngSize(f); return new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round(widthPx * h / w) } }); })()] }),
        new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: labels[k + d], color: SOFT, size: 17, bold: true })] })] : [new Paragraph("")];
      return new TableCell({ width: { size: cw, type: WidthType.DXA }, borders: { top: nob, bottom: nob, left: nob, right: nob }, children: kids });
    }) }));
  }
  return new Table({ layout: TableLayoutType.FIXED, width: { size: cw * 2, type: WidthType.DXA }, columnWidths: [cw, cw], rows });
}
const fig = (name) => path.join(FIG, name);
const shot = (name) => path.join(RES, "screens", name);

// ---------- cover ----------
const cover = [
  new Paragraph({ spacing: { before: 1800, after: 120 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "CS756 PROJECT-I REPORT", color: UV, bold: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "Deep Learning for Detecting AI-Generated Content", bold: true, size: 52, color: INK })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 900 }, children: [new TextRun({ text: "Provenance Lab: a two-layer system for essays, source code, images and voice", size: 28, color: SOFT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Submitted by", size: 22, color: SOFT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Sanatan Sharma (CO23355)", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 500 }, children: [new TextRun({ text: "Ryanveer Singh (CO23353)", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Under the guidance of", size: 22, color: SOFT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 900 }, children: [new TextRun({ text: "Prof. Varun Gupta", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "October 2026", size: 22, color: SOFT })] }),
];

// ---------- front matter ----------
const front = [
  h1("Abstract"),
  p("Generative AI now produces essays, source code, photorealistic images and cloned voices that are hard to tell apart from human work, which creates problems for academic integrity, software authorship, misinformation and voice fraud. This project builds Provenance Lab, a single system that detects AI-generated content across these four modalities and explains its decisions."),
  p("The system has two layers. The first checks for verifiable provenance: C2PA content credentials, AudioSeal audio watermarks and statistical text watermarks (green-list and SynthID-Text). When no such signal is present, the second layer runs a deep learning detector trained for that modality: BERT with perplexity and stylometric features for essays, CodeBERT with syntax-tree features for code, a Vision Transformer with a frequency-spectrum branch for images, and wav2vec 2.0 speech representations for voice. The text, code and image detectors are trained with binary cross-entropy plus a supervised contrastive loss."),
  p(`On held-out test data the deployed detectors reach ${pct(textM.accuracy)} on HC3 essays, ${pct(div.mage_accuracy)} on MAGE essays from 27 AI models, ${pct(code.accuracy)} on code, ${pct(img.accuracy)} on images from seven modern generators and ${pct(vm.new_model_on_modern)} on modern voice clones. On the same current content the best public open-source detectors reach ${pct(best("text", "mage"))}, ${pct(best("image", "genimage"))} and ${pct(best("audio", "modern"))} respectively. We also report honest negative findings: accuracy drops on AI models not seen in training, watermarks break under simple edits, and the contrastive loss gave no measurable gain at our scale.`),
  p([b("Keywords: "), "AI-generated content detection, deepfake detection, BERT, CodeBERT, Vision Transformer, wav2vec 2.0, contrastive learning, C2PA, watermarking."]),

  h1np("Contents"),
  ...[
    "1. Introduction", "2. Related work", "3. System design", "4. Datasets and experimental setup", "5. Results",
    "6. The web application", "7. Discussion and limitations", "8. Conclusion and future work", "References",
  ].map((t) => new Paragraph({ spacing: { after: 90 }, tabStops: [{ type: TabStopType.RIGHT, position: W }], children: [new TextRun({ text: t, size: 22 })] })),
];

// ---------- body ----------
const body = [
  h1("1. Introduction"),
  h2("1.1 Problem"),
  p("Large language models write fluent essays and working code; diffusion models create photorealistic images; voice-cloning services reproduce a person's voice from seconds of audio. These capabilities are useful, but they also enable academic dishonesty, misrepresented code authorship, image-based misinformation and voice-fraud calls. Reliable, automated detection is therefore an active research problem."),
  h2("1.2 Gaps in existing detectors"),
  bullet([b("Single modality. "), "Text detectors, image detectors and voice anti-spoofing systems are separate tools with separate interfaces."]),
  bullet([b("Poor generalisation. "), "Detectors learn the quirks of the generators they were trained on and fail on newer ones."]),
  bullet([b("Watermarks only help when present. "), "Provenance standards such as C2PA and SynthID give proof, but most tools do not apply them and simple edits remove them."]),
  bullet([b("Over-confident verdicts. "), "A forced yes/no answer can wrongly accuse a person; few tools express uncertainty."]),
  h2("1.3 Objectives"),
  num("Build one system that handles essays, Python code, images and voice recordings."),
  num("Combine verifiable provenance (credentials and watermarks) with trained deep learning detectors."),
  num("Train on public benchmarks with leakage-free splits, and evaluate against simple baselines and popular public detectors."),
  num("Test honestly on AI models and generators that were not seen in training."),
  num("Explain every verdict and report uncertainty instead of forcing a decision."),

  h1("2. Related work"),
  p([b("AI-generated text. "), "Early detectors used statistical signals such as perplexity: machine text tends to be more predictable to a language model. Fine-tuned classifiers such as OpenAI's RoBERTa detector followed. Benchmarks such as HC3 (human versus ChatGPT answers) and MAGE (27 language models, 10 domains) showed that detectors trained on one generator degrade on others, and RAID (ACL 2024) showed that detectors are easily fooled by unseen models and adversarial edits."]),
  p([b("AI-generated code. "), "CoDET-M4 provides human and LLM-written code across languages and generators. Stylometric and syntax-tree features, together with code language models such as CodeBERT, are the usual approach."]),
  p([b("AI-generated images. "), "Generators leave artefacts in the frequency spectrum from upsampling, which CNN and frequency-domain detectors exploit. CIFAKE is a small 32 by 32 benchmark; GenImage covers modern generators including Midjourney and Stable Diffusion at full resolution."]),
  p([b("Synthetic speech. "), "The ASVspoof challenges benchmark spoofed speech detection. Self-supervised speech models such as wav2vec 2.0 give strong representations for anti-spoofing."]),
  p([b("Provenance and watermarking. "), "C2PA attaches signed content credentials to media. SynthID-Text and green-list watermarks bias token choice so a detector can test for them statistically; AudioSeal embeds an imperceptible watermark in audio."]),
  p([b("Contrastive learning. "), "Supervised contrastive learning (Khosla et al., 2020) pulls same-class embeddings together and pushes different classes apart, and has been proposed to improve generalisation."]),

  h1("3. System design"),
  p("Figure 1 shows the architecture. Every input first passes through the provenance layer. If a verifiable AI signal is found, the system reports it as proof and stops. Otherwise the trained detector for that modality produces a probability, mapped to a verdict: above 70% is AI-generated, below 30% is human, and anything between is reported as uncertain for human review."),
  image(fig("architecture.png"), 610),
  figCap("System architecture: provenance first, then a trained detector, then a verdict with evidence."),
  h2("3.1 Layer 1: provenance"),
  bullet([b("C2PA credentials "), "for images and audio: the system reads the signed manifest and checks whether it declares AI generation (IPTC trainedAlgorithmicMedia)."]),
  bullet([b("AudioSeal "), "for voice: a neural watermark detector."]),
  bullet([b("Text watermarks "), "for essays and code: green-list (Kirchenbauer et al.) and SynthID-Text detection by z-score."]),
  h2("3.2 Layer 2: trained detectors"),
  p("Figure 2 shows the common training pipeline. Each detector fuses a pretrained deep encoder with hand-crafted forensic features through a small fusion network, which feeds a classifier head and a projection head for the contrastive loss."),
  image(fig("pipeline.png"), 610),
  figCap("Training pipeline shared by all detectors."),
  table([1500, 3000, 4526], ["Content", "Deep encoder", "Forensic features"], [
    ["Essays", "BERT (bert-base-uncased)", "Perplexity and its spread under DistilGPT-2, sentence burstiness, vocabulary richness, punctuation and length"],
    ["Code", "CodeBERT", "Syntax-tree depth and node mix, function length, identifier style, comment and docstring ratios"],
    ["Images", "ViT-Small at 224 px + FFT-spectrum CNN", "Radial power spectrum, high-frequency energy, noise-residual statistics"],
    ["Voice", "wav2vec 2.0 (frozen) + logistic regression", "Mean of hidden layers 3, 6, 9 and 12; a spectrogram CNN with MFCC and phase features is kept for comparison"],
  ]),
  tabCap("Detector design for each modality."),
  p([b("Loss. "), "The text, code and image detectors minimise L = BCE + 0.5 × SupCon (temperature 0.1). The essay model used in the application is additionally trained on diverse data and its decision threshold is calibrated on validation data."]),

  h1("4. Datasets and experimental setup"),
  p("All experiments use public benchmark datasets downloaded from Hugging Face. Table 2 lists each dataset, what its human and AI classes contain, and how many samples we used for training and testing."),
  table([1300, 2500, 2800, 2426], ["Content", "Dataset", "Human vs AI", "Train / test used"], [
    ["Essays", "HC3", "Human answers vs ChatGPT", "4,000 / 1,000"],
    ["Essays", "MAGE", "Human vs 27 LLMs, 14 domains", "12,000 (OPT held out) / 1,000"],
    ["Code", "CoDET-M4 (Python)", "GitHub, LeetCode, Codeforces vs 5 LLMs", "12,000 / 1,000"],
    ["Images", "Tiny-GenImage", "ImageNet photos vs 7 generators", `${img.n_train} / ${img.n_test}`],
    ["Images", "CIFAKE", "CIFAR-10 vs Stable Diffusion (32 px)", "6,000 / 2,000"],
    ["Voice", "ASVspoof 2019 LA", "Bona fide vs 13 TTS/VC attacks", "3,000 / 1,000"],
    ["Voice", "Modern TTS (v4)", "YouTube speech vs ElevenLabs, Polly, Hume and others", `${1866 - vm.modern_test_n} / ${vm.modern_test_n}`],
  ]),
  tabCap("Public datasets used. All subsets are class-balanced."),
  h2("4.1 Data hygiene"),
  bullet("Leakage-free splits: by question (HC3), by speaker (ASVspoof), by recording or generation (modern voices), and official splits for code and images."),
  bullet("Shortcuts removed before reporting: tokenisation spacing and literal \"\\n\" escapes in HC3 answers, leading and trailing silence in ASVspoof, and the JPEG-versus-PNG format difference in GenImage (every image is resized and re-encoded as JPEG)."),
  h2("4.2 Training protocol"),
  bullet("10% of each training set is held out for validation; training stops when validation loss stops improving for two epochs and the best epoch is restored."),
  bullet("Pretrained encoders are fine-tuned at a learning rate of 2e-5; new layers at 1e-3; AdamW with weight decay 0.01."),
  bullet("Hardware: Apple M4 laptop with 16 GB memory, PyTorch on the Metal GPU, with every job run under a memory watchdog."),
  bullet("Metrics: accuracy, precision, recall, F1 and AUC-ROC, with 95% confidence intervals for comparisons."),

  h1("5. Results"),
  h2("5.1 Deployed detectors"),
  image(fig("results.png"), 600),
  figCap("Test accuracy of the detectors used by the application."),
  table([1700, 3500, 1300, 1300, 1226], ["Content", "Test data", "Accuracy", "F1", "AUC"], [
    ["Essays", "HC3", pct(textM.accuracy), pct(textM.f1), textM.auc_roc.toFixed(3)],
    ["Essays", "MAGE (unseen models)", pct(div.mage_accuracy), "-", "-"],
    ["Code", "CoDET-M4", pct(code.accuracy), pct(code.f1), code.auc_roc.toFixed(3)],
    ["Images", "GenImage", pct(img.accuracy), pct(img.f1), img.auc_roc.toFixed(3)],
    ["Voice", "Modern TTS", pct(vm.new_model_on_modern), "-", "-"],
    ["Voice", "ASVspoof 2019", pct(vm.new_model_on_asvspoof), "-", "-"],
  ], 2),
  tabCap("Test metrics of the deployed detectors."),
  imageGrid([path.join(RES, "text/text_bert_supcon_roc.png"), path.join(RES, "code/code_codebert_supcon_12k_roc.png"),
    path.join(RES, "image/image_genimage_roc.png"), path.join(RES, "audio/audio_w2v_modern_roc.png")],
  ["Essays (HC3)", "Code (CoDET-M4)", "Images (GenImage)", "Voice (ASVspoof + modern TTS)"]),
  figCap("ROC curves on the test sets."),
  h2("5.2 Learning curves: neither overfitting nor underfitting"),
  p("Figure 5 plots training and validation loss and accuracy for each detector. Validation curves that track the training curves, with the best epoch restored, show that the models learned general patterns rather than memorising the training set."),
  imageGrid([path.join(RES, "text/text_bert_supcon_curves.png"), path.join(RES, "code/code_codebert_supcon_12k_curves.png"),
    path.join(RES, "image/image_genimage_curves.png"), path.join(RES, "audio/audio_cnn_supcon_curves.png")],
  ["Essays", "Code", "Images", "Voice (spectrogram CNN)"], 295),
  figCap(`Training and validation loss and accuracy per epoch; the dotted line marks the epoch kept. Train-test accuracy gaps: essays ${(textM.generalization_gap * 100).toFixed(1)}, code ${(code.generalization_gap * 100).toFixed(1)}, images ${(img.generalization_gap * 100).toFixed(1)} points.`),

  h2("5.3 Comparison with public detectors"),
  p("Popular open-source detectors from Hugging Face were run unchanged on exactly our test sets. On current AI content, where no public model was trained on the test set, our detectors lead in every modality (Figure 5)."),
  image(fig("public_compare.png"), 520),
  figCap("Accuracy on current AI content: best public detector versus our system."),
  table([2200, 3300, 1700, 1826], ["Content", "Public detector", "Benchmark set", "Current set"], [
    ...Object.entries(market.text).map(([n, r]) => ["Essays", n.split("/")[1], `HC3 ${pct(r.hc3.accuracy)}`, `MAGE ${pct(r.mage.accuracy)}`]),
    ...Object.entries(market.image).map(([n, r]) => ["Images", n.split("/")[1], `CIFAKE ${pct(r.cifake.accuracy)}`, `GenImage ${pct(r.genimage.accuracy)}`]),
    ...Object.entries(market.audio).map(([n, r]) => ["Voice", n.split("/")[1], `ASVspoof ${pct(r.asvspoof.accuracy)}`, `Modern ${pct(r.modern.accuracy)}`]),
  ]),
  tabCap("Public detectors on our test sets. dima806 was trained on CIFAKE and the HC3 ChatGPT detector on HC3, so they have a home advantage on those benchmark sets."),
  p(`Commercial tools such as GPTZero and Turnitin are closed and paid, so we rely on independent evaluations: Weber-Wulff et al. (2023) found 14 tools, Turnitin included, "neither accurate nor reliable", and RAID (Dugan et al., 2024) found detectors easily fooled by unseen models and simple edits.`),

  h2("5.4 Comparison with simple baselines"),
  p("To check that the deep models earn their complexity, we trained well-known simple methods on exactly the same splits (Table 5)."),
  table([2300, 3500, 1500, 1726], ["Content", "Best simple method", "Simple", "Ours"], [
    ["Essays (HC3)", "Forensic features + logistic regression", pct(bestBase("text")), pct(textM.accuracy)],
    ["Code", "TF-IDF character n-grams + logistic regression", pct(bestBase("code")), pct(code.accuracy)],
    ["Images (CIFAKE)", "Forensic features + logistic regression", pct(bestBase("image")), pct(imgCif.accuracy)],
    ["Voice (ASVspoof)", "Forensic features + logistic regression", pct(bestBase("audio")), pct(audW2v.accuracy)],
  ], 2),
  tabCap("Simple baselines trained and tested on the same splits."),
  p("The deep models clearly beat simple methods on code, images and voice. On HC3 a simple feature-based classifier already matches the deep model: HC3 is an easy benchmark, so the deep model's value for essays shows on unseen AI models instead (Section 5.5)."),

  h2("5.5 Generalisation to unseen AI models"),
  image(fig("generalisation.png"), 610),
  figCap("Left: essays trained only on HC3 collapse on MAGE; diverse data and calibration recover much of the gap. Right: a voice model trained on 2019 attacks fails on modern clones until modern TTS is added to training."),
  p(`Trained only on HC3, the essay detector scores ${pct(hc3Only.hc3)} on HC3 but ${pct(hc3Only.mage.accuracy)} on MAGE. Training on HC3 plus a diverse MAGE training subset (with the OPT model family held out entirely) and calibrating the threshold on MAGE validation data gives ${pct(div.mage_accuracy)} on MAGE while ${pct(div.human_specificity)} of human texts are still cleared. On the held-out OPT family, ${pct(div.ai_recall_by_family["OPT (held out)"].recall)} of AI texts are caught.`),
  image(fig("genimage.png"), 600),
  figCap(`Images: accuracy per source. With Midjourney removed from training, ${pct(imgHo.Midjourney.accuracy)} of Midjourney images are still caught, against ${pct(imgGen.Midjourney.accuracy)} when included.`),
  p(`Code shows a similar pattern: with all GPT-written code removed from training, the detector still catches ${pct(Math.min(hoS.gpt.accuracy, hoC.gpt.accuracy))} to ${pct(Math.max(hoS.gpt.accuracy, hoC.gpt.accuracy))} of GPT code, against ${pct(codePG.gpt.accuracy)} when GPT is included.`),

  h2("5.6 Effect of the contrastive loss"),
  p("We trained each detector twice with identical data and settings, once with and once without the supervised contrastive loss (Table 6)."),
  table([2400, 2200, 2200, 2226], ["Content", "Cross-entropy only", "With SupCon", "Difference (pts, 95% CI)"], [
    ["Essays (HC3)", pct(textCE.accuracy), pct(textM.accuracy), diff(textCE.accuracy, textM.accuracy, textM.n_test)],
    ["Code (4k)", pct(codeCE.accuracy), pct(code4.accuracy), diff(codeCE.accuracy, code4.accuracy, code4.n_test)],
    ["Images (CIFAKE)", pct(imgCifCE.accuracy), pct(imgCif.accuracy), diff(imgCifCE.accuracy, imgCif.accuracy, imgCif.n_test)],
    ["Voice (CNN)", pct(audCE.accuracy), pct(audCNN.accuracy), diff(audCE.accuracy, audCNN.accuracy, audCNN.n_test)],
    ["Code, unseen GPT", pct(hoC.gpt.accuracy), pct(hoS.gpt.accuracy), diff(hoC.gpt.accuracy, hoS.gpt.accuracy, hoS.gpt.n)],
  ], 1),
  tabCap("Same architecture and data, with and without the supervised contrastive loss."),
  p("Every difference lies within its 95% confidence interval, so at our data scale the contrastive loss gave no measurable gain, including on an unseen generator. We report this as a finding. Two other design choices did matter: pretrained speech features beat a spectrogram CNN trained from scratch on voice (" + `${pct(audCNN.accuracy)} to ${pct(audW2v.accuracy)}` + "), and training on more varied data improved both code (95.5% to " + pct(code.accuracy) + ") and essays on unseen models."),

  h2("5.7 Robustness of watermarks"),
  image(fig("watermarks.png"), 610),
  figCap("Share of watermarked content still detected after common edits. Orange bars mark edits that break the watermark."),
  p("Watermarks are detected every time when content is untouched, and false positives on unwatermarked human content were zero. But added noise or a speed change removes the audio watermark, shuffling words removes text watermarks, and a single JPEG re-save strips C2PA credentials. Provenance is decisive when present but fragile, which is why the trained detectors form the second layer."),

  h1("6. The web application"),
  p("The application runs locally (FastAPI back end, custom HTML/CSS/JavaScript front end). A user pastes text or drops a file; the system detects the content type, runs both layers and returns a verdict, a confidence score, the deciding layer and an explanation. Buttons load real held-out test samples and the report marks whether the model was right. An automated end-to-end test of real samples through the API and a headless browser test of every tab completed with no errors."),
  image(shot("01_hero.png"), 560),
  figCap("Home screen with the content tabs and the live accuracy of each detector."),
  image(shot("02_text.png"), 560),
  figCap("Essay: verdict, confidence scale and per-sentence highlighting (darker means more AI-like)."),
  image(shot("04_image.png"), 560),
  figCap("Image: the occlusion heatmap shows the regions that most changed the verdict."),
  image(shot("05_audio.png"), 560),
  figCap("Voice: a real recording correctly cleared, with the analysed spectrogram."),

  h1("7. Discussion and limitations"),
  bullet([b("No detector is perfect. "), "Benchmark accuracy overstates real-world accuracy: every detector we tested, ours included, loses accuracy on AI models it has not seen. Our response is to train on diverse, current data, calibrate thresholds on validation data, and report uncertainty rather than force a verdict."]),
  bullet([b("Training scale. "), "Models were trained on subsets on a single laptop. Larger and newer data would likely help further."]),
  bullet([b("Coverage. "), "Code detection covers Python only; essays are English; the image detector expects photograph-like content."]),
  bullet([b("Provenance access. "), "Google's SynthID detectors for images and audio are not public; we use open equivalents."]),
  bullet([b("Hard cases. "), "Some ASVspoof attack types and the held-out Midjourney and OPT generators remain difficult."]),

  h1("8. Conclusion and future work"),
  p(`Provenance Lab detects AI-generated essays, code, images and voice in one system, combining verifiable provenance with trained deep learning detectors and explaining every verdict. It reaches ${pct(code.accuracy)} on code, ${pct(img.accuracy)} on modern images, ${pct(vm.new_model_on_modern)} on modern voice clones and ${pct(div.mage_accuracy)} on essays from 27 unseen-domain AI models, ahead of every public detector we could run on the same current content. Equally important, the project measures where detection fails and builds that uncertainty into the system.`),
  h2("Future work"),
  bullet("Train on larger and newer datasets (more generators, ASVspoof 5) on a GPU."),
  bullet("Evaluate and train against paraphrasing, compression and other adversarial edits."),
  bullet("Extend code detection to more languages and essays to more languages."),
  bullet("Revisit contrastive learning with many more generators and larger batches."),

  h1("References"),
  ...[
    "Guo, B. et al. (2023). How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection (HC3). arXiv:2301.07597.",
    "Li, Y. et al. (2024). MAGE: Machine-generated Text Detection in the Wild. ACL 2024.",
    "Orel, D. et al. (2024). CoDET-M4: Detecting Machine-Generated Code in Multi-Lingual, Multi-Generator and Multi-Domain Settings.",
    "Bird, J. J. and Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access.",
    "Zhu, M. et al. (2023). GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image. NeurIPS 2023.",
    "Wang, X. et al. (2020). ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech. Computer Speech & Language.",
    "Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.",
    "Feng, Z. et al. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. EMNLP Findings.",
    "Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.",
    "Baevski, A. et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS.",
    "Khosla, P. et al. (2020). Supervised Contrastive Learning. NeurIPS.",
    "Kirchenbauer, J. et al. (2023). A Watermark for Large Language Models. ICML.",
    "Dathathri, S. et al. (2024). Scalable Watermarking for Identifying Large Language Model Outputs (SynthID-Text). Nature.",
    "San Roman, R. et al. (2024). Proactive Detection of Voice Cloning with Localized Watermarking (AudioSeal). ICML.",
    "Coalition for Content Provenance and Authenticity (C2PA). Technical Specification.",
    "Weber-Wulff, D. et al. (2023). Testing of Detection Tools for AI-Generated Text. International Journal for Educational Integrity, 19, 26.",
    "Dugan, L. et al. (2024). RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. ACL 2024.",
  ].map((r) => new Paragraph({ numbering: { reference: "ref", level: 0 }, spacing: { after: 80, line: 276 }, children: [new TextRun({ text: r, size: 20 })] })),
];

const doc = new Document({
  creator: "Sanatan Sharma, Ryanveer Singh",
  title: "Deep Learning for Detecting AI-Generated Content",
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 34, bold: true, color: INK, font: FONT }, paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, color: UV, font: FONT }, paragraph: { spacing: { before: 260, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "ref", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 520 } } } }] },
  ] },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: cover },
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      footers: { default: new Footer({ children: [new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: W }], children: [
        new TextRun({ text: "Deep Learning for Detecting AI-Generated Content", color: SOFT, size: 17 }), new TextRun({ text: "\t", size: 17 }),
        new TextRun({ children: [PageNumber.CURRENT], color: SOFT, size: 17 })] })] }) },
      children: [...front, ...body],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
