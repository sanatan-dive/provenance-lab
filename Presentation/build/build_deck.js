// Builds Presentation/Provenance_Lab.pptx from the real training results in code/results.
//   cd Presentation/build && node build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "../../code");
const RES = path.join(ROOT, "results");
const OUT = path.resolve(__dirname, "../Provenance_Lab.pptx");

const THEME = {
  name: "Provenance Lab",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "13202A", lt1: "FFFFFF", dk2: "4A5A66", lt2: "E6ECEE",
    accent1: "5A3FD0", accent2: "1D7A57", accent3: "C24A12", accent4: "B08A1E",
    accent5: "8E7BE6", accent6: "9AA8B1", hlink: "5A3FD0", folHlink: "4A31B8",
  },
};
const HEX = THEME.colors;

// ---------- results ----------
const readJson = (p) => (fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null);
const metric = (mod, name) => readJson(path.join(RES, mod, `${name}_metrics.json`));
const MODS = [
  { key: "text", label: "Essays", main: "text_bert_supcon", ce: "text_bert_ce", model: "BERT + perplexity/burstiness" },
  { key: "code", label: "Code", main: "code_codebert_supcon_12k", abl: "code_codebert_supcon", ablTag: " (4k)", ce: "code_codebert_ce", model: "CodeBERT + AST features" },
  { key: "image", label: "Images", main: "image_genimage", bench: "image_vit_fft_supcon", abl: "image_vit_fft_supcon", ablTag: " (CIFAKE)", ce: "image_vit_fft_ce", model: "ViT + FFT CNN" },
  { key: "audio", label: "Voice", main: "audio_w2v_modern", bench: "audio_w2v_lr", abl: "audio_cnn_supcon", ablTag: " (CNN)", ce: "audio_cnn_ce", model: "wav2vec 2.0 + linear head" },
];
// m.r: deployed model. m.rb: model scored on the original benchmark (HC3, CoDET-M4, CIFAKE, ASVspoof), used for
// baseline and public-detector comparisons on those sets. m.ra/m.rce: the contrastive-ablation pair.
MODS.forEach((m) => {
  m.r = metric(m.key, m.main); m.rb = metric(m.key, m.bench || m.main);
  m.ra = metric(m.key, m.abl || m.main); m.rce = metric(m.key, m.ce);
});
const done = MODS.filter((m) => m.r);
const rob = readJson(path.join(RES, "provenance/robustness.json"));
const perGen = readJson(path.join(RES, "code/code_codebert_supcon_per_generator.json"));
const perAtk = readJson(path.join(RES, "audio/audio_w2v_lr_per_attack.json"));
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const PUBLIC_NAMES = {
  "openai-community/roberta-base-openai-detector": "OpenAI RoBERTa",
  "fakespot-ai/roberta-base-ai-text-detection-v1": "Fakespot RoBERTa",
  "Hello-SimpleAI/chatgpt-detector-roberta": "HC3 ChatGPT detector",
  "umm-maybe/AI-image-detector": "umm-maybe",
  "dima806/ai_vs_real_image_detection": "dima806",
  "Ateeqq/ai-vs-human-image-detector": "Ateeqq",
  "mo-thecreator/Deepfake-audio-detection": "mo-thecreator",
  "MelodyMachine/Deepfake-audio-detection-V2": "MelodyMachine",
};
const pubName = (n) => PUBLIC_NAMES[n] || n.split("/")[1];
// 95% half-width for a difference of two accuracies on n test samples each (normal approximation)
const ci95 = (a, b, n) => 1.96 * Math.sqrt((a * (1 - a) + b * (1 - b)) / n);
const diffText = (a, b, n) => {
  const d = (b - a) * 100, ci = ci95(a, b, n) * 100;
  return `${d >= 0 ? "+" : ""}${d.toFixed(1)} pts (±${ci.toFixed(1)})${Math.abs(d) <= ci ? ", within noise" : ""}`;
};

// ---------- icons ----------
async function icon(Comp, color = "FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Provenance Lab";
  pres.author = "Sanatan Sharma, Ryanveer Singh";
  const C = pres.SchemeColor;

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "DARK_TITLE",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.35, w: 8.8, h: 1.5, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.0, w: 8.8, h: 0.9, fontSize: 18, color: C.accent5, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.5, 0.5, 0.5],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 26, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: "Provenance Lab", options: { x: 0.5, y: 5.22, w: 4, h: 0.25, fontSize: 10, color: C.text2, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.22, w: 0.5, h: 0.25, fontSize: 10, color: C.text2, align: "right" },
  });

  const ic = {
    grad: await icon(fa.FaGraduationCap), code: await icon(fa.FaCode), img: await icon(fa.FaImage),
    mic: await icon(fa.FaMicrophone), shield: await icon(fa.FaShieldAlt), brain: await icon(fa.FaBrain),
    stamp: await icon(fa.FaStamp), wave: await icon(fa.FaWaveSquare), news: await icon(fa.FaNewspaper),
    phone: await icon(fa.FaPhoneAlt), question: await icon(fa.FaQuestion), check: await icon(fa.FaCheck),
  };
  const dot = (s, x, y, img, color = C.accent1, d = 0.55) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 } });
    s.addImage({ data: img, x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52 });
  };
  const txt = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, color: C.text1, fontSize: 14, valign: "top", ...o });
  const content = (title, section) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };
  const chartStyle = () => ({
    catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: HEX.dk1,
    legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: HEX.dk2, titleFontFace: "+mn-lt", titleColor: HEX.dk1, titleFontSize: 13,
    valGridLine: { color: "D5DDE1", size: 0.5 }, catGridLine: { style: "none" },
  });

  // ===== 1. Title =====
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Introduction" });
  s.addText("Provenance Lab", { placeholder: "title" });
  s.addText("A two-layer, contrastively trained framework for detecting AI-generated essays, code, images and speech", { placeholder: "body" });
  txt(s, "Sanatan Sharma (CO23355)   Ryanveer Singh (CO23353)\nMentor: Prof. Varun Gupta   CS756 Project-I", { x: 0.6, y: 4.45, w: 8.8, h: 0.6, fontSize: 13, color: C.background2 });
  s.addNotes("Introduce the project: Provenance Lab detects whether content was made by a person or a generative AI model, across four kinds of content, and explains which layer of the system made the decision.");

  // ===== 2. Problem =====
  s = content("AI-made content is now hard to tell apart", "Introduction");
  const harms = [
    [ic.grad, "Academic integrity", "Essays and assignments written by chatbots"],
    [ic.code, "Code authenticity", "AI-written code submitted as original work"],
    [ic.news, "Misinformation", "Photorealistic synthetic images spread as real"],
    [ic.phone, "Voice fraud", "Cloned voices used in scam calls"],
  ];
  harms.forEach(([im, h, d], i) => {
    const x = 0.5 + i * 2.3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.45, w: 2.05, h: 2.6, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12 });
    dot(s, x + 0.25, 1.7, im);
    txt(s, h, { x: x + 0.25, y: 2.45, w: 1.6, h: 0.6, fontSize: 16, bold: true, fontFace: "+mj-lt" });
    txt(s, d, { x: x + 0.25, y: 3.05, w: 1.6, h: 0.9, fontSize: 13, color: C.text2 });
  });
  txt(s, "One detector per problem doesn't scale. We need one framework that covers all four, works on generators it hasn't seen, and says why it decided.", { x: 0.5, y: 4.3, w: 9.0, h: 0.7, fontSize: 15 });
  s.addNotes("Motivation. Each of these is a real, current harm. The common thread: content alone no longer tells you who made it.");

  // ===== 3. Gap =====
  s = content("What existing detectors get wrong", "Introduction");
  const gaps = [
    ["Single modality", "Text detectors, image detectors and voice anti-spoofing are separate systems with separate interfaces."],
    ["Break on new generators", "Classifiers learn one model's quirks and fail when a newer model arrives."],
    ["Watermarks only help if present", "SynthID and C2PA prove origin, but most tools don't add them and edits strip them."],
    ["Over-confident verdicts", "A forced yes/no wrongly accuses people; there is no 'not sure' outcome."],
  ];
  gaps.forEach(([h, d], i) => {
    const y = 1.3 + i * 0.95;
    dot(s, 0.5, y, ic.question, C.accent3, 0.5);
    txt(s, h, { x: 1.2, y: y - 0.02, w: 3.0, h: 0.5, fontSize: 16, bold: true, fontFace: "+mj-lt", valign: "middle" });
    txt(s, d, { x: 4.2, y: y - 0.02, w: 5.3, h: 0.6, fontSize: 14, color: C.text2, valign: "middle" });
  });
  s.addNotes("Research gap. Each of our design choices answers one of these four problems.");

  // ===== 4. Architecture =====
  pres.addSection({ title: "Method" });
  s = content("Two layers: proof first, then a trained detector", "Method");
  const box = (x, y, w, h, fill, title, sub, dark = false) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.1 });
    txt(s, title, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.35, fontSize: 14, bold: true, fontFace: "+mj-lt", color: dark ? C.background1 : C.text1 });
    txt(s, sub, { x: x + 0.15, y: y + 0.47, w: w - 0.3, h: h - 0.55, fontSize: 12, color: dark ? C.background2 : C.text2 });
  };
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: HEX.dk2, width: 1.5, endArrowType: "triangle" } });
  box(0.5, 2.05, 1.6, 1.3, C.background2, "Input", "Essay, Python file, image or voice clip");
  arrow(2.1, 2.7, 2.5, 2.7);
  box(2.5, 1.25, 2.7, 1.3, C.accent1, "Layer 1: provenance", "C2PA credentials, AudioSeal and text watermarks", true);
  arrow(5.2, 1.9, 7.4, 1.9);
  box(7.4, 1.25, 2.1, 1.3, C.accent3, "AI-generated", "Verifiable proof, stop here", true);
  arrow(3.85, 2.55, 3.85, 2.95);
  txt(s, "nothing found", { x: 3.95, y: 2.6, w: 1.4, h: 0.3, fontSize: 11, color: C.text2 });
  box(2.5, 2.95, 2.7, 1.65, C.text1, "Layer 2: forensic detector", "Deep encoder + forensic features, trained with contrastive loss", true);
  arrow(5.2, 3.75, 7.4, 3.75);
  box(7.4, 3.05, 2.1, 1.45, C.background2, "Verdict", "Over 70% AI, under 30% human; in between goes to human review");
  s.addNotes("The core design. Layer 1 is cheap and certain when a signal exists. Layer 2 covers everything else, including content from tools that never watermark. The uncertain band avoids false accusations.");

  // ===== 5. Layer 2 detectors =====
  s = content("Layer 2: a specialised detector per content type", "Method");
  const det = [
    [ic.grad, "Essays", "BERT (bert-base-uncased)", "Perplexity and its spread under DistilGPT-2, sentence burstiness, vocabulary richness"],
    [ic.code, "Code", "CodeBERT", "Syntax-tree depth and node mix, function length, identifier style, comment ratio"],
    [ic.img, "Images", "Vision Transformer + frequency CNN (224 px)", "2-D FFT spectrum branch, radial power profile, noise-residual statistics"],
    [ic.mic, "Voice", "wav2vec 2.0 (frozen) + linear head", "Compared with a spectrogram CNN using MFCC, spectral-shape and phase cues"],
  ];
  txt(s, "Deep encoder", { x: 2.65, y: 1.15, w: 2.6, h: 0.3, fontSize: 12, bold: true, color: C.text2 });
  txt(s, "Hand-crafted forensic signals (fused with the encoder)", { x: 5.4, y: 1.15, w: 4.1, h: 0.3, fontSize: 12, bold: true, color: C.text2 });
  det.forEach(([im, h, enc, f], i) => {
    const y = 1.5 + i * 0.9;
    dot(s, 0.5, y, im, C.accent1, 0.5);
    txt(s, h, { x: 1.15, y, w: 1.4, h: 0.5, fontSize: 16, bold: true, fontFace: "+mj-lt", valign: "middle" });
    txt(s, enc, { x: 2.65, y, w: 2.6, h: 0.6, fontSize: 13, valign: "middle" });
    txt(s, f, { x: 5.4, y, w: 4.1, h: 0.7, fontSize: 12, color: C.text2, valign: "middle" });
  });
  s.addNotes("Hybrid design: a pretrained deep encoder captures semantics; hand-crafted forensic features capture statistical fingerprints known to separate AI from human content. They are concatenated and fused by an MLP.");

  // ===== 6. Contrastive =====
  s = content("Contrastive learning: what all AI content shares", "Method");
  txt(s, [
    { text: "Every detector is trained with two losses on the fused embedding:", options: { breakLine: true } },
    { text: "Binary cross-entropy for the human/AI decision", options: { bullet: true, breakLine: true } },
    { text: "Supervised contrastive loss (SupCon, temperature 0.1) that pulls same-label samples together and pushes human and AI apart", options: { bullet: true, breakLine: true } },
    { text: "L = BCE + 0.5 × SupCon", options: { bold: true, breakLine: true } },
    { text: "Why: the model is rewarded for a shared notion of 'AI-made' across all generators in a batch, not one generator's quirks, which should help on unseen models.", options: {} },
  ], { x: 0.5, y: 1.3, w: 4.6, h: 3.6, fontSize: 14, paraSpaceAfter: 8 });
  // embedding-space sketch
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.5, y: 1.3, w: 4.0, h: 3.5, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.12 });
  const pts = [[6.0, 1.9], [6.4, 2.3], [5.9, 2.6], [6.5, 1.75], [6.2, 2.9], [6.7, 2.55]];
  pts.forEach(([x, y]) => s.addShape(pres.shapes.OVAL, { x, y, w: 0.22, h: 0.22, fill: { color: C.accent2 }, line: { color: C.accent2 } }));
  pts.forEach(([x, y]) => s.addShape(pres.shapes.OVAL, { x: x + 2.1, y: y + 1.0, w: 0.22, h: 0.22, fill: { color: C.accent3 }, line: { color: C.accent3 } }));
  txt(s, "Human", { x: 5.8, y: 3.35, w: 1.5, h: 0.3, fontSize: 13, bold: true, color: C.accent2 });
  txt(s, "AI (any generator)", { x: 7.6, y: 2.25, w: 1.8, h: 0.3, fontSize: 13, bold: true, color: C.accent3 });
  txt(s, "Learned embedding space", { x: 5.7, y: 4.4, w: 3.6, h: 0.3, fontSize: 11, color: C.text2 });
  s.addNotes("SupCon (Khosla et al., 2020). Positives are all samples with the same label in the batch. This is why batch sizes are kept at 16 to 32: each batch needs several positives.");

  // ===== 7. Datasets =====
  pres.addSection({ title: "Experiments" });
  s = content("Real public benchmarks, leakage-free splits", "Experiments");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } });
  const rows = [
    [hdr("Content"), hdr("Dataset"), hdr("Human vs AI"), hdr("Train / test"), hdr("Leakage control")],
    ["Essays", "HC3 + MAGE", "Humans vs ChatGPT and 27 LLMs", "16,000 / 1,000 + 1,000", "By question; OPT held out"],
    ["Code", "CoDET-M4 (Python)", "GitHub, LeetCode, Codeforces vs 5 LLMs", "12,000 / 1,000", "Official split"],
    ["Images", "GenImage + CIFAKE", "Real photos vs 7 modern generators", "14,000 / 3,500", "Official split"],
    ["Voice", "ASVspoof 2019 + modern TTS", "Bona fide vs 2019 attacks, ElevenLabs, Polly...", "4,339 / 1,527", "By speaker / recording"],
  ];
  s.addTable(rows, { x: 0.5, y: 1.3, w: 9.0, colW: [1.1, 1.75, 2.75, 1.6, 1.8], fontSize: 12, fontFace: "+mn-lt", color: C.text1,
    border: { type: "solid", pt: 0.5, color: "D5DDE1" }, rowH: 0.45, valign: "middle" });
  txt(s, "All subsets are class-balanced. 10% of each training set is held out for validation. These are deliberately small subsets so the full pipeline trains on one laptop; full-scale training is future work.", { x: 0.5, y: 4.15, w: 9.0, h: 0.8, fontSize: 13, color: C.text2 });
  s.addNotes("Every dataset is a real, public benchmark downloaded from Hugging Face. Splits avoid leakage: the same question, or the same speaker, never appears in both train and test.");

  // ===== 8. Training protocol =====
  s = content("Training built to avoid over- and underfitting", "Experiments");
  const proto = [
    [ic.check, "Validation hold-out", "10% of each training set, never trained on"],
    [ic.check, "Early stopping", "Stop after 2 epochs without validation improvement; keep the best epoch"],
    [ic.check, "Regularisation", "Dropout, weight decay, SpecAugment for voice, gradient checkpointing"],
    [ic.check, "Shortcut removal", "HC3 spacing and \\n escapes; ASVspoof silence; GenImage JPEG vs PNG"],
  ];
  proto.forEach(([im, h, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 4.6, y = 1.35 + row * 1.55;
    dot(s, x, y, im, C.accent2, 0.5);
    txt(s, h, { x: x + 0.7, y, w: 3.7, h: 0.4, fontSize: 16, bold: true, fontFace: "+mj-lt" });
    txt(s, d, { x: x + 0.7, y: y + 0.42, w: 3.7, h: 0.8, fontSize: 13, color: C.text2 });
  });
  txt(s, "Hardware: Apple M4 laptop, 16 GB unified memory, PyTorch on the Metal GPU, every job capped at 3 GB by a memory watchdog.", { x: 0.5, y: 4.55, w: 9.0, h: 0.5, fontSize: 12, color: C.text2 });
  s.addNotes("Shortcut removal matters: human HC3 answers contain tokenisation artefacts like \"it 's\", and ASVspoof bona fide clips have more silence. Left in, a model could score highly without learning anything about AI generation.");

  // ===== 9. Main results =====
  s = content("Results on held-out test data", "Experiments");
  if (done.length) {
    s.addChart(pres.charts.BAR, [
      { name: "Accuracy", labels: done.map((m) => m.label), values: done.map((m) => +(m.r.accuracy * 100).toFixed(1)) },
      { name: "F1", labels: done.map((m) => m.label), values: done.map((m) => +(m.r.f1 * 100).toFixed(1)) },
      { name: "AUC-ROC ×100", labels: done.map((m) => m.label), values: done.map((m) => +(m.r.auc_roc * 100).toFixed(1)) },
    ], { x: 0.5, y: 1.2, w: 5.4, h: 3.8, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent1, HEX.accent5, HEX.dk1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 100,
      showLegend: true, legendPos: "b", ...chartStyle() });
    const trow = [[hdr("Content"), hdr("Prec."), hdr("Recall"), hdr("Test n")]].concat(
      done.map((m) => [m.label, pct(m.r.precision), pct(m.r.recall), String(m.r.n_test)]));
    s.addTable(trow, { x: 6.2, y: 1.3, w: 3.3, colW: [1.05, 0.8, 0.8, 0.65], fontSize: 12, fontFace: "+mn-lt", color: C.text1,
      border: { type: "solid", pt: 0.5, color: "D5DDE1" }, rowH: 0.4, valign: "middle" });
    txt(s, "Baseline results on small subsets; full datasets and longer training are expected to improve them.", { x: 6.2, y: 3.6, w: 3.3, h: 1.2, fontSize: 12, color: C.text2 });
  } else {
    txt(s, "Training in progress.", { x: 0.5, y: 2.5, w: 9, h: 0.5, fontSize: 18, color: C.text2 });
  }
  s.addNotes("Main results of the full model (BCE + SupCon + forensic features) for each modality, on test sets never seen during training or validation.\n" +
    done.map((m) => `${m.label}: accuracy ${pct(m.r.accuracy)}, F1 ${pct(m.r.f1)}, AUC ${m.r.auc_roc.toFixed(3)}.`).join("\n"));

  // ===== 9a. Today's generators =====
  const vm = readJson(path.join(RES, "ood/voice_modern.json"));
  const gen = readJson(path.join(RES, "image/image_genimage_per_generator.json"));
  const genHo = readJson(path.join(RES, "image/image_genimage_nomidjourney_per_generator.json"));
  if (vm && gen) {
    s = content("Built for today's generators, not 2019's", "Experiments");
    s.addChart(pres.charts.BAR, [
      { name: "Trained on ASVspoof 2019 only", labels: ["ASVspoof 2019", "Modern TTS"], values: [+(vm.old_model_on_asvspoof * 100).toFixed(1), +(vm.old_model_on_modern * 100).toFixed(1)] },
      { name: "+ modern TTS in training", labels: ["ASVspoof 2019", "Modern TTS"], values: [+(vm.new_model_on_asvspoof * 100).toFixed(1), +(vm.new_model_on_modern * 100).toFixed(1)] },
    ], { x: 0.5, y: 1.15, w: 4.3, h: 3.5, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent6, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 100,
      showTitle: true, title: "Voice accuracy (%): ElevenLabs, Polly, Hume, ...", showLegend: true, legendPos: "b", ...chartStyle() });
    const g = Object.keys(gen);
    s.addChart(pres.charts.BAR, [{ name: "Correct", labels: g.map((k) => (k === "Real" ? "Real photos" : k)), values: g.map((k) => +(gen[k].accuracy * 100).toFixed(0)) }],
      { x: 5.1, y: 1.15, w: 4.4, h: 3.5, barDir: "bar", chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
        valAxisMinVal: 0, valAxisMaxVal: 100, showTitle: true, title: "Images at 224 px (GenImage): % correct", showLegend: false, ...chartStyle() });
    txt(s, `A 2019-trained voice model flags real speech and misses modern clones (${pct(vm.old_model_on_modern)}); adding modern TTS fixes it.` +
      (genHo ? ` With Midjourney removed from training, ${pct(genHo.Midjourney.accuracy)} of Midjourney images are still caught (vs ${pct(gen.Midjourney.accuracy)}).` : ""),
      { x: 0.5, y: 4.75, w: 9.0, h: 0.4, fontSize: 11, color: C.text2 });
    s.addNotes("The deployed models are trained on current generators: voice on ASVspoof plus modern text-to-speech (split by recording and generation, so no clip family is in both train and test), images on GenImage at full resolution with every image re-encoded the same way so file format can't give the answer away.");
  }

  // ===== 9b. Baselines =====
  const base = readJson(path.join(RES, "baselines/baselines.json"));
  if (base) {
    s = content("Beating simple methods on the same data", "Experiments");
    const bm = MODS.filter((m) => base[m.key] && m.rb);
    const bestBase = (k) => Math.max(...Object.values(base[k]).map((v) => v.accuracy));
    s.addChart(pres.charts.BAR, [
      { name: "Best simple method", labels: bm.map((m) => m.label), values: bm.map((m) => +(bestBase(m.key) * 100).toFixed(1)) },
      { name: "Our model", labels: bm.map((m) => m.label), values: bm.map((m) => +(m.rb.accuracy * 100).toFixed(1)) },
    ], { x: 0.5, y: 1.2, w: 5.6, h: 3.8, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent6, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 50, valAxisMaxVal: 100,
      showTitle: true, title: "Test accuracy (%)", showLegend: true, legendPos: "b", ...chartStyle() });
    const blines = [
      { text: "Simple methods: perplexity threshold, TF-IDF + logistic regression, forensic features + logistic regression", options: { breakLine: true, color: C.text2 } },
    ].concat(bm.map((m, i) => {
      return { text: `${m.label}: ${diffText(bestBase(m.key), m.rb.accuracy, m.rb.n_test)}`, options: { bullet: true, bold: true, breakLine: i < bm.length - 1 } };
    }));
    txt(s, blines, { x: 6.4, y: 1.35, w: 3.1, h: 3.5, fontSize: 13, paraSpaceAfter: 8 });
    s.addNotes("Baselines are trained and tested on exactly the same splits. On essays even a single perplexity number scores highly: HC3 is an easy benchmark, which is why we also test out of distribution on the next slides. The deep model's gain is largest where simple features fall short (images, voice, code).");
  }

  // ===== 9b2. Public detectors =====
  const market = readJson(path.join(RES, "market/market.json"));
  if (market) {
    s = content("Compared with detectors people actually use", "Experiments");
    const short = pubName;
    const groups = [["text", "hc3", "Essays (HC3)"], ["image", "cifake", "Images (CIFAKE)"], ["audio", "asvspoof", "Voice (ASVspoof)"]];
    const colW = 2.95;
    groups.forEach(([mod, set, title], gi) => {
      const ms = Object.entries(market[mod] || {}).filter(([, r]) => r[set]);
      const ours = MODS.find((m) => m.key === mod).rb;
      const labels = ms.map(([n, r]) => short(n) + (r.note ? "*" : "")).concat(ours ? ["Ours"] : []);
      const vals = ms.map(([, r]) => +(r[set].accuracy * 100).toFixed(1)).concat(ours ? [+(ours.accuracy * 100).toFixed(1)] : []);
      s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels, values: vals }], {
        x: 0.5 + gi * (colW + 0.1), y: 1.15, w: colW, h: 3.1, barDir: "bar", chartColors: labels.map((l) => (l === "Ours" ? HEX.accent1 : HEX.accent6)),
        showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisHidden: true,
        showTitle: true, title, showLegend: false, ...chartStyle(), catAxisLabelFontSize: 10 });
    });
    txt(s, "* trained on that same benchmark (home advantage). Commercial tools are closed: Weber-Wulff et al. (2023) found 14 tools incl. Turnitin \"neither accurate nor reliable\"; RAID (Dugan et al., ACL 2024) found detectors \"easily fooled\" by unseen models and simple attacks.",
      { x: 0.5, y: 4.35, w: 9.0, h: 0.75, fontSize: 11, color: C.text2 });
    s.addNotes("Public detectors from Hugging Face, run unchanged on our exact test sets. Be precise when presenting: on these test sets we have a home advantage because we trained on the same benchmarks, except where a public model was trained on it too (marked *). The fair comparison is MAGE, where everyone is out of domain. Commercial tools could not be run without paid API access, so we cite published independent evaluations.");
  }

  // ===== 9b3. Public detectors on today's content =====
  const div12 = readJson(path.join(RES, "ood/mage_diverse_12k.json"));
  if (market && vm && div12?.calibrated) {
    s = content("Today's AI content: ours vs public detectors", "Experiments");
    const groups = [
      ["text", "mage", "Essays: MAGE (27 LLMs)", div12.calibrated.mage_accuracy],
      ["image", "genimage", "Images: GenImage (7 generators)", MODS[2].r?.accuracy],
      ["audio", "modern", "Voice: modern TTS", vm.new_model_on_modern],
    ];
    const colW = 2.95;
    groups.forEach(([mod, set, title, ours], gi) => {
      const ms = Object.entries(market[mod] || {}).filter(([, r]) => r[set]);
      const labels = ms.map(([n]) => pubName(n)).concat(["Ours"]);
      const vals = ms.map(([, r]) => +(r[set].accuracy * 100).toFixed(1)).concat([+(ours * 100).toFixed(1)]);
      s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels, values: vals }], {
        x: 0.5 + gi * (colW + 0.1), y: 1.15, w: colW, h: 3.2, barDir: "bar", chartColors: labels.map((l) => (l === "Ours" ? HEX.accent1 : HEX.accent6)),
        showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 100, valAxisHidden: true,
        showTitle: true, title, showLegend: false, ...chartStyle(), catAxisLabelFontSize: 10 });
    });
    txt(s, "None of these test sets was used to train any public model, so this is the fair comparison. Every model is run unchanged; essays use our calibrated diverse-data model, images and voice our modern-data models.",
      { x: 0.5, y: 4.45, w: 9.0, h: 0.6, fontSize: 11, color: C.text2 });
    s.addNotes("The headline comparison. On benchmarks a specialised public model can win (dima806 on CIFAKE), but on current generators every public detector we could run drops to 52-78%, while our detectors trained on diverse, modern data stay at 78-94%.");
  }

  // ===== 9c. Out-of-distribution =====
  const ood = readJson(path.join(RES, "ood/mage.json"));
  const div = readJson(path.join(RES, "ood/mage_diverse_12k.json"));
  if (ood) {
    s = content("Honest check: no detector is 100% accurate", "Experiments");
    const all = { ...ood.models };
    Object.entries(market?.text || {}).forEach(([n, r]) => {
      if (r.hc3 && r.mage) all[pubName(n)] = { hc3: r.hc3.accuracy, mage: r.mage };
    });
    if (div?.calibrated) all["Ours, diverse data"] = { hc3: div.calibrated.hc3_accuracy, mage: { accuracy: div.calibrated.mage_accuracy } };
    const names = Object.keys(all).map((n) => n.replace("Ours (BERT + forensic + SupCon)", "Ours, HC3 only").replace(" + logistic regression", " + LR"));
    const keys = Object.keys(all);
    s.addChart(pres.charts.BAR, [
      { name: "HC3 test (same source as training)", labels: names, values: keys.map((n) => +(all[n].hc3 * 100).toFixed(1)) },
      { name: "MAGE (unseen models and topics)", labels: names, values: keys.map((n) => +(all[n].mage.accuracy * 100).toFixed(1)) },
    ], { x: 0.5, y: 1.2, w: 5.8, h: 3.8, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent6, HEX.accent3],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 100,
      showTitle: true, title: "Essay detectors: accuracy (%)", showLegend: true, legendPos: "b", ...chartStyle() });
    txt(s, [
      { text: `Tested without retraining on ${ood.n} MAGE texts: ${ood.domains} domains, ${ood.generators} AI models never seen in training.`, options: { breakLine: true } },
      { text: "Every method drops. Benchmark accuracy overstates real-world accuracy, which is why we report this gap, keep an 'uncertain' band, and put verifiable provenance first.", options: { bold: true } },
    ], { x: 6.6, y: 1.35, w: 2.9, h: 3.6, fontSize: 13, paraSpaceAfter: 10 });
    s.addNotes("This answers 'can a detector be 100% accurate?' directly: no. On familiar data we score very high; on new generators and topics every method, including ours, loses accuracy. Our contribution is a system that measures and admits this, rather than claiming perfection.");
  }

  // ===== 9d. Fix: diverse training =====
  if (ood && div) {
    s = content("The fix: train on many AI models, not one", "Experiments");
    const cal = div.calibrated || div;
    const famO = ood.ours_ai_recall_by_family, famD = cal.ai_recall_by_family;
    const fams = Object.keys(famD);
    const key = (f) => f.replace(" (held out)", "");
    s.addChart(pres.charts.BAR, [
      { name: "Trained on HC3 only (ChatGPT)", labels: fams, values: fams.map((f) => +((famO[key(f)]?.recall ?? 0) * 100).toFixed(0)) },
      { name: "HC3 + diverse MAGE data (calibrated)", labels: fams, values: fams.map((f) => +(famD[f].recall * 100).toFixed(0)) },
    ], { x: 0.5, y: 1.2, w: 5.8, h: 3.8, barDir: "bar", barGapWidthPct: 40, chartColors: [HEX.accent6, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 100,
      showTitle: true, title: "% of AI essays caught on MAGE, by model family", showLegend: true, legendPos: "b", ...chartStyle() });
    const oursMage = ood.models[Object.keys(ood.models)[0]].mage;
    txt(s, [
      { text: `MAGE accuracy: ${pct(oursMage.accuracy)} → ${pct(cal.mage_accuracy)}`, options: { bold: true, breakLine: true } },
      { text: `HC3 accuracy stays ${pct(cal.hc3_accuracy)}; ${pct(cal.human_specificity)} of human texts still cleared`, options: { breakLine: true } },
      { text: `Threshold calibrated on MAGE validation data (${(cal.threshold ?? 0.5).toFixed(2)}), never on test.`, options: { color: C.text2, breakLine: true } },
      { text: "OPT was never in training: its bar is a true unseen-generator result.", options: { color: C.text2 } },
    ], { x: 6.6, y: 1.35, w: 2.9, h: 3.6, fontSize: 13, paraSpaceAfter: 10 });
    s.addNotes("Diagnosis and fix. The HC3-only model learned ChatGPT's style. Adding diverse training data (many generators and topics from MAGE's training split) changes that. OPT was held out entirely, so its bar is a genuine unseen-generator result. Report the numbers exactly as they came out.");
  }

  // ===== 10. Contrastive ablation =====
  const abl = MODS.filter((m) => m.ra && m.rce);
  s = content("Does contrastive learning help?", "Experiments");
  if (abl.length) {
    s.addChart(pres.charts.BAR, [
      { name: "Cross-entropy only", labels: abl.map((m) => m.label + (m.ablTag || "")), values: abl.map((m) => +(m.rce.accuracy * 100).toFixed(1)) },
      { name: "Cross-entropy + SupCon", labels: abl.map((m) => m.label + (m.ablTag || "")), values: abl.map((m) => +(m.ra.accuracy * 100).toFixed(1)) },
    ], { x: 0.5, y: 1.2, w: 5.6, h: 3.8, barDir: "col", barGapWidthPct: 60, chartColors: [HEX.accent6, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 100,
      showTitle: true, title: "Test accuracy (%)", showLegend: true, legendPos: "b", ...chartStyle() });
    const lines = abl.map((m) => {
      return { text: `${m.label}${m.ablTag || ""}: ${diffText(m.rce.accuracy, m.ra.accuracy, m.ra.n_test)}`, options: { bullet: true, breakLine: true } };
    });
    lines[lines.length - 1].options.breakLine = false;
    txt(s, lines, { x: 6.4, y: 1.35, w: 3.1, h: 3.5, fontSize: 13, paraSpaceAfter: 8 });
  } else {
    txt(s, "Ablation runs pending.", { x: 0.5, y: 2.5, w: 9, h: 0.5, fontSize: 18, color: C.text2 });
  }
  txt(s, "± is the 95% confidence interval of the difference.", { x: 6.4, y: 4.75, w: 3.1, h: 0.3, fontSize: 11, color: C.text2 });
  s.addNotes("Same architecture, same data, same epochs; the only change is removing the SupCon term. In distribution, contrastive learning barely moves accuracy: differences sit inside the 95% confidence interval. Its claimed benefit is generalisation to unseen generators, tested on the next slide.");

  // ===== 11. Learning curves =====
  s = content("It generalises: train and validation move together", "Experiments");
  // The voice head (logistic regression on wav2vec features) has no epochs, so show the voice CNN's curves.
  const curves = done.map((m) => {
    const own = path.join(RES, m.key, `${m.main}_curves.png`);
    return fs.existsSync(own) ? { m, p: own, label: m.label } : { m, p: path.join(RES, m.key, `${m.abl}_curves.png`), label: `${m.label} (spectrogram CNN)` };
  }).filter((c) => fs.existsSync(c.p));
  curves.forEach(({ m, p, label }, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    txt(s, label, { x: 0.5 + col * 4.6, y: 1.08 + row * 1.92, w: 4.4, h: 0.25, fontSize: 12, bold: true, fontFace: "+mj-lt" });
    s.addImage({ path: p, x: 0.5 + col * 4.6, y: 1.33 + row * 1.92, w: 4.0, h: 1.6 });
  });
  const gaps2 = done.map((m) => `${m.label} ${m.r.generalization_gap >= 0 ? "+" : ""}${(m.r.generalization_gap * 100).toFixed(1)}`).join("   ");
  txt(s, `Train minus test accuracy (pts): ${gaps2}. Voice is tested on unseen speakers.`, { x: 0.5, y: 4.92, w: 8.3, h: 0.25, fontSize: 10, color: C.text2 });
  s.addNotes("Left of each plot: loss; right: accuracy. Validation loss tracking training loss means no overfitting; both being low means no underfitting. The best-validation epoch is restored before testing.");

  // ===== 12. Per generator / per attack =====
  s = content("Which generators and attacks are hardest", "Experiments");
  if (perGen) {
    const gens = Object.keys(perGen).filter((g) => g !== "human");
    s.addChart(pres.charts.BAR, [{ name: "Detected", labels: gens, values: gens.map((g) => +(perGen[g].accuracy * 100).toFixed(1)) }],
      { x: 0.5, y: 1.2, w: 4.4, h: 3.8, barDir: "bar", chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
        valAxisMinVal: 0, valAxisMaxVal: 100, showTitle: true, title: "Code: % of AI samples caught, per LLM", showLegend: false, ...chartStyle() });
  }
  if (perAtk) {
    const atks = Object.keys(perAtk).filter((a) => a !== "bonafide" && a !== "human").sort();
    s.addChart(pres.charts.BAR, [{ name: "Detected", labels: atks, values: atks.map((a) => +((perAtk[a].accuracy ?? perAtk[a]) * 100).toFixed(1)) }],
      { x: 5.1, y: 1.2, w: 4.4, h: 3.8, barDir: "bar", chartColors: [HEX.accent3], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
        valAxisMinVal: 0, valAxisMaxVal: 100, showTitle: true, title: "Voice: % of spoofs caught, per attack", showLegend: false, ...chartStyle() });
  }
  s.addNotes("Breaking results down by source shows where the detector is weakest. ASVspoof attacks A07 to A19 include some generators that are notoriously hard to detect (A17, A18).");

  // ===== 12b. Unseen generator =====
  const hoS = readJson(path.join(RES, "code/code_holdout_gpt_supcon_per_generator.json"));
  const hoC = readJson(path.join(RES, "code/code_holdout_gpt_ce_per_generator.json"));
  if (hoS && hoC) {
    s = content("Catching a generator it never saw in training", "Experiments");
    const g = ["gpt", "codellama", "llama3.1", "nxcode", "qwen1.5", "human"];
    const lab = g.map((x) => (x === "gpt" ? "GPT (unseen)" : x === "human" ? "Human (correct)" : x));
    s.addChart(pres.charts.BAR, [
      { name: "Cross-entropy only", labels: lab, values: g.map((x) => +(hoC[x].accuracy * 100).toFixed(1)) },
      { name: "Cross-entropy + SupCon", labels: lab, values: g.map((x) => +(hoS[x].accuracy * 100).toFixed(1)) },
    ], { x: 0.5, y: 1.2, w: 5.8, h: 3.8, barDir: "col", barGapWidthPct: 50, chartColors: [HEX.accent6, HEX.accent1],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 100,
      showTitle: true, title: "Code detector trained without any GPT code (% correct)", showLegend: true, legendPos: "b", ...chartStyle() });
    txt(s, [
      { text: "Leave-one-generator-out test", options: { bold: true, breakLine: true } },
      { text: "All GPT-written code removed from training; the test set still contains it.", options: { breakLine: true } },
      { text: `Unseen GPT code is still caught ${Math.round(Math.min(hoC.gpt.accuracy, hoS.gpt.accuracy) * 100)}–${Math.round(Math.max(hoC.gpt.accuracy, hoS.gpt.accuracy) * 100)}% of the time (vs 94% when GPT is in training).`, options: { bold: true, breakLine: true } },
      { text: `SupCon vs cross-entropy: ${diffText(hoC.gpt.accuracy, hoS.gpt.accuracy, hoS.gpt.n)}. At this scale contrastive learning gave no measurable gain.`, options: { color: C.text2 } },
    ], { x: 6.6, y: 1.35, w: 2.9, h: 3.6, fontSize: 13, paraSpaceAfter: 10 });
    s.addNotes("The abstract's generalisation claim, tested directly. Finding 1: the detector does transfer to an unseen LLM, losing about 10-16 points. Finding 2: contrastive learning did not improve that transfer here; the difference is inside the confidence interval (only 100 GPT samples). Present this as an honest negative result and future work: more generators, larger batches for SupCon.");
  }

  // ===== 13. Watermark robustness =====
  s = content("Watermarks hold only until the content is edited", "Experiments");
  if (rob) {
    const aa = Object.keys(rob.audio.attacks);
    s.addChart(pres.charts.BAR, [{ name: "AudioSeal", labels: aa, values: aa.map((a) => +(rob.audio.attacks[a].detection_rate * 100).toFixed(0)) }],
      { x: 0.5, y: 1.2, w: 4.6, h: 3.6, barDir: "bar", chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
        valAxisMinVal: 0, valAxisMaxVal: 100, showTitle: true, title: `Audio watermark detected (%), ${rob.audio.n_clips} real clips`, showLegend: false, ...chartStyle() });
    const ta = Object.keys(rob.text.schemes.kgw.attacks);
    s.addChart(pres.charts.BAR, [
      { name: "Green-list (KGW)", labels: ta, values: ta.map((a) => +(rob.text.schemes.kgw.attacks[a].detection_rate * 100).toFixed(0)) },
      { name: "SynthID-Text", labels: ta, values: ta.map((a) => +(rob.text.schemes.synthid.attacks[a].detection_rate * 100).toFixed(0)) },
    ], { x: 5.3, y: 1.2, w: 4.2, h: 3.6, barDir: "bar", barGapWidthPct: 40, chartColors: [HEX.accent2, HEX.accent4], showValue: false,
      valAxisMinVal: 0, valAxisMaxVal: 100, showTitle: true, title: `Text watermark detected (%), ${rob.text.n_prompts} generations`, showLegend: true, legendPos: "b", ...chartStyle() });
    txt(s, "C2PA image credentials were lost after a single JPEG re-save. False positives on unwatermarked human content: 0% in every test.", { x: 0.5, y: 4.85, w: 9.0, h: 0.35, fontSize: 12, color: C.text2 });
  }
  s.addNotes("Real experiment: we embedded real watermarks, then attacked them. Speed change and added noise erase the audio watermark; shuffling words erases text watermarks; re-saving a JPEG strips C2PA. This is why Layer 2 exists.");

  // ===== 14. Demo =====
  pres.addSection({ title: "System" });
  s = content("The working system: drop in anything, get a verdict", "System");
  const shot = path.resolve(__dirname, "../app_screenshot.png");
  if (fs.existsSync(shot)) s.addImage({ path: shot, x: 0.5, y: 1.15, w: 5.9, h: 3.85, sizing: { type: "contain", w: 5.9, h: 3.85 } });
  txt(s, [
    { text: "Web app (FastAPI + custom front end)", options: { bold: true, breakLine: true } },
    { text: "Auto-detects the kind of file dropped in", options: { bullet: true, breakLine: true } },
    { text: "Shows which layer decided, and the AI probability on a human-to-AI scale", options: { bullet: true, breakLine: true } },
    { text: "Loads real held-out test samples and marks whether the model was right", options: { bullet: true, breakLine: true } },
    { text: "Live results page with ROC and learning curves", options: { bullet: true } },
  ], { x: 6.7, y: 1.3, w: 2.8, h: 3.6, fontSize: 13, paraSpaceAfter: 8 });
  s.addNotes("Live demo: run `python server.py` in the code folder and open http://127.0.0.1:8000. Load an AI-made sample, examine it, show the verdict scale and the correct/wrong banner.");

  // ===== 14b. Key findings =====
  {
    s = content("What we found", "System");
    const oursMage = ood ? ood.models[Object.keys(ood.models)[0]].mage.accuracy : null;
    const marketWins = market ? (() => {
      let w = 0, t = 0;
      [["text", "hc3"], ["image", "cifake"], ["audio", "asvspoof"]].forEach(([mod, set]) => {
        const ours = MODS.find((m) => m.key === mod).rb;
        const best = Math.max(...Object.values(market[mod] || {}).filter((r) => r[set]).map((r) => r[set].accuracy));
        if (ours && isFinite(best)) { t++; if (ours.accuracy > best) w++; }
      });
      return `${w} of ${t}`;
    })() : null;
    const findings = [
      [ic.check, C.accent2, "High accuracy on held-out benchmark data", `${done.map((m) => `${m.label} ${pct(m.r.accuracy)}`).join(", ")}. Pretrained wav2vec 2.0 features lifted voice from ${MODS[3].ra ? pct(MODS[3].ra.accuracy) : "the CNN"}.`],
      [ic.check, C.accent2, "Ahead of popular public detectors", marketWins ? `Best in ${marketWins} benchmark comparisons; a CIFAKE-specialised model trained on 16× more data edges us on images.` : "Compared on identical test sets."],
      [ic.question, C.accent3, "Benchmarks overstate real-world accuracy", oursMage != null ? `All essay detectors drop on unseen models; ours fell to ${pct(oursMage)}${div ? `, diverse training lifted it to ${pct((div.calibrated || div).mage_accuracy)}` : ""}.` : "Out-of-distribution testing matters."],
      [ic.question, C.accent3, "Contrastive loss: no measurable gain at this scale", "Every with/without difference sits inside the 95% confidence interval, including on an unseen generator."],
      [ic.shield, C.accent1, "Proof beats prediction when it exists", "Watermarks and C2PA are decisive when present but break under simple edits, so both layers are needed."],
    ];
    findings.forEach(([im, col, h, d], i) => {
      const y = 1.15 + i * 0.8;
      dot(s, 0.5, y, im, col, 0.48);
      txt(s, h, { x: 1.15, y: y - 0.04, w: 8.3, h: 0.32, fontSize: 15, bold: true, fontFace: "+mj-lt" });
      txt(s, d, { x: 1.15, y: y + 0.28, w: 8.3, h: 0.4, fontSize: 12, color: C.text2 });
    });
    s.addNotes("The five takeaways. Lead with the honest ones: a detector that admits its limits and measures them is more credible than one claiming 100%.");
  }

  // ===== 15. Limitations & future =====
  s = content("Limitations and next steps", "System");
  txt(s, "Limitations", { x: 0.5, y: 1.2, w: 4.3, h: 0.4, fontSize: 18, bold: true, fontFace: "+mj-lt", color: C.accent3 });
  txt(s, [
    { text: "Accuracy drops on unseen generators and topics (see MAGE test)", options: { bullet: true, breakLine: true } },
    { text: "Small training subsets on one laptop", options: { bullet: true, breakLine: true } },
    { text: "Unseen generators stay hard (Midjourney held out: 56% caught)", options: { bullet: true, breakLine: true } },
    { text: "Google's SynthID image/audio detectors aren't public; we use open equivalents", options: { bullet: true, breakLine: true } },
    { text: "Text and code detectors see only English and Python", options: { bullet: true } },
  ], { x: 0.5, y: 1.7, w: 4.3, h: 3.2, fontSize: 14, paraSpaceAfter: 8 });
  txt(s, "Next steps", { x: 5.2, y: 1.2, w: 4.3, h: 0.4, fontSize: 18, bold: true, fontFace: "+mj-lt", color: C.accent2 });
  txt(s, [
    { text: "Full datasets and longer training on a GPU", options: { bullet: true, breakLine: true } },
    { text: "Test voice on newer synthesis (ASVspoof 5) and fine-tune wav2vec 2.0", options: { bullet: true, breakLine: true } },
    { text: "Train on many more generators; retest contrastive loss with larger batches", options: { bullet: true, breakLine: true } },
    { text: "Extend code detection beyond Python", options: { bullet: true, breakLine: true } },
    { text: "Robustness training against paraphrase and compression", options: { bullet: true } },
  ], { x: 5.2, y: 1.7, w: 4.3, h: 3.2, fontSize: 14, paraSpaceAfter: 8 });
  s.addNotes("Be upfront: these are baseline results. The architecture is in place for the next semester's work.");

  // ===== 16. Close =====
  s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "System" });
  s.addText("Thank you", { placeholder: "title" });
  s.addText("Questions, and a live demo", { placeholder: "body" });
  s.addNotes("Offer the live demo.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT, `(${done.length}/4 modalities with results)`);
})();
