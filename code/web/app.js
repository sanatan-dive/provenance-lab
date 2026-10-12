const $ = (id) => document.getElementById(id);
const form = $("form"), text = $("text"), drop = $("drop"), fileInput = $("file");
const tabs = [...document.querySelectorAll(".kinds button")];

const KINDS = {
  text: { name: "Essay", placeholder: "Paste the essay here…", accept: ".txt,.md", icon: "i-text", label: "Essays" },
  code: { name: "Python code", placeholder: "Paste Python code here…", accept: ".py", icon: "i-code", label: "Code" },
  image: { name: "Image", accept: "image/*", sub: "JPG, PNG or WebP", icon: "i-image", label: "Images" },
  audio: { name: "Voice recording", accept: "audio/*", sub: "WAV, FLAC, MP3 or OGG", icon: "i-audio", label: "Voice" },
};
const EXT = { py: "code", txt: "text", md: "text" };
// Deployed models (what the app runs and the headline cards show)
const MAIN = { text: "text_bert_supcon", code: "code_codebert_supcon_12k", image: "image_genimage", audio: "audio_w2v_modern" };
// Models scored on the original benchmarks, used wherever we compare against baselines or public detectors on those sets
const BENCH = { text: "text_bert_supcon", code: "code_codebert_supcon_12k", image: "image_vit_fft_supcon", audio: "audio_w2v_lr" };
const MODEL = {
  text: "BERT + perplexity and burstiness", code: "CodeBERT + syntax-tree features",
  image: "ViT + frequency-spectrum CNN", audio: "wav2vec 2.0 (frozen) + linear head",
};
const DATASET = { text: "HC3", code: "CoDET-M4", image: "GenImage (7 modern generators, 224 px)", audio: "ASVspoof 2019 + modern TTS" };
const BENCH_SET = { text: "HC3", code: "CoDET-M4", image: "CIFAKE", audio: "ASVspoof 2019" };

let kind = "text";
let file = null;
let truth = null; // set when a real test sample is loaded: {label, source}

const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const ico = (id) => `<svg><use href="#${id}"/></svg>`;

/* ---------- input ---------- */
function setKind(k) {
  kind = k;
  tabs.forEach((t) => t.setAttribute("aria-selected", t.dataset.kind === k));
  const typed = k === "text" || k === "code";
  text.hidden = !typed;
  drop.hidden = typed;
  text.placeholder = KINDS[k].placeholder || "";
  text.classList.toggle("code", k === "code");
  fileInput.accept = KINDS[k].accept;
  $("dropSub").textContent = KINDS[k].sub || "";
  if (!typed && file && kindOfFile(file) !== k) clearFile();
}

function kindOfFile(f) {
  if (f.type.startsWith("image/")) return "image";
  if (f.type.startsWith("audio/")) return "audio";
  return EXT[f.name.split(".").pop().toLowerCase()] || null;
}

function clearFile() {
  file = null;
  $("preview").hidden = true;
  $("audioPreview").hidden = true;
  $("dropMain").textContent = "Drop a file here, or click to choose one";
}

async function takeFile(f, keepTruth = false) {
  if (!keepTruth) truth = null;
  const k = kindOfFile(f);
  if (!k) return showError(`We can't examine ${f.name}. Use an image, an audio file, a .py file or a .txt file.`);
  setKind(k);
  if (k === "text" || k === "code") {
    text.value = await f.text();
    file = null;
    return;
  }
  file = f;
  $("dropMain").textContent = f.name;
  const url = URL.createObjectURL(f);
  $("preview").hidden = k !== "image";
  $("audioPreview").hidden = k !== "audio";
  if (k === "image") $("preview").src = url;
  else $("audioPreview").src = url;
}

tabs.forEach((t) => t.addEventListener("click", () => setKind(t.dataset.kind)));
text.addEventListener("input", () => (truth = null));
fileInput.addEventListener("change", () => fileInput.files[0] && takeFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((e) => form.addEventListener(e, (ev) => { ev.preventDefault(); form.classList.add("dragging"); }));
["dragleave", "drop"].forEach((e) => form.addEventListener(e, (ev) => { ev.preventDefault(); form.classList.remove("dragging"); }));
form.addEventListener("drop", (ev) => ev.dataTransfer.files[0] && takeFile(ev.dataTransfer.files[0]));

document.querySelectorAll(".sample").forEach((b) => b.addEventListener("click", async () => {
  const res = await fetch(`/api/sample?modality=${kind}&label=${b.dataset.label}`);
  const s = await res.json();
  if (!res.ok || s.error) return showError(s.error || "Couldn't load a sample.");
  if (s.content !== undefined) {
    text.value = s.content;
    file = null;
  } else {
    const blob = await (await fetch(s.data_url)).blob();
    await takeFile(new File([blob], s.filename, { type: blob.type }), true);
  }
  truth = { label: s.label, source: s.source };
  $("hint").textContent = `Loaded from ${s.source}. Choose Examine to test the model.`;
}));

/* ---------- examine ---------- */
function showError(msg) {
  $("empty").hidden = true;
  $("filled").hidden = true;
  $("error").hidden = false;
  $("error").textContent = msg;
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const typed = kind === "text" || kind === "code";
  if (typed && text.value.trim().split(/\s+/).length < 20) return showError("Add at least 20 words so there is enough to examine.");
  if (!typed && !file) return showError(`Choose ${kind === "image" ? "an image" : "a recording"} to examine.`);

  const body = new FormData();
  body.append("modality", kind);
  if (typed) body.append("text", text.value);
  else body.append("file", file);

  const go = $("go");
  go.disabled = true;
  go.textContent = "Examining…";
  form.classList.add("scanning");
  try {
    const res = await fetch("/api/detect", { method: "POST", body });
    const data = await res.json();
    if (!res.ok || data.error) return showError(data.error || `The server returned ${res.status}.`);
    render(data);
  } catch {
    showError("Couldn't reach the server. Check that server.py is running.");
  } finally {
    go.disabled = false;
    go.textContent = "Examine";
    form.classList.remove("scanning");
  }
});

function render(r) {
  $("empty").hidden = true;
  $("error").hidden = true;
  const filled = $("filled");
  filled.hidden = true;
  void filled.offsetWidth; // restart the entrance animation
  filled.hidden = false;
  $("kindLine").textContent = KINDS[r.modality].name;

  const v = $("verdict");
  v.textContent = r.verdict;
  v.className = "verdict " + (r.verdict.startsWith("AI") ? "ai" : r.verdict === "Human" ? "human" : "unsure");

  const t = $("truth");
  t.hidden = !truth;
  if (truth) {
    const said = r.verdict.startsWith("AI") ? 1 : r.verdict === "Human" ? 0 : null;
    const actual = truth.label ? "AI-made" : "human-made";
    if (said === null) {
      t.className = "truth";
      t.textContent = `Actually ${actual}. The model flagged it for review instead of guessing.`;
    } else {
      t.className = "truth " + (said === truth.label ? "right" : "wrong");
      t.textContent = said === truth.label ? `Correct: this sample really is ${actual}.` : `Wrong: this sample is actually ${actual}.`;
    }
  }

  const p = r.layer === "provenance" ? 1 : r.ai_probability;
  const needle = $("needle");
  const nl = $("needleLabel");
  nl.textContent = r.layer === "provenance" ? "proof" : `${Math.round(p * 100)}% AI`;
  nl.style.transform = p > 0.88 ? "translateX(-92%)" : p < 0.12 ? "translateX(-8%)" : "translateX(-50%)";
  needle.style.left = "50%";
  requestAnimationFrame(() => requestAnimationFrame(() => (needle.style.left = `${(p * 100).toFixed(1)}%`)));

  const prov = r.provenance || {};
  const decidedByProv = r.layer === "provenance";
  $("layer1").classList.toggle("decided", decidedByProv);
  $("layer2").classList.toggle("decided", !decidedByProv);
  $("layer1Text").textContent = decidedByProv
    ? `Found ${describeSource(prov.source)}. This is verifiable proof the content came from an AI tool.`
    : prov.source === "c2pa"
      ? `Content credentials found (from ${prov.details?.claim_generator || "an unknown tool"}), but they don't declare AI generation.`
      : "No content credentials or watermark found.";
  renderEvidence(r.explanation);
  $("layer2Text").textContent = decidedByProv
    ? "Not needed: provenance already settled it."
    : `${pct(r.ai_probability)} likely to be AI-generated.` +
      (r.verdict.startsWith("Uncertain") ? " Too close to call, so a person should review it." : "");
}

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function renderEvidence(e) {
  const box = $("evidence"), body = $("evidenceBody");
  box.hidden = !e;
  if (!e) return;
  if (e.kind === "sentences") {
    body.innerHTML = `<p class="ev-note">Each sentence scored on its own. Darker means more AI-like.</p><p class="ev-text">` +
      e.items.map((s) => s.p == null ? `<span>${esc(s.text)}</span>`
        : `<span class="ev-s" style="--a:${Math.max(0, (s.p - 0.3) / 0.7).toFixed(2)}" title="${Math.round(s.p * 100)}% AI">${esc(s.text)}</span>`).join(" ") + "</p>";
  } else if (e.kind === "blocks") {
    body.innerHTML = `<p class="ev-note">Each function or class scored on its own.</p>` +
      e.items.map((b) => `<div class="bar-row with"><span class="lbl"><code>${esc(b.name)}</code></span><span class="bar"><i style="width:${(b.p * 100).toFixed(0)}%"></i></span><span class="val">${Math.round(b.p * 100)}%</span></div>`).join("");
  } else {
    body.innerHTML = `<img class="ev-img ${e.kind}" src="${e.image}" alt="${e.kind === "heatmap" ? "Heatmap of the image regions that drove the verdict" : "Spectrogram of the recording"}">` +
      (e.note ? `<p class="ev-note">${esc(e.note)}</p>` : "");
  }
}

function describeSource(s) {
  return { c2pa: "signed C2PA content credentials", audioseal: "an AudioSeal audio watermark", text_watermark: "a statistical text watermark" }[s] || "a provenance signal";
}

/* ---------- results ---------- */
function diverseHtml(o, d) {
  if (!d) return "";
  const famO = o.ours_ai_recall_by_family || {};
  const c = d.calibrated || d;
  const rows = Object.entries(c.ai_recall_by_family).map(([f, v]) => {
    const before = famO[f.replace(" (held out)", "")]?.recall;
    return `<div class="cmp"><h4>${f}</h4><p class="cmp-sub">Share of AI essays caught on MAGE</p>
      ${bars([["HC3 only", before ?? 0], ["Diverse data", v.recall, "with"]])}</div>`;
  }).join("");
  return `<div class="sec-head"><h3 class="sub">The fix: train on many AI models, not one</h3>
    <p class="lede">The HC3-only model learned ChatGPT's style, so it missed text from other model families. Retraining on HC3 plus MAGE's diverse training split, with the decision threshold calibrated on MAGE validation data, lifts MAGE accuracy from ${pct(o.models[Object.keys(o.models)[0]].mage.accuracy)} to ${pct(c.mage_accuracy)} while HC3 accuracy stays ${pct(c.hc3_accuracy)} and ${pct(c.human_specificity)} of human texts are still cleared. OPT models were held out of training entirely, so that card is a true unseen-generator result. This is the model the essay tab uses.</p></div>
    <div class="compare">${rows}</div>`;
}

const COMPARE = [
  { mod: "text", without: "text_bert_ce", with: "text_bert_supcon", title: "Contrastive loss on essays" },
  { mod: "code", without: "code_codebert_ce", with: "code_codebert_supcon", title: "Contrastive loss on code" },
  { mod: "image", without: "image_vit_fft_ce", with: "image_vit_fft_supcon", title: "Contrastive loss on images" },
  { mod: "audio", without: "audio_cnn_ce", with: "audio_cnn_supcon", title: "Contrastive loss on voice" },
  { mod: "image", without: "image_vit_only_supcon", with: "image_vit_fft_supcon", title: "Frequency branch on images", labels: ["ViT only", "ViT + FFT"] },
  { mod: "audio", without: "audio_cnn_supcon", with: "audio_w2v_lr", title: "Pretrained speech encoder on voice", labels: ["Spectrogram CNN", "wav2vec 2.0"] },
  { mod: "image", without: "image_genimage", with: "image_genimage_nomidjourney", title: "Unseen generator: Midjourney images", gen: "Midjourney",
    labels: ["Midjourney in training", "Never seen"], sub: "Midjourney removed from training. Bars show how much Midjourney output each model still catches." },
  { mod: "code", without: "code_holdout_gpt_ce", with: "code_holdout_gpt_supcon", title: "Unseen generator: GPT code", gen: "gpt",
    sub: "GPT-written code removed from training. Bars show how much unseen GPT code each model still catches." },
];

function bars(rows) {
  return rows.map(([lbl, v, cls]) => `<div class="bar-row ${cls || ""}"><span class="lbl">${lbl}</span>
    <span class="bar"><i style="width:${(v * 100).toFixed(1)}%"></i></span><span class="val">${pct(v)}</span></div>`).join("");
}

// 95% half-width for the difference of two accuracies measured on n test samples each (normal approximation).
const ci95 = (a, b, n) => 1.96 * Math.sqrt((a * (1 - a) + b * (1 - b)) / n);

function deltaTag(d, ci) {
  const pts = d * 100;
  const sig = ci == null || Math.abs(d) > ci;
  const cls = !sig ? "flat" : pts > 0 ? "up" : "down";
  const tail = ci == null ? "" : sig ? "" : " (within noise)";
  return `<span class="delta ${cls}">${pts >= 0 ? "+" : ""}${pts.toFixed(1)} pts${tail}</span>` +
    (ci == null ? "" : ` <span class="muted small">95% CI ±${(ci * 100).toFixed(1)}</span>`);
}

async function loadMetrics() {
  let data;
  try { data = await (await fetch("/api/metrics")).json(); } catch { return; }
  const byName = Object.fromEntries(data.runs.map((m) => [m.name, m]));

  // Proof strip + cards
  const cards = [];
  for (const mod of Object.keys(MAIN)) {
    const m = byName[MAIN[mod]];
    const dd = document.querySelector(`.proof dd[data-mod="${mod}"]`);
    dd.textContent = m ? pct(m.accuracy) : "training";
    dd.classList.toggle("pending", !m);
    cards.push(m ? `<article class="card">
        <div class="card-head"><span class="ico">${ico(KINDS[mod].icon)}</span>${KINDS[mod].label}</div>
        <p class="card-model">${MODEL[mod]}, tested on ${DATASET[mod]}</p>
        <p class="card-big">${pct(m.accuracy)}<small>accuracy</small></p>
        <div class="card-stats">
          <div><b>${m.auc_roc.toFixed(3)}</b>AUC-ROC</div>
          <div><b>${pct(m.f1)}</b>F1</div>
          <div><b>${m.generalization_gap >= 0 ? "+" : ""}${(m.generalization_gap * 100).toFixed(1)}</b>gap (pts)</div>
        </div>
        ${mod === "text" && data.ood_diverse?.calibrated ? `<p class="card-model">The essay tab uses the diverse-data model: ${pct(data.ood_diverse.calibrated.hc3_accuracy)} on HC3, ${pct(data.ood_diverse.calibrated.mage_accuracy)} on unseen models (MAGE).</p>` : ""}
        ${m.roc ? `<img src="${m.roc}" alt="ROC curve for the ${KINDS[mod].label.toLowerCase()} detector" loading="lazy">` : ""}
      </article>`
      : `<article class="card pending"><div class="card-head"><span class="ico">${ico(KINDS[mod].icon)}</span>${KINDS[mod].label}</div>
        <p class="card-model">${MODEL[mod]}</p><p>Training. Results appear here when it finishes.</p></article>`);
  }
  $("cards").innerHTML = cards.join("");

  // Comparisons
  const cmp = COMPARE.filter((c) => byName[c.without] && byName[c.with]).map((c) => {
    const a = byName[c.without], b = byName[c.with];
    const va = c.gen ? a.per_generator?.[c.gen]?.accuracy : a.accuracy;
    const vb = c.gen ? b.per_generator?.[c.gen]?.accuracy : b.accuracy;
    if (va == null || vb == null) return "";
    const [la, lb] = c.labels || ["Without", "With"];
    return `<div class="cmp"><h4>${c.title}</h4>
      <p class="cmp-sub">${c.sub || `Test accuracy. AUC ${a.auc_roc.toFixed(3)} without, ${b.auc_roc.toFixed(3)} with.`}</p>
      ${bars([[la, va], [lb, vb, "with"]])}${deltaTag(vb - va, ci95(va, vb, c.gen ? (b.per_generator?.[c.gen]?.n || 100) : b.n_test))}</div>`;
  }).join("");
  if (cmp) $("compare").innerHTML = cmp;

  // Baselines vs ours
  if (data.baselines) {
    const html = Object.keys(MAIN).filter((mod) => data.baselines[mod]).map((mod) => {
      const ours = byName[BENCH[mod]];
      const rows = Object.entries(data.baselines[mod]).map(([k, v]) => [k.replace(" + logistic regression", " + LR"), v.accuracy]);
      if (ours) rows.push(["Our model", ours.accuracy, "with"]);
      const best = Math.max(...Object.values(data.baselines[mod]).map((v) => v.accuracy));
      return `<div class="cmp wide"><h4>${KINDS[mod].label}</h4><p class="cmp-sub">Test accuracy on ${BENCH_SET[mod]}</p>
        ${bars(rows)}${ours ? deltaTag(ours.accuracy - best, ci95(ours.accuracy, best, ours.n_test)) : ""}</div>`;
    }).join("");
    $("baselines").innerHTML = html;
  }

  // Public detectors
  if (data.market) {
    const SETS = { text: [["hc3", "HC3 test"], ["mage", "MAGE (unseen models)"]], image: [["cifake", "CIFAKE test"], ["genimage", "GenImage test (modern)"]],
      audio: [["asvspoof", "ASVspoof test"], ["modern", "Modern TTS test"]] };
    const short = (n) => n.split("/")[1].replace(/[-_]/g, " ");
    const oursOn = (mod, set) => set === "mage" ? data.ood?.models?.[Object.keys(data.ood.models)[0]]?.mage?.accuracy
      : set === "genimage" ? byName[MAIN.image]?.accuracy : set === "modern" ? data.voice_modern?.new_model_on_modern : byName[BENCH[mod]]?.accuracy;
    const cards = Object.entries(data.market).flatMap(([mod, models]) => SETS[mod].map(([set, setLabel]) => {
      const rows = Object.entries(models).filter(([, r]) => r[set]).map(([n, r]) => [short(n) + (r.note ? " *" : ""), r[set].accuracy]);
      const ours = oursOn(mod, set);
      if (ours != null) rows.push([set === "mage" ? "Ours, HC3 only" : "Our model", ours, "with"]);
      if (set === "mage" && data.ood_diverse?.calibrated) rows.push(["Ours, diverse data", data.ood_diverse.calibrated.mage_accuracy, "with"]);
      const notes = Object.entries(models).filter(([, r]) => r.note).map(([n, r]) => `* ${short(n)}: ${r.note}`).join(" ");
      return `<div class="cmp wide"><h4>${KINDS[mod].label}: ${setLabel}</h4><p class="cmp-sub">Accuracy, every model as released</p>${bars(rows)}
        ${notes ? `<p class="cmp-sub small" style="margin:0.6rem 0 0">${notes}</p>` : ""}</div>`;
    })).join("");
    $("market").innerHTML = `<div class="sec-head"><h3 class="sub">Compared with public detectors</h3>
      <p class="lede">Popular open detectors from Hugging Face, run unchanged on exactly our test sets. Commercial tools such as GPTZero and Turnitin are closed and paid, so we cite independent studies instead: Weber-Wulff et al. (2023) found 14 tools, Turnitin included, "neither accurate nor reliable", and the RAID benchmark (Dugan et al., ACL 2024) found detectors "easily fooled" by unseen models and simple attacks.</p></div>
      <div class="compare">${cards}</div>`;
  }

  // Out-of-distribution
  if (data.ood) {
    const o = data.ood;
    const rows = Object.entries(o.models).map(([k, v]) => `<div class="cmp wide"><h4>${k}</h4>
      ${bars([["HC3 (seen)", v.hc3], ["MAGE (unseen)", v.mage.accuracy, "with"]])}
      <p class="cmp-sub" style="margin:0.4rem 0 0">MAGE AUC ${v.mage.auc_roc.toFixed(3)}</p></div>`).join("");
    $("ood").innerHTML = `<div class="sec-head"><h3 class="sub">Honest check: new AI models, new topics</h3>
      <p class="lede">No detector is 100% accurate, and scores on a familiar benchmark overstate real-world performance. Here every essay detector trained on HC3 is tested, without retraining, on ${o.n} MAGE texts from ${o.domains} domains and ${o.generators} AI models it never saw. Accuracy drops for every method. This gap is the open problem, and it's why the system can say "uncertain".</p></div>
      <div class="compare">${rows}</div>${diverseHtml(o, data.ood_diverse)}`;
  }

  // Learning curves
  $("curves").innerHTML = Object.keys(MAIN).map((mod) => byName[MAIN[mod]]).filter((m) => m && m.curves).map((m) =>
    `<figure><img src="${m.curves}" alt="Training and validation curves for ${m.name}" loading="lazy">
     <figcaption>${KINDS[m.modality].label}: best epoch ${m.best_epoch} of ${m.epochs_run}, train–test gap ${(m.generalization_gap * 100).toFixed(1)} pts</figcaption></figure>`).join("");

  // Watermark robustness
  const rob = data.robustness;
  if (rob) {
    const row = (name, v) => [name, v, v < 0.5 ? "broke" : ""];
    const audio = Object.entries(rob.audio.attacks).map(([k, v]) => row(k, v.detection_rate));
    const kgw = Object.entries(rob.text.schemes.kgw.attacks).map(([k, v]) => row(k, v.detection_rate));
    const syn = Object.entries(rob.text.schemes.synthid.attacks).map(([k, v]) => row(k, v.detection_rate));
    $("robust").innerHTML = `<div class="sec-head"><h3 class="sub">Watermarks break under simple edits</h3>
      <p class="lede">We embedded real watermarks, then attacked the content. Detection is perfect when it's untouched and collapses under noise, speed change or word shuffling. Re-saving a JPEG also strips C2PA credentials. That's why the trained detector is the second layer. False positives on unwatermarked human content were 0% throughout.</p></div>
      <div class="robust-grid">
        <div class="cmp"><h4>AudioSeal audio watermark</h4><p class="cmp-sub">${rob.audio.n_clips} real speech clips, share still detected</p>${bars(audio)}</div>
        <div class="cmp"><h4>Green-list text watermark</h4><p class="cmp-sub">${rob.text.n_prompts} GPT-2 generations, share still detected</p>${bars(kgw)}</div>
        <div class="cmp"><h4>SynthID-Text watermark</h4><p class="cmp-sub">${rob.text.n_prompts} GPT-2 generations, share still detected</p>${bars(syn)}</div>
      </div>`;
  }

  // All runs
  const tbody = document.querySelector("#metrics tbody");
  tbody.innerHTML = data.runs.length ? data.runs.map((m) => `<tr>
      <td>${KINDS[m.modality]?.label || m.modality}</td><td>${m.name}</td>
      <td class="num">${pct(m.accuracy)}</td><td class="num">${pct(m.precision)}</td>
      <td class="num">${pct(m.recall)}</td><td class="num">${pct(m.f1)}</td><td class="num">${m.auc_roc.toFixed(3)}</td>
      <td class="num">${m.generalization_gap == null ? "–" : (m.generalization_gap >= 0 ? "+" : "") + (m.generalization_gap * 100).toFixed(1) + " pts"}</td></tr>`).join("")
    : `<tr><td colspan="8" class="muted">No trained models yet.</td></tr>`;
}

setKind("text");
loadMetrics();
