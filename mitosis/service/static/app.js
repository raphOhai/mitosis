const SVG_NS = "http://www.w3.org/2000/svg";

const STAGES = {
  ingest: { label: "Ingest", hint: "Probe the video, find shots, align the script" },
  understand: { label: "Understand", hint: "Extract the creative DNA of the winner" },
  hero: { label: "Hero lock", hint: "Collect identity references for the hero" },
  hooks: { label: "Hooks", hint: "Write new hooks and policy-check them" },
  plan: { label: "Plan", hint: "Choose the variant mix inside the budget" },
  produce: { label: "Produce", hint: "Hook shots, voices, renders and QA" },
  report: { label: "Report", hint: "Report, Meta sheet and bundle" },
};

const DEFAULT_STAGE_ORDER = ["ingest", "understand", "hero", "hooks", "plan", "produce", "report"];

const STAGE_STATUS_TEXT = {
  pending: "waiting",
  running: "in progress",
  done: "done",
  error: "failed",
};

const RUN_STATUS = {
  queued: { label: "Queued", tone: "" },
  running: { label: "Running", tone: "pill-accent" },
  done: { label: "Done", tone: "pill-success" },
  error: { label: "Failed", tone: "pill-danger" },
};

const VARIANT_STATUS = {
  passed: { label: "QA passed", short: "Passed" },
  flagged: { label: "Flagged", short: "Flagged" },
  failed: { label: "Failed", short: "Failed" },
};

const MODE_COPY = {
  live: {
    label: "Live models",
    tone: "success",
    text: "API keys found. Mitosis uses production models and falls back down each chain automatically.",
  },
  hybrid: {
    label: "Hybrid mode",
    tone: "info",
    text: "Some providers are live. The rest fall back to free local models.",
  },
  offline: {
    label: "Offline mode",
    tone: "info",
    text: "No API keys found. Every step runs on free local models, so you can rehearse the whole flow at zero cost.",
  },
  unknown: {
    label: "Models unavailable",
    tone: "warning",
    text: "The model registry could not be loaded.",
  },
};

const VARIANT_FILTERS = [
  { id: "all", label: "All", matches: () => true },
  { id: "passed", label: "Passed", matches: (variant) => variant.status === "passed" },
  { id: "flagged", label: "Flagged", matches: (variant) => variant.status === "flagged" },
  { id: "failed", label: "Failed", matches: (variant) => variant.status === "failed" },
  { id: "approved", label: "Approved", matches: (variant) => variant.review === "approved" },
  { id: "rejected", label: "Rejected", matches: (variant) => variant.review === "rejected" },
];

const LIVE_VARIANT_LABELS = {
  queued: "queued",
  start: "starting",
  rendering: "rendering",
  keyframe_rejected: "new keyframe",
  "qa:accept": "QA passed",
  "qa:retry_shot": "reshooting",
  "qa:retry_render": "re-rendering",
  "qa:flag": "flagged",
  passed: "passed",
  flagged: "flagged",
  failed: "failed",
};

const REVIEW_FILTERS = new Set(["approved", "rejected"]);

const VARIANT_SORTS = {
  plan: (left, right) => left.index - right.index,
  score: (left, right) => prelaunchValue(right) - prelaunchValue(left),
  cost: (left, right) => (left.shadow_usd ?? 0) - (right.shadow_usd ?? 0),
};

const FACTOR_KEYS = {
  archetype: ["hook", "archetype"],
  voice: ["voice"],
  location: ["location", "place", "setting"],
};

const FINISHED_VARIANT_STATES = new Set(["passed", "flagged", "failed"]);
const ACTIVE_RUN_STATES = new Set(["queued", "running"]);
const VIDEO_EXTENSIONS = /\.(mp4|mov|m4v|webm|mkv)$/i;
const POLL_INTERVAL_MS = 2500;
const REFRESH_DEBOUNCE_MS = 350;
const ACTIVITY_LOG_LIMIT = 60;
const WORDS_PER_SECOND = 2.6;
const DEFAULT_TITLE = document.title;

const ICONS = {
  check: "M5 12.5l4.5 4.5L19 7.5",
  x: "M6.5 6.5l11 11M17.5 6.5l-11 11",
  alert: "M12 8.5v4.5M12 16.5v.01M10.3 3.9L2.6 17.3A2 2 0 0 0 4.3 20h15.4a2 2 0 0 0 1.7-2.7L13.7 3.9a2 2 0 0 0-3.4 0z",
  play: "M8 5.5v13l11-6.5z",
  download: "M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19.5h14",
  external: "M14 4h6v6M20 4l-9 9M19 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4",
  table: "M4 5h16v14H4zM4 10h16M4 15h16M10 5v14",
};

const DOM_IDS = [
  "brand-client",
  "health",
  "health-label",
  "health-panel",
  "health-banner",
  "composer",
  "dropzone",
  "video-input",
  "dropzone-empty",
  "dropzone-preview",
  "video-preview",
  "video-name",
  "video-details",
  "video-error",
  "script-input",
  "script-stats",
  "script-error",
  "winner-input",
  "variants-input",
  "variants-output",
  "renders-hint",
  "aspect-toggles",
  "aspects-error",
  "voice-chips",
  "location-chips",
  "archetype-field",
  "archetype-chips",
  "hooks-input",
  "budget-input",
  "offline-input",
  "body-mode-input",
  "body-mode-hint",
  "music-input",
  "notes-input",
  "composer-summary",
  "estimate-button",
  "multiply-button",
  "learn-form",
  "learn-input",
  "learn-button",
  "learn-result",
  "workspace",
  "run-select",
  "refresh-button",
  "empty-state",
  "run-view",
  "run-header",
  "run-error",
  "pipeline-aside",
  "timeline",
  "progress-text",
  "progress-count",
  "progress-bar",
  "progress-fill",
  "activity",
  "activity-log",
  "live-variants",
  "activity-list",
  "kpis",
  "estimate-panel",
  "hooks-panel",
  "hero-panel",
  "variants-panel",
  "variant-filter",
  "variant-sort",
  "variant-grid",
  "run-actions",
  "toasts",
];

const state = {
  config: null,
  health: null,
  runs: [],
  runId: null,
  run: null,
  lastSeq: 0,
  filter: "all",
  sort: "plan",
  formatByVariant: new Map(),
  stream: null,
  streamFailed: false,
  pollTimer: 0,
  refreshTimer: 0,
  loadToken: 0,
  gridKey: "",
  panelKeys: { hooks: "", hero: "", estimate: "" },
  audibleVideo: null,
  composer: {
    video: null,
    previewUrl: "",
    aspects: new Set(),
    voices: new Set(),
    locations: new Set(),
    archetypes: new Set(),
    winnerEdited: false,
    busy: false,
  },
};

const dom = {};

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function camelize(id) {
  return id.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function bindDom() {
  for (const id of DOM_IDS) {
    dom[camelize(id)] = document.getElementById(id);
  }
}

function isMissing(value) {
  return value === null || value === undefined;
}

function applyProps(node, props) {
  for (const [key, value] of Object.entries(props)) {
    if (isMissing(value) || value === false) {
      continue;
    }
    if (key === "class") {
      node.className = value;
    } else if (key === "text") {
      node.textContent = value;
    } else if (key === "dataset") {
      Object.assign(node.dataset, value);
    } else if (key === "style") {
      Object.entries(value).forEach(([name, styleValue]) => node.style.setProperty(name, String(styleValue)));
    } else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      node.setAttribute(key, value === true ? "" : String(value));
    }
  }
}

function compact(children) {
  return children.flat(Infinity).filter((child) => !isMissing(child) && child !== false && child !== "");
}

function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  applyProps(node, props);
  node.append(...compact(children));
  return node;
}

function setChildren(node, ...children) {
  node.replaceChildren(...compact(children));
}

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", ICONS[name]);
  svg.append(path);
  return svg;
}

function spinner() {
  return h("span", { class: "spinner", "aria-hidden": "true" });
}

const usdFormatter = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

function formatUsd(value) {
  if (isMissing(value) || Number.isNaN(Number(value))) {
    return "–";
  }
  const amount = Number(value);
  return amount > 0 && amount < 0.01 ? "<$0.01" : usdFormatter.format(amount);
}

function formatSeconds(value) {
  if (isMissing(value) || Number.isNaN(Number(value))) {
    return "–";
  }
  const seconds = Number(value);
  if (seconds < 10) {
    return `${seconds.toFixed(1)}s`;
  }
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
  }
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
}

function formatPercent(fraction) {
  return isMissing(fraction) || Number.isNaN(Number(fraction)) ? "–" : `${Math.round(Number(fraction) * 100)}%`;
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let size = Number(bytes) || 0;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDateTime(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatClock(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function plural(count, singular, pluralForm = `${singular}s`) {
  return `${count} ${Number(count) === 1 ? singular : pluralForm}`;
}

function humanize(identifier) {
  const text = String(identifier ?? "").replace(/[_-]+/g, " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function padIndex(index) {
  return String(index ?? 0).padStart(2, "0");
}

function normalizeScore(score) {
  if (isMissing(score) || Number.isNaN(Number(score))) {
    return null;
  }
  const value = Number(score);
  return Math.max(0, Math.min(100, value <= 1 ? value * 100 : value));
}

function prelaunchValue(variant) {
  return normalizeScore(variant.prelaunch?.score) ?? -1;
}

function scoreColor(value) {
  if (value >= 70) {
    return "var(--success)";
  }
  return value >= 45 ? "var(--warning)" : "var(--danger)";
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 48);
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function describeApiError(payload, status) {
  const detail = payload?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map((item) => `${humanize(item.loc?.at(-1) ?? "field")}: ${item.msg}`).join(" · ");
  }
  return `The server answered with an error (${status}).`;
}

async function api(path, { method = "GET", body, json } = {}) {
  const init = { method, headers: { Accept: "application/json" } };
  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the Mitosis server. Check that it is still running.", 0);
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(describeApiError(payload, response.status), response.status);
  }
  return payload;
}

function runPath(runId, suffix = "") {
  return `/api/runs/${encodeURIComponent(runId)}${suffix}`;
}

function toast(message, tone = "info", timeout = 5000) {
  const item = h(
    "div",
    { class: `toast toast-${tone}`, role: tone === "error" ? "alert" : "status" },
    h("span", { class: "toast-message", text: message }),
    h("button", { class: "toast-close", type: "button", "aria-label": "Dismiss notification", onclick: () => dismissToast(item) }, icon("x")),
  );
  dom.toasts.append(item);
  while (dom.toasts.children.length > 4) {
    dom.toasts.firstElementChild.remove();
  }
  if (timeout) {
    window.setTimeout(() => dismissToast(item), timeout);
  }
}

function dismissToast(item) {
  if (!item.isConnected || item.classList.contains("is-leaving")) {
    return;
  }
  item.classList.add("is-leaving");
  window.setTimeout(() => item.remove(), 200);
}

function showFieldError(element, message) {
  element.textContent = message;
  element.hidden = false;
}

function hideFieldError(element) {
  element.textContent = "";
  element.hidden = true;
}

function setButtonBusy(button, busy, label = "") {
  if (busy) {
    button.dataset.label = button.textContent;
    setChildren(button, spinner(), label);
    button.classList.add("is-busy");
    button.setAttribute("aria-busy", "true");
    return;
  }
  if (button.dataset.label) {
    button.textContent = button.dataset.label;
    delete button.dataset.label;
  }
  button.classList.remove("is-busy");
  button.removeAttribute("aria-busy");
}

async function loadHealth() {
  try {
    state.health = await api("/api/health");
  } catch (error) {
    dom.health.dataset.tone = "danger";
    dom.healthLabel.textContent = "Server unreachable";
    setChildren(dom.healthPanel, h("h2", { text: "Server unreachable" }), h("p", { text: error.message }));
    return;
  }
  renderHealth(state.health);
}

function chainRows(chains) {
  return Object.entries(chains ?? {}).flatMap(([kind, names]) => [
    h("dt", { text: kind }),
    h("dd", { text: names.length ? names.join(" → ") : "not configured" }),
  ]);
}

function renderHealth(health) {
  const copy = MODE_COPY[health.mode] ?? MODE_COPY.unknown;
  dom.health.dataset.tone = health.ffmpeg ? copy.tone : "danger";
  dom.healthLabel.textContent = health.ffmpeg ? copy.label : "ffmpeg missing";
  setChildren(
    dom.healthPanel,
    h("h2", { text: copy.label }),
    h("p", { text: health.error ? `${copy.text} ${health.error}` : copy.text }),
    h("dl", { class: "chain-list", "aria-label": "Model fallback chains" }, chainRows(health.chains)),
  );
  dom.healthBanner.hidden = Boolean(health.ffmpeg);
  if (!health.ffmpeg) {
    dom.healthBanner.textContent = "ffmpeg is not installed on this machine, so renders will fail. Install it (for example: brew install ffmpeg) and restart Mitosis.";
  }
  if (health.mode === "offline" && !dom.offlineInput.dataset.touched) {
    dom.offlineInput.checked = true;
  }
}

async function loadConfig() {
  try {
    state.config = await api("/api/config");
  } catch (error) {
    toast(`Couldn't load voices and locations. ${error.message}`, "error", 0);
    return;
  }
  renderComposerOptions(state.config);
}

function renderComposerOptions(config) {
  if (config.brand?.name) {
    dom.brandClient.textContent = `for ${config.brand.name}`;
    dom.brandClient.hidden = false;
  }
  const aspects = config.aspects ?? [];
  state.composer.aspects = new Set(aspects.map((aspect) => aspect.value));
  setChildren(dom.aspectToggles, ...aspects.map(aspectToggle));
  setChildren(
    dom.voiceChips,
    ...(config.voices ?? []).map((voice) => choiceChip(voice.label, voice.tone, state.composer.voices, voice.id)),
  );
  setChildren(
    dom.locationChips,
    ...(config.locations ?? []).map((place) => choiceChip(place.label, locationCaption(place), state.composer.locations, place.id)),
  );
  const archetypes = config.archetypes ?? [];
  dom.archetypeField.hidden = archetypes.length === 0;
  setChildren(
    dom.archetypeChips,
    ...archetypes.map((archetype) => choiceChip(archetype.label, archetype.description, state.composer.archetypes, archetype.id)),
  );
  renderBodyModes(config.body_modes ?? [], config.defaults?.body_mode ?? "auto");
  applyDefaults(config.defaults ?? {}, config.limits ?? {});
  updateComposerSummary();
}

function locationCaption(place) {
  return place.original ? "Same set as the winner" : `${humanize(place.grade)} grade`;
}

function aspectToggle(aspect) {
  const scale = 18 / Math.max(aspect.width, aspect.height);
  const button = h(
    "button",
    { type: "button", class: "segment", "aria-pressed": "true", title: `${aspect.width}×${aspect.height}` },
    h("span", {
      class: "aspect-glyph",
      "aria-hidden": "true",
      style: { "--glyph-w": Math.round(aspect.width * scale), "--glyph-h": Math.round(aspect.height * scale) },
    }),
    aspect.value,
  );
  button.addEventListener("click", () => toggleAspect(button, aspect.value));
  return button;
}

function toggleAspect(button, value) {
  const selected = state.composer.aspects;
  if (selected.has(value) && selected.size === 1) {
    showFieldError(dom.aspectsError, "Keep at least one format.");
    return;
  }
  if (selected.has(value)) {
    selected.delete(value);
  } else {
    selected.add(value);
  }
  button.setAttribute("aria-pressed", String(selected.has(value)));
  hideFieldError(dom.aspectsError);
  updateComposerSummary();
}

function choiceChip(title, caption, selection, value) {
  const button = h(
    "button",
    { type: "button", class: "chip-toggle", "aria-pressed": String(selection.has(value)), title: caption || null },
    h("span", { class: "chip-title", text: title }),
    caption ? h("span", { class: "chip-sub", text: caption }) : null,
  );
  button.addEventListener("click", () => {
    if (selection.has(value)) {
      selection.delete(value);
    } else {
      selection.add(value);
    }
    button.setAttribute("aria-pressed", String(selection.has(value)));
    updateComposerSummary();
  });
  return button;
}

function renderBodyModes(modes, selected) {
  setChildren(dom.bodyModeInput, ...modes.map((mode) => h("option", { value: mode.id, text: mode.label })));
  dom.bodyModeInput.value = selected;
  updateBodyModeHint();
}

function updateBodyModeHint() {
  const mode = (state.config?.body_modes ?? []).find((item) => item.id === dom.bodyModeInput.value);
  dom.bodyModeHint.textContent = mode?.description ?? "";
}

function applyDefaults(defaults, limits) {
  const [minVariants, maxVariants] = limits.variants ?? [3, 30];
  dom.variantsInput.min = String(minVariants);
  dom.variantsInput.max = String(maxVariants);
  dom.variantsInput.value = String(Math.min(maxVariants, Math.max(minVariants, defaults.variants ?? 10)));
  dom.variantsOutput.textContent = dom.variantsInput.value;
  const [minHooks, maxHooks] = limits.hook_count ?? [2, 12];
  dom.hooksInput.min = String(minHooks);
  dom.hooksInput.max = String(maxHooks);
  dom.hooksInput.value = String(defaults.hook_count ?? 6);
  dom.budgetInput.value = String(defaults.budget_usd ?? 40);
  if (defaults.offline === true && !dom.offlineInput.dataset.touched) {
    dom.offlineInput.checked = true;
  }
}

function orderedAspects() {
  const known = (state.config?.aspects ?? []).map((aspect) => aspect.value);
  return known.filter((value) => state.composer.aspects.has(value));
}

function updateComposerSummary() {
  const variants = Number(dom.variantsInput.value) || 0;
  const formats = state.composer.aspects.size;
  const renders = variants * formats;
  dom.variantsOutput.textContent = String(variants);
  dom.rendersHint.textContent = `${plural(variants, "variant")} × ${plural(formats, "format")} = ${plural(renders, "render")}`;
  const voices = state.composer.voices.size || "all";
  const places = state.composer.locations.size || "all";
  dom.composerSummary.textContent = `${plural(renders, "render")} · voices: ${voices} · locations: ${places} · budget ${formatUsd(Number(dom.budgetInput.value) || 0)}`;
}

function updateScriptStats() {
  const words = dom.scriptInput.value.trim().split(/\s+/).filter(Boolean).length;
  const seconds = words / WORDS_PER_SECOND;
  dom.scriptStats.textContent = words ? `${plural(words, "word")} · about ${formatSeconds(seconds)} spoken` : "0 words";
  if (words) {
    hideFieldError(dom.scriptError);
    dom.scriptInput.removeAttribute("aria-invalid");
  }
}

function setVideo(file) {
  if (!file.type.startsWith("video/") && !VIDEO_EXTENSIONS.test(file.name)) {
    showFieldError(dom.videoError, "That file isn't a video. Use MP4, MOV or WebM.");
    dom.dropzone.classList.add("is-invalid");
    return;
  }
  hideFieldError(dom.videoError);
  dom.dropzone.classList.remove("is-invalid");
  state.composer.video = file;
  if (state.composer.previewUrl) {
    URL.revokeObjectURL(state.composer.previewUrl);
  }
  state.composer.previewUrl = URL.createObjectURL(file);
  dom.videoPreview.muted = true;
  dom.videoPreview.src = state.composer.previewUrl;
  dom.videoName.textContent = file.name;
  dom.videoDetails.textContent = formatBytes(file.size);
  dom.dropzoneEmpty.hidden = true;
  dom.dropzonePreview.hidden = false;
  dom.dropzone.classList.add("has-file");
  if (!state.composer.winnerEdited) {
    dom.winnerInput.value = slugify(file.name);
  }
}

function describePreview() {
  const preview = dom.videoPreview;
  const file = state.composer.video;
  if (!file || !preview.videoWidth) {
    return;
  }
  dom.videoDetails.textContent = `${formatSeconds(preview.duration)} · ${preview.videoWidth}×${preview.videoHeight} · ${formatBytes(file.size)}`;
}

function bindDropzone() {
  const zone = dom.dropzone;
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
  }
  zone.addEventListener("dragleave", (event) => {
    if (!zone.contains(event.relatedTarget)) {
      zone.classList.remove("is-dragging");
    }
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      setVideo(file);
    }
  });
  dom.videoInput.addEventListener("change", () => {
    const file = dom.videoInput.files?.[0];
    if (file) {
      setVideo(file);
    }
  });
  dom.videoPreview.addEventListener("loadedmetadata", describePreview);
  for (const type of ["dragover", "drop"]) {
    window.addEventListener(type, (event) => {
      if (event.dataTransfer?.types?.includes("Files")) {
        event.preventDefault();
      }
    });
  }
}

function validateComposer() {
  const invalid = [];
  if (!state.composer.video) {
    showFieldError(dom.videoError, "Add the winning video first.");
    dom.dropzone.classList.add("is-invalid");
    invalid.push(dom.videoInput);
  }
  if (!dom.scriptInput.value.trim()) {
    showFieldError(dom.scriptError, "Paste the script so the new hooks match the winner's voice.");
    dom.scriptInput.setAttribute("aria-invalid", "true");
    invalid.push(dom.scriptInput);
  }
  if (state.composer.aspects.size === 0) {
    showFieldError(dom.aspectsError, "Pick at least one format.");
    invalid.push(dom.aspectToggles.querySelector("button"));
  }
  invalid[0]?.focus();
  return invalid.length === 0;
}

function composerFormData(dryRun) {
  const data = new FormData();
  const video = state.composer.video;
  data.append("video", video, video.name);
  data.append("script", dom.scriptInput.value.trim());
  data.append("winner_id", dom.winnerInput.value.trim());
  data.append("variants", dom.variantsInput.value);
  data.append("aspects", orderedAspects().join(","));
  data.append("voices", [...state.composer.voices].join(","));
  data.append("locations", [...state.composer.locations].join(","));
  data.append("archetypes", [...state.composer.archetypes].join(","));
  data.append("hook_count", dom.hooksInput.value || "6");
  data.append("budget_usd", dom.budgetInput.value || "0");
  data.append("offline", String(dom.offlineInput.checked));
  data.append("body_mode", dom.bodyModeInput.value || "auto");
  data.append("dry_run", String(dryRun));
  data.append("notes", dom.notesInput.value.trim());
  const music = dom.musicInput.files?.[0];
  if (music) {
    data.append("music", music, music.name);
  }
  return data;
}

function setComposerBusy(busy, dryRun) {
  state.composer.busy = busy;
  dom.estimateButton.disabled = busy;
  dom.multiplyButton.disabled = busy;
  const active = dryRun ? dom.estimateButton : dom.multiplyButton;
  setButtonBusy(active, busy, "Uploading");
}

async function submitComposer(dryRun) {
  if (state.composer.busy || !validateComposer()) {
    return;
  }
  setComposerBusy(true, dryRun);
  try {
    const created = await api("/api/runs", { method: "POST", body: composerFormData(dryRun) });
    toast(dryRun ? "Estimating. Hooks and the plan are on their way." : "Multiplying. Follow the pipeline on the right.", "success");
    await refreshRuns({ quiet: true });
    await selectRun(created.run_id);
    dom.workspace.focus({ preventScroll: true });
  } catch (error) {
    toast(error.message, "error", 9000);
  } finally {
    setComposerBusy(false, dryRun);
  }
}

function bindComposer() {
  bindDropzone();
  dom.scriptInput.addEventListener("input", updateScriptStats);
  dom.winnerInput.addEventListener("input", () => {
    state.composer.winnerEdited = dom.winnerInput.value.trim() !== "";
  });
  for (const input of [dom.variantsInput, dom.hooksInput, dom.budgetInput]) {
    input.addEventListener("input", updateComposerSummary);
  }
  dom.offlineInput.addEventListener("change", () => {
    dom.offlineInput.dataset.touched = "true";
  });
  dom.bodyModeInput.addEventListener("change", updateBodyModeHint);
  dom.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    submitComposer(false);
  });
  dom.estimateButton.addEventListener("click", () => submitComposer(true));
  dom.learnForm.addEventListener("submit", submitLearn);
}

async function submitLearn(event) {
  event.preventDefault();
  const file = dom.learnInput.files?.[0];
  if (!file) {
    toast("Choose the results CSV first.", "error");
    dom.learnInput.focus();
    return;
  }
  const data = new FormData();
  data.append("file", file, file.name);
  setButtonBusy(dom.learnButton, true, "Learning");
  dom.learnButton.disabled = true;
  try {
    const summary = await api("/api/learn", { method: "POST", body: data });
    renderLearnResult(summary);
    toast(`Learned from ${file.name}. The next batches lean on what worked.`, "success");
  } catch (error) {
    toast(error.message, "error", 9000);
  } finally {
    dom.learnButton.disabled = false;
    setButtonBusy(dom.learnButton, false);
  }
}

function learnedLevel(level) {
  return h("li", {
    class: "pill",
    title: `${plural(level.wins, "win")} in ${plural(level.trials, "test")}`,
    text: `${humanize(level.level)} ${formatPercent(level.mean)}`,
  });
}

function renderLearnResult(summary) {
  const factors = Object.entries(summary.factors ?? {});
  if (factors.length === 0) {
    setChildren(dom.learnResult, h("p", { class: "field-hint", text: "Nothing learned yet. Check that the CSV has one row per variant." }));
    return;
  }
  setChildren(
    dom.learnResult,
    ...factors.map(([factor, levels]) =>
      h(
        "div",
        { class: "learn-factor" },
        h("h3", { text: `Best ${humanize(factor).toLowerCase()}` }),
        h("ul", { class: "learn-levels" }, levels.slice(0, 3).map(learnedLevel)),
      ),
    ),
  );
}

async function refreshRuns({ quiet = false } = {}) {
  try {
    const payload = await api("/api/runs");
    state.runs = payload.runs ?? [];
  } catch (error) {
    if (!quiet) {
      toast(error.message, "error");
    }
    return;
  }
  renderRunSelect();
}

function runOptionLabel(run) {
  const status = RUN_STATUS[run.status]?.label ?? humanize(run.status);
  const kind = run.kind === "estimate" ? "Estimate" : plural(run.variants ?? 0, "variant");
  return `${run.winner_id || run.run_id} · ${kind} · ${status} · ${formatDateTime(run.created_at)}`;
}

function renderRunSelect() {
  const select = dom.runSelect;
  if (state.runs.length === 0) {
    setChildren(select, h("option", { text: "No runs yet" }));
    select.disabled = true;
    return;
  }
  select.disabled = false;
  setChildren(select, ...state.runs.map((run) => h("option", { value: run.run_id, text: runOptionLabel(run) })));
  if (state.runId && state.runs.some((run) => run.run_id === state.runId)) {
    select.value = state.runId;
  }
}

function readHashRun() {
  const match = /^#run=(.+)$/.exec(window.location.hash);
  return match ? decodeURIComponent(match[1]) : null;
}

function writeHash(runId) {
  const next = `#run=${encodeURIComponent(runId)}`;
  if (window.location.hash !== next) {
    window.history.replaceState(null, "", next);
  }
}

function showEmptyState() {
  state.runId = null;
  state.run = null;
  dom.emptyState.hidden = false;
  dom.runView.hidden = true;
  document.title = DEFAULT_TITLE;
}

function resetRunView() {
  closeStream();
  window.clearTimeout(state.pollTimer);
  window.clearTimeout(state.refreshTimer);
  state.run = null;
  state.lastSeq = 0;
  state.streamFailed = false;
  state.gridKey = "";
  state.panelKeys = { hooks: "", hero: "", estimate: "" };
  state.formatByVariant.clear();
  state.filter = "all";
}

async function selectRun(runId) {
  if (!runId) {
    return;
  }
  resetRunView();
  state.runId = runId;
  if (state.runs.some((run) => run.run_id === runId)) {
    dom.runSelect.value = runId;
  }
  writeHash(runId);
  await loadRun();
}

async function loadRun() {
  const runId = state.runId;
  if (!runId) {
    return;
  }
  const token = ++state.loadToken;
  let run;
  try {
    run = await api(runPath(runId));
  } catch (error) {
    if (token === state.loadToken) {
      handleLoadFailure(runId, error);
    }
    return;
  }
  if (token !== state.loadToken || runId !== state.runId) {
    return;
  }
  state.run = run;
  renderRun();
  if (ACTIVE_RUN_STATES.has(run.status)) {
    followRun(runId);
  } else {
    closeStream();
    window.clearTimeout(state.pollTimer);
  }
}

function handleLoadFailure(runId, error) {
  if (error.status === 404) {
    toast(`Run ${runId} no longer exists.`, "error");
    window.history.replaceState(null, "", window.location.pathname);
    const fallback = state.runs.find((run) => run.run_id !== runId);
    if (fallback) {
      selectRun(fallback.run_id);
    } else {
      showEmptyState();
    }
    return;
  }
  toast(error.message, "error");
  schedulePoll();
}

function followRun(runId) {
  if (state.stream) {
    return;
  }
  if (state.streamFailed || typeof EventSource === "undefined") {
    schedulePoll();
    return;
  }
  const source = new EventSource(runPath(runId, "/events"));
  state.stream = source;
  source.addEventListener("message", (message) => handleStreamMessage(runId, message));
  source.addEventListener("end", (message) => handleStreamEnd(runId, message));
  source.addEventListener("error", () => {
    if (source.readyState === EventSource.CLOSED) {
      closeStream();
      state.streamFailed = true;
      schedulePoll();
    }
  });
}

function closeStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
}

function schedulePoll() {
  window.clearTimeout(state.pollTimer);
  state.pollTimer = window.setTimeout(loadRun, POLL_INTERVAL_MS);
}

function scheduleRefresh() {
  window.clearTimeout(state.refreshTimer);
  state.refreshTimer = window.setTimeout(loadRun, REFRESH_DEBOUNCE_MS);
}

function handleStreamMessage(runId, message) {
  if (runId !== state.runId || !state.run) {
    return;
  }
  const event = parseJson(message.data);
  if (!event || event.seq <= state.lastSeq) {
    return;
  }
  state.lastSeq = event.seq;
  applyLiveEvent(state.run, event);
  renderLive(state.run);
  if (event.type === "stage" && event.status === "done") {
    scheduleRefresh();
  }
}

function handleStreamEnd(runId, message) {
  closeStream();
  if (runId !== state.runId) {
    return;
  }
  const outcome = parseJson(message.data) ?? {};
  const isEstimate = state.run?.kind === "estimate";
  if (outcome.status === "done") {
    toast(isEstimate ? "Estimate ready. Review the hooks and the plan." : "All variants are rendered and checked.", "success");
  } else {
    toast(outcome.error ? `The run stopped: ${outcome.error}` : "The run stopped.", "error", 9000);
  }
  window.clearTimeout(state.refreshTimer);
  loadRun();
  refreshRuns({ quiet: true });
}

function findStage(run, name) {
  let stage = run.stages.find((item) => item.name === name);
  if (!stage) {
    stage = { name, status: "pending", started_at: null, seconds: null, message: "" };
    run.stages.push(stage);
  }
  return stage;
}

function applyStageEvent(run, event) {
  const stage = findStage(run, event.stage);
  if (event.status === "start") {
    Object.assign(stage, { status: "running", started_at: event.ts, message: event.message });
  } else if (event.status === "done") {
    const elapsed = stage.started_at ? event.ts - stage.started_at : null;
    Object.assign(stage, { status: "done", seconds: event.data?.seconds ?? elapsed });
  }
}

function applyVariantEvent(run, event) {
  let variant = run.variants.find((item) => item.id === event.variant);
  if (!variant) {
    variant = { id: event.variant, status: "queued", hook: "" };
    run.variants.push(variant);
  }
  variant.status = event.status ?? variant.status;
  if (event.message) {
    variant.hook = event.message;
  }
  if (!FINISHED_VARIANT_STATES.has(event.status)) {
    return;
  }
  Object.assign(variant, {
    usd: event.data?.usd ?? null,
    shadow_usd: event.data?.shadow_usd ?? null,
    seconds: event.data?.seconds ?? null,
    score: event.data?.score ?? null,
  });
  run.variants_done = run.variants.filter((item) => FINISHED_VARIANT_STATES.has(item.status)).length;
  if (event.progress) {
    run.variants_total = Math.max(run.variants_done, Math.round(run.variants_done / event.progress));
  }
  run.variant_progress = run.variants_total ? run.variants_done / run.variants_total : 0;
}

function applyLiveEvent(run, event) {
  run.events = [...(run.events ?? []), event].slice(-ACTIVITY_LOG_LIMIT * 2);
  if (run.status === "queued") {
    run.status = "running";
  }
  if (event.type === "stage" && event.stage) {
    applyStageEvent(run, event);
  } else if (event.type === "variant" && event.variant) {
    applyVariantEvent(run, event);
  } else if (event.type === "error" && event.stage) {
    findStage(run, event.stage).status = "error";
  }
}

function renderRun() {
  const run = state.run;
  dom.emptyState.hidden = true;
  dom.runView.hidden = false;
  state.lastSeq = Math.max(state.lastSeq, ...(run.events ?? []).map((event) => event.seq ?? 0));
  renderLive(run);
  renderRunError(run);
  renderEstimate(run);
  renderHooks(run);
  renderHero(run);
  renderVariants(run);
  renderActions(run);
}

function renderLive(run) {
  renderRunHeader(run);
  renderTimeline(run);
  renderProgress(run);
  renderActivity(run);
  renderLiveVariants(run);
  renderKpis(run);
  updateDocumentTitle(run);
}

function updateDocumentTitle(run) {
  const name = run.brief?.winner_id || run.run_id;
  document.title = ACTIVE_RUN_STATES.has(run.status) ? `${formatPercent(run.progress)} · ${name} · Mitosis` : `${name} · Mitosis`;
}

function statusPill(status) {
  const meta = RUN_STATUS[status] ?? { label: humanize(status), tone: "" };
  return h("span", { class: `pill ${meta.tone}` }, status === "running" ? spinner() : null, meta.label);
}

function modePill(mode) {
  const copy = MODE_COPY[mode];
  return copy ? h("span", { class: "pill", title: copy.text, text: copy.label }) : null;
}

function separator() {
  return h("span", { "aria-hidden": "true", text: "·" });
}

function renderRunHeader(run) {
  const brief = run.brief ?? {};
  setChildren(
    dom.runHeader,
    h(
      "div",
      { class: "run-heading" },
      h(
        "div",
        { class: "run-title-row" },
        h("h2", { class: "run-title", id: "run-title", text: brief.winner_id || run.run_id }),
        statusPill(run.status),
        run.kind === "estimate" ? h("span", { class: "pill pill-accent", text: "Cost estimate" }) : null,
        modePill(run.mode),
      ),
      h(
        "div",
        { class: "run-meta" },
        h("span", { class: "mono", text: run.run_id }),
        separator(),
        h("time", { datetime: run.created_at, text: formatDateTime(run.created_at) }),
        brief.video ? [separator(), h("span", { text: brief.video })] : null,
      ),
    ),
    h(
      "div",
      { class: "run-brief", "aria-label": "Brief" },
      h("span", { class: "pill", text: plural(brief.variants ?? 0, "variant") }),
      h("span", { class: "pill", text: (brief.aspects ?? []).join(" · ") || "No formats" }),
      h("span", { class: "pill", text: plural(brief.hook_count ?? 0, "hook") }),
      h("span", { class: "pill", text: `Budget ${formatUsd(brief.budget_usd)}` }),
    ),
  );
}

function renderRunError(run) {
  const message = run.status === "error" ? run.error || "The run stopped before it finished." : run.manifest_error;
  dom.runError.hidden = !message;
  if (message) {
    setChildren(dom.runError, h("strong", { text: run.status === "error" ? "This run stopped. " : "Results unavailable. " }), message);
  }
}

function stageGlyph(status) {
  if (status === "done") {
    return icon("check");
  }
  if (status === "error") {
    return icon("x");
  }
  return status === "running" ? spinner() : null;
}

function stageTime(stage) {
  if (stage.status === "running" && stage.started_at) {
    const elapsed = Math.max(0, Date.now() / 1000 - stage.started_at);
    return h("span", { class: "stage-time", dataset: { since: stage.started_at }, text: formatSeconds(elapsed) });
  }
  const text = isMissing(stage.seconds) ? "" : formatSeconds(stage.seconds);
  return h("span", { class: "stage-time", text });
}

function stageItem(stage) {
  const info = STAGES[stage.name] ?? { label: humanize(stage.name), hint: "" };
  return h(
    "li",
    { class: `stage stage-${stage.status}`, title: stage.message || info.hint },
    h("span", { class: "stage-dot", "aria-hidden": "true" }, stageGlyph(stage.status)),
    h("span", { class: "stage-name", text: info.label }),
    h("span", { class: "visually-hidden", text: `: ${STAGE_STATUS_TEXT[stage.status] ?? stage.status}` }),
    stageTime(stage),
  );
}

function pipelineAside(run) {
  if (run.status === "queued") {
    return "Waiting for a free worker";
  }
  if (run.status === "running") {
    return `${formatPercent(run.progress)} overall`;
  }
  if (run.status === "error") {
    return "Stopped";
  }
  const seconds = finishedIn(run);
  return isMissing(seconds) ? "Finished" : `Finished in ${formatSeconds(seconds)}`;
}

function finishedIn(run) {
  const recorded = run.manifest?.totals?.seconds ?? run.totals?.seconds;
  if (recorded) {
    return recorded;
  }
  const started = Date.parse(run.started_at ?? "");
  const finished = Date.parse(run.finished_at ?? "");
  return Number.isNaN(started) || Number.isNaN(finished) ? null : (finished - started) / 1000;
}

function renderTimeline(run) {
  const stages = run.stages?.length ? run.stages : DEFAULT_STAGE_ORDER.map((name) => ({ name, status: "pending" }));
  dom.timeline.style.setProperty("--stage-count", String(stages.length));
  setChildren(dom.timeline, ...stages.map(stageItem));
  dom.pipelineAside.textContent = pipelineAside(run);
}

function currentStageLabel(run) {
  const stage = (run.stages ?? []).find((item) => item.status === "running");
  return stage ? STAGES[stage.name]?.label ?? humanize(stage.name) : "starting";
}

function progressModel(run) {
  const total = run.variants_total || run.brief?.variants || 0;
  const done = run.variants_done || 0;
  if (run.kind === "estimate") {
    const label = run.status === "done" ? "Estimate ready. Nothing was rendered and nothing was spent." : "Dry run: writing hooks and planning, no renders";
    return { label, value: (run.progress ?? 0) * 100, count: formatPercent(run.progress) };
  }
  const producing = (run.stages ?? []).some((stage) => stage.name === "produce" && stage.status !== "pending");
  if (producing || run.status === "done") {
    const value = total ? (done / total) * 100 : 0;
    return { label: `${done} of ${plural(total, "variant")} produced`, value, count: formatPercent(value / 100) };
  }
  if (run.status === "queued") {
    return { label: "Queued. It starts as soon as a worker is free.", value: 0, count: "" };
  }
  if (run.status === "error") {
    return { label: "Stopped before production started", value: 0, count: "" };
  }
  return { label: `Preparing: ${currentStageLabel(run)}. Variants start after planning.`, value: 0, count: "" };
}

function renderProgress(run) {
  const model = progressModel(run);
  const value = Math.max(0, Math.min(100, model.value));
  dom.progressText.textContent = model.label;
  dom.progressCount.textContent = model.count;
  dom.progressBar.setAttribute("aria-valuenow", String(Math.round(value)));
  dom.progressBar.setAttribute("aria-valuetext", model.label);
  dom.progressBar.classList.toggle("is-error", run.status === "error");
  dom.progressFill.style.setProperty("--value", `${value}%`);
}

function eventTag(event) {
  return event.variant || event.stage || event.type;
}

function eventText(event) {
  if (event.type === "variant") {
    return [humanize(String(event.status ?? "").replace("qa:", "QA ")), event.message].filter(Boolean).join(" · ");
  }
  if (event.type === "stage") {
    return [event.status === "start" ? "started" : event.status, event.message].filter(Boolean).join(" · ");
  }
  return event.message || event.type;
}

function activityItem(event) {
  return h(
    "li",
    { class: event.type === "error" ? "is-error" : null },
    h("span", { class: "activity-time", text: formatClock(event.ts) }),
    h("span", { class: "activity-tag", text: eventTag(event) }),
    h("span", { text: eventText(event) }),
  );
}

function renderActivity(run) {
  const events = run.events ?? [];
  const latest = [...events].reverse().find((event) => event.message && event.type !== "variant");
  const text = ACTIVE_RUN_STATES.has(run.status) && latest ? latest.message : "";
  if (dom.activity.textContent !== text) {
    dom.activity.textContent = text;
  }
  dom.activityLog.hidden = events.length === 0;
  setChildren(dom.activityList, events.slice(-ACTIVITY_LOG_LIMIT).reverse().map(activityItem));
}

function liveVariantState(status) {
  return FINISHED_VARIANT_STATES.has(status) ? status : "working";
}

function liveVariantLabel(status) {
  const key = String(status ?? "queued");
  return LIVE_VARIANT_LABELS[key] ?? humanize(key.replace("qa:", "QA ")).toLowerCase();
}

function renderLiveVariants(run) {
  const showLive = !run.manifest && run.kind === "run" && (run.variants ?? []).length > 0;
  if (!showLive) {
    setChildren(dom.liveVariants);
    return;
  }
  setChildren(
    dom.liveVariants,
    ...run.variants.map((variant) =>
      h(
        "li",
        { class: "live-variant", dataset: { state: liveVariantState(variant.status) }, title: variant.hook || variant.id },
        h("span", { class: "live-variant-dot", "aria-hidden": "true" }),
        `${variant.id} ${liveVariantLabel(variant.status)}`,
      ),
    ),
  );
}

function kpiTile({ label, value, sub }) {
  return h(
    "div",
    { class: "kpi" },
    h("span", { class: "kpi-label", text: label }),
    h("span", { class: "kpi-value" }, value),
    sub ? h("span", { class: "kpi-sub", text: sub }) : null,
  );
}

function splitValue(parts) {
  const spoken = parts.map(({ value, label }) => `${value} ${label}`).join(", ");
  const visible = parts.flatMap(({ value, tone }, index) => [
    index ? h("span", { class: "kpi-sep", text: "/" }) : null,
    h("span", { class: tone, text: value }),
  ]);
  return h("span", { class: "kpi-split" }, h("span", { "aria-hidden": "true" }, visible), h("span", { class: "visually-hidden", text: spoken }));
}

function outcomeSplit(passed, flagged, failed) {
  return splitValue([
    { value: passed, tone: "tone-success", label: "passed" },
    { value: flagged, tone: "tone-warning", label: "flagged" },
    { value: failed, tone: "tone-danger", label: "failed" },
  ]);
}

function sumOf(items, key) {
  return items.reduce((total, item) => total + (Number(item[key]) || 0), 0);
}

function liveTiles(run) {
  const finished = (run.variants ?? []).filter((variant) => FINISHED_VARIANT_STATES.has(variant.status));
  if (finished.length === 0) {
    return [];
  }
  const count = (status) => finished.filter((variant) => variant.status === status).length;
  const passed = count("passed");
  return [
    { label: "Variants", value: `${finished.length} / ${run.variants_total || run.brief?.variants || finished.length}`, sub: "produced so far" },
    { label: "QA outcome", value: outcomeSplit(passed, count("flagged"), count("failed")), sub: "passed / flagged / failed" },
    { label: "Pass rate", value: formatPercent(passed / finished.length), sub: "so far" },
    { label: "Actual spend", value: formatUsd(sumOf(finished, "usd")), sub: "so far" },
    { label: "Production-equivalent", value: formatUsd(sumOf(finished, "shadow_usd")), sub: "at live model prices" },
    { label: "Time per variant", value: formatSeconds(sumOf(finished, "seconds") / finished.length), sub: "average so far" },
  ];
}

function runTiles(run) {
  const totals = run.manifest?.totals ?? run.totals;
  if (!totals) {
    return liveTiles(run);
  }
  const finished = (totals.passed ?? 0) + (totals.flagged ?? 0) + (totals.failed ?? 0);
  return [
    { label: "Variants", value: String(totals.variants ?? 0), sub: plural(totals.renders ?? 0, "render") },
    { label: "QA outcome", value: outcomeSplit(totals.passed ?? 0, totals.flagged ?? 0, totals.failed ?? 0), sub: "passed / flagged / failed" },
    { label: "Pass rate", value: finished ? formatPercent((totals.passed ?? 0) / finished) : "–", sub: "cleared every QA gate" },
    { label: "Actual spend", value: formatUsd(totals.usd), sub: run.mode === "offline" ? "free local models" : "billed by providers" },
    { label: "Production-equivalent", value: formatUsd(totals.shadow_usd), sub: "at live model prices" },
    { label: "Cost per variant", value: formatUsd(totals.usd_per_variant), sub: `${formatUsd(totals.shadow_per_variant)} at live prices` },
    { label: "Time per variant", value: formatSeconds(totals.seconds_per_variant), sub: `${formatSeconds(totals.seconds)} in total` },
    { label: "Cache savings", value: formatUsd(totals.saved_usd), sub: plural(totals.cache_hits ?? 0, "cache hit") },
  ];
}

function estimateTiles(run) {
  const estimate = run.estimate;
  const plan = estimate?.plan;
  if (!plan) {
    return [];
  }
  const variants = (plan.variants ?? []).length;
  const formats = (run.brief?.aspects ?? []).length || 1;
  const cost = plan.estimate ?? {};
  const budget = run.brief?.budget_usd;
  const tiles = [
    { label: "Planned variants", value: String(variants), sub: plan.design ? `${humanize(plan.design)} design` : "" },
    { label: "Renders", value: String(variants * formats), sub: plural(formats, "format") },
    { label: "Production cost", value: formatUsd(cost.usd), sub: "if run on live models" },
    { label: "Per variant", value: formatUsd(cost.per_variant_usd), sub: "at live prices" },
    {
      label: "Hooks",
      value: splitValue([
        { value: (estimate.hooks ?? []).length, tone: "tone-success", label: "accepted" },
        { value: (estimate.rejected_hooks ?? []).length, tone: "tone-danger", label: "blocked" },
      ]),
      sub: "accepted / blocked by policy",
    },
  ];
  if (cost.seconds) {
    tiles.push({ label: "Estimated time", value: formatSeconds(cost.seconds), sub: "wall clock" });
  }
  if (!isMissing(budget)) {
    tiles.push({ label: "Budget", value: formatUsd(budget), sub: `${formatUsd(Math.max(0, budget - (cost.usd ?? 0)))} headroom` });
  }
  return tiles;
}

function renderKpis(run) {
  const tiles = run.kind === "estimate" ? estimateTiles(run) : runTiles(run);
  setChildren(dom.kpis, ...tiles.map(kpiTile));
}

function panelHead(id, title, aside = "") {
  return h(
    "div",
    { class: "panel-head" },
    h("h2", { class: "panel-title", id, text: title }),
    aside ? h("span", { class: "panel-aside", text: aside }) : null,
  );
}

function levelPills(plan) {
  const levels = Object.entries(plan.levels ?? {}).map(([factor, values]) =>
    h("span", { class: "pill", text: `${values.length} ${humanize(factor).toLowerCase()}` }),
  );
  const coverage = Object.entries(plan.coverage ?? {}).map(([factor, value]) =>
    h("span", { class: "pill", title: "Share of levels covered", text: `${humanize(factor)} coverage ${Number(value) <= 1 ? formatPercent(value) : value}` }),
  );
  return [...levels, ...coverage];
}

function planRow(spec) {
  return h(
    "tr",
    {},
    h("td", { class: "num", text: padIndex(spec.index) }),
    h(
      "td",
      {},
      h("div", { text: spec.hook?.text ?? "" }),
      spec.hook?.on_screen_text ? h("div", { class: "hook-onscreen", text: `On screen: ${spec.hook.on_screen_text}` }) : null,
    ),
    h("td", { text: humanize(spec.hook?.archetype) }),
    h("td", { text: spec.voice?.label ?? "" }),
    h("td", { text: spec.location?.label ?? "" }),
    h("td", { text: (spec.changed ?? []).map(humanize).join(", ") || "–" }),
  );
}

function planTable(variants) {
  if (variants.length === 0) {
    return h("p", { class: "panel-empty", text: "The planner returned no variants. Raise the budget or add more hooks." });
  }
  return h(
    "div",
    { class: "table-scroll" },
    h(
      "table",
      { class: "plan-table" },
      h(
        "thead",
        {},
        h("tr", {}, ["#", "Hook", "Style", "Voice", "Location", "Changes"].map((label) => h("th", { scope: "col", text: label }))),
      ),
      h("tbody", {}, variants.map(planRow)),
    ),
  );
}

function renderEstimate(run) {
  const plan = run.estimate?.plan;
  const visible = run.kind === "estimate" && Boolean(plan);
  dom.estimatePanel.hidden = !visible;
  if (!visible) {
    state.panelKeys.estimate = "";
    return;
  }
  const key = `${run.run_id}:${(plan.variants ?? []).length}`;
  if (state.panelKeys.estimate === key) {
    return;
  }
  state.panelKeys.estimate = key;
  const pills = levelPills(plan);
  setChildren(
    dom.estimatePanel,
    panelHead("estimate-title", "Planned variants", plan.design ? `${humanize(plan.design)} design` : ""),
    pills.length ? h("div", { class: "plan-summary" }, pills) : null,
    (plan.notes ?? []).length ? h("ul", { class: "plan-notes" }, plan.notes.map((note) => h("li", { text: note }))) : null,
    planTable(plan.variants ?? []),
    h(
      "div",
      { class: "estimate-cta" },
      h("p", { text: "Happy with the plan? Production reuses the settings in the panel on the left." }),
      h("button", { type: "button", class: "button button-primary", onclick: () => submitComposer(false) }, "Multiply winner"),
    ),
  );
}

function hookScore(score) {
  const value = normalizeScore(score);
  if (!value) {
    return null;
  }
  return h(
    "span",
    { class: "hook-score", title: "Hook score" },
    h("span", { class: "hook-score-track", "aria-hidden": "true" }, h("span", { class: "hook-score-fill", style: { "--value": `${value}%` } })),
    h("span", { text: `${Math.round(value)}` }),
    h("span", { class: "visually-hidden", text: " out of 100" }),
  );
}

function flagItem(flag) {
  return h(
    "li",
    { class: `flag flag-${flag.severity}` },
    h("span", { class: "flag-rule", text: flag.rule }),
    h("span", { class: "flag-message", text: flag.message }),
    flag.suggestion ? h("span", { class: "flag-suggestion", text: `Try instead: ${flag.suggestion}` }) : null,
  );
}

function hookItem(hook, blocked) {
  const flags = blocked ? hook.flags ?? [] : (hook.flags ?? []).filter((flag) => flag.severity === "warn");
  return h(
    "li",
    { class: `hook${blocked ? " hook-blocked" : ""}` },
    h("div", { class: "hook-head" }, h("span", { class: "factor", text: humanize(hook.archetype) }), blocked ? null : hookScore(hook.score)),
    h("p", { class: "hook-text", text: hook.text }),
    hook.on_screen_text ? h("p", { class: "hook-onscreen", text: `On screen: ${hook.on_screen_text}` }) : null,
    flags.length ? h("ul", { class: "flag-list" }, flags.map(flagItem)) : null,
  );
}

function hooksEmptyCopy(run) {
  if (run.status === "error") {
    return "No hooks were written before the run stopped.";
  }
  return "Hooks appear here as soon as the copywriter stage finishes.";
}

function renderHooks(run) {
  const source = run.manifest ?? run.estimate ?? run.partial ?? {};
  const accepted = source.hooks ?? [];
  const blocked = source.rejected_hooks ?? [];
  const key = JSON.stringify([run.run_id, run.status === "error", accepted.map((hook) => hook.id), blocked.map((hook) => hook.id)]);
  if (state.panelKeys.hooks === key) {
    return;
  }
  state.panelKeys.hooks = key;
  setChildren(
    dom.hooksPanel,
    panelHead("hooks-title", "Hooks", accepted.length ? `${accepted.length} accepted` : ""),
    accepted.length
      ? h("ul", { class: "hook-list" }, accepted.map((hook) => hookItem(hook, false)))
      : h("p", { class: "panel-empty", text: hooksEmptyCopy(run) }),
    blocked.length
      ? [
          h("h3", { class: "subheading" }, "Blocked before Meta could reject them", h("span", { class: "blocked-count", text: blocked.length })),
          h("ul", { class: "hook-list" }, blocked.map((hook) => hookItem(hook, true))),
        ]
      : null,
  );
}

function heroStrip(references) {
  return h(
    "ul",
    { class: "hero-strip", "aria-label": "Hero reference frames" },
    references.map((url, index) =>
      h(
        "li",
        { class: "hero-ref" },
        h(
          "a",
          { href: url, target: "_blank", rel: "noopener", "aria-label": `Open hero reference ${index + 1}` },
          h("img", {
            src: url,
            alt: `Hero reference ${index + 1}`,
            loading: "lazy",
            decoding: "async",
            onerror: (event) => {
              event.currentTarget.hidden = true;
            },
          }),
        ),
      ),
    ),
  );
}

function dnaList(dna) {
  const rows = [
    ["Angle", dna.angle],
    ["Promise", dna.promise],
    ["Audience", dna.audience],
    ["Tone", dna.tone],
    ["Original hook", dna.hook_text],
    ["Format", humanize(dna.format)],
  ].filter(([, value]) => value);
  return h("dl", { class: "dna" }, rows.flatMap(([label, value]) => [h("dt", { text: label }), h("dd", { text: value })]));
}

function heroBadges(hero) {
  return h(
    "div",
    { class: "hero-badge-row" },
    hero.identity_locked
      ? h("span", { class: "pill pill-success" }, icon("check"), "Identity locked")
      : h("span", { class: "pill pill-warning", text: "No identity embedding" }),
    h("span", { class: "pill", text: plural(hero.references.length, "reference") }),
  );
}

function renderHero(run) {
  const source = run.manifest ?? run.partial ?? {};
  const hero = source.hero;
  const dna = source.dna;
  const key = JSON.stringify([run.run_id, hero?.references ?? null, Boolean(dna)]);
  if (state.panelKeys.hero === key) {
    return;
  }
  state.panelKeys.hero = key;
  const children = [panelHead("hero-title", "Hero and creative DNA")];
  if (!hero && !dna) {
    children.push(h("p", { class: "panel-empty", text: "The hero lock appears once the winner has been analysed." }));
  }
  if (hero) {
    children.push(heroBadges(hero));
    if (hero.description) {
      children.push(h("p", { class: "hero-description", text: hero.description }));
    }
    if (hero.references.length) {
      children.push(heroStrip(hero.references));
    }
  }
  if (dna) {
    children.push(h("h3", { class: "subheading", text: "What made the winner work" }), dnaList(dna));
  }
  setChildren(dom.heroPanel, ...children);
}

function findVariant(variantId) {
  return (state.run?.manifest?.results ?? []).find((variant) => variant.id === variantId) ?? null;
}

function currentRender(variant) {
  const renders = variant.renders ?? [];
  const chosen = state.formatByVariant.get(variant.id);
  return renders.find((render) => render.aspect === chosen) ?? renders.find((render) => render.aspect === "9:16") ?? renders[0] ?? null;
}

function qaBadge(variant) {
  const meta = VARIANT_STATUS[variant.status] ?? { label: humanize(variant.status) };
  return h("span", { class: "media-badge", dataset: { state: variant.status }, text: meta.label });
}

function playVideo(video) {
  if (!video.muted && state.audibleVideo && state.audibleVideo !== video) {
    state.audibleVideo.pause();
  }
  if (!video.muted) {
    state.audibleVideo = video;
  }
  video.play().catch(() => {});
}

function bindPlayback(button, video) {
  video.addEventListener("play", () => button.classList.add("is-playing"));
  video.addEventListener("pause", () => button.classList.remove("is-playing"));
  button.addEventListener("mouseenter", () => {
    if (video.paused) {
      video.muted = true;
      playVideo(video);
    }
  });
  button.addEventListener("mouseleave", () => {
    if (video.muted) {
      video.pause();
    }
  });
  button.addEventListener("click", () => {
    if (video.paused || video.muted) {
      video.muted = false;
      playVideo(video);
    } else {
      video.pause();
    }
  });
}

function showRender(media, variant, render) {
  const { button, video } = media;
  const wasPlaying = !video.paused;
  video.pause();
  video.poster = render.thumbnail ?? "";
  video.src = render.url;
  button.setAttribute("aria-label", `Play variant ${padIndex(variant.index)} in ${render.aspect}`);
  if (wasPlaying) {
    playVideo(video);
  }
}

function createMedia(variant) {
  const badges = h("span", { class: "media-badges" }, h("span", { class: "media-index", text: `#${padIndex(variant.index)}` }), qaBadge(variant));
  const render = currentRender(variant);
  if (!render?.url) {
    const placeholder = h(
      "div",
      { class: "media", role: "img", "aria-label": `Variant ${padIndex(variant.index)} has no render` },
      badges,
      h("span", { class: "media-empty" }, icon("alert"), variant.error ? "Render failed. Details below." : "No render available"),
    );
    return { element: placeholder, button: null, video: null };
  }
  const video = h("video", { preload: "none", loop: true, playsinline: true });
  video.muted = true;
  video.playsInline = true;
  const button = h("button", { type: "button", class: "media" }, video, badges, h("span", { class: "media-play", "aria-hidden": "true" }, icon("play")));
  const media = { element: button, button, video };
  showRender(media, variant, render);
  bindPlayback(button, video);
  return media;
}

function formatSwitch(variant, media) {
  const group = h("div", { class: "segmented segmented-compact", role: "group", "aria-label": `Format for variant ${padIndex(variant.index)}` });
  const active = currentRender(variant)?.aspect;
  for (const render of variant.renders) {
    const button = h("button", { type: "button", class: "segment", "aria-pressed": String(render.aspect === active), text: render.aspect });
    button.addEventListener("click", () => {
      state.formatByVariant.set(variant.id, render.aspect);
      group.querySelectorAll("button").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      if (media.video) {
        showRender(media, variant, render);
      }
    });
    group.append(button);
  }
  return group;
}

function factorChips(variant) {
  const changed = new Set((variant.changed ?? []).map((factor) => String(factor).toLowerCase()));
  const chips = [
    { factor: "archetype", text: humanize(variant.hook?.archetype), title: "Hook style" },
    { factor: "voice", text: variant.voice?.label, title: variant.voice?.tone ? `Voice: ${variant.voice.tone}` : "Voice" },
    { factor: "location", text: variant.location?.label, title: "Hook location" },
  ].filter((chip) => chip.text);
  return h(
    "div",
    { class: "factors" },
    chips.map((chip) => {
      const isChanged = FACTOR_KEYS[chip.factor].some((key) => changed.has(key));
      return h(
        "span",
        { class: `factor${isChanged ? " is-changed" : ""}`, title: isChanged ? `${chip.title}, changed from the winner` : chip.title },
        chip.text,
        isChanged ? h("span", { class: "visually-hidden", text: " (changed from the winner)" }) : null,
      );
    }),
  );
}

function groupIssues(issues) {
  const groups = new Map();
  for (const issue of issues) {
    const key = [issue.name, issue.severity, issue.detail, issue.score, issue.threshold].join("|");
    const group = groups.get(key) ?? { ...issue, aspects: [] };
    group.aspects.push(issue.aspect);
    groups.set(key, group);
  }
  return [...groups.values()];
}

function issueText(issue, formatCount) {
  const measured = !isMissing(issue.score) && !isMissing(issue.threshold) ? `${Number(issue.score).toFixed(2)} vs ${Number(issue.threshold).toFixed(2)} needed` : "";
  const detail = issue.detail || measured;
  const scope = formatCount > 1 && issue.aspects.length === formatCount ? "all formats" : issue.aspects.join(", ");
  return `${humanize(issue.name)} (${scope})${detail ? `: ${detail}` : ""}`;
}

function qaSection(variant) {
  const qa = variant.qa ?? { issues: [], checks: 0 };
  if (variant.error && !(variant.renders ?? []).length) {
    return h("div", { class: "qa" }, h("p", { class: "qa-error", text: variant.error }));
  }
  if (!qa.issues.length) {
    const text = qa.checks ? `All ${plural(qa.checks, "QA check")} passed` : "No QA checks recorded";
    return h("div", { class: "qa" }, h("p", { class: "qa-ok" }, icon("check"), text));
  }
  return h(
    "div",
    { class: "qa" },
    h(
      "ul",
      { class: "qa-issues", "aria-label": "QA findings" },
      groupIssues(qa.issues).map((issue) => {
        const warning = issue.severity === "warn";
        const text = issueText(issue, (variant.renders ?? []).length);
        return h("li", { class: `qa-issue${warning ? " is-warning" : ""}` }, icon(warning ? "alert" : "x"), h("span", { text }));
      }),
    ),
  );
}

function prelaunchScore(variant) {
  const value = normalizeScore(variant.prelaunch?.score);
  if (isMissing(value)) {
    return null;
  }
  const tooltipId = `prelaunch-${variant.id}`;
  const reasons = variant.prelaunch.reasons ?? [];
  return h(
    "div",
    { class: "score", tabindex: "0", "aria-describedby": tooltipId, style: { "--value": `${value}%`, "--score-color": scoreColor(value) } },
    h("span", { class: "score-label", text: "Pre-launch" }),
    h("span", { class: "score-track", "aria-hidden": "true" }, h("span", { class: "score-fill" })),
    h("span", { class: "score-value", text: `${Math.round(value)}` }),
    h(
      "span",
      { class: "score-tooltip", role: "tooltip", id: tooltipId },
      reasons.length ? h("ul", {}, reasons.map((reason) => h("li", { text: reason }))) : "No reasons were recorded for this score.",
    ),
  );
}

function variantMeta(variant) {
  return h(
    "div",
    { class: "variant-meta" },
    h("span", {}, h("strong", { text: formatUsd(variant.usd) }), ` actual · ${formatUsd(variant.shadow_usd)} production`),
    h("span", { text: `${plural(variant.attempts ?? 0, "attempt")} · ${formatSeconds(variant.seconds)}` }),
  );
}

function reviewButton(variant, decision, label, iconName) {
  return h(
    "button",
    {
      type: "button",
      class: "button button-secondary review-button",
      dataset: { decision },
      "aria-pressed": String(variant.review === decision),
      onclick: () => reviewVariant(variant.id, decision),
    },
    icon(iconName),
    label,
  );
}

function noteInput(variant) {
  const input = h("input", {
    type: "text",
    class: "review-note",
    maxlength: "500",
    placeholder: variant.review === "rejected" ? "Why not? It helps the next batch" : "Note for the media buyer",
    "aria-label": `Review note for variant ${padIndex(variant.index)}`,
  });
  input.value = variant.note ?? "";
  input.addEventListener("change", () => {
    const current = findVariant(variant.id);
    const note = input.value.trim();
    if (current && note !== (current.note ?? "")) {
      reviewVariant(variant.id, current.review, note);
    }
  });
  return input;
}

function fillReview(container, variant) {
  setChildren(
    container,
    h(
      "div",
      { class: "review-buttons", role: "group", "aria-label": `Review variant ${padIndex(variant.index)}` },
      reviewButton(variant, "approved", "Approve", "check"),
      reviewButton(variant, "rejected", "Reject", "x"),
    ),
    variant.review === "pending" ? null : noteInput(variant),
  );
}

function variantCard(variant) {
  const media = createMedia(variant);
  const review = h("div", { class: "review" });
  fillReview(review, variant);
  return h(
    "article",
    { class: "variant-card", dataset: { variantId: variant.id, review: variant.review }, "aria-labelledby": `variant-${variant.id}-hook` },
    media.element,
    h(
      "div",
      { class: "variant-body" },
      (variant.renders ?? []).length > 1 ? formatSwitch(variant, media) : null,
      h("p", { class: "variant-hook", id: `variant-${variant.id}-hook`, title: variant.hook?.text ?? "", text: variant.hook?.text ?? variant.id }),
      factorChips(variant),
      qaSection(variant),
      prelaunchScore(variant),
      variantMeta(variant),
      review,
    ),
  );
}

function cardFor(variantId) {
  return [...dom.variantGrid.querySelectorAll(".variant-card")].find((card) => card.dataset.variantId === variantId) ?? null;
}

function refreshCardReview(variant, focusDecision) {
  const card = cardFor(variant.id);
  if (!card) {
    return;
  }
  card.dataset.review = variant.review;
  fillReview(card.querySelector(".review"), variant);
  if (focusDecision) {
    card.querySelector(`.review-button[data-decision="${focusDecision}"]`)?.focus();
  }
}

function reviewCounts(results) {
  return {
    approved: results.filter((variant) => variant.review === "approved").length,
    rejected: results.filter((variant) => variant.review === "rejected").length,
    pending: results.filter((variant) => variant.review === "pending").length,
  };
}

function afterReviewChange() {
  const manifest = state.run?.manifest;
  if (!manifest) {
    return;
  }
  manifest.review = reviewCounts(manifest.results);
  renderFilter(manifest.results);
  renderActions(state.run);
  if (REVIEW_FILTERS.has(state.filter)) {
    state.gridKey = "";
    renderVariants(state.run);
  }
}

async function reviewVariant(variantId, decision, note) {
  const runId = state.runId;
  const variant = findVariant(variantId);
  if (!variant) {
    return;
  }
  const previous = { review: variant.review, note: variant.note };
  const nextNote = note ?? variant.note ?? "";
  const decisionChanged = variant.review !== decision;
  Object.assign(variant, { review: decision, note: nextNote });
  if (decisionChanged) {
    refreshCardReview(variant, decision);
    afterReviewChange();
  }
  try {
    const updated = await api(runPath(runId, `/variants/${encodeURIComponent(variantId)}/review`), {
      method: "POST",
      json: { decision, note: nextNote },
    });
    Object.assign(variant, { review: updated.review, note: updated.note });
    toast(decisionChanged ? `Variant #${padIndex(variant.index)} ${decision}.` : "Note saved.", "success", 2500);
  } catch (error) {
    Object.assign(variant, previous);
    if (runId === state.runId) {
      refreshCardReview(variant);
      afterReviewChange();
    }
    toast(`Couldn't save the review. ${error.message}`, "error");
  }
}

function renderFilter(results) {
  setChildren(
    dom.variantFilter,
    ...VARIANT_FILTERS.map((filter) => {
      const count = results.filter(filter.matches).length;
      return h(
        "button",
        {
          type: "button",
          class: "segment",
          "aria-pressed": String(state.filter === filter.id),
          onclick: () => {
            state.filter = filter.id;
            state.gridKey = "";
            renderVariants(state.run);
          },
        },
        filter.label,
        h("span", { class: "segment-count", text: count }),
      );
    }),
  );
}

function renderVariants(run) {
  const manifest = run.manifest;
  dom.variantsPanel.hidden = !manifest;
  if (!manifest) {
    setChildren(dom.variantGrid);
    return;
  }
  const results = manifest.results ?? [];
  renderFilter(results);
  const reviewSignature = results.map((variant) => `${variant.id}:${variant.review}`).join(",");
  const key = [run.run_id, manifest.created_at, state.filter, state.sort, reviewSignature].join("|");
  if (state.gridKey === key) {
    return;
  }
  state.gridKey = key;
  const filter = VARIANT_FILTERS.find((item) => item.id === state.filter) ?? VARIANT_FILTERS[0];
  const visible = results.filter(filter.matches).sort(VARIANT_SORTS[state.sort] ?? VARIANT_SORTS.plan);
  if (visible.length === 0) {
    const message = results.length ? "No variants match this filter." : "This run produced no variants.";
    setChildren(dom.variantGrid, h("p", { class: "grid-empty", text: message }));
    return;
  }
  setChildren(dom.variantGrid, ...visible.map(variantCard));
}

function actionLink(href, label, iconName, extra = {}, primary = false) {
  const className = `button ${primary ? "button-primary" : "button-secondary"}`;
  if (!href) {
    return h("span", { class: className, "aria-disabled": "true", title: "Not available for this run" }, icon(iconName), label);
  }
  return h("a", { class: className, href, ...extra }, icon(iconName), label);
}

function renderActions(run) {
  const manifest = run.manifest;
  dom.runActions.hidden = !manifest;
  if (!manifest) {
    return;
  }
  const links = manifest.links ?? {};
  const review = manifest.review ?? reviewCounts(manifest.results ?? []);
  const variants = manifest.totals?.variants ?? (manifest.results ?? []).length;
  setChildren(
    dom.runActions,
    h("p", { class: "run-actions-summary", text: `${plural(variants, "variant")} · ${review.approved} approved · ${review.rejected} rejected` }),
    h(
      "div",
      { class: "run-actions-buttons" },
      actionLink(links.report, "Open report", "external", { target: "_blank", rel: "noopener" }),
      actionLink(links.meta_csv, "Meta CSV", "table", { download: "" }),
      actionLink(links.bundle, "Download bundle", "download", { download: "" }, true),
    ),
  );
}

function tickElapsed() {
  const now = Date.now() / 1000;
  for (const node of document.querySelectorAll("[data-since]")) {
    node.textContent = formatSeconds(Math.max(0, now - Number(node.dataset.since)));
  }
}

async function refreshWorkspace() {
  dom.refreshButton.classList.add("is-spinning");
  try {
    await Promise.all([refreshRuns(), loadRun(), loadHealth()]);
  } finally {
    dom.refreshButton.classList.remove("is-spinning");
  }
}

function bindWorkspace() {
  dom.runSelect.addEventListener("change", () => selectRun(dom.runSelect.value));
  dom.refreshButton.addEventListener("click", refreshWorkspace);
  dom.variantSort.addEventListener("change", () => {
    state.sort = dom.variantSort.value;
    state.gridKey = "";
    if (state.run) {
      renderVariants(state.run);
    }
  });
  window.addEventListener("hashchange", () => {
    const runId = readHashRun();
    if (runId && runId !== state.runId) {
      selectRun(runId);
    }
  });
}

async function init() {
  bindDom();
  bindComposer();
  bindWorkspace();
  updateScriptStats();
  updateComposerSummary();
  window.setInterval(tickElapsed, 1000);
  await Promise.all([loadConfig(), loadHealth(), refreshRuns()]);
  const initial = readHashRun() ?? state.runs[0]?.run_id;
  if (initial) {
    await selectRun(initial);
  } else {
    showEmptyState();
  }
}

init();
