const state = {
  token: sessionStorage.getItem("shadow-annotation-token") || "",
  sessionId: sessionStorage.getItem("shadow-annotation-session") || crypto.randomUUID(),
  assignment: null,
  selected: null,
  saving: false,
  history: [],
  imageUrls: [],
  role: null,
};
sessionStorage.setItem("shadow-annotation-session", state.sessionId);

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function request(path, options = {}) {
  const headers = {...(options.headers || {}), Authorization: `Bearer ${state.token}`};
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    const body = await response.json().catch(() => ({error: `HTTP ${response.status}`}));
    throw new Error(body.error || `HTTP ${response.status}`);
  }
  return response;
}

function setSaving(saving, failed = false) {
  state.saving = saving;
  const node = $("#save-state");
  node.textContent = failed ? "● 保存失败" : saving ? "● 正在保存" : "● 已保存";
  node.className = failed ? "failed" : saving ? "saving" : "saved";
}

function clearImages() {
  state.imageUrls.forEach((url) => URL.revokeObjectURL(url));
  state.imageUrls = [];
}

async function loadImage(side) {
  const response = await request(state.assignment[`${side}_media_url`], {
    headers: {"X-Lease-Token": state.assignment.lease_token},
  });
  const url = URL.createObjectURL(await response.blob());
  state.imageUrls.push(url);
  $(`#${side}-image`).src = url;
}

async function updateProgress() {
  const progress = await (await request("/api/progress")).json();
  const completed = state.role === "reviewer" ? progress.adjudicated : progress.answered;
  $("#progress").textContent = `${completed} 已完成 · ${progress.remaining} 待评`;
}

function resetDecision() {
  state.selected = null;
  $$('[data-decision]').forEach((button) => button.classList.remove("selected"));
  $$("#reasons input").forEach((input) => { input.checked = false; });
  $("#reasons").hidden = false;
  $("#submit").disabled = true;
}

async function nextTask() {
  clearImages();
  resetDecision();
  $("#task-error").textContent = "";
  const payload = await (await request("/api/next")).json();
  state.assignment = payload.assignment;
  if (!state.assignment) {
    $("#task").hidden = true;
    $("#complete").hidden = false;
    await updateProgress();
    return;
  }
  $("#complete").hidden = true;
  $("#task").hidden = false;
  $("#prompt").textContent = state.assignment.prompt_text;
  $("#review-reason").hidden = state.role !== "reviewer";
  if (state.role === "reviewer") {
    const labels = {flagged: "标注者标记", disagreement: "答案分歧"};
    $("#review-reason").textContent = `进入复核：${state.assignment.queue_reasons.map((reason) => labels[reason]).join(" + ")}`;
  }
  $("#text-reason").hidden = state.role === "reviewer" || !state.assignment.has_text_target;
  $("#reasons").hidden = state.role === "reviewer";
  await Promise.all([loadImage("left"), loadImage("right"), updateProgress()]);
}

function choose(decision) {
  if (!state.assignment || state.saving) return;
  state.selected = decision;
  $$('[data-decision]').forEach((button) => {
    button.classList.toggle("selected", button.dataset.decision === decision);
  });
  $("#reasons").hidden = decision === "skip";
  if (state.role === "reviewer") $("#reasons").hidden = true;
  $("#submit").disabled = false;
}

async function submit() {
  if (!state.assignment || !state.selected || state.saving) return;
  setSaving(true);
  $("#task-error").textContent = "";
  const assignmentId = state.role === "reviewer"
    ? state.assignment.review_assignment_id
    : state.assignment.assignment_id;
  try {
    const endpoint = state.role === "reviewer" ? "/api/reviews" : "/api/answers";
    const identity = state.role === "reviewer"
      ? {review_assignment_id: assignmentId}
      : {assignment_id: assignmentId};
    await request(endpoint, {
      method: "POST",
      body: JSON.stringify({
        ...identity,
        lease_token: state.assignment.lease_token,
        idempotency_key: crypto.randomUUID(),
        session_id: state.sessionId,
        decision: state.selected,
        ...(state.role === "reviewer" ? {} : {
          reason_codes: $$("#reasons input:checked").map((input) => input.value),
          reason_target: state.selected === "left" || state.selected === "right" ? state.selected : "none",
        }),
        client_version: "web-0.2",
      }),
    });
    state.history.push(assignmentId);
    setSaving(false);
    await nextTask();
  } catch (error) {
    setSaving(false, true);
    $("#task-error").textContent = error.message;
  }
}

async function undo() {
  if (state.saving || state.history.length === 0) return;
  setSaving(true);
  const assignmentId = state.history.pop();
  try {
    const collection = state.role === "reviewer" ? "reviews" : "answers";
    await request(`/api/${collection}/${assignmentId}/undo`, {
      method: "POST",
      body: JSON.stringify({idempotency_key: crypto.randomUUID(), session_id: state.sessionId}),
    });
    setSaving(false);
    await nextTask();
  } catch (error) {
    state.history.push(assignmentId);
    setSaving(false, true);
    $("#task-error").textContent = error.message;
  }
}

async function login() {
  state.token = $("#token").value.trim();
  $("#login-error").textContent = "";
  try {
    const me = await (await request("/api/me")).json();
    if (!["annotator", "reviewer"].includes(me.role)) {
      throw new Error("此页面需要 annotator 或 reviewer 角色");
    }
    state.role = me.role;
    sessionStorage.setItem("shadow-annotation-token", state.token);
    $("#annotator-name").textContent = me.display_name;
    $("#mode-label").textContent = me.role === "reviewer" ? "BLIND ADJUDICATION" : "SHADOW BLIND REVIEW";
    $("#workspace-title").textContent = me.role === "reviewer" ? "复核裁决" : "图像质量盲测";
    $("#login").hidden = true;
    $("#workspace").hidden = false;
    await nextTask();
  } catch (error) {
    $("#login-error").textContent = error.message;
  }
}

$("#login-button").addEventListener("click", login);
$("#token").addEventListener("keydown", (event) => { if (event.key === "Enter") login(); });
$$('[data-decision]').forEach((button) => button.addEventListener("click", () => choose(button.dataset.decision)));
$("#submit").addEventListener("click", submit);
$("#undo").addEventListener("click", undo);
$("#help-button").addEventListener("click", () => $("#help-dialog").showModal());
$("#help-close").addEventListener("click", () => $("#help-dialog").close());
$("#image-close").addEventListener("click", () => $("#image-dialog").close());
$$('.image-card').forEach((card) => card.addEventListener("click", () => {
  $("#large-image").src = $(`#${card.dataset.side}-image`).src;
  $("#image-dialog").showModal();
}));
$("#zoom").addEventListener("input", (event) => {
  $$(".image-card img").forEach((image) => { image.style.width = `${event.target.value}%`; });
});
let syncingScroll = false;
$$('.image-card').forEach((card) => card.addEventListener("scroll", () => {
  if (syncingScroll) return;
  syncingScroll = true;
  const other = $(`.image-card[data-side="${card.dataset.side === "left" ? "right" : "left"}"]`);
  const xRange = Math.max(card.scrollWidth - card.clientWidth, 1);
  const yRange = Math.max(card.scrollHeight - card.clientHeight, 1);
  other.scrollLeft = (card.scrollLeft / xRange) * Math.max(other.scrollWidth - other.clientWidth, 0);
  other.scrollTop = (card.scrollTop / yRange) * Math.max(other.scrollHeight - other.clientHeight, 0);
  requestAnimationFrame(() => { syncingScroll = false; });
}));
document.addEventListener("keydown", (event) => {
  if ($("#workspace").hidden || event.target.matches("input[type=password]")) return;
  if (event.key === "1") choose("left");
  if (event.key === "2") choose("tie");
  if (event.key === "3") choose("right");
  if (event.key === "0") choose("skip");
  if (event.key === "Enter") submit();
  if (event.key === "Backspace") { event.preventDefault(); undo(); }
  if (event.key.toLowerCase() === "h") $("#help-dialog").showModal();
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
    event.preventDefault(); submit();
  }
});
window.addEventListener("beforeunload", (event) => {
  if (state.saving) { event.preventDefault(); event.returnValue = ""; }
});
if (state.token) { $("#token").value = state.token; login(); }
