"use strict";
const $ = (id) => document.getElementById(id);
let job = new URLSearchParams(location.search).get("job") || "",
  state = null,
  objectData = [],
  busy = false,
  lastSlice = "",
  lastObjects = "",
  plot = null;
const number = (value, digits = 2) =>
  value == null ? "—" : Number(value).toFixed(digits);
async function api(path, options) {
  const r = await fetch(path, options);
  const data = await r.json();
  if (!r.ok) throw Error(data.error || r.statusText);
  return data;
}
function fail(error) {
  $("error").textContent = error.message;
}
function viewer(source, id = 0) {
  return `/predicted_train.html?job=${job}&source=${source}&new_object=${id}&focus=${id ? 1 : 0}&width=1`;
}
function selectJob(id) {
  job = id;
  state = null;
  lastSlice = "";
  lastObjects = "";
  $("objectDetail").replaceChildren();
  $("objects").replaceChildren();
  $("objectCount").textContent = "0";
  const url = new URL(location.href);
  url.searchParams.set("job", id);
  history.replaceState(null, "", url);
  $("log").href = `/api/log?job=${job}`;
  $("csv").href = `/api/export?job=${job}&format=csv`;
  $("json").href = `/api/export?job=${job}`;
  tick();
}
async function loadJobs() {
  const jobs = await api("/api/jobs");
  $("jobs").replaceChildren(new Option("Выберите запись", ""));
  for (const j of jobs)
    $("jobs").add(new Option(j.name + " · " + j.status, j.id));
  if (job) $("jobs").value = job;
}
function upload(file) {
  if (!file) return;
  const request = new XMLHttpRequest();
  $("error").textContent = "";
  $("uploadProgress").hidden = false;
  $("state").textContent = "Загружаем файл в Docker…";
  request.open("PUT", "/api/upload");
  request.setRequestHeader("X-Filename", encodeURIComponent(file.name));
  request.upload.onprogress = (e) => {
    if (e.lengthComputable)
      $("uploadProgress").value = (e.loaded / e.total) * 100;
  };
  request.onload = async () => {
    try {
      const data = JSON.parse(request.responseText);
      if (request.status >= 400) throw Error(data.error);
      job = data.id;
      await loadJobs();
      selectJob(job);
    } catch (e) {
      fail(e);
    } finally {
      $("uploadProgress").hidden = true;
    }
  };
  request.onerror = () =>
    fail(Error("Загрузка прервалась. Проверьте соединение с Docker."));
  request.send(file);
}
$("file").onchange = () => upload($("file").files[0]);
$("drop").ondragover = (e) => {
  e.preventDefault();
  $("drop").classList.add("drag");
};
$("drop").ondragleave = () => $("drop").classList.remove("drag");
$("drop").ondrop = (e) => {
  e.preventDefault();
  $("drop").classList.remove("drag");
  upload(e.dataTransfer.files[0]);
};
$("jobs").onchange = () => selectJob($("jobs").value);
$("start").onclick = async () => {
  try {
    $("start").disabled = true;
    await api("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job, topic: Number($("topic").value) }),
    });
    await tick();
  } catch (e) {
    fail(e);
  }
};
const names = {
  READY: "Файл готов к расчёту",
  PREPARING: "Подготавливаем выбранный топик…",
  RUNNING: "Идёт расчёт",
  LIVE_RUNNING: "Принимаем поток ROS 2",
  COMPLETE: "Расчёт завершён",
  FAILED: "Расчёт остановлен с ошибкой",
  INTERRUPTED: "Прогон прерван",
  UPLOADING: "Загрузка",
};
async function tick() {
  if (!job || busy) return;
  busy = true;
  try {
    const expected = job;
    const next = await api(`/api/status?job=${job}`);
    if (expected !== job) return;
    state = next;
    $("state").textContent = names[state.status] || state.status;
    $("start").disabled = state.status !== "READY";
    if (state.error) $("error").textContent = state.error;
    if (!$("topic").options.length || $("topic").dataset.job !== job) {
      $("topic").replaceChildren();
      for (const t of state.topics?.length ? state.topics : state.topic ? [state.topic] : [])
        $("topic").add(new Option(t.count == null ? t.name : `${t.name} · ${t.count} кадров`, t.id));
      $("topic").dataset.job = job;
    }
    $("processed").textContent = state.processed;
    $("progressText").textContent =
      `Получено ${state.received} из ${state.topic?.count || state.topics?.[0]?.count || "—"} кадров. Готово ${state.processed} расчётов.`;
    if (state.frames.length) {
      $("timeline").max = state.frames.length - 1;
      if ($("follow").checked) $("timeline").value = state.frames.length - 1;
      await showSlice();
      $("inspect").disabled = false;
    }
    if ((state.status === "COMPLETE" && lastObjects !== job) || state.status === "LIVE_RUNNING") {
      await loadObjects();
      lastObjects = job;
    }
  } catch (e) {
    fail(e);
  } finally {
    busy = false;
  }
}
async function showSlice(force = false) {
  if (!state?.frames.length) return;
  const source = state.frames[Number($("timeline").value)],
    distance = Number($("distance").value),
    key = `${job}:${source}:${distance}`;
  if (!force && key === lastSlice) return;
  lastSlice = key;
  const data = await api(
    `/api/slice?job=${job}&source=${source}&distance=${distance}`,
  );
  if (key !== lastSlice) return;
  plot = data;
  $("source").textContent = source;
  $("distanceLabel").textContent = number(distance, 1);
  $("empty").hidden = data.available;
  $("empty").textContent = data.status;
  $("compute").textContent =
    data.algorithm_ms == null ? "—" : `${number(data.algorithm_ms, 0)} мс`;
  $("sliceNote").textContent = data.available
    ? `${data.slice_points} точек · попаданий ${data.intrusion_points} · ближайшее ${number(data.nearest_m)} м`
    : "Нет принятого прогноза в этом срезе";
  draw();
}
function draw() {
  const canvas = $("slice"),
    ratio = devicePixelRatio || 1,
    box = canvas.getBoundingClientRect();
  canvas.width = box.width * ratio;
  canvas.height = box.height * ratio;
  const c = canvas.getContext("2d");
  c.scale(ratio, ratio);
  const w = box.width,
    h = box.height,
    scale = Math.min((w - 70) / 7, (h - 65) / 4.4),
    X = (x) => w / 2 + x * scale,
    Y = (y) => h - 48 - y * scale;
  c.strokeStyle = "#203040";
  c.fillStyle = "#72899e";
  c.font = "10px system-ui";
  for (let x = -3; x <= 3; x++) {
    c.beginPath();
    c.moveTo(X(x), 20);
    c.lineTo(X(x), h - 30);
    c.stroke();
    c.fillText(x + " м", X(x) + 4, h - 28);
  }
  for (let y = 0; y <= 3; y++) {
    c.beginPath();
    c.moveTo(30, Y(y));
    c.lineTo(w - 30, Y(y));
    c.stroke();
    c.fillText(y + " м", 32, Y(y) - 4);
  }
  c.strokeStyle = "#80dbc4";
  c.lineWidth = 1.5;
  c.strokeRect(X(-1.05), Y(3), 2.1 * scale, 3 * scale);
  c.setLineDash([4, 5]);
  c.strokeStyle = "#668780";
  c.strokeRect(X(-1.02), Y(2.97), 2.04 * scale, 2.87 * scale);
  c.setLineDash([]);
  if (!plot?.available) return;
  for (const [x, z, kind] of plot.points) {
    c.fillStyle = kind === 3 ? "#ff737f" : kind === 1 ? "#80dbc4" : "#90a7bb";
    c.globalAlpha = kind === 3 ? 1 : 0.65;
    c.fillRect(X(x), Y(z), 2, 2);
  }
  c.globalAlpha = 1;
  for (const [points, color] of [
    [plot.rails, "#80dbc4"],
    [plot.contact, "#eaca7c"],
  ]) {
    c.fillStyle = color;
    for (const [x, z] of points) {
      c.beginPath();
      c.arc(X(x), Y(z), 5, 0, Math.PI * 2);
      c.fill();
    }
  }
}
new ResizeObserver(draw).observe($("slice"));
$("distance").oninput = () => showSlice().catch(fail);
for (const [id, delta] of [
  ["back", -1],
  ["forward", 1],
])
  $(id).onclick = () => {
    $("distance").value = Math.max(0, Number($("distance").value) + delta);
    showSlice().catch(fail);
  };
$("timeline").oninput = () => {
  $("follow").checked = false;
  showSlice().catch(fail);
};
for (const [id, delta] of [
  ["prevFrame", -1],
  ["nextFrame", 1],
])
  $(id).onclick = () => {
    $("follow").checked = false;
    $("timeline").value = Number($("timeline").value) + delta;
    showSlice().catch(fail);
  };
$("inspect").onclick = () =>
  window.open(viewer(state.frames[Number($("timeline").value)]), "_blank");
async function loadObjects() {
  const data = await api(`/api/objects?job=${job}`);
  objectData = data.tracks;
  renderObjects();
}
function renderObjects() {
  const shown = objectData.filter(
    (t) => $("singles").checked || t.status === "CONFIRMED",
  );
  $("objectCount").textContent = shown.length;
  $("objects").replaceChildren();
  for (const [index, t] of shown.entries()) {
    const row = document.createElement("tr");
    for (const value of [
      `${index + 1}`,
      `${t.first_frame}–${t.last_frame}`,
      `${number(t.first_distance_m)} м`,
      `${number(t.width_m)} × ${number(t.height_m)} м`,
      `${number(t.bbox_area_m2, 3)} м²`,
    ]) {
      const td = document.createElement("td");
      td.textContent = value;
      row.append(td);
    }
    const action = document.createElement("td"),
      button = document.createElement("button");
    button.textContent = "Посмотреть";
    button.onclick = () => showObject(t, index + 1);
    action.append(button);
    row.append(action);
    $("objects").append(row);
  }
}
function showObject(t, number) {
  const detail = $("objectDetail");
  detail.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = `Объект №${number} · кадры обнаружения:`;
  detail.append(title, document.createElement("br"));
  for (const source of t.frames) {
    const link = document.createElement("a");
    link.href = viewer(source, t.id);
    link.target = "_blank";
    link.textContent = source;
    detail.append(link);
  }
  window.open(viewer(t.size_frame, t.id), "_blank");
}
$("singles").onchange = renderObjects;
api("/api/health")
  .then(async (health) => {
    if (health.live) {
      $("drop").style.display = "none";
      $("start").hidden = true;
      $("connection").textContent = "ROS 2 · живой поток";
      $("inputHelp").textContent = "Облака поступают из ROS 2. Выберите текущий прогон, чтобы следить за расчётом.";
      $("state").textContent = "Ожидаем поток ROS 2";
    }
    await loadJobs();
    if (health.live && !job) {
      const jobs = await api("/api/jobs");
      job = (jobs.find((j) => j.status === "LIVE_RUNNING") || jobs[0])?.id || "";
      $("jobs").value = job;
    }
    if (job) selectJob(job);
  })
  .catch(fail);
setInterval(tick, 500);
