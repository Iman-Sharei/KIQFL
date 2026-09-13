(function () {
  const STORAGE_KEY = "kiqfl-checkpoint";

  function detectRunType(checkpoint) {
    if (checkpoint.type) return checkpoint.type;
    const p = checkpoint.params || {};
    if (p.algorithms !== undefined) return "comparison";
    if (p.fl_type !== undefined) return "classification";
    if (p.num_clients !== undefined && p.fl_type === undefined && p.algorithms === undefined) {
      return "regression";
    }
    return "classification";
  }

  function applyCheckpointToForm(checkpoint) {
    const p = checkpoint.params || {};
    const type = detectRunType(checkpoint);

    const simForm = document.getElementById("simulationForm");
    if (simForm && type === "classification") {
      Object.keys(p).forEach((key) => {
        const el = document.getElementById(key) || simForm.querySelector(`[name="${key}"]`);
        if (el) el.value = p[key];
      });
      return true;
    }

    const compareForm = document.getElementById("comparison-form");
    if (compareForm && type === "comparison") {
      Object.keys(p).forEach((key) => {
        if (key === "algorithms") return;
        const el = document.getElementById(key) || compareForm.querySelector(`[name="${key}"]`);
        if (el) el.value = p[key];
      });
      const selected = (p.algorithms || "").split(",").map((s) => s.trim()).filter(Boolean);
      compareForm.querySelectorAll('input[name="algorithms"]').forEach((box) => {
        box.checked = selected.length ? selected.includes(box.value) : box.checked;
      });
      return true;
    }

    return false;
  }

  function applyUrlParamsToForm() {
    const params = new URLSearchParams(window.location.search);
    if (params.get("resume") !== "1") return null;
    const p = {};
    params.forEach((value, key) => {
      if (key !== "resume") p[key] = value;
    });
    if (!Object.keys(p).length) return null;
    const type = p.algorithms !== undefined ? "comparison" : "classification";
    return { type, params: p, stopped: true, completed_epochs: p.completed_epochs || "?" };
  }

  function showResumeBanner(checkpoint) {
    let banner = document.getElementById("resumeBanner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "resumeBanner";
      banner.className = "checkpoint-panel";
      const anchor = document.querySelector(".page-header") || document.querySelector(".form-card");
      if (anchor) anchor.insertAdjacentElement("afterend", banner);
    }
    const done = checkpoint.completed_epochs ?? "?";
    const total = checkpoint.total_epochs ?? checkpoint.params?.global_epochs ?? "?";
    banner.innerHTML = `
      <h3>▶️ Parameters restored</h3>
      <p class="help-text">Loaded saved config (stopped at epoch <strong>${done}</strong> of <strong>${total}</strong>). Review and click Start again.</p>`;
  }

  function showCheckpointPanel(checkpoint) {
    let panel = document.getElementById("checkpointPanel");
    if (!panel) {
      panel = document.createElement("div");
      panel.id = "checkpointPanel";
      panel.className = "checkpoint-panel";
      const anchor = document.querySelector(".btn-group") || document.querySelector(".summary-box");
      if (anchor) anchor.insertAdjacentElement("afterend", panel);
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(checkpoint));
    const json = JSON.stringify(checkpoint, null, 2);
    const resumeUrl = buildResumeUrl(checkpoint);
    const total = checkpoint.total_epochs ?? checkpoint.params?.global_epochs ?? "?";
    panel.innerHTML = `
      <h3>💾 Saved run configuration</h3>
      <p class="help-text">Stopped at epoch <strong>${checkpoint.completed_epochs ?? "?"}</strong> of <strong>${total}</strong>. Copy this config or resume with the same parameters.</p>
      <pre id="checkpointJson">${json}</pre>
      <div class="btn-group">
        <button type="button" class="btn btn-secondary" id="copyCheckpointBtn">📋 Copy JSON</button>
        <a class="btn btn-primary" id="resumeSetupBtn" href="${resumeUrl}">▶️ Resume setup</a>
      </div>`;
    document.getElementById("copyCheckpointBtn").onclick = () => {
      navigator.clipboard.writeText(json).then(() => alert("Checkpoint copied to clipboard."));
    };
    document.getElementById("resumeSetupBtn").addEventListener("click", (e) => {
      localStorage.setItem(STORAGE_KEY, json);
    });
  }

  function buildResumeUrl(checkpoint) {
    const p = checkpoint.params || {};
    const type = detectRunType(checkpoint);
    if (type === "regression") return "/simulate_own?resume=1";
    if (type === "comparison") {
      const q = new URLSearchParams(p);
      q.set("resume", "1");
      return "/compare?" + q.toString();
    }
    const q = new URLSearchParams(p);
    q.set("resume", "1");
    return "/simulate?" + q.toString();
  }

  function tryRestoreFromSaved() {
    const fromUrl = applyUrlParamsToForm();
    const saved = localStorage.getItem(STORAGE_KEY);
    let checkpoint = null;
    try {
      checkpoint = saved ? JSON.parse(saved) : null;
    } catch (_) {
      checkpoint = null;
    }
    if (fromUrl) {
      const merged = checkpoint ? { ...checkpoint, params: { ...checkpoint.params, ...fromUrl.params } } : fromUrl;
      if (applyCheckpointToForm(merged)) showResumeBanner(merged);
      return;
    }
    if (checkpoint && new URLSearchParams(window.location.search).get("resume") === "1") {
      if (applyCheckpointToForm(checkpoint)) showResumeBanner(checkpoint);
    }
  }

  window.KiqflCheckpoint = {
    saveLocal(checkpoint) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(checkpoint));
    },
    show(checkpoint) {
      showCheckpointPanel(checkpoint);
    },
    applyCheckpointToForm,
    stopAndSave(evtSource, runType, params, completedEpochs) {
      const status = document.getElementById("statusPill") || document.getElementById("currentStatus");
      const stopBtn = document.getElementById("stopBtn");
      function markStopping() {
        if (status) status.textContent = "⏹ Stopping… waiting for the current epoch to finish";
        if (stopBtn) {
          stopBtn.disabled = true;
          stopBtn.textContent = "⏹ Stopping…";
        }
      }
      function markStopped() {
        if (status) status.textContent = "⏹ Stopped — configuration saved below";
        if (stopBtn) {
          stopBtn.disabled = true;
          stopBtn.textContent = "⏹ Stopped";
        }
      }

      markStopping();

      const localCheckpoint = {
        type: runType,
        params: params || {},
        completed_epochs: completedEpochs,
        total_epochs: (params && (params.global_epochs || params.total_epochs)) || "?",
        stopped: true,
      };

      // Ask the server to stop; keep EventSource open so we get complete.stopped
      // after the in-flight epoch (closing early made the UI lie about being stopped).
      return fetch("/stop_simulation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_type: runType,
          params,
          completed_epochs: completedEpochs,
        }),
      })
        .then((r) => {
          if (!r.ok) throw new Error("stop endpoint returned " + r.status);
          return r.json();
        })
        .then((checkpoint) => {
          if (!checkpoint.type && runType) checkpoint.type = runType;
          if (!checkpoint.params) checkpoint.params = params;
          if (completedEpochs != null) checkpoint.completed_epochs = completedEpochs;
          // Save config now; final "Stopped" label comes when SSE sends complete/stopped,
          // or after a short grace if the stream is already closed.
          showCheckpointPanel(checkpoint);
          window.__kiqflStopPending = true;
          window.__kiqflMarkStopped = markStopped;
          if (!evtSource || evtSource.readyState === 2) {
            markStopped();
            window.__kiqflStopPending = false;
          }
          return checkpoint;
        })
        .catch(() => {
          showCheckpointPanel(localCheckpoint);
          window.__kiqflStopPending = true;
          window.__kiqflMarkStopped = markStopped;
          if (status) {
            status.textContent =
              "⏹ Stop requested locally — waiting for current epoch (or stream already closed)";
          }
          if (!evtSource || evtSource.readyState === 2) {
            markStopped();
            window.__kiqflStopPending = false;
          }
          return localCheckpoint;
        });
    },
  };

  document.addEventListener("DOMContentLoaded", tryRestoreFromSaved);
})();
