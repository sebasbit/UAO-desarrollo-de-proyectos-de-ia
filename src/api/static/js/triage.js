// Triage UI — lógica de interacción con el endpoint /api/predict
// Responsabilidad única: enviar imagen, manejar respuesta y actualizar la UI.

const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const previewArea = document.getElementById("preview-area");
const previewImg  = document.getElementById("preview-img");
const previewName = document.getElementById("preview-name");
const analyzeBtn  = document.getElementById("analyze-btn");
const loading     = document.getElementById("loading");
const errorBox    = document.getElementById("error-box");
const resultCard  = document.getElementById("result");
const resetBtn    = document.getElementById("reset-btn");

let selectedFile = null;

// File selection
function handleFile(file) {
  if (!file) return;
  selectedFile = file;

  previewImg.src = URL.createObjectURL(file);
  previewName.textContent =
    file.name + " (" + (file.size / 1024).toFixed(1) + " KB)";
  previewArea.style.display = "block";
  analyzeBtn.disabled = false;
  hideResult();
  hideError();
}

fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("dragover")
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

// Analyze
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  loading.style.display = "block";
  hideResult();
  hideError();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.detail || "Error al procesar la imagen.");
      return;
    }

    showResult(data);
  } catch {
    showError(
      "No se pudo conectar con el servidor. Verifique que este corriendo."
    );
  } finally {
    loading.style.display = "none";
    analyzeBtn.disabled = false;
  }
});

// ── Reset ────────────────────────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewArea.style.display = "none";
  previewImg.src = "";
  analyzeBtn.disabled = true;
  hideResult();
  hideError();
});

// ── Helpers ──────────────────────────────────────────────────────────────
function showResult(data) {
  document.getElementById("res-category").textContent = data.category;
  document.getElementById("res-team").textContent     = data.team;

  const pct = Math.round(data.score * 100);
  document.getElementById("res-score").textContent = pct + "%";

  const fill = document.getElementById("score-bar-fill");
  fill.style.width = pct + "%";
  fill.className =
    "bar-fill" + (pct < 40 ? " very-low" : pct < 65 ? " low" : "");

  const ts = new Date(data.timestamp);
  document.getElementById("res-timestamp").textContent =
    ts.toLocaleDateString("es-CO") + " " + ts.toLocaleTimeString("es-CO");

  const alertHuman = document.getElementById("alert-human");
  alertHuman.style.display = data.human_review_required ? "block" : "none";

  resultCard.style.display = "block";
}

function hideResult() {
  resultCard.style.display = "none";
}

function showError(msg) {
  errorBox.textContent = "Error: " + msg;
  errorBox.style.display = "block";
}

function hideError() {
  errorBox.style.display = "none";
}
