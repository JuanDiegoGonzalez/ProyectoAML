const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const classNameP = document.getElementById("class-name");
const confidenceP = document.getElementById("confidence");

// Mostrar previsualización de la imagen
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = e => {
    preview.innerHTML = `<img src="${e.target.result}" alt="preview">`;
    resultDiv.classList.add("hidden");
  };
  reader.readAsDataURL(file);
});

// Manejar envío
form.addEventListener("submit", async e => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return alert("Selecciona una imagen.");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });
    const data = await response.json();

    classNameP.textContent = `Clase predicha: ${data.class_name}`;
    confidenceP.textContent = `Confianza: ${(data.confidence * 100).toFixed(2)} %`;

    // Ordenar probabilidades descendente
    const sortedProbs = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1]);

    // Crear un div con clase para las probabilidades
    let probsHTML = `<div class="probabilities-list">`;
    probsHTML += `<h3>Probabilidades por clase:</h3>`;

    for (const [className, prob] of sortedProbs) {
    const percent = (prob * 100).toFixed(2);
    probsHTML += `<div class="probability-item"><span class="class-name">${className}</span> <span class="prob-value">${percent} %</span></div>`;
    }

    probsHTML += `</div>`;

    // Eliminar previo y añadir nuevo
    const oldProbsDiv = document.querySelector(".probabilities-list");
    if (oldProbsDiv) oldProbsDiv.remove();

    resultDiv.insertAdjacentHTML("beforeend", probsHTML);

    resultDiv.classList.remove("hidden");
  } catch (err) {
    alert("Ocurrió un error al predecir. Revisa la consola.");
    console.error(err);
  }
});
