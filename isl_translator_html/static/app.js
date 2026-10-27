const video = document.getElementById('video');
const canvas = document.getElementById('captureCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cameraState = document.getElementById('cameraState');
const notice = document.getElementById('notice');
const letterBox = document.getElementById('letter');
const confidenceBox = document.getElementById('confidence');
const meterFill = document.getElementById('meterFill');
const predictionMessage = document.getElementById('predictionMessage');
const topPredictions = document.getElementById('topPredictions');
const wordBox = document.getElementById('word');
const sentenceBox = document.getElementById('sentence');
let stream = null;
let timer = null;
let busy = false;
let word = '';
let sentence = '';
let candidate = null;
let candidateSince = 0;
let committedForThisHold = false;
const HOLD_MS = 1200;

function renderWords() {
  wordBox.textContent = word || '—';
  sentenceBox.textContent = sentence || 'Your sentence will appear here.';
}
function setNotice(text) { notice.textContent = text; }
function resetCandidate() { candidate = null; candidateSince = 0; committedForThisHold = false; }

async function predictFrame() {
  if (!stream || busy || video.readyState < 2) return;
  busy = true;
  try {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;
    canvas.width = width;
    canvas.height = height;
    // Keep the original camera orientation for inference; preview is mirrored only for the user.
    ctx.drawImage(video, 0, 0, width, height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.82));
    if (!blob) return;
    const form = new FormData();
    form.append('image', blob, 'frame.jpg');
    const response = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Prediction request failed.');
    if (!data.letter) {
      letterBox.textContent = '—'; confidenceBox.textContent = '0%'; meterFill.style.width = '0%';
      predictionMessage.textContent = data.message || 'No hand detected'; topPredictions.replaceChildren(); resetCandidate();
      setNotice('Move your hand clearly into the camera frame.');
      return;
    }
    const conf = Number(data.confidence || 0);
    const pct = Math.max(0, Math.min(100, conf * 100));
    letterBox.textContent = data.letter;
    confidenceBox.textContent = `${pct.toFixed(1)}%`;
    meterFill.style.width = `${pct}%`;
    predictionMessage.textContent = conf < 0.4 ? 'Low confidence — adjust your hand shape.' : 'Hand detected';
    topPredictions.replaceChildren();
    (data.top || []).forEach(item => {
      const chip = document.createElement('span'); chip.className = 'pred-chip';
      chip.textContent = `${item.label}: ${(Number(item.score) * 100).toFixed(0)}%`; topPredictions.appendChild(chip);
    });
    if (conf < 0.4) { resetCandidate(); setNotice('Prediction uncertain. Try better lighting and a steady sign.'); return; }
    const now = Date.now();
    if (candidate !== data.letter) { candidate = data.letter; candidateSince = now; committedForThisHold = false; }
    const elapsed = now - candidateSince;
    if (!committedForThisHold && elapsed >= HOLD_MS) {
      word += data.letter; committedForThisHold = true; renderWords();
      setNotice(`Added “${data.letter}” to the current word. Change or remove the sign before adding it again.`);
    } else if (!committedForThisHold) {
      setNotice(`Hold “${data.letter}” steady for ${Math.max(0, (HOLD_MS - elapsed) / 1000).toFixed(1)} more seconds.`);
    } else {
      setNotice(`Recognizing “${data.letter}”. Change the sign before adding another letter.`);
    }
  } catch (err) {
    setNotice(`Could not get a prediction: ${err.message}`);
    predictionMessage.textContent = 'Check that the Python server is running.';
  } finally { busy = false; }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setNotice('Camera access is unavailable. Use localhost or HTTPS and a supported browser.'); return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    await video.play();
    document.querySelector('.camera-frame').classList.add('active');
    cameraState.textContent = 'Camera on'; cameraState.classList.add('on');
    startBtn.disabled = true; stopBtn.disabled = false;
    setNotice('Camera started. Hold a sign steadily in the frame.');
    timer = setInterval(predictFrame, 350);
  } catch (err) {
    setNotice(`Could not start camera: ${err.message}. Check browser camera permissions.`);
  }
}
function stopCamera() {
  if (timer) clearInterval(timer); timer = null;
  if (stream) stream.getTracks().forEach(track => track.stop()); stream = null;
  video.srcObject = null; document.querySelector('.camera-frame').classList.remove('active');
  cameraState.textContent = 'Camera off'; cameraState.classList.remove('on');
  startBtn.disabled = false; stopBtn.disabled = true; resetCandidate();
  setNotice('Camera is off.'); predictionMessage.textContent = 'Waiting for camera…';
}
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
document.getElementById('spaceBtn').addEventListener('click', () => { if (word) { sentence = `${sentence} ${word}`.trim(); word = ''; renderWords(); resetCandidate(); } });
document.getElementById('backspaceBtn').addEventListener('click', () => { word = word.slice(0, -1); renderWords(); resetCandidate(); });
document.getElementById('clearBtn').addEventListener('click', () => { word = ''; sentence = ''; renderWords(); resetCandidate(); });
window.addEventListener('beforeunload', stopCamera);
renderWords();
