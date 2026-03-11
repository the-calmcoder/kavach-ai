/**
 * VoiceGuard AI — Frontend Application
 * File upload, API integration, localStorage history, scroll animations
 */

const API_BASE_URL = window.location.origin;
const HISTORY_KEY = "voiceguard_history";

// DOM
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileSelected = document.getElementById("fileSelected");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const fileRemove = document.getElementById("fileRemove");
const audioPlayer = document.getElementById("audioPlayer");
const analyzeBtn = document.getElementById("analyzeBtn");

const loadingOverlay = document.getElementById("loadingOverlay");
const loadingStep = document.getElementById("loadingStep");
const progressFill = document.getElementById("progressFill");

const resultsOverlay = document.getElementById("resultsOverlay");
const resultsClose = document.getElementById("resultsClose");
const resultsNewBtn = document.getElementById("resultsNewBtn");

const historyContainer = document.getElementById("historyContainer");
const historyEmpty = document.getElementById("historyEmpty");
const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

const navMenuBtn = document.getElementById("navMenuBtn");
const navMobileOverlay = document.getElementById("navMobileOverlay");

let selectedFile = null;

// Mobile Nav
navMenuBtn.addEventListener("click", () => {
    navMobileOverlay.classList.toggle("open");
});
document.querySelectorAll(".mobile-link").forEach(link => {
    link.addEventListener("click", () => navMobileOverlay.classList.remove("open"));
});

// Nav Active State
const navLinks = document.querySelectorAll(".nav-link");
const sections = document.querySelectorAll("section[id]");
window.addEventListener("scroll", () => {
    let current = "";
    sections.forEach(sec => {
        const top = sec.offsetTop - 100;
        if (window.scrollY >= top) current = sec.id;
    });
    navLinks.forEach(link => {
        link.classList.remove("active");
        if (link.getAttribute("href") === "#" + current) link.classList.add("active");
    });
});

// Scroll Reveal
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1, rootMargin: "0px 0px -50px 0px" });
document.querySelectorAll(".section").forEach(sec => observer.observe(sec));

// File Upload
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileRemove.addEventListener("click", clearFile);

function handleFile(file) {
    const validExts = [".mp3",".wav",".m4a",".ogg",".flac"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!validExts.includes(ext)) { alert("Unsupported format. Please upload MP3, WAV, M4A, OGG, or FLAC."); return; }
    if (file.size > 10 * 1024 * 1024) { alert("File too large. Max 10MB."); return; }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    dropZone.style.display = "none";
    fileSelected.style.display = "block";
    analyzeBtn.disabled = false;
    audioPlayer.src = URL.createObjectURL(file);
}

function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    dropZone.style.display = "block";
    fileSelected.style.display = "none";
    analyzeBtn.disabled = true;
    audioPlayer.src = "";
}

function formatSize(b) {
    if (b < 1024) return b + " B";
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
    return (b / (1024 * 1024)).toFixed(1) + " MB";
}

// Analysis
analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
    if (!selectedFile) return;
    showLoading();
    animateLoading();

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const res = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.message || `Server error: ${res.status}`);
        }

        const data = await res.json();
        stopLoading();
        renderResults(data);
        saveToHistory(data, selectedFile.name);
        showResults();
    } catch (err) {
        stopLoading();
        hideLoading();
        alert("Analysis failed: " + (err.message || "Unknown error"));
    }
}

// Loading
const STEPS = [
    { text: "Preprocessing audio signal...", p: 15 },
    { text: "Extracting voice features with HuBERT...", p: 35 },
    { text: "Classifying voice (AI vs Human)...", p: 50 },
    { text: "Transcribing audio with Whisper...", p: 70 },
    { text: "Detecting language...", p: 80 },
    { text: "Running scam analysis...", p: 90 },
    { text: "Building report...", p: 95 },
];
let loadingInterval = null;

function showLoading() { loadingOverlay.style.display = "flex"; }
function hideLoading() { loadingOverlay.style.display = "none"; }

function animateLoading() {
    let i = 0;
    progressFill.style.width = "5%";
    loadingInterval = setInterval(() => {
        if (i < STEPS.length) {
            loadingStep.textContent = STEPS[i].text;
            progressFill.style.width = STEPS[i].p + "%";
            i++;
        } else clearInterval(loadingInterval);
    }, 2500);
}
function stopLoading() {
    if (loadingInterval) clearInterval(loadingInterval);
    progressFill.style.width = "100%";
}

// Results
function showResults() { hideLoading(); resultsOverlay.style.display = "flex"; }
function hideResults() { resultsOverlay.style.display = "none"; }

resultsClose.addEventListener("click", hideResults);
resultsNewBtn.addEventListener("click", () => { hideResults(); clearFile(); });

function renderResults(d) {
    const vVerdict = d.voice_analysis?.verdict || "UNKNOWN";
    const vConf = d.voice_analysis?.confidence_score || 0;
    const vExpl = d.voice_analysis?.explanation || "";
    const vProb = d.voice_analysis?.ai_probability ?? d.detailed_analysis?.ai_probability_raw ?? 0;

    document.getElementById("voiceVerdict").textContent = vVerdict.replace("_"," ");
    document.getElementById("voiceConf").textContent = `${(vConf*100).toFixed(1)}% confidence`;
    document.getElementById("voiceBar").style.width = `${vConf*100}%`;
    document.getElementById("voiceExpl").textContent = vExpl;
    document.getElementById("aiProbRaw").textContent = vProb.toFixed(4);

    const voiceCard = document.getElementById("voiceResultCard");
    voiceCard.className = "result-card voice-result " + (vVerdict === "HUMAN" ? "res-human" : "res-ai");

    const sVerdict = d.fraud_analysis?.verdict || "UNKNOWN";
    const sConf = d.fraud_analysis?.confidence_score || 0;
    const sReason = d.fraud_analysis?.reason || "";

    document.getElementById("scamVerdict").textContent = sVerdict;
    document.getElementById("scamConf").textContent = `${(sConf*100).toFixed(1)}% confidence`;
    document.getElementById("scamBar").style.width = `${sConf*100}%`;
    document.getElementById("scamExpl").textContent = sReason;

    const scamCard = document.getElementById("scamResultCard");
    scamCard.className = "result-card scam-result " + (sVerdict === "SAFE" ? "res-safe" : sVerdict === "SCAM" ? "res-scam" : "");

    document.getElementById("langVerdict").textContent = d.language_detected || "Unknown";
    const dur = d.detailed_analysis?.audio_duration_seconds || 0;
    document.getElementById("audioDur").textContent = `Duration: ${dur}s`;

    document.getElementById("transcriptBox").textContent = d.transcript || "No transcript available.";

    const pt = d.processing_time || {};
    setText("mPre", fmtMs(pt.preprocessing_ms));
    setText("mVoice", fmtMs(pt.voice_classification_ms));
    setText("mTrans", fmtMs(pt.transcription_ms));
    setText("mFraud", fmtMs(pt.fraud_analysis_ms));
    setText("mTotal", fmtMs(pt.total_ms));
}

function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }
function fmtMs(ms) { if (ms == null) return "—"; return ms < 1000 ? ms + "ms" : (ms/1000).toFixed(1) + "s"; }

// History (localStorage)
function getHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
    catch { return []; }
}

function saveToHistory(data, filename) {
    const history = getHistory();
    history.unshift({
        id: Date.now(),
        filename: filename,
        date: new Date().toLocaleString(),
        voiceVerdict: data.voice_analysis?.verdict || "UNKNOWN",
        scamVerdict: data.fraud_analysis?.verdict || "UNKNOWN",
        language: data.language_detected || "Unknown",
        data: data,
    });
    if (history.length > 20) history.pop();
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    const history = getHistory();
    if (history.length === 0) {
        historyEmpty.style.display = "block";
        historyList.style.display = "none";
        clearHistoryBtn.style.display = "none";
        return;
    }

    historyEmpty.style.display = "none";
    historyList.style.display = "flex";
    clearHistoryBtn.style.display = "inline-block";

    historyList.innerHTML = history.map(h => {
        const voiceClass = h.voiceVerdict === "HUMAN" ? "safe" : "danger";
        const scamClass = h.scamVerdict === "SAFE" ? "safe" : h.scamVerdict === "SCAM" ? "danger" : "neutral";
        return `
            <div class="history-item" data-id="${h.id}">
                <span class="hi-icon">🎵</span>
                <div class="hi-meta">
                    <span class="hi-name">${escapeHtml(h.filename)}</span>
                    <span class="hi-date">${h.date}</span>
                </div>
                <div class="hi-verdicts">
                    <span class="hi-tag ${voiceClass}">${h.voiceVerdict.replace("_"," ")}</span>
                    <span class="hi-tag ${scamClass}">${h.scamVerdict}</span>
                </div>
            </div>
        `;
    }).join("");

    document.querySelectorAll(".history-item").forEach(item => {
        item.addEventListener("click", () => {
            const id = parseInt(item.dataset.id);
            const entry = history.find(h => h.id === id);
            if (entry && entry.data) {
                renderResults(entry.data);
                showResults();
            }
        });
    });
}

clearHistoryBtn.addEventListener("click", () => {
    if (confirm("Clear all analysis history?")) {
        localStorage.removeItem(HISTORY_KEY);
        renderHistory();
    }
});

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// Init
renderHistory();
