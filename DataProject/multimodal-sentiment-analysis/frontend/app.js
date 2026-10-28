// Multimodal Sentiment Analysis - Frontend Logic

(function () {
  "use strict";

  const form = document.getElementById("analyzeForm");
  const textInput = document.getElementById("textInput");
  const charCount = document.getElementById("charCount");
  const textError = document.getElementById("textError");
  const imageInput = document.getElementById("imageInput");
  const dropZone = document.getElementById("dropZone");
  const dropContent = document.getElementById("dropContent");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const removeBtn = document.getElementById("removeImage");
  const imageError = document.getElementById("imageError");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const btnText = analyzeBtn.querySelector(".btn-text");
  const spinner = document.getElementById("spinner");
  const resultsSection = document.getElementById("resultsSection");

  // Voice controls
  const voiceInput = document.getElementById("voiceInput");
  const voiceActions = document.getElementById("voiceActions");
  const recordBtn = document.getElementById("recordBtn");
  const stopRecBtn = document.getElementById("stopRecBtn");
  const recordingState = document.getElementById("recordingState");
  const recordTimer = document.getElementById("recordTimer");
  const voicePreviewBox = document.getElementById("voicePreviewBox");
  const audioPlayer = document.getElementById("audioPlayer");
  const voiceStatusText = document.getElementById("voiceStatusText");
  const removeVoiceBtn = document.getElementById("removeVoice");
  const voiceError = document.getElementById("voiceError");

  let mediaRecorder = null;
  let audioChunks = [];
  let recordInterval = null;
  let recordStartTime = 0;
  let recordedAudioBlob = null;
  let uploadedAudioFile = null;

  // Character counter
  textInput.addEventListener("input", () => {
    charCount.textContent = textInput.value.length;
  });

  // Image Drop zone: click to open file picker
  dropZone.addEventListener("click", (e) => {
    if (e.target === removeBtn || removeBtn.contains(e.target)) return;
    imageInput.click();
  });

  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      imageInput.click();
    }
  });

  // Image Drag and drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) setImageFile(file);
  });

  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) setImageFile(file);
  });

  function setImageFile(file) {
    imageError.textContent = "";
    if (!["image/jpeg", "image/png"].includes(file.type)) {
      imageError.textContent = "Only JPEG or PNG images are accepted.";
      clearImage();
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropContent.hidden = true;
      previewWrap.hidden = false;
    };
    reader.readAsDataURL(file);
  }

  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearImage();
  });

  function clearImage() {
    imageInput.value = "";
    previewImg.src = "";
    previewWrap.hidden = true;
    dropContent.hidden = false;
  }

  // Voice file upload
  voiceInput.addEventListener("change", () => {
    const file = voiceInput.files[0];
    if (file) setVoiceFile(file);
  });

  function setVoiceFile(file) {
    voiceError.textContent = "";
    recordedAudioBlob = null;
    uploadedAudioFile = file;

    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    voiceStatusText.textContent = file.name + " (" + Math.round(file.size / 1024) + " KB)";
    voiceActions.hidden = true;
    recordingState.hidden = true;
    voicePreviewBox.hidden = false;
  }

  // Voice recording with MediaRecorder
  recordBtn.addEventListener("click", async () => {
    voiceError.textContent = "";
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      voiceError.textContent = "Microphone recording is not supported in this browser.";
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];

      // Determine supported mimeType
      let mimeType = "audio/webm";
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        mimeType = "audio/webm;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
        mimeType = "audio/ogg;codecs=opus";
      }

      mediaRecorder = new MediaRecorder(stream, { mimeType });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        clearInterval(recordInterval);

        const blob = new Blob(audioChunks, { type: mimeType });
        recordedAudioBlob = blob;
        uploadedAudioFile = null;

        const audioUrl = URL.createObjectURL(blob);
        audioPlayer.src = audioUrl;
        voiceStatusText.textContent = "Recorded voice clip (" + Math.round(blob.size / 1024) + " KB)";

        recordingState.hidden = true;
        voiceActions.hidden = true;
        voicePreviewBox.hidden = false;
      };

      mediaRecorder.start(250);
      recordStartTime = Date.now();
      recordTimer.textContent = "00:00";
      recordInterval = setInterval(updateTimer, 1000);

      voiceActions.hidden = true;
      recordingState.hidden = false;
    } catch (err) {
      voiceError.textContent = "Could not access microphone: " + err.message;
    }
  });

  function updateTimer() {
    const elapsedSec = Math.floor((Date.now() - recordStartTime) / 1000);
    const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
    const secs = String(elapsedSec % 60).padStart(2, "0");
    recordTimer.textContent = mins + ":" + secs;
  }

  stopRecBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  });

  removeVoiceBtn.addEventListener("click", () => {
    clearVoice();
  });

  function clearVoice() {
    uploadedAudioFile = null;
    recordedAudioBlob = null;
    voiceInput.value = "";
    audioPlayer.src = "";
    voicePreviewBox.hidden = true;
    recordingState.hidden = true;
    voiceActions.hidden = false;
    voiceError.textContent = "";
  }

  // Form submit
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    textError.textContent = "";
    imageError.textContent = "";
    voiceError.textContent = "";

    const text = textInput.value.trim();
    const hasVoice = Boolean(uploadedAudioFile || recordedAudioBlob);

    // If neither text nor voice is provided
    if (!text && !hasVoice) {
      textError.textContent = "Please enter text or record/upload a voice clip to analyze.";
      textInput.focus();
      return;
    }

    if (text) {
      if (text.length < 3) {
        textError.textContent = "Text must be at least 3 characters.";
        textInput.focus();
        return;
      }
      if (text.length > 500) {
        textError.textContent = "Text must be at most 500 characters.";
        textInput.focus();
        return;
      }
    }

    setLoading(true);

    const formData = new FormData();
    if (text) {
      formData.append("text", text);
    }
    if (imageInput.files[0]) {
      formData.append("image", imageInput.files[0]);
    }

    if (uploadedAudioFile) {
      formData.append("voice", uploadedAudioFile);
    } else if (recordedAudioBlob) {
      const ext = recordedAudioBlob.type.includes("ogg") ? "ogg" : "webm";
      formData.append("voice", recordedAudioBlob, "voice_recording." + ext);
    }

    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        let msg = "Analysis failed. Please try again.";
        try {
          const err = await resp.json();
          if (err.detail) msg = err.detail;
        } catch (_) {}
        showError(msg);
        return;
      }

      const data = await resp.json();
      renderResults(data);
    } catch (err) {
      showError("Could not reach the server. Is it running?");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(active) {
    analyzeBtn.disabled = active;
    spinner.hidden = !active;
    if (active) {
      btnText.textContent = "Analyzing...";
      document.querySelector("main").classList.add("loading");
    } else {
      btnText.textContent = "Analyze";
      document.querySelector("main").classList.remove("loading");
    }
  }

  function showError(msg) {
    textError.textContent = msg;
  }

  const EMOJIS = { positive: "😊", neutral: "😐", negative: "😔" };
  const VERDICTS = { positive: "Good (Positive)", neutral: "Neutral", negative: "Bad (Negative)" };

  function renderResults(data) {
    const overall = data.overall;
    const badge = document.getElementById("overallBadge");
    badge.className = "overall-badge " + overall.label;
    document.getElementById("sentimentEmoji").textContent = EMOJIS[overall.label] || "";
    document.getElementById("overallLabel").textContent =
      VERDICTS[overall.label] || overall.label;
    document.getElementById("overallConf").textContent =
      "Confidence: " + pct(overall.confidence);

    // Text card
    const noTextNote = document.getElementById("noTextNote");
    const textResults = document.getElementById("textResults");
    if (data.text) {
      noTextNote.hidden = true;
      textResults.hidden = false;
      setCardLabel("textCardLabel", data.text.label);
      setConfBar("textConfBar", data.text.label, data.text.confidence);
      renderScores("textScores", data.text.scores);
    } else {
      noTextNote.hidden = false;
      textResults.hidden = true;
    }

    // Image card
    const noImageNote = document.getElementById("noImageNote");
    const imageResults = document.getElementById("imageResults");
    if (data.image) {
      noImageNote.hidden = true;
      imageResults.hidden = false;
      setCardLabel("imageCardLabel", data.image.label);
      setConfBar("imageConfBar", data.image.label, data.image.confidence);
      renderScores("imageScores", data.image.scores);
      document.getElementById("captionNote").textContent =
        'Caption generated: "' + data.image.caption + '"';
    } else {
      noImageNote.hidden = false;
      imageResults.hidden = true;
    }

    // Voice card
    const noVoiceNote = document.getElementById("noVoiceNote");
    const voiceResults = document.getElementById("voiceResults");
    if (data.voice) {
      noVoiceNote.hidden = true;
      voiceResults.hidden = false;
      const voiceVerdict =
        VERDICTS[data.voice.label] || data.voice.sentiment || data.voice.label;
      setCardLabel("voiceCardLabel", data.voice.label, voiceVerdict);
      setConfBar("voiceConfBar", data.voice.label, data.voice.confidence);
      renderScores("voiceScores", data.voice.scores);
      document.getElementById("transcriptionNote").textContent =
        'Transcribed speech: "' + data.voice.transcription + '"';
    } else {
      noVoiceNote.hidden = false;
      voiceResults.hidden = true;
    }

    // Fusion note
    const fw = data.fusion_weights;
    const parts = [];
    if (fw.text) parts.push("Text " + Math.round(fw.text * 100) + "%");
    if (fw.image) parts.push("Image " + Math.round(fw.image * 100) + "%");
    if (fw.voice) parts.push("Voice " + Math.round(fw.voice * 100) + "%");

    document.getElementById("fusionNote").textContent =
      (parts.length ? "Fusion weights: " + parts.join(" / ") + " | " : "") +
      "Modalities used: " +
      data.modalities_used.join(", ");

    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function setCardLabel(id, label, customText) {
    const el = document.getElementById(id);
    el.textContent = customText || label;
    el.className = "card-label " + label;
  }

  function setConfBar(id, label, conf) {
    const bar = document.getElementById(id);
    bar.style.width = pct(conf);
    bar.className = "conf-bar " + label;
  }

  function renderScores(containerId, scores) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    const order = ["positive", "neutral", "negative"];
    order.forEach((key) => {
      const val = scores[key] !== undefined ? scores[key] : 0;
      const row = document.createElement("div");
      row.className = "score-row";
      row.innerHTML =
        '<span class="score-name">' +
        key +
        '</span><span class="score-mini-bar-wrap"><span class="score-mini-bar ' +
        key +
        '" style="width:' +
        pct(val) +
        '"></span></span><span class="score-value">' +
        pct(val) +
        "</span>";
      container.appendChild(row);
    });
  }

  function pct(val) {
    return Math.round(val * 100) + "%";
  }
})();
