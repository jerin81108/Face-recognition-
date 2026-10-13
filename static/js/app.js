/**
 * AURA VISION AI - Frontend Logic
 * Real-time face detection, recognition streaming, roster management, and biometric inspector.
 */

(function () {
  'use strict';

  // State Management
  const state = {
    currentTab: 'tab-recognition',
    currentMode: 'webcam', // 'webcam' | 'upload'
    isCameraRunning: false,
    cameraStream: null,
    isProcessingFrame: false,
    enrollCameraStream: null,
    enrollSnapshotData: null,
    enrolledPersons: [],
    lastDetectedNames: new Set(),
    lastAnnounceTime: {},
    fpsHistory: [],
    lastFrameTime: performance.now(),
    currentFps: 0,
    recognitionSettings: {
      threshold: 0.42,
      confidence: 0.65,
      showLandmarks: true,
      voiceAnnounce: true,
    },
    detectionIntervalId: null,
  };

  // DOM Elements
  const elements = {
    // Nav
    navTabs: document.querySelectorAll('.nav-tab'),
    tabPanes: document.querySelectorAll('.tab-pane'),

    // Header Stats
    statEngineName: document.getElementById('stat-engine-name'),
    statRosterCount: document.getElementById('stat-roster-count'),
    statSamplesCount: document.getElementById('stat-samples-count'),

    // Viewport & Mode
    modeBtnWebcam: document.getElementById('mode-btn-webcam'),
    modeBtnUpload: document.getElementById('mode-btn-upload'),
    webcamVideo: document.getElementById('webcam-video'),
    renderCanvas: document.getElementById('render-canvas'),
    scannerLaser: document.getElementById('scanner-laser'),
    idleOverlay: document.getElementById('idle-overlay'),
    dropzoneOverlay: document.getElementById('dropzone-overlay'),
    fileInputUpload: document.getElementById('file-input-upload'),
    latencyTag: document.getElementById('latency-tag'),

    // Controls
    btnToggleCamera: document.getElementById('btn-toggle-camera'),
    txtCameraBtn: document.getElementById('txt-camera-btn'),
    btnStartCameraPrompt: document.getElementById('btn-start-camera-prompt'),
    btnCaptureSnapshot: document.getElementById('btn-capture-snapshot'),
    btnBrowseFile: document.getElementById('btn-browse-file'),
    webcamControlsGroup: document.getElementById('webcam-controls-group'),
    uploadControlsGroup: document.getElementById('upload-controls-group'),
    sampleChipsContainer: document.getElementById('sample-chips-container'),

    // Inspector
    detectedCountBadge: document.getElementById('detected-count-badge'),
    metricVerifiedCount: document.getElementById('metric-verified-count'),
    metricUnknownCount: document.getElementById('metric-unknown-count'),
    metricFps: document.getElementById('metric-fps'),
    facesRosterList: document.getElementById('faces-roster-list'),
    chkSpeechAnnounce: document.getElementById('chk-speech-announce'),
    chkLandmarks: document.getElementById('chk-landmarks'),

    // Roster Tab
    personsGalleryGrid: document.getElementById('persons-gallery-grid'),
    rosterSearchInput: document.getElementById('roster-search-input'),
    btnReloadDatabase: document.getElementById('btn-reload-database'),
    btnRosterAddPerson: document.getElementById('btn-roster-add-person'),

    // Logs Tab
    logsTableBody: document.getElementById('logs-table-body'),
    btnRefreshLogs: document.getElementById('btn-refresh-logs'),
    btnExportCsv: document.getElementById('btn-export-csv'),
    btnClearLogs: document.getElementById('btn-clear-logs'),

    // Enroll Modal
    modalEnroll: document.getElementById('modal-enroll'),
    btnQuickEnroll: document.getElementById('btn-quick-enroll'),
    btnCloseEnrollModal: document.getElementById('btn-close-enroll-modal'),
    btnCancelEnroll: document.getElementById('btn-cancel-enroll'),
    btnSubmitEnroll: document.getElementById('btn-submit-enroll'),
    enrollPersonName: document.getElementById('enroll-person-name'),
    enrollModeUpload: document.getElementById('enroll-mode-upload'),
    enrollModeCamera: document.getElementById('enroll-mode-camera'),
    enrollPaneUpload: document.getElementById('enroll-pane-upload'),
    enrollPaneCamera: document.getElementById('enroll-pane-camera'),
    enrollUploadZone: document.getElementById('enroll-upload-zone'),
    enrollFileInput: document.getElementById('enroll-file-input'),
    enrollImagePreview: document.getElementById('enroll-image-preview'),
    enrollPreviewWrap: document.getElementById('enroll-preview-wrap'),
    enrollUploadPrompt: document.getElementById('enroll-upload-prompt'),
    enrollWebcamVideo: document.getElementById('enroll-webcam-video'),
    enrollWebcamCanvas: document.getElementById('enroll-webcam-canvas'),
    enrollSnapPreviewWrap: document.getElementById('enroll-snap-preview-wrap'),
    enrollSnapPreview: document.getElementById('enroll-snap-preview'),
    btnSnapPhoto: document.getElementById('btn-snap-photo'),
    btnRetakePhoto: document.getElementById('btn-retake-photo'),
    enrollStatusMsg: document.getElementById('enroll-status-msg'),

    // Settings Modal
    modalSettings: document.getElementById('modal-settings'),
    btnOpenSettings: document.getElementById('btn-open-settings'),
    btnCloseSettingsModal: document.getElementById('btn-close-settings-modal'),
    btnCloseSettings: document.getElementById('btn-close-settings'),
    btnSaveSettings: document.getElementById('btn-save-settings'),
    rangeThreshold: document.getElementById('range-threshold'),
    valThreshold: document.getElementById('val-threshold'),
    rangeConfidence: document.getElementById('range-confidence'),
    valConfidence: document.getElementById('val-confidence'),
  };

  // Offscreen canvas for frame capture
  const offscreenCanvas = document.createElement('canvas');
  const offscreenCtx = offscreenCanvas.getContext('2d');

  /* ==========================================================================
     INITIALIZATION
     ========================================================================== */
  async function init() {
    setupTabNavigation();
    setupEventListeners();
    setupSettingsEvents();
    setupEnrollModal();
    await fetchSystemStatus();
    await fetchSampleImages();
    await loadRoster();
  }

  /* ==========================================================================
     NAVIGATION TABS
     ========================================================================== */
  function setupTabNavigation() {
    elements.navTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const targetId = tab.getAttribute('data-tab');
        elements.navTabs.forEach((t) => t.classList.remove('active'));
        elements.tabPanes.forEach((p) => p.classList.remove('active'));

        tab.classList.add('active');
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add('active');
        state.currentTab = targetId;

        if (targetId === 'tab-roster') {
          loadRoster();
        } else if (targetId === 'tab-logs') {
          loadLogs();
        }
      });
    });
  }

  /* ==========================================================================
     API INTEGRATION: SYSTEM STATUS & ROSTER
     ========================================================================== */
  async function fetchSystemStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.status === 'online') {
        elements.statRosterCount.textContent = `${data.persons_count} Persons`;
        elements.statSamplesCount.textContent = `${data.samples_count} Photos`;
        state.recognitionSettings.threshold = data.cosine_threshold;
        state.recognitionSettings.confidence = data.detector_confidence;
        elements.rangeThreshold.value = data.cosine_threshold;
        elements.valThreshold.textContent = data.cosine_threshold.toFixed(2);
        elements.rangeConfidence.value = data.detector_confidence;
        elements.valConfidence.textContent = data.detector_confidence.toFixed(2);
      }
    } catch (err) {
      console.warn('Status fetch error:', err);
    }
  }

  async function loadRoster() {
    try {
      const res = await fetch('/api/persons');
      const data = await res.json();
      state.enrolledPersons = data.persons || [];
      renderRoster(state.enrolledPersons);
      elements.statRosterCount.textContent = `${state.enrolledPersons.length} Persons`;
    } catch (err) {
      console.error('Failed to load roster:', err);
    }
  }

  function renderRoster(persons) {
    elements.personsGalleryGrid.innerHTML = '';
    const query = elements.rosterSearchInput.value.toLowerCase().trim();
    const filtered = persons.filter((p) => p.name.toLowerCase().includes(query));

    if (filtered.length === 0) {
      elements.personsGalleryGrid.innerHTML = `
        <div style="grid-column: 1/-1; padding: 40px; text-align: center; color: var(--text-dim);">
          <p>No persons match your search or roster is empty.</p>
        </div>
      `;
      return;
    }

    filtered.forEach((p) => {
      const card = document.createElement('div');
      card.className = 'person-card';
      const thumb = p.thumbnail || '/static/img/avatar_placeholder.png';
      card.innerHTML = `
        <div class="person-card-cover">
          <img src="${thumb}" alt="${escapeHtml(p.name)}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100\\' height=\\'100\\'><rect fill=\\'%23111827\\' width=\\'100\\' height=\\'100\\'/><text fill=\\'%2394a3b8\\' x=\\'50%\\' y=\\'50%\\' text-anchor=\\'middle\\' dy=\\'.3em\\'>No Photo</text></svg>'">
          <div class="person-card-badges">
            <span class="badge-pill">${p.samples_count} Sample${p.samples_count > 1 ? 's' : ''}</span>
          </div>
        </div>
        <div class="person-card-info">
          <h3 class="person-name">${escapeHtml(p.name)}</h3>
          <div class="person-meta">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            <span>Verified Identity</span>
          </div>
          <div class="person-card-actions">
            <button class="btn btn-secondary btn-sm btn-delete-person" data-name="${escapeHtml(p.name)}" title="Delete Person">
              <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
              Delete
            </button>
          </div>
        </div>
      `;

      card.querySelector('.btn-delete-person').addEventListener('click', async (e) => {
        const name = e.currentTarget.getAttribute('data-name');
        if (confirm(`Are you sure you want to delete '${name}' from the face database?`)) {
          await deletePerson(name);
        }
      });

      elements.personsGalleryGrid.appendChild(card);
    });
  }

  async function deletePerson(name) {
    try {
      const res = await fetch(`/api/persons/${encodeURIComponent(name)}`, { method: 'DELETE' });
      const data = await res.json();
      if (data.success) {
        await loadRoster();
        await fetchSystemStatus();
      } else {
        alert(data.error || 'Failed to delete person');
      }
    } catch (err) {
      alert('Network error while deleting person.');
    }
  }

  /* ==========================================================================
     WEBCAM & STREAMING RECOGNITION
     ========================================================================== */
  function setupEventListeners() {
    // Mode switcher
    elements.modeBtnWebcam.addEventListener('click', () => switchMode('webcam'));
    elements.modeBtnUpload.addEventListener('click', () => switchMode('upload'));

    // Camera buttons
    elements.btnToggleCamera.addEventListener('click', toggleCamera);
    elements.btnStartCameraPrompt.addEventListener('click', toggleCamera);

    // Snapshot button
    elements.btnCaptureSnapshot.addEventListener('click', captureInspectionSnapshot);

    // Upload & Dropzone
    elements.btnBrowseFile.addEventListener('click', () => elements.fileInputUpload.click());
    elements.dropzoneOverlay.addEventListener('click', () => elements.fileInputUpload.click());
    elements.fileInputUpload.addEventListener('change', handleFileUpload);

    // Drag and drop
    ['dragenter', 'dragover'].forEach((eventName) => {
      elements.dropzoneOverlay.addEventListener(eventName, (e) => {
        e.preventDefault();
        elements.dropzoneOverlay.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach((eventName) => {
      elements.dropzoneOverlay.addEventListener(eventName, (e) => {
        e.preventDefault();
        elements.dropzoneOverlay.classList.remove('dragover');
      });
    });
    elements.dropzoneOverlay.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleImageFile(e.dataTransfer.files[0]);
      }
    });

    // Inspector toggles
    elements.chkSpeechAnnounce.addEventListener('change', (e) => {
      state.recognitionSettings.voiceAnnounce = e.target.checked;
    });
    elements.chkLandmarks.addEventListener('change', (e) => {
      state.recognitionSettings.showLandmarks = e.target.checked;
    });

    // Roster Search
    elements.rosterSearchInput.addEventListener('input', () => renderRoster(state.enrolledPersons));
    elements.btnReloadDatabase.addEventListener('click', async () => {
      await fetchSystemStatus();
      await loadRoster();
    });

    // Logs Actions
    elements.btnRefreshLogs.addEventListener('click', loadLogs);
    elements.btnExportCsv.addEventListener('click', () => {
      window.location.href = '/api/logs/export';
    });
    elements.btnClearLogs.addEventListener('click', async () => {
      if (confirm('Clear all detection activity logs?')) {
        await fetch('/api/logs/clear', { method: 'POST' });
        loadLogs();
      }
    });
  }

  function switchMode(mode) {
    state.currentMode = mode;
    if (mode === 'webcam') {
      elements.modeBtnWebcam.classList.add('active');
      elements.modeBtnUpload.classList.remove('active');
      elements.webcamControlsGroup.classList.remove('hidden-element');
      elements.uploadControlsGroup.classList.add('hidden-element');
      elements.dropzoneOverlay.style.display = 'none';
      if (!state.isCameraRunning) {
        elements.idleOverlay.style.display = 'flex';
      }
    } else {
      elements.modeBtnUpload.classList.add('active');
      elements.modeBtnWebcam.classList.remove('active');
      elements.webcamControlsGroup.classList.add('hidden-element');
      elements.uploadControlsGroup.classList.remove('hidden-element');
      stopCamera();
      elements.idleOverlay.style.display = 'none';
      elements.dropzoneOverlay.style.display = 'flex';
    }
  }

  async function toggleCamera() {
    if (state.isCameraRunning) {
      stopCamera();
    } else {
      await startCamera();
    }
  }

  async function startCamera() {
    try {
      const constraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      state.cameraStream = stream;
      elements.webcamVideo.srcObject = stream;
        elements.webcamVideo.classList.remove('hidden-element');
      await elements.webcamVideo.play();

      state.isCameraRunning = true;
      elements.idleOverlay.style.display = 'none';
      elements.scannerLaser.classList.add('active');
      elements.txtCameraBtn.textContent = 'Stop Camera';
      elements.btnToggleCamera.classList.remove('btn-emerald');
      elements.btnToggleCamera.classList.add('btn-coral');
      elements.btnCaptureSnapshot.disabled = false;

      // Start recognition frame loop
      startStreamRecognitionLoop();
    } catch (err) {
      console.error('Camera access failed:', err);
      alert('Unable to access camera. Please allow camera permissions in your browser or use the Upload mode.');
    }
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((track) => track.stop());
      state.cameraStream = null;
    }
    state.isCameraRunning = false;
    elements.scannerLaser.classList.remove('active');
    elements.txtCameraBtn.textContent = 'Start Camera';
    elements.btnToggleCamera.classList.remove('btn-coral');
    elements.btnToggleCamera.classList.add('btn-emerald');
    elements.btnCaptureSnapshot.disabled = true;

    if (state.currentMode === 'webcam') {
      elements.idleOverlay.style.display = 'flex';
    }

    if (state.detectionIntervalId) {
      clearInterval(state.detectionIntervalId);
      state.detectionIntervalId = null;
    }
  }

  function startStreamRecognitionLoop() {
    // Process frames at ~6 FPS for smooth real-time recognition without CPU saturation
    if (state.detectionIntervalId) clearInterval(state.detectionIntervalId);
    state.detectionIntervalId = setInterval(captureAndSendFrame, 180);
  }

  async function captureAndSendFrame() {
    if (!state.isCameraRunning || state.isProcessingFrame) return;
    if (elements.webcamVideo.videoWidth === 0) return;

    state.isProcessingFrame = true;

    try {
      const video = elements.webcamVideo;
      const vWidth = video.videoWidth;
      const vHeight = video.videoHeight;

      offscreenCanvas.width = vWidth;
      offscreenCanvas.height = vHeight;
      offscreenCtx.drawImage(video, 0, 0, vWidth, vHeight);

      const frameDataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.8);

      const payload = {
        image: frameDataUrl,
        threshold: state.recognitionSettings.threshold,
        show_landmarks: state.recognitionSettings.showLandmarks,
        log_events: true,
      };

      const startTime = performance.now();
      const res = await fetch('/api/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      const totalLatency = Math.round(performance.now() - startTime);

      if (data.success && state.isCameraRunning) {
        // Render annotated frame to primary canvas
        renderAnnotatedFrame(data.annotated_image);
        updateInspector(data, totalLatency);
        calculateFps();
      }
    } catch (err) {
      console.warn('Frame processing dropped:', err);
    } finally {
      state.isProcessingFrame = false;
    }
  }

  function renderAnnotatedFrame(base64Uri) {
    if (!base64Uri) return;
    const img = new Image();
    img.onload = () => {
      const canvas = elements.renderCanvas;
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
    };
    img.src = base64Uri;
  }

  function calculateFps() {
    const now = performance.now();
    const delta = (now - state.lastFrameTime) / 1000;
    state.lastFrameTime = now;
    if (delta > 0) {
      const fps = Math.round(1 / delta);
      state.fpsHistory.push(fps);
      if (state.fpsHistory.length > 5) state.fpsHistory.shift();
      const avgFps = Math.round(state.fpsHistory.reduce((a, b) => a + b, 0) / state.fpsHistory.length);
      elements.metricFps.textContent = avgFps;
    }
  }

  /* ==========================================================================
     IMAGE UPLOAD & QUICK SAMPLE PICKER
     ========================================================================== */
  async function fetchSampleImages() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      elements.sampleChipsContainer.innerHTML = '';

      if (data.samples && data.samples.length > 0) {
        data.samples.forEach((sample) => {
          const chip = document.createElement('button');
          chip.className = 'sample-chip';
          chip.textContent = sample.title;
          chip.title = `Test with ${sample.filename}`;
          chip.addEventListener('click', () => loadAndProcessSample(sample.url));
          elements.sampleChipsContainer.appendChild(chip);
        });
      }
    } catch (err) {
      console.warn('Samples fetch error:', err);
    }
  }

  async function loadAndProcessSample(sampleUrl) {
    switchMode('upload');
    elements.dropzoneOverlay.style.display = 'none';

    try {
      elements.latencyTag.textContent = 'Processing sample...';
      const imgRes = await fetch(sampleUrl);
      const blob = await imgRes.blob();
      const file = new File([blob], 'sample.jpg', { type: 'image/jpeg' });
      await processImageFile(file);
    } catch (err) {
      console.error('Failed to load sample:', err);
      alert('Error loading sample image.');
    }
  }

  function handleFileUpload(e) {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0]);
    }
  }

  function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file.');
      return;
    }
    switchMode('upload');
    elements.dropzoneOverlay.style.display = 'none';
    processImageFile(file);
  }

  async function processImageFile(file) {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('threshold', state.recognitionSettings.threshold);
    formData.append('show_landmarks', state.recognitionSettings.showLandmarks);
    formData.append('log_events', 'true');

    elements.latencyTag.textContent = 'Analyzing...';
    elements.scannerLaser.classList.add('active');

    const startTime = performance.now();
    try {
      const res = await fetch('/api/recognize', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      const totalLatency = Math.round(performance.now() - startTime);

      elements.scannerLaser.classList.remove('active');

      if (data.success) {
        renderAnnotatedFrame(data.annotated_image);
        updateInspector(data, totalLatency);
      } else {
        alert(data.error || 'Recognition failed');
      }
    } catch (err) {
      elements.scannerLaser.classList.remove('active');
      console.error('Recognition error:', err);
      alert('Error communicating with recognition server.');
    }
  }

  function captureInspectionSnapshot() {
    if (!state.isCameraRunning) return;
    const canvas = elements.renderCanvas;
    const link = document.createElement('a');
    link.download = `AURA_Snapshot_${Date.now()}.jpg`;
    link.href = canvas.toDataURL('image/jpeg', 0.95);
    link.click();
  }

  /* ==========================================================================
     INSPECTOR SIDEBAR & VOICE ANNOUNCEMENT
     ========================================================================== */
  function updateInspector(data, latencyMs) {
    elements.latencyTag.textContent = `Latency: ${data.latency_ms || latencyMs} ms`;
    elements.detectedCountBadge.textContent = `${data.count} Detected`;
    elements.metricVerifiedCount.textContent = data.verified_count;
    elements.metricUnknownCount.textContent = data.unknown_count;

    elements.facesRosterList.innerHTML = '';

    if (!data.faces || data.faces.length === 0) {
      elements.facesRosterList.innerHTML = `
        <div class="empty-roster-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M8 15h8M9 9h.01M15 9h.01"></path>
          </svg>
          <p>No faces currently detected in view.</p>
        </div>
      `;
      return;
    }

    const currentNames = new Set();

    data.faces.forEach((face, idx) => {
      currentNames.add(face.name);
      const isKnown = face.is_known;
      const card = document.createElement('div');
      card.className = `face-item-card ${isKnown ? 'is-known' : 'is-unknown'}`;

      const thumb = face.thumbnail || '';
      const chipClass = isKnown ? 'verified' : 'unknown';
      const fillClass = isKnown ? 'fill-emerald' : 'fill-coral';

      card.innerHTML = `
        <img class="face-item-thumb" src="${thumb}" alt="Face Crop" onerror="this.style.display='none'">
        <div class="face-item-details">
          <div class="face-item-header">
            <span class="face-item-name">${escapeHtml(face.name)}</span>
            <span class="confidence-chip ${chipClass}">${face.confidence.toFixed(1)}%</span>
          </div>
          <div class="confidence-bar-bg">
            <div class="confidence-bar-fill ${fillClass}" style="width: ${Math.min(100, Math.max(5, face.confidence))}%"></div>
          </div>
          <div class="face-item-meta">
            <span>Cosine: ${face.similarity.toFixed(3)}</span>
            <span>Box: [${face.bbox[0]}, ${face.bbox[1]}]</span>
          </div>
          ${!isKnown ? `
            <button class="btn btn-secondary btn-enroll-detected" data-crop="${thumb}">
              + Enroll Identity
            </button>
          ` : ''}
        </div>
      `;

      if (!isKnown) {
        const btnEnroll = card.querySelector('.btn-enroll-detected');
        if (btnEnroll) {
          btnEnroll.addEventListener('click', () => {
            openEnrollModalWithCrop(thumb);
          });
        }
      }

      elements.facesRosterList.appendChild(card);
    });

    // Voice announcement for newly recognized individuals
    if (state.recognitionSettings.voiceAnnounce && 'speechSynthesis' in window) {
      announceRecognizedPersons(data.faces);
    }
  }

  function announceRecognizedPersons(faces) {
    const now = Date.now();
    faces.forEach((face) => {
      if (face.is_known && face.name !== 'Unknown') {
        const lastTime = state.lastAnnounceTime[face.name] || 0;
        // 12 second cooldown per person
        if (now - lastTime > 12000) {
          state.lastAnnounceTime[face.name] = now;
          const utterance = new SpeechSynthesisUtterance(`Recognized ${face.name}`);
          utterance.rate = 1.0;
          utterance.pitch = 1.0;
          window.speechSynthesis.speak(utterance);
        }
      }
    });
  }

  /* ==========================================================================
     ENROLL MODAL (FILE UPLOAD & WEBCAM SNAPSHOT)
     ========================================================================== */
  function setupEnrollModal() {
    elements.btnQuickEnroll.addEventListener('click', openEnrollModal);
    elements.btnRosterAddPerson.addEventListener('click', openEnrollModal);
    elements.btnCloseEnrollModal.addEventListener('click', closeEnrollModal);
    elements.btnCancelEnroll.addEventListener('click', closeEnrollModal);

    // Modal tabs
    elements.enrollModeUpload.addEventListener('click', () => switchEnrollMode('upload'));
    elements.enrollModeCamera.addEventListener('click', () => switchEnrollMode('camera'));

    // Upload box
    elements.enrollUploadZone.addEventListener('click', () => elements.enrollFileInput.click());
    elements.enrollFileInput.addEventListener('change', handleEnrollFileSelect);

    // Camera snap
    elements.btnSnapPhoto.addEventListener('click', takeEnrollSnapshot);
    elements.btnRetakePhoto.addEventListener('click', retakeEnrollSnapshot);

    // Submit
    elements.btnSubmitEnroll.addEventListener('click', submitEnrollment);
  }

  function openEnrollModal() {
    elements.enrollPersonName.value = '';
    elements.enrollStatusMsg.className = 'enroll-status-msg';
    elements.enrollStatusMsg.textContent = '';
    state.enrollSnapshotData = null;
    elements.enrollPreviewWrap.classList.add('hidden-element');
    elements.enrollUploadPrompt.classList.remove('hidden-element');
    elements.modalEnroll.classList.add('active');
    switchEnrollMode('upload');
  }

  function openEnrollModalWithCrop(cropDataUri) {
    openEnrollModal();
    if (cropDataUri) {
      state.enrollSnapshotData = cropDataUri;
      elements.enrollImagePreview.src = cropDataUri;
      elements.enrollPreviewWrap.classList.remove('hidden-element');
      elements.enrollUploadPrompt.classList.add('hidden-element');
    }
  }

  function closeEnrollModal() {
    elements.modalEnroll.classList.remove('active');
    stopEnrollCamera();
  }

  function switchEnrollMode(mode) {
    if (mode === 'upload') {
      elements.enrollModeUpload.classList.add('active');
      elements.enrollModeCamera.classList.remove('active');
      elements.enrollPaneUpload.classList.add('active');
      elements.enrollPaneCamera.classList.remove('active');
      stopEnrollCamera();
    } else {
      elements.enrollModeCamera.classList.add('active');
      elements.enrollModeUpload.classList.remove('active');
      elements.enrollPaneCamera.classList.add('active');
      elements.enrollPaneUpload.classList.remove('active');
      startEnrollCamera();
    }
  }

  async function startEnrollCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 480 } });
      state.enrollCameraStream = stream;
      elements.enrollWebcamVideo.srcObject = stream;
      elements.enrollWebcamVideo.classList.remove('hidden-element');
      elements.enrollSnapPreviewWrap.classList.add('hidden-element');
      elements.btnSnapPhoto.classList.remove('hidden-element');
      elements.btnRetakePhoto.classList.add('hidden-element');
    } catch (err) {
      console.warn('Enroll camera access failed:', err);
      alert('Could not access camera for enrollment snapshot.');
    }
  }

  function stopEnrollCamera() {
    if (state.enrollCameraStream) {
      state.enrollCameraStream.getTracks().forEach((track) => track.stop());
      state.enrollCameraStream = null;
    }
  }

  function takeEnrollSnapshot() {
    const video = elements.enrollWebcamVideo;
    const canvas = elements.enrollWebcamCanvas;
    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    state.enrollSnapshotData = canvas.toDataURL('image/jpeg', 0.9);
    elements.enrollSnapPreview.src = state.enrollSnapshotData;

    elements.enrollWebcamVideo.classList.add('hidden-element');
    elements.enrollSnapPreviewWrap.classList.remove('hidden-element');
    elements.btnSnapPhoto.classList.add('hidden-element');
    elements.btnRetakePhoto.classList.remove('hidden-element');
    stopEnrollCamera();
  }

  function retakeEnrollSnapshot() {
    state.enrollSnapshotData = null;
    startEnrollCamera();
  }

  function handleEnrollFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        state.enrollSnapshotData = evt.target.result;
        elements.enrollImagePreview.src = evt.target.result;
        elements.enrollPreviewWrap.classList.remove('hidden-element');
        elements.enrollUploadPrompt.classList.add('hidden-element');
      };
      reader.readAsDataURL(file);
    }
  }

  async function submitEnrollment() {
    const name = elements.enrollPersonName.value.trim();
    if (!name) {
      showEnrollError('Please enter a name for the identity.');
      return;
    }

    if (!state.enrollSnapshotData) {
      showEnrollError('Please upload an image or snap a photo.');
      return;
    }

    elements.btnSubmitEnroll.disabled = true;
    elements.btnSubmitEnroll.textContent = 'Enrolling...';

    try {
      const payload = {
        name: name,
        image: state.enrollSnapshotData,
      };

      const res = await fetch('/api/persons', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (data.success) {
        showEnrollSuccess(data.message || 'Identity enrolled successfully!');
        await fetchSystemStatus();
        await loadRoster();
        setTimeout(closeEnrollModal, 1200);
      } else {
        showEnrollError(data.error || 'Enrollment failed.');
      }
    } catch (err) {
      showEnrollError('Server communication error during enrollment.');
    } finally {
      elements.btnSubmitEnroll.disabled = false;
      elements.btnSubmitEnroll.textContent = 'Save & Register';
    }
  }

  function showEnrollError(msg) {
    elements.enrollStatusMsg.className = 'enroll-status-msg error';
    elements.enrollStatusMsg.textContent = msg;
  }

  function showEnrollSuccess(msg) {
    elements.enrollStatusMsg.className = 'enroll-status-msg success';
    elements.enrollStatusMsg.textContent = msg;
  }

  /* ==========================================================================
     LOGS & ATTENDANCE TABLE
     ========================================================================== */
  async function loadLogs() {
    try {
      const res = await fetch('/api/logs?limit=100');
      const data = await res.json();
      renderLogsTable(data.logs || []);
    } catch (err) {
      console.warn('Failed to load logs:', err);
    }
  }

  function renderLogsTable(logs) {
    elements.logsTableBody.innerHTML = '';
    if (logs.length === 0) {
      elements.logsTableBody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align: center; color: var(--text-dim); padding: 40px;">
            No detection records logged yet.
          </td>
        </tr>
      `;
      return;
    }

    logs.forEach((log) => {
      const tr = document.createElement('tr');
      const thumb = log.thumbnail || '';
      const isKnown = log.is_known;
      const statusClass = isKnown ? 'verified' : 'unknown';
      const statusText = isKnown ? 'Verified' : 'Unknown';

      tr.innerHTML = `
        <td>
          <img class="log-thumb-img" src="${thumb}" alt="Thumb" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'44\\' height=\\'44\\'><rect fill=\\'%23111827\\' width=\\'44\\' height=\\'44\\'/></svg>'">
        </td>
        <td><strong>${escapeHtml(log.name)}</strong></td>
        <td><span class="confidence-chip ${statusClass}">${statusText}</span></td>
        <td><span style="font-family: var(--font-mono); font-weight: 600;">${log.confidence.toFixed(1)}%</span></td>
        <td><span style="font-family: var(--font-mono); color: var(--text-dim);">${log.similarity.toFixed(3)}</span></td>
        <td><span style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">${log.timestamp}</span></td>
      `;
      elements.logsTableBody.appendChild(tr);
    });
  }

  /* ==========================================================================
     SETTINGS MODAL
     ========================================================================== */
  function setupSettingsEvents() {
    elements.btnOpenSettings.addEventListener('click', () => {
      elements.modalSettings.classList.add('active');
    });
    elements.btnCloseSettingsModal.addEventListener('click', () => {
      elements.modalSettings.classList.remove('active');
    });
    elements.btnCloseSettings.addEventListener('click', () => {
      elements.modalSettings.classList.remove('active');
    });

    elements.rangeThreshold.addEventListener('input', (e) => {
      elements.valThreshold.textContent = parseFloat(e.target.value).toFixed(2);
    });
    elements.rangeConfidence.addEventListener('input', (e) => {
      elements.valConfidence.textContent = parseFloat(e.target.value).toFixed(2);
    });

    elements.btnSaveSettings.addEventListener('click', async () => {
      const thresh = parseFloat(elements.rangeThreshold.value);
      const conf = parseFloat(elements.rangeConfidence.value);

      state.recognitionSettings.threshold = thresh;
      state.recognitionSettings.confidence = conf;

      try {
        await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            cosine_threshold: thresh,
            detector_confidence: conf,
          }),
        });
      } catch (err) {
        console.warn('Settings update error:', err);
      }

      elements.modalSettings.classList.remove('active');
    });
  }

  /* ==========================================================================
     HELPERS
     ========================================================================== */
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Initialize on DOM load
  document.addEventListener('DOMContentLoaded', init);
})();
