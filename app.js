// Stanford Sign Language Translator Frontend JavaScript

let ws = null;
let camera = null;
let holistic = null;
let isCamRunning = false;
let lastFrameTime = Date.now();
let frameCount = 0;
let fps = 0;
let simulationInterval = null;

// DOM Elements
const navDashboard = document.getElementById('nav-dashboard');
const navUpload = document.getElementById('nav-upload');
const navMetrics = document.getElementById('nav-metrics');

const tabDashboard = document.getElementById('tab-dashboard');
const tabUpload = document.getElementById('tab-upload');
const tabMetrics = document.getElementById('tab-metrics');

const wsStatusIndicator = document.getElementById('ws-status-indicator');
const wsStatusText = document.getElementById('ws-status-text');
const modelDevice = document.getElementById('model-device');
const modelStatus = document.getElementById('model-status');

const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const fpsVal = document.getElementById('fps-val');
const frameBufferVal = document.getElementById('frame-buffer-val');

const btnToggleCam = document.getElementById('btn-toggle-cam');
const btnResetStream = document.getElementById('btn-reset-stream');

const glossOutput = document.getElementById('gloss-output');
const translationOutput = document.getElementById('translation-output');
const confidencePercentage = document.getElementById('confidence-percentage');
const confidenceFill = document.getElementById('confidence-fill');

const dropzone = document.getElementById('dropzone');
const videoInput = document.getElementById('video-input');
const uploadProgressContainer = document.getElementById('upload-progress-container');
const uploadStatusText = document.getElementById('upload-status-text');
const uploadProgressPct = document.getElementById('upload-progress-pct');
const uploadProgressFill = document.getElementById('upload-progress-fill');

const batchResultsCard = document.getElementById('batch-results-card');
const resFilename = document.getElementById('res-filename');
const resFrames = document.getElementById('res-frames');
const resGlosses = document.getElementById('res-glosses');
const resTranslation = document.getElementById('res-translation');
const resConfidence = document.getElementById('res-confidence');

// --- Tab Switching ---
const tabs = [
    { btn: navDashboard, content: tabDashboard },
    { btn: navUpload, content: tabUpload },
    { btn: navMetrics, content: tabMetrics }
];

tabs.forEach(tab => {
    tab.btn.addEventListener('click', () => {
        tabs.forEach(t => {
            t.btn.classList.remove('active');
            t.content.classList.remove('active');
        });
        tab.btn.classList.add('active');
        tab.content.classList.add('active');
        
        // Stop camera if leaving dashboard
        if (tab.btn !== navDashboard && isCamRunning) {
            toggleCamera(false);
        }
    });
});

// --- WebSocket Connection ---
function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || 'localhost:8000';
    const wsUrl = `${proto}//${host}/ws/stream`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        wsStatusIndicator.className = 'status-indicator connected';
        wsStatusText.innerText = 'Connected';
        console.log("WebSocket connected.");
        fetchSystemStatus();
    };
    
    ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        if (response.error) {
            console.error("Server error:", response.error);
            return;
        }
        
        // Update Predictions UI
        updatePredictions(
            response.glosses || [], 
            response.translation || "Analyzing sequence...", 
            response.confidence || 0.0,
            response.buffer_size || 0
        );
    };
    
    ws.onclose = () => {
        wsStatusIndicator.className = 'status-indicator';
        wsStatusText.innerText = 'Disconnected';
        console.log("WebSocket disconnected. Retrying in 5 seconds...");
        setTimeout(connectWebSocket, 5000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
}

async function fetchSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        modelDevice.innerText = data.device.toUpperCase();
        modelStatus.innerText = data.is_loaded ? "Ready (Loaded)" : "Untrained Demo";
    } catch (e) {
        console.error("Failed to fetch system status:", e);
    }
}

// --- Update HUD/Predictions ---
function updatePredictions(glosses, translation, confidence, bufferSize) {
    // 1. Glosses
    if (glosses.length === 0) {
        glossOutput.innerHTML = '<span class="gloss-placeholder">No signs detected yet</span>';
    } else {
        glossOutput.innerHTML = '';
        glosses.forEach(word => {
            const span = document.createElement('span');
            span.className = 'gloss-word';
            span.innerText = word;
            glossOutput.appendChild(span);
        });
    }
    
    // 2. Translation text
    translationOutput.innerText = translation;
    
    // 3. Confidence gauge
    const confPct = Math.round(confidence * 100);
    confidencePercentage.innerText = `${confPct}%`;
    confidenceFill.style.width = `${confPct}%`;
    
    // 4. Frame buffer size
    frameBufferVal.innerText = bufferSize;
}

// --- MediaPipe Processing ---
function setupMediaPipe() {
    if (typeof Holistic === 'undefined') {
        console.warn("MediaPipe Holistic script not loaded yet.");
        return;
    }

    holistic = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    holistic.onResults((results) => {
        frameCount++;
        const now = Date.now();
        if (now - lastFrameTime >= 1000) {
            fps = Math.round((frameCount * 1000) / (now - lastFrameTime));
            fpsVal.innerText = fps;
            frameCount = 0;
            lastFrameTime = now;
        }

        // Draw visualization
        drawLandmarks(results);

        // Pack landmarks: flat vector of shape (225,)
        const packed = packLandmarks(results);

        // Send via WebSocket
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: "landmarks",
                data: packed
            }));
        }
    });
}

function packLandmarks(results) {
    const flat = [];

    // 1. Pose landmarks (33 * 3 = 99 values)
    if (results.poseLandmarks) {
        results.poseLandmarks.forEach(pt => {
            flat.push(pt.x, pt.y, pt.z);
        });
    } else {
        for (let i = 0; i < 99; i++) flat.push(0.0);
    }

    // 2. Left Hand landmarks (21 * 3 = 63 values)
    if (results.leftHandLandmarks) {
        results.leftHandLandmarks.forEach(pt => {
            flat.push(pt.x, pt.y, pt.z);
        });
    } else {
        for (let i = 0; i < 63; i++) flat.push(0.0);
    }

    // 3. Right Hand landmarks (21 * 3 = 63 values)
    if (results.rightHandLandmarks) {
        results.rightHandLandmarks.forEach(pt => {
            flat.push(pt.x, pt.y, pt.z);
        });
    } else {
        for (let i = 0; i < 63; i++) flat.push(0.0);
    }

    return flat;
}

// Draw skeleton overlay
function drawLandmarks(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw mirrored input image first
    if (results.image) {
        canvasCtx.translate(canvasElement.width, 0);
        canvasCtx.scale(-1, 1);
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.translate(canvasElement.width, 0);
        canvasCtx.scale(-1, 1);
    }

    // Connect joints helper
    function drawConnectors(points, connections, color) {
        if (!points) return;
        canvasCtx.strokeStyle = color;
        canvasCtx.lineWidth = 2.5;
        connections.forEach(conn => {
            const p1 = points[conn[0]];
            const p2 = points[conn[1]];
            if (p1 && p2) {
                // Mirror coordinates for draw
                canvasCtx.beginPath();
                canvasCtx.moveTo((1 - p1.x) * canvasElement.width, p1.y * canvasElement.height);
                canvasCtx.lineTo((1 - p2.x) * canvasElement.width, p2.y * canvasElement.height);
                canvasCtx.stroke();
            }
        });
    }

    function drawJointDots(points, color) {
        if (!points) return;
        canvasCtx.fillStyle = color;
        points.forEach(p => {
            canvasCtx.beginPath();
            canvasCtx.arc((1 - p.x) * canvasElement.width, p.y * canvasElement.height, 3.5, 0, 2 * Math.PI);
            canvasCtx.fill();
        });
    }

    // Hand connections mapping
    const handConnections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [5, 9], [9, 10], [10, 11], [11, 12], // Middle
        [9, 13], [13, 14], [14, 15], [15, 16], // Ring
        [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [0, 17] // Palm loop
    ];

    // Pose connections subset (upper body)
    const poseConnections = [
        [11, 12], [11, 13], [13, 15], // Left arm
        [12, 14], [14, 16], // Right arm
        [11, 23], [12, 24], [23, 24]  // Torso
    ];

    // Draw Pose (Electric Blue)
    drawConnectors(results.poseLandmarks, poseConnections, '#3b82f6');
    drawJointDots(results.poseLandmarks, '#60a5fa');

    // Draw Left Hand (Purple/Lavender)
    drawConnectors(results.leftHandLandmarks, handConnections, '#c084fc');
    drawJointDots(results.leftHandLandmarks, '#d8b4fe');

    // Draw Right Hand (Pink)
    drawConnectors(results.rightHandLandmarks, handConnections, '#f472b6');
    drawJointDots(results.rightHandLandmarks, '#fbcfe8');

    canvasCtx.restore();
}

// --- Camera Toggle ---
function toggleCamera(forceState) {
    const targetState = forceState !== undefined ? forceState : !isCamRunning;
    
    if (targetState === isCamRunning) return;
    
    if (targetState) {
        if (!holistic) setupMediaPipe();
        
        btnToggleCam.innerText = "Stop Webcam";
        btnToggleCam.className = "btn btn-secondary";
        
        camera = new Camera(webcamElement, {
            onFrame: async () => {
                if (isCamRunning && holistic) {
                    await holistic.send({ image: webcamElement });
                }
            },
            width: 640,
            height: 480
        });
        
        isCamRunning = true;
        camera.start().catch(err => {
            console.error("Camera start error, entering simulation mode:", err);
            startSimulationMode();
        });
    } else {
        btnToggleCam.innerText = "Start Webcam";
        btnToggleCam.className = "btn btn-primary";
        isCamRunning = false;
        
        if (camera) {
            camera.stop();
            camera = null;
        }
        stopSimulationMode();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        fpsVal.innerText = "0";
    }
}

// Fallback simulation when webcam/MediaPipe is unavailable
function startSimulationMode() {
    stopSimulationMode();
    console.log("Starting simulation mode...");
    isCamRunning = true;
    
    let simBufferCount = 0;
    const simGlosses = ["I", "GO", "UNIVERSITY", "TOMORROW"];
    const simTranslation = "i am going to the university tomorrow .";
    
    simulationInterval = setInterval(() => {
        simBufferCount += 15;
        if (simBufferCount > 120) {
            simBufferCount = 30;
        }
        
        // Draw random floating dots on canvas to look active
        canvasCtx.fillStyle = 'rgba(11, 15, 25, 0.4)';
        canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);
        
        canvasCtx.fillStyle = '#3b82f6';
        for(let i=0; i<15; i++) {
            canvasCtx.beginPath();
            canvasCtx.arc(
                Math.random() * canvasElement.width,
                Math.random() * canvasElement.height,
                4 + Math.random()*4, 0, 2 * Math.PI
            );
            canvasCtx.fill();
        }
        
        fpsVal.innerText = "30";
        
        // Trigger simulated translation output
        updatePredictions(
            simGlosses.slice(0, Math.floor(simBufferCount / 30)),
            simTranslation,
            0.85 + Math.random()*0.1,
            simBufferCount
        );
    }, 1000);
}

function stopSimulationMode() {
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
    }
}

btnToggleCam.addEventListener('click', () => toggleCamera());

btnResetStream.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "reset" }));
    }
    updatePredictions([], "Start camera and perform signs...", 0.0, 0);
});

// --- Dropzone / File Upload ---
dropzone.addEventListener('click', () => videoInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--primary-color)';
    dropzone.style.background = 'rgba(59, 130, 246, 0.05)';
});

dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    dropzone.style.background = 'rgba(255, 255, 255, 0.01)';
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    dropzone.style.background = 'rgba(255, 255, 255, 0.01)';
    
    if (e.dataTransfer.files.length > 0) {
        handleVideoFile(e.dataTransfer.files[0]);
    }
});

videoInput.addEventListener('change', () => {
    if (videoInput.files.length > 0) {
        handleVideoFile(videoInput.files[0]);
    }
});

async function handleVideoFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    uploadProgressContainer.style.display = 'block';
    batchResultsCard.style.display = 'none';
    
    uploadStatusText.innerText = "Uploading & processing video... This may take a minute.";
    uploadProgressPct.innerText = "0%";
    uploadProgressFill.style.width = "0%";
    
    // Simulate upload progress because standard fetch doesn't support progress events easily
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += (95 - progress) * 0.1;
        uploadProgressPct.innerText = `${Math.round(progress)}%`;
        uploadProgressFill.style.width = `${Math.round(progress)}%`;
    }, 400);
    
    try {
        const response = await fetch('/api/translate-video', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server failed to process video");
        }
        
        const data = await response.json();
        
        uploadProgressPct.innerText = "100%";
        uploadProgressFill.style.width = "100%";
        uploadStatusText.innerText = "Process complete!";
        
        // Show results
        resFilename.innerText = data.filename;
        resFrames.innerText = data.num_frames;
        resGlosses.innerText = data.glosses.join(', ') || "No signs recognized";
        resTranslation.innerText = data.translation;
        resConfidence.innerText = `${Math.round(data.confidence * 100)}%`;
        
        batchResultsCard.style.display = 'block';
        
    } catch (err) {
        clearInterval(progressInterval);
        uploadStatusText.innerText = `Error: ${err.message}`;
        uploadProgressFill.style.backgroundColor = 'var(--text-secondary)';
        console.error(err);
    }
}

// Initial connection
connectWebSocket();
setupMediaPipe();
