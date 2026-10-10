
const $ = (id) => document.getElementById(id);

const video = $("camera");
const canvas = $("output");
const ctx = canvas.getContext("2d");

const startButton = $("startCamera");
const stopButton = $("stopCamera");
const calibrateButton = $("calibrateButton");
const beginButton = $("beginExercise");
const resetButton = $("resetExercise");
const retryButton = $("retryModel");

const bodySelect = $("bodyPosition");
const exerciseSelect = $("exercise");

const cameraStatus = $("cameraStatus");
const cameraPlaceholder = $("cameraPlaceholder");
const poseStatus = $("poseStatus");
const mediaPipeStatus = $("mediapipeStatus");

const exerciseName = $("exerciseName");
const exerciseStatus = $("exerciseStatus");
const angleValue = $("angleValue");
const timerValue = $("timerValue");
const repValue = $("repValue");
const roundValue = $("roundValue");
const progressBar = $("progressBar");
const progressText = $("progressText");

const TASKS_VISION_VERSION = "0.10.14";

const VISION_MODULE_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/vision_bundle.mjs`;

const WASM_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const LOAD_TIMEOUT_MS = 45000;
const FRAME_INTERVAL_MS = 120;
const MOVEMENT_THRESHOLD = 5;
const RETURN_THRESHOLD = 3;
const SMOOTHING = 0.25;

const EXERCISES = {
  lying_hold: {
    name: "ท่าที่ 1: นอนยกศีรษะค้าง",
    type: "hold",
    target: 60,
    rounds: 3,
    rest: 60
  },

  lying_reps: {
    name: "ท่าที่ 2: นอนยกและผ่อนศีรษะ",
    type: "reps",
    target: 30,
    rounds: 1
  },

  sitting_hold: {
    name: "ท่าที่ 3: นั่งกดศีรษะต้านบอล",
    type: "hold",
    target: 60,
    rounds: 1,
    rest: 0
  },

  sitting_reps: {
    name: "ท่าที่ 4: นั่งกดบอลและปล่อย",
    type: "reps",
    target: 30,
    rounds: 1
  }
};

const CONNECTIONS = [
  [0, 7], [0, 8], [7, 11], [8, 12],
  [11, 12], [11, 13], [13, 15],
  [12, 14], [14, 16], [11, 23],
  [12, 24], [23, 24], [23, 25],
  [25, 27], [24, 26], [26, 28],
  [27, 29], [29, 31], [28, 30],
  [30, 32], [27, 31], [28, 32]
];

// ---------- STATE ----------
let PoseLandmarker = null;
let FilesetResolver = null;
let landmarker = null;
let cameraStream = null;

let modelLoading = false;
let modelReady = false;
let cameraStarting = false;
let animationRunning = false;
let processing = false;
let lastFrameTime = 0;
let lastVideoTime = -1;

let rawAngle = null;
let smoothedAngle = null;
let baselineAngle = null;
let angleDelta = null;

let active = false;
let phase = "idle";
let repetitions = 0;
let completedRounds = 0;
let movementDetected = false;

let holdStartTime = null;
let holdElapsed = 0;
let restStartTime = null;

function withTimeout(promise, ms, message) {
  let timer;

  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(message)), ms);
    })
  ]).finally(() => clearTimeout(timer));
}

function setStatus(element, text, color = "") {
  element.textContent = text;

  if (color) {
    element.style.color = color;
  } else {
    element.style.removeProperty("color");
  }
}

function currentExercise() {
  return EXERCISES[exerciseSelect.value] || EXERCISES.lying_hold;
}

function currentTarget() {
  if (exerciseSelect.value === "sitting_hold") return 60;
  return currentExercise().target;
}

function formatTime(seconds) {
  const n = Math.max(0, Math.ceil(seconds));
  const min = Math.floor(n / 60);
  const sec = n % 60;

  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function updateExerciseName() {
  const ex = currentExercise();

  exerciseName.textContent = ex.name;
  exerciseStatus.textContent =
    ex.type === "hold"
      ? `เป้าหมายค้าง ${currentTarget()} วินาที`
      : `เป้าหมาย ${ex.target} ครั้ง`;

  updateResults();
}

function updateResults() {
  const ex = currentExercise();
  const target = currentTarget();

  repValue.textContent = repetitions;

  roundValue.textContent =
    ex.type === "hold"
      ? `${completedRounds} / ${ex.rounds}`
      : "—";

  let progress = 0;

  if (ex.type === "hold") {
    progress = phase === "done"
      ? 100
      : ((completedRounds + Math.min(holdElapsed / target, 1)) /
         ex.rounds) * 100;
  } else {
    progress = repetitions / ex.target * 100;
  }

  progress = Math.max(0, Math.min(100, progress));
  progressBar.style.width = `${progress}%`;
  progressText.textContent = `${Math.round(progress)}%`;
}

function updateButtons() {
  const cameraOn = Boolean(cameraStream);

  startButton.disabled = cameraOn || cameraStarting || !modelReady;
  stopButton.disabled = !cameraOn;
  calibrateButton.disabled = !cameraOn || !modelReady || active;

  beginButton.disabled =
    !cameraOn || !modelReady || active || baselineAngle === null;
}

async function loadMediaPipe() {
  if (modelLoading || modelReady) return;

  modelLoading = true;
  retryButton.hidden = true;
  startButton.disabled = true;

  setStatus(mediaPipeStatus, "กำลังโหลดไลบรารี...");
  poseStatus.textContent = "กำลังดาวน์โหลด MediaPipe...";

  try {
    const module = await withTimeout(
      import(VISION_MODULE_URL),
      LOAD_TIMEOUT_MS,
      "ดาวน์โหลดไลบรารีไม่สำเร็จภายใน 45 วินาที"
    );

    PoseLandmarker = module.PoseLandmarker;
    FilesetResolver = module.FilesetResolver;

    if (!PoseLandmarker || !FilesetResolver) {
      throw new Error("ไม่พบส่วนประกอบ Pose Landmarker ในไลบรารี");
    }

    setStatus(mediaPipeStatus, "กำลังโหลดระบบประมวลผล...");
    poseStatus.textContent = "กำลังเตรียม WASM สำหรับตรวจจับร่างกาย...";

    const vision = await withTimeout(
      FilesetResolver.forVisionTasks(WASM_URL),
      LOAD_TIMEOUT_MS,
      "โหลดไฟล์ WASM ไม่สำเร็จภายใน 45 วินาที"
    );

    setStatus(mediaPipeStatus, "กำลังโหลดโมเดล...");
    poseStatus.textContent = "กำลังดาวน์โหลดโมเดลตรวจจับร่างกาย...";

    landmarker = await withTimeout(
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "CPU"
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      }),
      LOAD_TIMEOUT_MS,
      "โหลดโมเดลไม่สำเร็จภายใน 45 วินาที"
    );

    modelReady = true;

    setStatus(mediaPipeStatus, "พร้อมใช้งาน", "#198754");
    poseStatus.textContent = "ระบบพร้อมใช้งาน กดเปิดกล้องเพื่อเริ่มต้น";

    updateButtons();
  } catch (error) {
    console.error("MediaPipe loading error:", error);

    modelReady = false;
    landmarker = null;

    setStatus(mediaPipeStatus, "โหลดไม่สำเร็จ", "#b42318");
    poseStatus.textContent = `โหลด MediaPipe ไม่สำเร็จ: ${error.message}`;

    retryButton.hidden = false;
    startButton.disabled = true;
  } finally {
    modelLoading = false;
  }
}

retryButton.addEventListener("click", () => {
  if (!modelLoading) loadMediaPipe();
});

async function startCamera() {
  if (cameraStarting || cameraStream) return;

  if (!modelReady || !landmarker) {
    poseStatus.textContent = "MediaPipe ยังไม่พร้อม กรุณารอให้โหลดเสร็จก่อน";
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    poseStatus.textContent =
      "เบราว์เซอร์ไม่รองรับกล้อง โปรดใช้ localhost หรือ HTTPS";
    return;
  }

  cameraStarting = true;
  updateButtons();

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 320 },
        height: { ideal: 240 },
        frameRate: { ideal: 15, max: 20 }
      }
    });

    video.srcObject = cameraStream;
    await video.play();

    cameraStatus.textContent = "เปิดกล้องแล้ว";
    cameraPlaceholder.hidden = true;
    poseStatus.textContent = "กำลังตรวจจับร่างกาย...";

    lastVideoTime = -1;
    lastFrameTime = 0;
    animationRunning = true;

    updateButtons();
    requestAnimationFrame(processFrame);
  } catch (error) {
    console.error("Camera error:", error);

    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
    }

    cameraStream = null;
    cameraStatus.textContent = "เปิดกล้องไม่สำเร็จ";

    const messages = {
      NotAllowedError: "กรุณาอนุญาตสิทธิ์การใช้กล้อง",
      NotFoundError: "ไม่พบกล้องที่เชื่อมต่อ",
      NotReadableError: "กล้องอาจกำลังใช้งานโดยโปรแกรมอื่น"
    };

    const message =
      messages[error.name] || error.message || "เปิดกล้องไม่สำเร็จ";

    poseStatus.textContent = message;
  } finally {
    cameraStarting = false;
    updateButtons();
  }
}

function stopCamera() {
  animationRunning = false;
  active = false;
  processing = false;

  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
  }

  cameraStream = null;
  video.pause();
  video.srcObject = null;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  cameraStatus.textContent = "ยังไม่เปิดกล้อง";
  cameraPlaceholder.hidden = false;
  poseStatus.textContent = "กล้องปิดอยู่";

  rawAngle = null;
  smoothedAngle = null;
  angleDelta = null;

  resetExerciseState(false);
  angleValue.textContent = "--°";

  updateButtons();
}

function getVisibility(point) {
  return point?.visibility ?? 0;
}

function getSideLandmarks(landmarks) {
  const leftScore = Math.min(
    getVisibility(landmarks[7]),
    getVisibility(landmarks[11]),
    getVisibility(landmarks[23])
  );

  const rightScore = Math.min(
    getVisibility(landmarks[8]),
    getVisibility(landmarks[12]),
    getVisibility(landmarks[24])
  );

  const left = leftScore >= rightScore;

  const ear = landmarks[left ? 7 : 8];
  const shoulder = landmarks[left ? 11 : 12];
  const hip = landmarks[left ? 23 : 24];

  if (
    !ear || !shoulder || !hip ||
    getVisibility(ear) < 0.35 ||
    getVisibility(shoulder) < 0.35 ||
    getVisibility(hip) < 0.35
  ) {
    return null;
  }

  return { ear, shoulder, hip };
}

function calculateAngle(a, vertex, c) {
  const ax = a.x - vertex.x;
  const ay = a.y - vertex.y;
  const cx = c.x - vertex.x;
  const cy = c.y - vertex.y;

  const lenA = Math.hypot(ax, ay);
  const lenC = Math.hypot(c.x - vertex.x, c.y - vertex.y);

  if (!lenA || !lenC) return null;

  const dot = ax * cx + ay * cy;
  const cosine = Math.max(-1, Math.min(1, dot / (lenA * lenC)));

  return Math.acos(cosine) * 180 / Math.PI;
}

function analyzePose(landmarks) {
  const side = getSideLandmarks(landmarks);
  if (!side) return null;

  return calculateAngle(side.ear, side.shoulder, side.hip);
}

function drawPose(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!landmarks) return;

  function point(index) {
    const p = landmarks[index];

    return {
      x: (1 - p.x) * canvas.width,
      y: p.y * canvas.height,
      visible: getVisibility(p) >= 0.35
    };
  }

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#32d399";
  ctx.fillStyle = "#ffca5c";

  for (const [from, to] of CONNECTIONS) {
    const a = point(from);
    const b = point(to);

    if (!a.visible || !b.visible) continue;

    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (let i = 0; i < landmarks.length; i++) {
    const p = point(i);
    if (!p.visible) continue;

    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function processFrame(timestamp) {
  if (!animationRunning || !cameraStream || !landmarker) return;

  requestAnimationFrame(processFrame);

  if (processing) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  if (video.videoWidth === 0 || video.videoHeight === 0) return;
  if (timestamp - lastFrameTime < FRAME_INTERVAL_MS) return;
  if (video.currentTime === lastVideoTime) return;

  lastFrameTime = timestamp;
  lastVideoTime = video.currentTime;
  processing = true;

  try {
    const width = video.videoWidth;
    const height = video.videoHeight;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const result = landmarker.detectForVideo(video, timestamp);
    const landmarks = result.landmarks?.[0] || null;

    drawPose(landmarks);

    if (!landmarks) {
      rawAngle = null;
      angleValue.textContent = "--°";
      poseStatus.textContent =
        "ไม่พบร่างกาย กรุณาจัดกล้องให้เห็นศีรษะและลำตัวด้านข้าง";
      return;
    }

    rawAngle = analyzePose(landmarks);

    if (rawAngle === null) {
      angleValue.textContent = "--°";
      poseStatus.textContent =
        "ตรวจจับหู ไหล่ หรือสะโพกไม่ครบ กรุณาปรับตำแหน่งกล้อง";
      return;
    }

    smoothedAngle =
      smoothedAngle === null
        ? rawAngle
        : smoothedAngle * (1 - SMOOTHING) + rawAngle * SMOOTHING;

    if (baselineAngle !== null) {
      angleDelta = Math.abs(smoothedAngle - baselineAngle);
      angleValue.textContent = `${angleDelta.toFixed(1)}°`;
    } else {
      angleDelta = null;
      angleValue.textContent = "--°";
    }

    poseStatus.textContent = "ตรวจพบร่างกายแล้ว";

    if (active && angleDelta !== null) {
      if (currentExercise().type === "hold") {
        processHold(timestamp);
      } else {
        processRepetitions();
      }
    }

    updateResults();
  } catch (error) {
    console.error("Pose detection error:", error);
    poseStatus.textContent =
      "เกิดข้อผิดพลาดระหว่างตรวจจับ ดู Console ในเบราว์เซอร์";
  } finally {
    processing = false;
  }
}

function calibratePose() {
  if (!cameraStream) {
    poseStatus.textContent = "กรุณาเปิดกล้องก่อน";
    return;
  }

  if (smoothedAngle === null) {
    poseStatus.textContent =
      "ยังตรวจจับมุมไม่ได้ กรุณาให้เห็นหู ไหล่ และสะโพกชัดเจน";
    return;
  }

  baselineAngle = smoothedAngle;
  resetExerciseState(false);

  exerciseStatus.textContent = "บันทึกท่าเริ่มต้นแล้ว";
  poseStatus.textContent = "ปรับเทียบท่าเริ่มต้นแล้ว พร้อมเริ่มฝึก";

  updateButtons();
  updateResults();
}

function resetExerciseState(clearCalibration = false) {
  active = false;
  phase = "idle";
  repetitions = 0;
  completedRounds = 0;
  movementDetected = false;

  holdStartTime = null;
  holdElapsed = 0;
  restStartTime = null;

  timerValue.textContent = "00:00";

  if (clearCalibration) {
    baselineAngle = null;
    angleDelta = null;
    exerciseStatus.textContent = "กรุณาปรับเทียบท่าเริ่มต้น";
  } else {
    exerciseStatus.textContent = "พร้อมเริ่มฝึก";
  }

  updateResults();
}

function beginExercise() {
  if (!cameraStream || !modelReady) {
    poseStatus.textContent =
      "กรุณาเปิดกล้องและรอให้ MediaPipe พร้อมก่อน";
    return;
  }

  if (baselineAngle === null) {
    poseStatus.textContent = "กรุณาปรับเทียบท่าเริ่มต้นก่อนเริ่มฝึก";
    return;
  }

  resetExerciseState(false);
  active = true;
  phase = "working";

  exerciseStatus.textContent = "กำลังฝึก";
  poseStatus.textContent = "เริ่มฝึกแล้ว ระบบกำลังตรวจจับการเคลื่อนไหว";

  updateButtons();
}

function resetExercise() {
  resetExerciseState(true);
  angleValue.textContent = "--°";

  poseStatus.textContent = "รีเซ็ตแล้ว กรุณาปรับเทียบท่าเริ่มต้นใหม่";
  updateButtons();
}

function processHold(now) {
  const ex = currentExercise();
  const target = currentTarget();

  if (phase === "rest") {
    const elapsed = (now - restStartTime) / 1000;
    const remaining = Math.max(0, ex.rest - elapsed);

    timerValue.textContent = formatTime(remaining);

    if (remaining <= 0) {
      phase = "working";
      restStartTime = null;
      holdStartTime = null;
      holdElapsed = 0;

      poseStatus.textContent = "หมดเวลาพัก ระบบพร้อมจับเวลารอบถัดไป";
    }

    return;
  }

  if (phase !== "working") return;

  if (angleDelta >= MOVEMENT_THRESHOLD) {
    if (holdStartTime === null) {
      holdStartTime = now;
      holdElapsed = 0;
    }

    holdElapsed = (now - holdStartTime) / 1000;
    timerValue.textContent = formatTime(target - holdElapsed);

    exerciseStatus.textContent =
      `กำลังค้างท่า รอบ ${completedRounds + 1} / ${ex.rounds}`;

    if (holdElapsed >= target) {
      completedRounds++;
      holdStartTime = null;
      holdElapsed = 0;
      timerValue.textContent = "00:00";

      if (completedRounds >= ex.rounds) {
        active = false;
        phase = "done";
        exerciseStatus.textContent = "ครบตามเป้าหมาย";
        poseStatus.textContent = "ครบจำนวนรอบตามเป้าหมายแล้ว";
        updateButtons();
      } else {
        phase = "rest";
        restStartTime = now;
        exerciseStatus.textContent = "พักระหว่างรอบ";
        poseStatus.textContent = `พัก ${ex.rest} วินาทีก่อนเริ่มรอบถัดไป`;
      }
    }
  } else {
    if (holdStartTime !== null) {
      holdStartTime = null;
      holdElapsed = 0;
      timerValue.textContent = "00:00";
      poseStatus.textContent =
        "มุมเปลี่ยนกลับใกล้ท่าเริ่มต้น ตัวจับเวลาหยุดแล้ว";
    }
  }
}

function processRepetitions() {
  const ex = currentExercise();

  if (phase !== "working" || angleDelta === null) return;

  if (angleDelta >= MOVEMENT_THRESHOLD && !movementDetected) {
    movementDetected = true;
    exerciseStatus.textContent = "ตรวจพบการเคลื่อนไหว";
  }

  if (movementDetected && angleDelta <= RETURN_THRESHOLD) {
    movementDetected = false;
    repetitions = Math.min(repetitions + 1, ex.target);

    if (repetitions >= ex.target) {
      active = false;
      phase = "done";
      exerciseStatus.textContent = "ครบตามเป้าหมาย";
      poseStatus.textContent = "ครบจำนวนครั้งตามเป้าหมายแล้ว";
      updateButtons();
    } else {
      poseStatus.textContent = `นับได้ ${repetitions} ครั้ง`;
    }

    updateResults();
  }
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
calibrateButton.addEventListener("click", calibratePose);
beginButton.addEventListener("click", beginExercise);
resetButton.addEventListener("click", resetExercise);

exerciseSelect.addEventListener("change", () => {
  bodySelect.value = exerciseSelect.value.startsWith("lying_")
    ? "lying"
    : "sitting";

  resetExerciseState(true);
  updateExerciseName();
  updateButtons();

  poseStatus.textContent =
    "เปลี่ยนท่าฝึกแล้ว กรุณาปรับเทียบท่าเริ่มต้นใหม่";
});

bodySelect.addEventListener("change", () => {
  exerciseSelect.value =
    bodySelect.value === "lying" ? "lying_hold" : "sitting_hold";

  resetExerciseState(true);
  updateExerciseName();
  updateButtons();

  poseStatus.textContent =
    "เปลี่ยนตำแหน่งร่างกายแล้ว กรุณาปรับเทียบท่าเริ่มต้นใหม่";
});

updateExerciseName();
updateButtons();
loadMediaPipe();

window.addEventListener("pagehide", () => {
  animationRunning = false;

  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
  }

  if (landmarker) {
    landmarker.close();
  }
});
