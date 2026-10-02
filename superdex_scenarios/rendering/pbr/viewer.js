import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

import { HumanPreview } from "./human.js";

const viewport = document.querySelector("#viewport");
const loading = document.querySelector("#loading");
const errorPanel = document.querySelector("#error");
const playButton = document.querySelector("#play");
const speedButton = document.querySelector("#speed");
const cameraButton = document.querySelector("#camera");
const scrubber = document.querySelector("#scrubber");
const timeLabel = document.querySelector("#time");

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
viewport.append(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x252a31);
scene.fog = new THREE.FogExp2(0x252a31, 0.16);

const camera = new THREE.PerspectiveCamera(56, window.innerWidth / window.innerHeight, 0.01, 40);
const fallbackCameraPosition = new THREE.Vector3(1.17, 0.91, 1.29);
const fallbackTarget = new THREE.Vector3(-0.02, 0.39, 0.0);
camera.position.copy(fallbackCameraPosition);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(fallbackTarget);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.55;
controls.maxDistance = 4.0;
controls.maxPolarAngle = Math.PI * 0.49;
controls.update();

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.045).texture;
pmrem.dispose();

const keyLight = new THREE.DirectionalLight(0xfff2df, 4.2);
keyLight.position.set(1.7, 2.8, 1.9);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.left = -1.7;
keyLight.shadow.camera.right = 1.7;
keyLight.shadow.camera.top = 1.7;
keyLight.shadow.camera.bottom = -1.7;
keyLight.shadow.camera.near = 0.2;
keyLight.shadow.camera.far = 6.0;
keyLight.shadow.bias = -0.00008;
keyLight.shadow.normalBias = 0.018;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x9fc2ff, 1.25);
fillLight.position.set(-1.4, 1.4, -1.8);
scene.add(fillLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(4.2, 96),
  new THREE.MeshStandardMaterial({ color: 0x444a52, roughness: 0.78, metalness: 0.02 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.002;
floor.receiveShadow = true;
scene.add(floor);

const backWall = new THREE.Mesh(
  new THREE.PlaneGeometry(8, 4),
  new THREE.MeshStandardMaterial({ color: 0x343a42, roughness: 0.9 }),
);
backWall.position.set(-2.25, 1.5, 0);
backWall.rotation.y = Math.PI / 2;
backWall.receiveShadow = true;
scene.add(backWall);

const gltfLoader = new GLTFLoader();
const humanArmAssets = new Map([
  ["human_upper_arm", "upper_arm"],
  ["human_forearm", "forearm"],
  ["human_shoulder_anchor", "shoulder_anchor"],
]);

function anatomicalArmPath(actor) {
  const asset = humanArmAssets.get(actor.split("/").at(-1));
  return asset
    ? new URL(`../../embodiments/assets/${asset}.glb`, import.meta.url).href
    : null;
}
const axisConversion = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  -Math.PI / 2,
);
const inverseAxisConversion = axisConversion.clone().invert();
const objects = new Map();
const speedOptions = [0.5, 1, 1.5, 2];
let replay;
let humanSkin = null;
let humanPreview = null;
const query = new URLSearchParams(window.location.search);
let exportMode = query.get("export") === "1";
let cameraOrder = [];
let cameraLabels = {};
let selectedCameraName = query.get("camera") ?? "desk_gemini_335";
// A free perspective camera for exports: {position, target} in Three.js space, fov in degrees.
let freeCamera = null;
let playing = !exportMode;
let elapsed = 0;
let previousTime;
let speedIndex = 1;

window.superdexExport = {
  ready: false,
  error: null,
  configure: configureExport,
  metadata: exportMetadata,
  renderAt,
};

function actorNameForLink(robotName, linkName) {
  return `${robotName}/${linkName}`;
}

function simScaleToThree(scale = [1, 1, 1]) {
  return [scale[0], scale[2], scale[1]];
}

function simulationVectorToThree(vector) {
  return new THREE.Vector3(vector[0], vector[2], -vector[1]);
}

function selectedCameraSpec() {
  if (selectedCameraName === "free") return { name: "free", kind: "free", label: "Free" };
  const spec = replay?.cameras?.find((candidate) => candidate.name === selectedCameraName);
  if (!spec) throw new Error(`Unknown camera: ${selectedCameraName}`);
  return spec;
}

function updateCalibratedProjection(width, height) {
  if (!replay?.cameras?.length || selectedCameraName === "free") {
    camera.fov = freeCamera?.fov ?? 45;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    return;
  }
  const intrinsics = selectedCameraSpec().intrinsics;
  const scaleX = width / intrinsics.width_px;
  const scaleY = height / intrinsics.height_px;
  const fx = intrinsics.fx_px * scaleX;
  const fy = intrinsics.fy_px * scaleY;
  const cx = intrinsics.cx_px * scaleX;
  const cy = intrinsics.cy_px * scaleY;
  const near = camera.near;
  const far = camera.far;
  camera.projectionMatrix.set(
    2 * fx / width, 0, 1 - 2 * cx / width, 0,
    0, 2 * fy / height, 2 * cy / height - 1, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, -1, 0,
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

function applyFreeCamera() {
  if (!freeCamera) throw new Error("Free camera pose is not configured.");
  camera.position.copy(freeCamera.position);
  camera.up.set(0, 1, 0);
  camera.lookAt(freeCamera.target);
  controls.target.copy(freeCamera.target);
  controls.update();
  camera.updateMatrixWorld();
  updateCalibratedProjection(renderer.domElement.width, renderer.domElement.height);
}

function applyFixedCamera() {
  const pose = selectedCameraSpec().world_from_camera_cv;
  if (!pose) throw new Error(`Fixed camera ${selectedCameraName} has no world pose.`);
  const position = simulationVectorToThree(pose.position_m);
  const forward = simulationVectorToThree(pose.forward_world).normalize();
  const up = simulationVectorToThree(pose.up_world).normalize();
  const target = position.clone().addScaledVector(forward, 0.6);
  camera.position.copy(position);
  camera.up.copy(up);
  camera.lookAt(target);
  controls.target.copy(target);
  controls.update();
  updateCalibratedProjection(renderer.domElement.width, renderer.domElement.height);
}

function interpolatedActorPose(actorName, frameA, frameB, mix) {
  const actorIndex = replay.actors.indexOf(actorName);
  if (actorIndex < 0) {
    throw new Error(`Replay does not contain camera actor ${actorName}`);
  }
  const a = frameA[actorIndex];
  const b = frameB[actorIndex];
  const position = new THREE.Vector3(a[0], a[2], -a[1]).lerp(
    new THREE.Vector3(b[0], b[2], -b[1]),
    mix,
  );
  const quaternionA = new THREE.Quaternion(a[3], a[4], a[5], a[6]);
  const quaternionB = new THREE.Quaternion(b[3], b[4], b[5], b[6]);
  quaternionA.slerp(quaternionB, mix);
  return { position, simulationQuaternion: quaternionA };
}

function applyActorCamera(frameA, frameB, mix) {
  const spec = selectedCameraSpec();
  const pose = interpolatedActorPose(spec.actor_suffix, frameA, frameB, mix);
  const forwardSimulation = new THREE.Vector3(0, 0, 1)
    .applyQuaternion(pose.simulationQuaternion);
  const upSimulation = new THREE.Vector3(0, -1, 0)
    .applyQuaternion(pose.simulationQuaternion);
  const forward = new THREE.Vector3(
    forwardSimulation.x,
    forwardSimulation.z,
    -forwardSimulation.y,
  );
  const up = new THREE.Vector3(
    upSimulation.x,
    upSimulation.z,
    -upSimulation.y,
  );
  camera.position.copy(pose.position);
  // Camera-site poses use OpenCV RDF: site +Z is optical forward and -Y is
  // image up. Build the view from those rays after converting FLU to Three.js.
  camera.up.copy(up);
  camera.lookAt(pose.position.clone().add(forward));
  camera.updateMatrixWorld();
}

function updateFirstPersonVisibility() {
  // Camera-site actors have no render model. Keep embodiment geometry visible;
  // this is useful for inspecting physical hand contact from the wrist view.
}

function selectCamera(name) {
  if (name !== "free" && !cameraOrder.includes(name)) throw new Error(`Unknown camera: ${name}`);
  selectedCameraName = name;
  const label = name === "free" ? "Free" : cameraLabels[name];
  cameraButton.textContent = label;
  cameraButton.setAttribute("aria-label", `Switch camera; showing ${label}`);
  controls.enabled = !exportMode && name === "desk_gemini_335";
  updateFirstPersonVisibility();
  updateCalibratedProjection(renderer.domElement.width, renderer.domElement.height);
  if (name === "free") {
    applyFreeCamera();
  } else if (selectedCameraSpec().kind === "fixed") {
    applyFixedCamera();
  } else if (replay) {
    applyReplayTime(elapsed);
  }
}

function collectAssetEntries(prefab, manifestUrl) {
  const entries = [];
  const manifestDirectory = new URL(".", new URL(manifestUrl, window.location.href));
  for (const robot of prefab.actors.articulated ?? []) {
    for (const link of robot.links ?? []) {
      if (!link.renderModel) continue;
      const actor = actorNameForLink(robot.name, link.name);
      const armPath = anatomicalArmPath(actor);
      entries.push({
        actor,
        path: armPath
          ?? new URL(link.renderModel.replace(/^\.\//, ""), manifestDirectory).href,
        scale: simScaleToThree(link.renderModelScale),
        color: humanSkin && !armPath && link.name.startsWith("bone_")
          ? humanSkin.base_color_linear : undefined,
        skin: Boolean(humanSkin && link.name.startsWith("bone_")),
      });
    }
  }
  for (const actor of prefab.actors.rigid ?? []) {
    if (!actor.renderModel) continue;
    const override = replay.renderOverrides?.[actor.name] ?? {};
    entries.push({
      actor: actor.name,
      path: anatomicalArmPath(actor.name)
        ?? new URL(actor.renderModel.replace(/^\.\//, ""), manifestDirectory).href,
      scale: simScaleToThree(override.scale ?? actor.renderModelScale),
      color: override.color,
    });
  }
  // The anatomical upper arm includes Blender's shoulder cap. The continuous
  // forearm also replaces the standalone hand's rigid wrist cuff.
  const suffixes = new Set(entries.map((entry) => entry.actor.split("/").at(-1)));
  return entries.filter((entry) => {
    const suffix = entry.actor.split("/").at(-1);
    if (suffix === "human_shoulder_anchor" && suffixes.has("human_upper_arm")) return false;
    if (suffix === "bone_01_wrist_stub" && suffixes.has("human_forearm")) return false;
    return true;
  });
}

async function loadModel(entry) {
  const gltf = await gltfLoader.loadAsync(entry.path);
  const root = new THREE.Group();
  const model = gltf.scene;
  model.scale.fromArray(entry.scale);
  model.traverse((node) => {
    if (!node.isMesh) return;
    node.castShadow = true;
    node.receiveShadow = true;
    if (node.material) {
      const updateMaterial = (source) => {
        const material = source.clone();
        material.envMapIntensity = entry.actor === "gray_bowl" ? 1.35 : 0.9;
        if (entry.color && material.color) material.color.fromArray(entry.color);
        if (entry.skin) {
          material.roughness = humanSkin.roughness;
          material.metalness = 0;
        }
        material.needsUpdate = true;
        return material;
      };
      node.material = Array.isArray(node.material)
        ? node.material.map(updateMaterial)
        : updateMaterial(node.material);
    }
  });
  root.add(model);
  root.matrixAutoUpdate = true;
  scene.add(root);
  objects.set(entry.actor, root);
}

function setActorTransform(object, transform) {
  object.position.set(transform[0], transform[2], -transform[1]);
  const simulationQuaternion = new THREE.Quaternion(
    transform[3], transform[4], transform[5], transform[6],
  );
  object.quaternion
    .copy(axisConversion)
    .multiply(simulationQuaternion)
    .multiply(inverseAxisConversion);
}

function applyReplayTime(seconds) {
  const duration = (replay.frames.length - 1) / replay.fps;
  const framePosition = THREE.MathUtils.clamp(seconds * replay.fps, 0, replay.frames.length - 1);
  const frameAIndex = Math.floor(framePosition);
  const frameBIndex = Math.min(frameAIndex + 1, replay.frames.length - 1);
  const mix = framePosition - frameAIndex;
  const frameA = replay.frames[frameAIndex];
  const frameB = replay.frames[frameBIndex];

  for (let actorIndex = 0; actorIndex < replay.actors.length; actorIndex += 1) {
    const object = objects.get(replay.actors[actorIndex]);
    if (!object) continue;
    const a = frameA[actorIndex];
    const b = frameB[actorIndex];
    const positionA = new THREE.Vector3(a[0], a[2], -a[1]);
    const positionB = new THREE.Vector3(b[0], b[2], -b[1]);
    object.position.lerpVectors(positionA, positionB, mix);

    const quaternionA = new THREE.Quaternion(a[3], a[4], a[5], a[6]);
    const quaternionB = new THREE.Quaternion(b[3], b[4], b[5], b[6]);
    quaternionA.slerp(quaternionB, mix);
    object.quaternion
      .copy(axisConversion)
      .multiply(quaternionA)
      .multiply(inverseAxisConversion);
  }

  humanPreview?.update(frameA, frameB, mix);

  if (selectedCameraSpec().kind === "actor") {
    applyActorCamera(frameA, frameB, mix);
  }

  scrubber.value = duration > 0 ? seconds / duration : 0;
  timeLabel.textContent = `${formatTime(seconds)} / ${formatTime(duration)}`;
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function animate(now) {
  requestAnimationFrame(animate);
  if (previousTime === undefined) previousTime = now;
  const delta = Math.min((now - previousTime) / 1000, 0.1);
  previousTime = now;
  if (replay && playing) {
    const duration = (replay.frames.length - 1) / replay.fps;
    elapsed = (elapsed + delta * speedOptions[speedIndex]) % duration;
    applyReplayTime(elapsed);
  }
  if (controls.enabled) controls.update();
  renderer.render(scene, camera);
}

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Ⅱ" : "▶";
  playButton.setAttribute("aria-label", playing ? "Pause replay" : "Play replay");
});

speedButton.addEventListener("click", () => {
  speedIndex = (speedIndex + 1) % speedOptions.length;
  speedButton.textContent = `${speedOptions[speedIndex]}×`;
});

cameraButton.addEventListener("click", () => {
  const index = cameraOrder.indexOf(selectedCameraName);
  selectCamera(cameraOrder[(index + 1) % cameraOrder.length]);
});

scrubber.addEventListener("input", () => {
  if (!replay) return;
  const duration = (replay.frames.length - 1) / replay.fps;
  elapsed = Number(scrubber.value) * duration;
  applyReplayTime(elapsed);
});

window.addEventListener("resize", () => {
  if (exportMode) return;
  renderer.setSize(window.innerWidth, window.innerHeight);
  updateCalibratedProjection(window.innerWidth, window.innerHeight);
});

function configureExport(width, height, cameraName = "desk_gemini_335", free = null) {
  if (!window.superdexExport.ready) throw new Error("Replay is not ready.");
  exportMode = true;
  playing = false;
  document.body.classList.add("export-mode");
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  if (free) {
    // look_from / look_at are simulation-frame (FLU) metres.
    freeCamera = {
      position: simulationVectorToThree(free.look_from),
      target: simulationVectorToThree(free.look_at),
      fov: free.fov_deg ?? 45,
    };
    cameraName = "free";
  }
  selectCamera(cameraName);
  controls.enabled = false;
  applyReplayTime(0);
  renderer.render(scene, camera);
  return exportMetadata();
}

function exportMetadata() {
  if (!replay) return {};
  return {
    replayFps: replay.fps,
    replayFrames: replay.frames.length,
    durationSeconds: (replay.frames.length - 1) / replay.fps,
    cameraName: selectedCameraName,
    scenario: replay.scenario ?? null,
    width: renderer.domElement.width,
    height: renderer.domElement.height,
  };
}

function renderAt(seconds) {
  if (!window.superdexExport.ready) throw new Error("Replay is not ready.");
  playing = false;
  elapsed = THREE.MathUtils.clamp(
    seconds,
    0,
    (replay.frames.length - 1) / replay.fps,
  );
  applyReplayTime(elapsed);
  renderer.render(scene, camera);
  return renderer.domElement.toDataURL("image/png");
}

async function start() {
  try {
    const recordingUrl = query.get("recording") ?? "./episode.json";
    const recordingResponse = await fetch(recordingUrl);
    if (!recordingResponse.ok) throw new Error("No episode recording found. Run runner.py --record-pbr first.");
    replay = await recordingResponse.json();
    if (replay.format !== "superdex-transform-replay-v2") {
      throw new Error(`Unsupported replay format: ${replay.format}`);
    }
    if (!replay.renderManifest) throw new Error("Replay has no render manifest.");
    const prefabResponse = await fetch(replay.renderManifest);
    if (!prefabResponse.ok) throw new Error("Could not load the Studio asset manifest.");
    const prefab = await prefabResponse.json();
    cameraOrder = replay.cameras.map((cameraSpec) => cameraSpec.name);
    cameraLabels = Object.fromEntries(
      replay.cameras.map((cameraSpec) => [cameraSpec.name, cameraSpec.label ?? cameraSpec.name]),
    );
    if (!cameraOrder.includes(selectedCameraName)) {
      throw new Error(`Unknown camera: ${selectedCameraName}`);
    }
    for (const cameraSpec of replay.cameras.filter((candidate) => candidate.kind === "actor")) {
      if (!replay.actors.includes(cameraSpec.actor_suffix)) {
        throw new Error(`Replay does not contain camera actor ${cameraSpec.actor_suffix}`);
      }
    }
    if (replay.actors.some((actor) => actor.split("/").at(-1) === "human_upper_arm")) {
      const appearanceResponse = await fetch(new URL("../human_appearance.json", import.meta.url));
      if (!appearanceResponse.ok) throw new Error("Could not load the human appearance settings.");
      humanSkin = (await appearanceResponse.json()).skin;
    }
    await Promise.all(collectAssetEntries(prefab, replay.renderManifest).map(loadModel));
    humanPreview = await HumanPreview.load(scene, gltfLoader, replay, objects);
    applyReplayTime(0);
    selectCamera(selectedCameraName);
    if (exportMode) document.body.classList.add("export-mode");
    loading.style.display = "none";
    window.superdexExport.ready = true;
  } catch (error) {
    loading.style.display = "none";
    errorPanel.style.display = "grid";
    errorPanel.textContent = `Unable to start the PBR viewer.\n\n${error.message}`;
    window.superdexExport.error = error.message;
    console.error(error);
  }
}

requestAnimationFrame(animate);
start();
