from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from mitosis.config import Settings
from mitosis.models import Box, Check, HeroPack
from mitosis.qa.footage import fit_within, grab_frames, read_image, spread, video_duration, write_jpeg

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "opencv-sface-2021dec"
SFACE_MATCH_COSINE = 0.363
FLICKER_MARGIN = 0.12
MAX_MISSING_SHARE = 0.4
MAX_FACES_COMPARED = 3
DETECT_SIDES = (640, 480, 360, 1024)
DETECT_SCORE = 0.6
DETECT_NMS = 0.3
DETECT_TOP_K = 50
HAAR_MAX_SIDE = 640
REFERENCE_COUNT = 3
REFERENCE_MIN_SIDE = 768
TEMPLATE_TOP_K = 8
SHARPNESS_WIDTH = 256
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
DOWNLOAD_DEADLINE = 240.0
LOAD_RETRY_AFTER = 600.0

HERO_NEGATIVE = ", ".join(
    (
        "different person",
        "different face",
        "face swap",
        "identity drift",
        "changed facial structure",
        "different hairline",
        "different hairstyle",
        "different facial hair",
        "different eye color",
        "different skin tone",
        "different age",
        "different body type",
        "different wardrobe",
        "twin",
        "duplicate person",
        "extra people",
        "distorted face",
        "asymmetric eyes",
        "warped features",
        "melting skin",
        "plastic skin",
        "waxy skin",
        "airbrushed",
        "uncanny",
        "deformed hands",
        "extra fingers",
        "missing fingers",
        "extra limbs",
        "blurry",
        "low resolution",
        "jpeg artifacts",
        "text",
        "watermark",
        "logo",
        "subtitles",
    )
)


class ModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    urls: tuple[str, ...]
    min_bytes: int
    sha256: str


DETECTOR_MODEL = ModelAsset(
    filename="face_detection_yunet_2023mar.onnx",
    urls=(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ),
    min_bytes=200_000,
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)

RECOGNIZER_MODEL = ModelAsset(
    filename="face_recognition_sface_2021dec.onnx",
    urls=(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ),
    min_bytes=30_000_000,
    sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)


@dataclass(frozen=True, eq=False)
class Face:
    row: np.ndarray
    landmarks: bool = True

    @property
    def x(self) -> float:
        return float(self.row[0])

    @property
    def y(self) -> float:
        return float(self.row[1])

    @property
    def width(self) -> float:
        return max(0.0, float(self.row[2]))

    @property
    def height(self) -> float:
        return max(0.0, float(self.row[3]))

    @property
    def confidence(self) -> float:
        return float(self.row[14])

    @property
    def area(self) -> float:
        return self.width * self.height

    def frontalness(self) -> float:
        if not self.landmarks:
            return 1.0
        right_eye_x, left_eye_x, nose_x = float(self.row[4]), float(self.row[6]), float(self.row[8])
        to_right = abs(nose_x - right_eye_x)
        to_left = abs(left_eye_x - nose_x)
        widest = max(to_right, to_left)
        if widest <= 1e-6:
            return 0.0
        return min(to_right, to_left) / widest

    def normalized(self, image_width: int, image_height: int) -> Box:
        left = min(max(self.x / image_width, 0.0), 1.0)
        top = min(max(self.y / image_height, 0.0), 1.0)
        right = min(max((self.x + self.width) / image_width, 0.0), 1.0)
        bottom = min(max((self.y + self.height) / image_height, 0.0), 1.0)
        return Box(x=left, y=top, w=right - left, h=bottom - top)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_model(path: Path, asset: ModelAsset) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= asset.min_bytes and checksum(path) == asset.sha256
    except OSError:
        return False


def download_model(url: str, target: Path, asset: ModelAsset) -> None:
    handle = tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False)
    partial = Path(handle.name)
    digest = hashlib.sha256()
    deadline = time.monotonic() + DOWNLOAD_DEADLINE
    try:
        with handle, httpx.stream("GET", url, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            for block in response.iter_bytes(1 << 20):
                if time.monotonic() > deadline:
                    raise ModelUnavailable(f"download exceeded {DOWNLOAD_DEADLINE:.0f}s")
                handle.write(block)
                digest.update(block)
        size = partial.stat().st_size
        if size < asset.min_bytes:
            raise ModelUnavailable(f"got {size} bytes, expected at least {asset.min_bytes}")
        if digest.hexdigest() != asset.sha256:
            raise ModelUnavailable("checksum mismatch")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def fetch_model(asset: ModelAsset, directory: Path) -> Path:
    target = directory / asset.filename
    if is_valid_model(target, asset):
        return target
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelUnavailable(f"cannot create {directory}: {exc}") from exc
    failures: list[str] = []
    for url in asset.urls:
        log.info("downloading %s from %s", asset.filename, url)
        try:
            download_model(url, target, asset)
        except (httpx.HTTPError, OSError, ModelUnavailable) as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            log.warning("download of %s from %s failed: %s", asset.filename, url, exc)
            continue
        return target
    raise ModelUnavailable(f"cannot obtain {asset.filename} ({' | '.join(failures)})")


def l2_normalized(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 1e-12 else flat


class FaceEngine:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._load_lock = threading.Lock()
        self._detect_lock = threading.Lock()
        self._embed_lock = threading.Lock()
        self._detector: cv2.FaceDetectorYN | None = None
        self._recognizer: cv2.FaceRecognizerSF | None = None
        self._failure = ""
        self._failed_at: float | None = None

    @property
    def available(self) -> bool:
        return self._load()

    @property
    def failure(self) -> str:
        return self._failure

    def _load(self) -> bool:
        if self._recognizer is not None:
            return True
        with self._load_lock:
            if self._recognizer is not None:
                return True
            if self._failed_at is not None and time.monotonic() - self._failed_at < LOAD_RETRY_AFTER:
                return False
            try:
                detector_path = fetch_model(DETECTOR_MODEL, self.models_dir)
                recognizer_path = fetch_model(RECOGNIZER_MODEL, self.models_dir)
                detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), DETECT_SCORE, DETECT_NMS, DETECT_TOP_K)
                recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
            except (ModelUnavailable, cv2.error) as exc:
                self._failure = str(exc)
                self._failed_at = time.monotonic()
                log.warning("face models unavailable, identity lock degraded: %s", exc)
                return False
            self._detector = detector
            self._recognizer = recognizer
            self._failure = ""
            self._failed_at = None
            log.info("face engine ready (YuNet + SFace) from %s", self.models_dir)
            return True

    def _require(self) -> None:
        if not self._load():
            raise ModelUnavailable(self._failure or "face models are not loaded")

    def detect(self, pixels: np.ndarray) -> list[Face]:
        self._require()
        tried: set[tuple[int, int]] = set()
        for side in DETECT_SIDES:
            working, scale = fit_within(pixels, side)
            size = (working.shape[1], working.shape[0])
            if size in tried:
                continue
            tried.add(size)
            faces = self._detect_scaled(working, scale)
            if faces:
                return faces
        return []

    def _detect_scaled(self, working: np.ndarray, scale: float) -> list[Face]:
        with self._detect_lock:
            self._detector.setInputSize((working.shape[1], working.shape[0]))
            _, rows = self._detector.detect(working)
        if rows is None:
            return []
        faces: list[Face] = []
        for row in rows:
            restored = np.array(row, dtype=np.float32)
            restored[:14] /= scale
            faces.append(Face(row=restored))
        return faces

    def embed(self, pixels: np.ndarray, face: Face) -> np.ndarray:
        self._require()
        with self._embed_lock:
            aligned = self._recognizer.alignCrop(pixels, face.row)
            feature = self._recognizer.feature(aligned)
        return l2_normalized(np.array(feature, dtype=np.float32))


_ENGINES: dict[Path, FaceEngine] = {}
_ENGINES_LOCK = threading.Lock()
_HAAR_LOCK = threading.Lock()


def face_engine(settings: Settings) -> FaceEngine:
    key = Path(settings.models_dir).expanduser().resolve()
    with _ENGINES_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = FaceEngine(key)
            _ENGINES[key] = engine
        return engine


@lru_cache(maxsize=1)
def haar_cascade() -> Any | None:
    factory = getattr(cv2, "CascadeClassifier", None)
    folder = getattr(getattr(cv2, "data", None), "haarcascades", "")
    path = Path(folder) / "haarcascade_frontalface_default.xml"
    if factory is None or not path.is_file():
        log.warning("Haar cascade fallback is not available in OpenCV %s", cv2.__version__)
        return None
    cascade = factory(str(path))
    if cascade.empty():
        log.warning("Haar cascade at %s could not be loaded", path)
        return None
    return cascade


def haar_faces(pixels: np.ndarray) -> list[Face]:
    cascade = haar_cascade()
    if cascade is None:
        return []
    working, scale = fit_within(pixels, HAAR_MAX_SIDE)
    gray = cv2.equalizeHist(cv2.cvtColor(working, cv2.COLOR_BGR2GRAY))
    minimum = max(24, min(gray.shape[:2]) // 12)
    with _HAAR_LOCK:
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(minimum, minimum))
    faces: list[Face] = []
    for x, y, w, h in np.asarray(boxes).reshape(-1, 4):
        row = np.zeros(15, dtype=np.float32)
        row[:4] = np.array([x, y, w, h], dtype=np.float32) / scale
        row[14] = 1.0
        faces.append(Face(row=row, landmarks=False))
    return faces


def detect_faces(pixels: np.ndarray, settings: Settings) -> list[Face]:
    engine = face_engine(settings)
    if engine.available:
        return engine.detect(pixels)
    return haar_faces(pixels)


def face_box(image: Path, settings: Settings) -> Box | None:
    pixels = read_image(image)
    if pixels is None:
        return None
    faces = detect_faces(pixels, settings)
    if not faces:
        return None
    height, width = pixels.shape[:2]
    return max(faces, key=lambda face: face.area).normalized(width, height)


@dataclass(eq=False)
class Candidate:
    time: float
    path: Path
    width: int
    height: int
    sharpness: float
    face: Face | None = None
    embedding: np.ndarray | None = None
    score: float = 0.0


def sharpness(pixels: np.ndarray) -> float:
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if pixels.ndim == 3 else pixels
    height, width = gray.shape[:2]
    if width == 0 or height == 0:
        return 0.0
    if width != SHARPNESS_WIDTH:
        gray = cv2.resize(gray, (SHARPNESS_WIDTH, max(1, round(height * SHARPNESS_WIDTH / width))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_region(pixels: np.ndarray, face: Face, padding: float = 0.2) -> np.ndarray:
    height, width = pixels.shape[:2]
    left = int(max(0, face.x - face.width * padding))
    top = int(max(0, face.y - face.height * padding))
    right = int(min(width, face.x + face.width * (1 + padding)))
    bottom = int(min(height, face.y + face.height * (1 + padding)))
    if right - left < 2 or bottom - top < 2:
        return pixels
    return pixels[top:bottom, left:right]


def inspect_frame(moment: float, path: Path, engine: FaceEngine | None) -> Candidate | None:
    pixels = read_image(path)
    if pixels is None:
        return None
    height, width = pixels.shape[:2]
    faces = engine.detect(pixels) if engine else haar_faces(pixels)
    face = max(faces, key=lambda item: item.area) if faces else None
    region = face_region(pixels, face) if face else pixels
    candidate = Candidate(time=moment, path=path, width=width, height=height, sharpness=sharpness(region), face=face)
    if face is not None and engine is not None:
        candidate.embedding = engine.embed(pixels, face)
    return candidate


def score_candidates(candidates: list[Candidate]) -> None:
    peak = max((c.sharpness for c in candidates), default=0.0) or 1.0
    for candidate in candidates:
        face = candidate.face
        if face is None:
            candidate.score = 0.0
            continue
        width_share = min(1.0, face.width / max(1, candidate.width))
        candidate.score = width_share * face.confidence * (candidate.sharpness / peak) * face.frontalness()


def dominant_identity(ranked: list[Candidate]) -> list[Candidate]:
    embedded = [c for c in ranked if c.embedding is not None]
    if len(embedded) < 2:
        return ranked
    matrix = np.stack([c.embedding for c in embedded])
    similar = (matrix @ matrix.T) >= SFACE_MATCH_COSINE
    support = similar.sum(axis=1)
    anchor = int(np.argmax(support))
    members = {id(embedded[index]) for index in np.flatnonzero(similar[anchor])}
    dropped = len(embedded) - len(members)
    if dropped:
        log.info("hero lock ignored %d frame(s) showing a different face", dropped)
    return [c for c in ranked if id(c) in members]


def diverse_picks(ranked: list[Candidate], count: int) -> list[Candidate]:
    if not ranked:
        return []
    times = [c.time for c in ranked]
    gap = max(1.0, (max(times) - min(times)) / (2 * count))
    chosen: list[Candidate] = []
    for spacing in (gap, gap / 2, gap / 4, 0.0):
        for candidate in ranked:
            if len(chosen) >= count:
                return chosen
            if any(candidate is picked for picked in chosen):
                continue
            if all(abs(candidate.time - picked.time) >= spacing for picked in chosen):
                chosen.append(candidate)
    return chosen


def portrait_crop(pixels: np.ndarray, face: Face) -> np.ndarray:
    height, width = pixels.shape[:2]
    crop_height = min(float(height), face.height * 3.2)
    crop_width = min(float(width), max(crop_height * 0.8, face.width * 1.8))
    center_x = face.x + face.width / 2
    left = min(max(center_x - crop_width / 2, 0.0), width - crop_width)
    top = min(max(face.y - crop_height * 0.24, 0.0), height - crop_height)
    crop = pixels[int(top) : int(top + crop_height), int(left) : int(left + crop_width)]
    return upscaled(crop, REFERENCE_MIN_SIDE)


def upscaled(pixels: np.ndarray, min_side: int) -> np.ndarray:
    height, width = pixels.shape[:2]
    shortest = min(height, width)
    if shortest >= min_side or shortest == 0:
        return pixels
    factor = min_side / shortest
    size = (max(min_side, round(width * factor)), max(min_side, round(height * factor)))
    return cv2.resize(pixels, size, interpolation=cv2.INTER_LANCZOS4)


def write_full_frame(candidate: Candidate, target: Path) -> Path | None:
    pixels = read_image(candidate.path)
    if pixels is None:
        return None
    return write_jpeg(pixels, target)


def write_portrait(candidate: Candidate, target: Path) -> Path | None:
    pixels = read_image(candidate.path)
    if pixels is None or candidate.face is None:
        return None
    return write_jpeg(portrait_crop(pixels, candidate.face), target)


def write_references(picks: list[Candidate], out_dir: Path, with_faces: bool) -> list[Path]:
    if not picks:
        return []
    references: list[Path] = []
    first = write_full_frame(picks[0], out_dir / "ref_01.jpg")
    if first:
        references.append(first)
    for slot in range(1, REFERENCE_COUNT):
        target = out_dir / f"ref_{slot + 1:02d}.jpg"
        if with_faces:
            written = write_portrait(picks[slot % len(picks)], target)
        elif slot < len(picks):
            written = write_full_frame(picks[slot], target)
        else:
            written = None
        if written:
            references.append(written)
    return references


def identity_template(cluster: list[Candidate]) -> list[float] | None:
    embedded = [c.embedding for c in cluster if c.embedding is not None][:TEMPLATE_TOP_K]
    if not embedded:
        return None
    template = l2_normalized(np.mean(np.stack(embedded), axis=0))
    return [round(float(value), 6) for value in template]


def hero_id(description: str, seed: int, picks: list[Candidate], embedding: list[float] | None) -> str:
    digest = hashlib.sha256()
    digest.update(description.encode())
    digest.update(str(seed).encode())
    for candidate in picks:
        digest.update(f"{candidate.time:.3f}:{candidate.path.name}".encode())
    if embedding:
        digest.update(np.asarray(embedding, dtype=np.float32).tobytes())
    return "hero-" + digest.hexdigest()[:10]


def build_hero(frames: list[tuple[float, Path]], out_dir: Path, description: str, seed: int, settings: Settings) -> HeroPack:
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = face_engine(settings)
    active = engine if engine.available else None
    candidates = [c for c in (inspect_frame(moment, Path(path), active) for moment, path in frames) if c is not None]
    if not candidates:
        log.warning("no readable frames for hero lock; hero has no references")
        return HeroPack(id=hero_id(description, seed, [], None), description=description, seed=seed, negative=HERO_NEGATIVE)
    with_faces = [c for c in candidates if c.face is not None]
    if not with_faces:
        log.warning("no face found in %d frames; using the sharpest frames as hero references", len(candidates))
        picks = diverse_picks(sorted(candidates, key=lambda c: c.sharpness, reverse=True), REFERENCE_COUNT)
        references = write_references(picks, out_dir, with_faces=False)
        return HeroPack(
            id=hero_id(description, seed, picks, None),
            description=description,
            references=references,
            seed=seed,
            negative=HERO_NEGATIVE,
        )
    score_candidates(with_faces)
    ranked = sorted(with_faces, key=lambda c: c.score, reverse=True)
    cluster = dominant_identity(ranked)
    picks = diverse_picks(cluster, REFERENCE_COUNT)
    references = write_references(picks, out_dir, with_faces=True)
    embedding = identity_template(cluster)
    log.info(
        "hero locked from %d/%d face frames, picks at %s, embedding %s",
        len(cluster),
        len(candidates),
        ", ".join(f"{c.time:.1f}s" for c in picks),
        "ready" if embedding else "unavailable",
    )
    return HeroPack(
        id=hero_id(description, seed, picks, embedding),
        description=description,
        references=references,
        embedding=embedding,
        embedding_model=EMBEDDING_MODEL if embedding else None,
        seed=seed,
        negative=HERO_NEGATIVE,
    )


def skip_reason(hero: HeroPack, engine: FaceEngine) -> str | None:
    if not hero.embedding:
        return "hero has no identity embedding (no face found in the winner); identity not enforced"
    if hero.embedding_model and hero.embedding_model != EMBEDDING_MODEL:
        return f"hero embedding from {hero.embedding_model} is not comparable with {EMBEDDING_MODEL}"
    if not engine.available:
        return f"face models unavailable ({engine.failure or 'not loaded'}); identity not verified"
    return None


def best_match(engine: FaceEngine, pixels: np.ndarray, anchor: np.ndarray) -> tuple[float | None, int]:
    faces = sorted(engine.detect(pixels), key=lambda face: face.area, reverse=True)[:MAX_FACES_COMPARED]
    if not faces:
        return None, 0
    return max(float(engine.embed(pixels, face) @ anchor) for face in faces), len(faces)


def check_image(image: Path, hero: HeroPack, threshold: float, settings: Settings) -> Check:
    name = "identity_keyframe"
    engine = face_engine(settings)
    reason = skip_reason(hero, engine)
    if reason:
        return Check(name=name, passed=False, skipped=True, threshold=threshold, retry="shot", detail=reason)
    pixels = read_image(Path(image))
    if pixels is None:
        return Check(name=name, passed=False, score=0.0, threshold=threshold, retry="shot", detail=f"keyframe {Path(image).name} could not be read")
    similarity, faces = best_match(engine, pixels, l2_normalized(np.asarray(hero.embedding)))
    if similarity is None:
        return Check(name=name, passed=False, score=0.0, threshold=threshold, retry="shot", detail="no face detected in keyframe")
    passed = similarity >= threshold
    verdict = "same person as the hero" if passed else "identity drift from the hero"
    return Check(
        name=name,
        passed=passed,
        score=round(similarity, 4),
        threshold=threshold,
        retry="shot",
        detail=f"cosine {similarity:.3f} vs threshold {threshold:.2f} ({faces} face(s) compared): {verdict}",
    )


def frame_readings(engine: FaceEngine, video: Path, times: list[float], anchor: np.ndarray) -> list[tuple[float, float | None]]:
    readings: list[tuple[float, float | None]] = []
    for moment, pixels in grab_frames(video, times):
        if pixels is None:
            continue
        similarity, _ = best_match(engine, pixels, anchor)
        readings.append((moment, similarity))
    return readings


def describe_readings(readings: list[tuple[float, float | None]]) -> str:
    return ", ".join(f"{moment:.2f}s={'no face' if score is None else f'{score:.2f}'}" for moment, score in readings)


def check_video(
    video: Path,
    hero: HeroPack,
    threshold: float,
    settings: Settings,
    start: float = 0.0,
    end: float | None = None,
    samples: int = 6,
) -> Check:
    name = "identity_hook"
    engine = face_engine(settings)
    reason = skip_reason(hero, engine)
    if reason:
        return Check(name=name, passed=False, skipped=True, threshold=threshold, retry="shot", detail=reason)
    video = Path(video)
    duration = video_duration(video)
    stop = end if end is not None else duration
    if stop is None:
        return Check(name=name, passed=False, score=0.0, threshold=threshold, retry="shot", detail=f"cannot read {video.name}")
    if duration is not None:
        stop = min(stop, duration)
    begin = min(max(0.0, start), stop)
    readings = frame_readings(engine, video, spread(begin, stop, max(1, samples)), l2_normalized(np.asarray(hero.embedding)))
    if not readings:
        return Check(name=name, passed=False, score=0.0, threshold=threshold, retry="shot", detail=f"no decodable frames in {video.name}")
    listing = describe_readings(readings)
    scored = [(moment, score) for moment, score in readings if score is not None]
    if not scored:
        return Check(
            name=name,
            passed=False,
            score=0.0,
            threshold=threshold,
            retry="shot",
            detail=f"no face in any of {len(readings)} sampled frames | frames: {listing}",
        )
    median = float(np.median([score for _, score in scored]))
    worst_time, worst = min(scored, key=lambda item: item[1])
    missing = len(readings) - len(scored)
    floor = threshold - FLICKER_MARGIN
    problems: list[str] = []
    if median < threshold:
        problems.append(f"median cosine {median:.3f} below {threshold:.2f}")
    if worst < floor:
        problems.append(f"identity flicker at {worst_time:.2f}s ({worst:.3f} < {floor:.2f})")
    if missing / len(readings) > MAX_MISSING_SHARE:
        problems.append(f"no face in {missing}/{len(readings)} frames")
    summary = "; ".join(problems) if problems else f"median cosine {median:.3f} >= {threshold:.2f}, lowest {worst:.3f}"
    return Check(
        name=name,
        passed=not problems,
        score=round(median, 4),
        threshold=threshold,
        retry="shot",
        detail=f"{summary} | frames: {listing}",
    )
