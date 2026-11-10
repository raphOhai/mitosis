from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from mitosis.config import ModelSpec, Settings
from mitosis.models import Aspect
from mitosis.providers.base import GenResult, PermanentError, TransientError, register
from mitosis.providers.remote import HttpProvider, download, env, media_type

log = logging.getLogger(__name__)

Graph = dict[str, dict[str, Any]]
NodePath = list[str]

MEMORY_MARKERS = ("out of memory", "outofmemory", "cuda error", "allocation on device")


def is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def binding_paths(binding: Any) -> list[NodePath]:
    if not binding:
        return []
    if not any(isinstance(step, (list, tuple)) for step in binding):
        return [[str(step) for step in binding]]
    return [[str(step) for step in path] for path in binding]


def assign(graph: Graph, path: NodePath, value: Any) -> None:
    if len(path) < 2:
        raise PermanentError(f"binding {path} must name a node and a field")
    target: Any = graph
    for step in path[:-1]:
        if not isinstance(target, dict) or step not in target:
            raise PermanentError(f"workflow has no {'/'.join(path)}")
        target = target[step]
    if not isinstance(target, dict):
        raise PermanentError(f"workflow path {'/'.join(path)} does not point into an object")
    target[path[-1]] = value


def bypass(graph: Graph, node_id: str) -> None:
    node = graph.get(node_id)
    if node is None:
        raise PermanentError(f"workflow has no node {node_id} to bypass")
    upstream = [value for value in node.get("inputs", {}).values() if is_link(value)]
    if not upstream:
        raise PermanentError(f"node {node_id} has no linked input to pass through")
    del graph[node_id]
    for other in graph.values():
        inputs = other.get("inputs", {})
        for key, value in inputs.items():
            if is_link(value) and value[0] == node_id:
                slot = value[1] if value[1] < len(upstream) else 0
                inputs[key] = list(upstream[slot])


def snap(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def fit(aspect: Aspect, megapixels: float, multiple: int) -> tuple[int, int]:
    height = math.sqrt(megapixels * 1_000_000 / aspect.ratio)
    return snap(height * aspect.ratio, multiple), snap(height, multiple)


def execution_error(entry: dict[str, Any]) -> str:
    for message in entry.get("status", {}).get("messages") or []:
        if isinstance(message, list) and len(message) == 2 and message[0] == "execution_error":
            detail = message[1] or {}
            return f"{detail.get('node_type', 'node')} {detail.get('node_id', '')}: {detail.get('exception_message', '').strip()}"
    return "execution failed without details"


def load_graph(path: Path) -> Graph:
    try:
        graph = json.loads(path.read_text())
    except OSError as exc:
        raise PermanentError(f"cannot read ComfyUI workflow {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PermanentError(f"ComfyUI workflow {path} is not valid JSON: {exc}") from exc
    if not isinstance(graph, dict) or "nodes" in graph:
        raise PermanentError(f"ComfyUI workflow {path} must be exported in API format")
    if not all(isinstance(node, dict) and "class_type" in node for node in graph.values()):
        raise PermanentError(f"ComfyUI workflow {path} contains nodes without class_type")
    return graph


@register("comfyui", "image")
class ComfyUIImage(HttpProvider):
    service = "comfyui"

    def __init__(self, spec: ModelSpec, settings: Settings) -> None:
        super().__init__(spec, settings)
        self.client_id = uuid.uuid4().hex
        self._graph: Graph | None = None
        self._graph_lock = threading.Lock()

    def base_url(self) -> str:
        return env("COMFYUI_URL").rstrip("/")

    def generate(
        self,
        prompt: str,
        references: list[Path],
        aspect: Aspect,
        seed: int,
        out: Path,
        negative: str = "",
    ) -> GenResult:
        bindings: dict[str, Any] = self.params.get("bindings", {})
        width, height = fit(aspect, float(self.params.get("megapixels", 1.0)), int(self.params.get("multiple", 16)))
        values: dict[str, Any] = {
            **self.params.get("values", {}),
            "prompt": prompt,
            "negative": negative or self.params.get("negative", ""),
            "seed": seed,
            "width": width,
            "height": height,
        }
        if references:
            values["reference"] = self.upload(references[0])
        elif "reference" in bindings:
            raise PermanentError("the ComfyUI identity workflow needs a hero reference image")
        graph = self.workflow()
        self.apply(graph, bindings, values)
        prompt_id = self.queue(graph)
        entry = self.wait(lambda: self.history(prompt_id), f"comfyui prompt {prompt_id}")
        self.fetch(self.output_image(entry), out)
        return GenResult(
            path=out,
            usage={"image": 1},
            meta={"prompt_id": prompt_id, "seed": seed, "width": width, "height": height},
        )

    def workflow(self) -> Graph:
        with self._graph_lock:
            if self._graph is None:
                source = Path(self.spec.model)
                self._graph = load_graph(source if source.is_absolute() else self.settings.root / source)
            return copy.deepcopy(self._graph)

    def apply(self, graph: Graph, bindings: dict[str, Any], values: dict[str, Any]) -> None:
        optional = set(self.params.get("bypass", []))
        removed: set[str] = set()
        for name in optional:
            if values.get(name) is None:
                for node_id in {path[0] for path in binding_paths(bindings.get(name))}:
                    if node_id not in removed:
                        bypass(graph, node_id)
                        removed.add(node_id)
        for name, binding in bindings.items():
            value = values.get(name)
            if value is None:
                continue
            for path in binding_paths(binding):
                if path[0] not in removed:
                    assign(graph, path, value)

    def upload(self, path: Path) -> str:
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise PermanentError(f"cannot read reference image {path}: {exc}") from exc
        name = f"mitosis_{hashlib.sha256(content).hexdigest()[:20]}{path.suffix.lower() or '.png'}"
        stored = self.call_json(
            "POST",
            "/upload/image",
            files={"image": (name, content, media_type(path))},
            data={"overwrite": "true", "type": "input"},
        )
        if not isinstance(stored, dict) or not stored.get("name"):
            raise TransientError(f"comfyui upload returned {str(stored)[:200]}")
        subfolder = stored.get("subfolder") or ""
        return f"{subfolder}/{stored['name']}" if subfolder else str(stored["name"])

    def queue(self, graph: Graph) -> str:
        queued = self.call_json("POST", "/prompt", json={"prompt": graph, "client_id": self.client_id})
        if not isinstance(queued, dict):
            raise TransientError(f"comfyui returned {str(queued)[:200]}")
        if queued.get("node_errors"):
            raise PermanentError(f"comfyui rejected the workflow: {json.dumps(queued['node_errors'])[:400]}")
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise TransientError(f"comfyui did not return a prompt id: {str(queued)[:200]}")
        log.info("comfyui queued prompt %s at position %s", prompt_id, queued.get("number"))
        return str(prompt_id)

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = self.call_json("GET", f"/history/{prompt_id}")
        entry = data.get(prompt_id) if isinstance(data, dict) else None
        if not entry:
            return None
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            detail = execution_error(entry)
            if any(marker in detail.lower() for marker in MEMORY_MARKERS):
                raise TransientError(f"comfyui ran out of memory: {detail}")
            raise PermanentError(f"comfyui execution failed: {detail}")
        if not status.get("completed", True):
            return None
        return entry

    def output_image(self, entry: dict[str, Any]) -> dict[str, Any]:
        outputs: dict[str, Any] = entry.get("outputs") or {}
        preferred = str(self.params.get("output", ""))
        nodes = [outputs[preferred]] if preferred in outputs else list(outputs.values())
        images = [image for node in nodes for image in node.get("images") or []]
        saved = [image for image in images if image.get("type") == "output"]
        if not (saved or images):
            raise PermanentError("comfyui finished without producing an image")
        return (saved or images)[0]

    def fetch(self, image: dict[str, Any], out: Path) -> Path:
        query = urlencode(
            {"filename": image.get("filename", ""), "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")}
        )
        return download(self.api, f"/view?{query}", out, self.service)
