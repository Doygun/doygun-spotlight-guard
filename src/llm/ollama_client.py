import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

class OllamaClientError(RuntimeError):
    pass

@dataclass
class ModelRequest:
    model: str
    prompt: str
    system: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = None

class OllamaClient:

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        max_retries: int = 2,
        backoff: float = 2.0,
        num_ctx: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_predict: int = 512,
        repeat_penalty: float = 1.3,
        repeat_last_n: int = 256,
        keep_alive: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.keep_alive = keep_alive
        self._log = logging.getLogger(self.__class__.__name__)

    def _build_options(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
        }
        if overrides:
            options.update(overrides)
        return options

    def generate(self, request: ModelRequest) -> str:
        payload: Dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": self._build_options(request.options),
        }
        if request.system:
            payload["system"] = request.system
        keep_alive = request.keep_alive if request.keep_alive is not None else self.keep_alive
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        url = f"{self.base_url}/api/generate"
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                if response.status_code != 200:
                    raise OllamaClientError(
                        f"Ollama generate {response.status_code}: {response.text}"
                    )
                data = response.json()
                if "response" not in data:
                    raise OllamaClientError(
                        f"Ollama dönüşünde 'response' alanı eksik: {json.dumps(data)}"
                    )
                return data["response"]
            except (requests.RequestException, ValueError, OllamaClientError) as exc:
                last_error = exc
                self._log.warning(
                    "Ollama istegi %s/%s basarisiz: %s",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt > self.max_retries:
                    break
                time.sleep(self.backoff * attempt)
        raise OllamaClientError(f"Ollama istegi tamamlanamadi: {last_error}")

    def health_check(self) -> bool:
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=min(10, self.timeout))
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            self._log.debug("Ollama health check failed: %s", exc)
            return False
