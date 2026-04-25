# llm_utils.py  (spatchat-connect)
import os, sys, time, random, json, re
from pathlib import Path
from typing import Iterator
from huggingface_hub import InferenceClient
from together import Together
from together.error import InvalidRequestError, RateLimitError, ServiceUnavailableError

HF_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
ANSWER_MAX_TOKENS = 360


def _load_local_env() -> None:
    """Load simple KEY=VALUE pairs from local .env files if present."""
    root = Path(__file__).resolve().parent
    for name in (".env", ".env.local"):
        path = root / name
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value
        except Exception:
            continue


_load_local_env()

def _choice_content(choice):
    msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
    content = None if msg is None else (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None))
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text",""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    return content or ""

def _delta_text(delta):
    return delta.get("content","") if isinstance(delta, dict) else getattr(delta, "content", "")


def _stream_choice_text(chunk) -> str:
    choices = getattr(chunk, "choices", None) or (chunk.get("choices") if isinstance(chunk, dict) else None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None) or (choices[0].get("delta") if isinstance(choices[0], dict) else None)
    if delta is None:
        return ""
    return _delta_text(delta)

class _SpacedCallLimiter:
    def __init__(self, min_interval_seconds: float): self.min, self._last = float(min_interval_seconds), 0.0
    def wait(self):
        now = time.monotonic(); gap = now - self._last
        if gap < self.min: time.sleep(self.min - gap)
        self._last = time.monotonic()

class UnifiedLLM:
    def __init__(self):
        token = (os.getenv("HF_TOKEN") or "").strip()
        endpoint_url = (os.getenv("HF_ENDPOINT_URL") or os.getenv("HF_BASE_URL") or "").strip()
        self.hf_model = (os.getenv("HF_MODEL") or HF_MODEL_DEFAULT).strip()
        self.hf_token = token
        self.hf_default = InferenceClient(token=token, timeout=300)

        if endpoint_url:
            self.hf = InferenceClient(base_url=endpoint_url, token=token, timeout=300)
            self.hf_chat_enabled = True
        else:
            self.hf = InferenceClient(model=self.hf_model, token=token, timeout=300)
            self.hf_chat_enabled = False

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        self.together_model_candidates = []
        for model_name in (
            self.together_model,
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
        ):
            if model_name and model_name not in self.together_model_candidates:
                self.together_model_candidates.append(model_name)
        tk = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tk:
            self.together = Together(api_key=tk)
            min_interval = float((os.getenv("TOGETHER_MIN_INTERVAL_SECONDS") or "0.0").strip() or "0.0")
            self._lim = _SpacedCallLimiter(min_interval)

    @staticmethod
    def _is_not_found_error(exc) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code == 404 or "404" in str(exc)

    def _hf_chat(self, messages, max_tokens=256, temperature=0.0, stream=False):
        prompt = self._messages_to_prompt(messages)

        # If no explicit backend is configured, prefer the model-less HF route first.
        # That gives Hugging Face a chance to pick an available default provider/model
        # instead of failing immediately on a model-specific public endpoint.
        if not self.hf_chat_enabled and not self.hf_token:
            return self.hf_default.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                return_full_text=False,
            )

        tries, delay, last = 3, 2.5, None
        for _ in range(tries):
            try:
                if self.hf_chat_enabled and hasattr(self.hf, "chat_completion"):
                    resp = self.hf.chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                    )
                    return "".join(_delta_text(ch.choices[0].delta) for ch in resp) if stream else _choice_content(resp.choices[0])
                return self.hf.text_generation(
                    prompt,
                    model=None if self.hf_chat_enabled else self.hf_model,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                    return_full_text=False,
                )
            except Exception as e:
                if self._is_not_found_error(e):
                    # Model-specific public routes can 404 even with a valid token.
                    # Fall back to the HF default recommended text-generation route.
                    return self.hf_default.text_generation(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                        return_full_text=False,
                    )
                last = e; time.sleep(delay); delay *= 1.8
        raise last

    @staticmethod
    def _messages_to_prompt(messages):
        parts=[]
        for m in messages:
            role=m.get("role","user"); content=m.get("content","")
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _together_chat(self, messages, temperature=0.0, max_tokens=256):
        if not self.together:
            raise RuntimeError("Together client is not configured")
        last = None
        for model_name in self.together_model_candidates:
            self._lim.wait()
            backoff=12.0
            for i in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=model_name, messages=messages,
                        temperature=temperature, max_tokens=max_tokens, stream=False
                    )
                    self.together_model = model_name
                    return _choice_content(resp.choices[0])
                except InvalidRequestError as e:
                    last = e
                    break
                except (RateLimitError, ServiceUnavailableError) as e:
                    last = e
                    if i==3: break
                    time.sleep(backoff + random.uniform(0,3)); backoff*=1.8
                except Exception as e:
                    last = e
                    break
        raise last if last is not None else RuntimeError("Together chat failed without an error")

    def _together_chat_stream(self, messages, temperature=0.0, max_tokens=256) -> Iterator[str]:
        if not self.together:
            raise RuntimeError("Together client is not configured")
        last = None
        for model_name in self.together_model_candidates:
            self._lim.wait()
            backoff=12.0
            for i in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                    self.together_model = model_name
                    for chunk in resp:
                        text = _stream_choice_text(chunk)
                        if text:
                            yield text
                    return
                except InvalidRequestError as e:
                    last = e
                    break
                except (RateLimitError, ServiceUnavailableError) as e:
                    last = e
                    if i==3: break
                    time.sleep(backoff + random.uniform(0,3)); backoff*=1.8
                except Exception as e:
                    last = e
                    break
        raise last if last is not None else RuntimeError("Together chat stream failed without an error")

    def _hf_chat_stream(self, messages, max_tokens=256, temperature=0.0) -> Iterator[str]:
        prompt = self._messages_to_prompt(messages)
        if not self.hf_chat_enabled and not self.hf_token:
            for token in self.hf_default.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                return_full_text=False,
            ):
                if token:
                    yield str(token)
            return

        if self.hf_chat_enabled and hasattr(self.hf, "chat_completion"):
            resp = self.hf.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in resp:
                text = _stream_choice_text(chunk)
                if text:
                    yield text
            return

        try:
            for token in self.hf.text_generation(
                prompt,
                model=self.hf_model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                return_full_text=False,
            ):
                if token:
                    yield str(token)
            return
        except Exception as e:
            if self._is_not_found_error(e):
                for token in self.hf_default.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    return_full_text=False,
                ):
                    if token:
                        yield str(token)
                return
            raise

    def chat(self, messages, temperature=0.0, max_tokens=256, stream=False):
        if self.together:
            try:
                return self._together_chat(messages, temperature=temperature, max_tokens=max_tokens)
            except Exception:
                pass
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as e:
            if not self.together: raise
            return self._together_chat(messages, temperature=temperature, max_tokens=max_tokens)

    def stream_chat(self, messages, temperature=0.0, max_tokens=256) -> Iterator[str]:
        if self.together:
            try:
                yield from self._together_chat_stream(messages, temperature=temperature, max_tokens=max_tokens)
                return
            except Exception:
                pass
        try:
            yield from self._hf_chat_stream(messages, max_tokens=max_tokens, temperature=temperature)
            return
        except Exception:
            if not self.together:
                raise
        yield from self._together_chat_stream(messages, temperature=temperature, max_tokens=max_tokens)

# ---- Connectivity-focused prompt (adds tool schema) -------------------------
SYSTEM_PROMPT = """
You are SpatChat, a landscape ecologist and connectivity analysis expert.

Your role:
- Answer questions about landscape ecology, habitat connectivity, resistance surfaces, conductance, least-cost paths, graph connectivity, resistant kernels, and circuit-style analyses.
- Be scientifically grounded, concise, and practical.
- When appropriate, guide the user back to what this app can do with the currently loaded data.
- Prefer app-oriented guidance such as uploading a raster, adding nodes, converting a surface, running least-cost paths, or running the graph workflow.

You are given a JSON object called dataset_context with facts about the currently loaded inputs
(e.g., raster CRS/bounds/dimensions/resolution, nodes columns/count/samples, and analysis result
summaries or small result-table previews). When the user asks anything about the loaded data or
analysis results, answer STRICTLY using dataset_context. If the answer is not present or cannot be
derived from it, say you don't know and briefly suggest what the user could do to compute it. For
questions about raster dimensions, use dataset_context.raster_dimensions and raster_resolution.
For questions about where the study area is, use dataset_context.raster_location first: report the
WGS84 centroid and/or bounds, and you may give a coarse approximate geographic description from
those coordinates if it is obvious. Do not invent exact administrative names; say reverse geocoding
or an administrative boundary lookup would be needed for an exact place name.
For questions about result counts, metrics, or outputs, use dataset_context.analysis_results.
For questions asking what a result layer, edge, metric, or output "means", use
dataset_context.result_layer_meanings plus the current values in analysis_results. Explain the
interpretation in plain language; do not just repeat the completion/status message.

When the user requests an analysis, prefer to emit a TOOL CALL as a single, raw JSON object with:
  {
    "tool": "connectivity",
    "method": "<one of: lcp | kernel | circuit>",
    // Optional common parameters:
    "n_jobs": <int>,
    "preview_n": <int>,            // # paths to render in the UI preview
    // Method-specific:
    // LCP:
    "simplify_tol_m": <float|null>,
    // Kernel:
    "kernel": "gaussian" | "epanechnikov",
    "sigma": <float>,              // in cost-distance units
    "cutoff": <float|null>,        // cost-distance cutoff
    "normalize": <bool>,
    "source": "all" | ["id1","id2",...],
    // Circuit:
    "normalize": <bool>,
    "source": "all" | ["id1","id2",...],
    "source_weights": {"id1": <float>, ...}
  }

Rules:
- If you emit a tool call, output ONLY the JSON object, no prose.
- Do not emit a tool call for questions asking about currently loaded data, metadata, previous outputs, or analysis results; answer from dataset_context instead.
- If you are not sure which tool to call, ask a brief clarifying question (<= 2 sentences).
- If no tool call is needed, answer naturally in <= 4 sentences.
- Make every prose answer end on a complete sentence; keep it shorter rather than ending mid-thought.
- For general topic questions, answer as a landscape ecologist first, then briefly connect the answer to a concrete action the user can take in this app when relevant.
- If the user asks for something outside the app's supported workflows, say that plainly and redirect them to the closest supported workflow.
- Do not fabricate numbers. Do not invent files or columns that aren't in dataset_context.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a landscape ecologist and connectivity analysis expert. If you can't map to a tool,
answer naturally in <=4 sentences and guide the user toward a supported app workflow when relevant.
""".strip()

_llm = UnifiedLLM()

# Try hard to extract a JSON object (handles ```json ... ```, leading prose, etc.)
_JSON_BLOCK_RE = re.compile(r"\\{[\\s\\S]*\\}")

def _extract_json(s: str):
    if not s: return None
    # code-fence strip
    if s.strip().startswith("```"):
        s = re.sub(r"^```(json)?\\s*|\\s*```$", "", s.strip(), flags=re.IGNORECASE)
    m = _JSON_BLOCK_RE.search(s)
    if not m: return None
    txt = m.group(0)
    try:
        return json.loads(txt)
    except Exception:
        return None

def ask_llm(chat_history, user_input, context=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context is not None:
        messages.append({"role": "system", "content": "dataset_context:\n" + json.dumps(context, ensure_ascii=False)})
    for m in chat_history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    resp = _llm.chat(messages, temperature=0.0, max_tokens=ANSWER_MAX_TOKENS, stream=False)

    # return (tool_dict_or_None, raw_text_output)
    call = _extract_json(resp)
    if isinstance(call, dict):
        return call, resp
    return None, resp


def ask_llm_stream(chat_history, user_input, context=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context is not None:
        messages.append({"role": "system", "content": "dataset_context:\n" + json.dumps(context, ensure_ascii=False)})
    for m in chat_history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    yield from _llm.stream_chat(messages, temperature=0.0, max_tokens=ANSWER_MAX_TOKENS)
