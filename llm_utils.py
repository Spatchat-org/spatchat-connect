# llm_utils.py  (spatchat-connect)
import os, sys, time, random, json, re
from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, ServiceUnavailableError

HF_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

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

class _SpacedCallLimiter:
    def __init__(self, min_interval_seconds: float): self.min, self._last = float(min_interval_seconds), 0.0
    def wait(self):
        now = time.monotonic(); gap = now - self._last
        if gap < self.min: time.sleep(self.min - gap)
        self._last = time.monotonic()

class UnifiedLLM:
    def __init__(self):
        model_or_url = (os.getenv("HF_ENDPOINT_URL") or HF_MODEL_DEFAULT).strip()
        token = (os.getenv("HF_TOKEN") or "").strip()
        self.hf = InferenceClient(model=model_or_url, token=token, timeout=300)

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tk = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tk:
            self.together = Together(api_key=tk)
            self._lim = _SpacedCallLimiter(100.0)  # ~0.6 QPM

    def _hf_chat(self, messages, max_tokens=256, temperature=0.0, stream=False):
        tries, delay, last = 3, 2.5, None
        for _ in range(tries):
            try:
                if hasattr(self.hf, "chat_completion"):
                    resp = self.hf.chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
                    return "".join(_delta_text(ch.choices[0].delta) for ch in resp) if stream else _choice_content(resp.choices[0])
                prompt = self._messages_to_prompt(messages)
                return self.hf.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature, stream=False, return_full_text=False)
            except Exception as e:
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

    def chat(self, messages, temperature=0.0, max_tokens=256, stream=False):
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as e:
            if not self.together: raise
            self._lim.wait()
            backoff=12.0
            for i in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=self.together_model, messages=messages,
                        temperature=temperature, max_tokens=max_tokens, stream=False
                    )
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError):
                    if i==3: raise
                    time.sleep(backoff + random.uniform(0,3)); backoff*=1.8

# ---- Connectivity-focused prompt (adds tool schema) -------------------------
SYSTEM_PROMPT = """
You are SpatChat, a connectivity analysis expert.

You are given a JSON object called dataset_context with facts about the currently loaded inputs
(e.g., whether a cost raster is present, nodes columns, small samples). When the user asks anything
about the data, answer STRICTLY using dataset_context. If the answer is not present or cannot be
derived from it, say you don't know and briefly suggest what the user could do to compute it.

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
- If you are not sure which tool to call, ask a brief clarifying question (<= 2 sentences).
- If no tool call is needed, answer naturally in <= 3 sentences.
- Do not fabricate numbers. Do not invent files or columns that aren't in dataset_context.
""".strip()

FALLBACK_PROMPT = """
You are SpatChat, a connectivity analysis expert. If you can't map to a tool,
answer naturally in <=3 sentences.
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

    resp = _llm.chat(messages, temperature=0.0, max_tokens=256, stream=False)

    # return (tool_dict_or_None, raw_text_output)
    call = _extract_json(resp)
    if isinstance(call, dict):
        return call, resp
    return None, resp
