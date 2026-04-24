"""Session analysis tracking utilities for SpatChat.

This module provides lightweight helpers to record which analytical routines
were executed during a user session.  Each step captures the function that ran,
its key inputs, derived outputs, and an optional code snippet (function body)
for provenance.  The :mod:`app` module pulls the recorded steps into the export
bundle, but any other workflow can import and reuse these helpers without
creating circular dependencies.
"""
from __future__ import annotations

import inspect
import json
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Set
import os

_LOCK = threading.Lock()
_DEFAULT_SESSION = "__default__"
_STEPS_BY_SESSION: dict[str, List[dict]] = {}
_RECORDED_MODULES_BY_SESSION: dict[str, Set[str]] = {}
_SESSION = threading.local()


def _resolve_session_id(session_id: Optional[str] = None) -> str:
    if session_id:
        return str(session_id)
    return str(getattr(_SESSION, "session_id", _DEFAULT_SESSION))


def set_current_session(session_id: Optional[str]) -> Optional[str]:
    previous = getattr(_SESSION, "session_id", None)
    if session_id:
        _SESSION.session_id = str(session_id)
    elif hasattr(_SESSION, "session_id"):
        delattr(_SESSION, "session_id")
    return previous


def _json_default(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _json_safe(data: Any) -> Any:
    try:
        return json.loads(json.dumps(data, default=_json_default))
    except Exception:
        return data


def reset(session_id: Optional[str] = None) -> None:
    """Clear the recorded steps."""
    resolved = _resolve_session_id(session_id)
    with _LOCK:
        _STEPS_BY_SESSION.pop(resolved, None)
        _RECORDED_MODULES_BY_SESSION.pop(resolved, None)


def record_step(
    *,
    kind: str,
    function: str,
    inputs: dict,
    outputs: dict,
    code_obj: Optional[Any] = None,
    extra_snippets: Optional[Iterable[dict]] = None,
    session_id: Optional[str] = None,
) -> None:
    """Record that an analytical routine ran.

    Parameters
    ----------
    kind:
        Logical category, e.g. ``"least_cost_paths"``.
    function:
        Fully-qualified function name that was invoked.
    inputs / outputs:
        JSON-serialisable dictionaries summarising key arguments/results.
    code_obj:
        Optional live function object; if provided we include its source.
    extra_snippets:
        Optional iterable of ``{"label": str, "source": str}`` dicts to append.
    """
    resolved = _resolve_session_id(session_id)
    step = {
        "kind": kind,
        "function": function,
        "invoked_at": datetime.now(timezone.utc).isoformat(),
        "inputs": _json_safe(inputs),
        "outputs": _json_safe(outputs),
        "code_snippets": [],
    }
    if code_obj is not None:
        try:
            source = inspect.getsource(code_obj)
        except Exception as exc:  # pragma: no cover - capture warnings only
            print(f"Warning: could not capture source for {function}: {exc}")
        else:
            step["code_snippets"].append({"label": function, "source": source})

        module = inspect.getmodule(code_obj)
        module_source = None
        module_label = None
        module_key = None
        if module is not None:
            module_label = f"{module.__name__} module"
            module_file = getattr(module, "__file__", None)
            if module_file:
                module_key = os.path.abspath(module_file)
            else:
                module_key = module.__name__
            recorded_modules = _RECORDED_MODULES_BY_SESSION.setdefault(resolved, set())
            if module_key not in recorded_modules:
                try:
                    module_source = inspect.getsource(module)
                except (OSError, TypeError):
                    if module_file and os.path.exists(module_file):
                        try:
                            with open(module_file, "r", encoding="utf-8") as handle:
                                module_source = handle.read()
                        except Exception as exc:  # pragma: no cover
                            print(f"Warning: could not read module file for {module.__name__}: {exc}")
                except Exception as exc:  # pragma: no cover
                    print(f"Warning: could not capture source for module {module.__name__}: {exc}")
        if module_source:
            step["code_snippets"].append({"label": module_label, "source": module_source})
            if module_key:
                _RECORDED_MODULES_BY_SESSION.setdefault(resolved, set()).add(module_key)

    if extra_snippets:
        for snippet in extra_snippets:
            if not snippet:
                continue
            label = snippet.get("label", "analysis snippet")
            source = snippet.get("source", "")
            step["code_snippets"].append({"label": label, "source": source})

    with _LOCK:
        _STEPS_BY_SESSION.setdefault(resolved, []).append(step)


def get_steps(session_id: Optional[str] = None) -> List[dict]:
    """Return a shallow copy of the recorded steps."""
    resolved = _resolve_session_id(session_id)
    with _LOCK:
        return list(_STEPS_BY_SESSION.get(resolved, ()))


__all__ = ["record_step", "get_steps", "reset", "set_current_session"]
