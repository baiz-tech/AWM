"""Config validation for the learned-token CLEVRER QA v2 protocol."""

from __future__ import annotations


PROTOCOLS = {"cjepa_native", "compatibility_stride2"}
TOKEN_SOURCES = {"online", "precomputed"}
FUTURE_GENERATION = {"none", "rollout", "ground_truth"}
REPRESENTATIONS = {"object_slots", "frame_tokens", "global_tokens"}
BACKENDS = {"native_aloe", "aloe_like_legacy"}


def resolve_qa_protocol(config):
    qa = config.get("qa") or {}
    protocol = str(qa.get("protocol", "compatibility_stride2"))
    if protocol not in PROTOCOLS:
        raise ValueError(f"unsupported qa.protocol={protocol!r}; expected one of {sorted(PROTOCOLS)}")
    data = qa.get("data") or {}
    token_source = str(data.get("token_source", "precomputed"))
    future_generation = str(data.get("future_generation", "none"))
    if token_source not in TOKEN_SOURCES:
        raise ValueError(f"unsupported qa.data.token_source={token_source!r}")
    if future_generation not in FUTURE_GENERATION:
        raise ValueError(f"unsupported qa.data.future_generation={future_generation!r}")
    representation = qa.get("representation") or {}
    representation_type = str(representation.get("type", "object_slots"))
    if representation_type not in REPRESENTATIONS:
        raise ValueError(f"unsupported qa.representation.type={representation_type!r}")
    backend = str(qa.get("backend", "aloe_like_legacy"))
    if backend not in BACKENDS:
        raise ValueError(f"unsupported qa.backend={backend!r}; expected one of {sorted(BACKENDS)}")
    learned_tokens = int(qa.get("learned_tokens_per_step", 16))
    if learned_tokens <= 0:
        raise ValueError("qa.learned_tokens_per_step must be positive")
    use_probes = bool(qa.get("use_probe_tokens", False))
    probe_adapter = (config.get("evaluation") or {}).get("probe_output_adapter")
    if use_probes and not probe_adapter:
        raise ValueError(
            "qa.use_probe_tokens=true requires evaluation.probe_output_adapter"
        )
    if protocol == "cjepa_native":
        temporal = qa.get("temporal") or {}
        if int(temporal.get("trajectory_length", 160)) != 160:
            raise ValueError("qa.protocol=cjepa_native requires qa.temporal.trajectory_length=160")
        if int(qa.get("video_len", 128)) != 128:
            raise ValueError("qa.protocol=cjepa_native requires qa.video_len=128")
    return {
        "protocol": protocol,
        "token_source": token_source,
        "future_generation": future_generation,
        "representation": representation_type,
        "backend": backend,
        "learned_tokens_per_step": learned_tokens,
        "use_probe_tokens": use_probes,
    }
