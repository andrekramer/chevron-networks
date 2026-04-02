from __future__ import annotations

from collections import defaultdict


def summarize_episode_infos(episode_infos: list[dict]) -> dict[str, float]:
    if not episode_infos:
        return {}
    accum = defaultdict(float)
    for info in episode_infos:
        accum["episodes"] += 1
        accum["ambiguous_episodes"] += float(info.get("was_ambiguous", False))
        accum["lure_triggered_rate"] += float(info.get("lure_triggered", False))
    total = accum.pop("episodes")
    return {key: value / total for key, value in accum.items()}
