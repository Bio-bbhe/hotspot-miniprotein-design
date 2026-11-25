"""Bound the expensive embedding pool while retaining exploration."""

import random


def limit_embedding_pool(candidates, limit, exploration_fraction=0.2):
    limit = int(limit)
    if limit < 1:
        raise ValueError("embedding_pool_size must be positive")
    if len(candidates) <= limit:
        return candidates

    ranked = sorted(
        candidates,
        key=lambda item: item.get("mpnn_score", float("inf")),
    )
    exploit_count = max(
        1, min(limit, round(limit * (1.0 - exploration_fraction)))
    )
    selected = ranked[:exploit_count]
    selected_objects = {id(item) for item in selected}
    remaining = [
        item for item in candidates if id(item) not in selected_objects
    ]
    random.shuffle(remaining)
    return selected + remaining[:limit - len(selected)]
