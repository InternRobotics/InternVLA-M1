from __future__ import annotations

from types import SimpleNamespace

class NamespaceWithGet(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

def dict_to_namespace(d): 
    if isinstance(d, dict):
        return NamespaceWithGet(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d