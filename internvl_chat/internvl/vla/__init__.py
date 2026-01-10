__all__ = ["get_vla_dataset_and_collator"]


def __getattr__(name: str):
    # Keep `internvl.vla` import lightweight: `materialize` pulls optional deps (e.g. prismatic).
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator

        return get_vla_dataset_and_collator
    raise AttributeError(name)
