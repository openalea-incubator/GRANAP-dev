"""Schema-equivalence guard for the parameter-schema unification.

The parameter classes in ``input_data.py`` are being de-duplicated (shared bases /
mixins for the xylem / phloem / cambium / planttype / stomata / mesophyll / ICS /
aerenchyma families).  Consumers (``build_bundle``, the organ classes, ``from_xml``)
read parameters off plain dicts via ``.get(name, default)``, so the refactor is
**schema-only**: it is correct iff every preset's ``to_dict_list()`` output is
unchanged — same param blocks, same field names, same values, same order.

This test freezes that output as a golden JSON (``golden/param_dicts.json``) and
asserts it never drifts.  Regenerate deliberately with::

    python test_param_schema_equivalence.py --update

(only when a *default* is intentionally changed, never to paper over a refactor).

Comparison is **field-order-insensitive within each param block** (each block is
compared as a name-keyed dict) but preserves block sequence: consumers read every
field by name, so unifying two classes onto a shared base — which naturally
reorders fields — is a no-op to them and must pass, while any changed *key* or
*value* is a real, caught regression.
"""

import json
import os

from openalea.granap.input_data import OrganInputData

PRESETS = [
    "for_root", "for_dicot_root", "for_woody_dicot", "for_dicot_secondary",
    "for_monocot_stem", "for_dicot_stem", "for_dicot_stem_continuous",
    "for_needle", "for_monocot_leaf", "for_dicot_leaf", "for_layer",
]

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "param_dicts.json")


def _emit() -> dict:
    """Every preset -> its ``to_dict_list()`` as a list of name-keyed dicts (the exact
    fields/values consumers read; within-block field order is not part of the contract)."""
    out = {}
    for name in PRESETS:
        data = getattr(OrganInputData, name)()
        out[name] = [dict(block) for block in data.to_dict_list()]
    return out


def _norm(obj):
    """JSON-normalise for a value-only compare (tuples -> lists, as JSON round-trips)."""
    if isinstance(obj, (list, tuple)):
        return [_norm(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _norm(v) for k, v in obj.items()}
    return obj


def _load() -> dict:
    with open(_GOLDEN) as f:
        return json.load(f)


def test_preset_param_dicts_unchanged():
    current = _norm(_emit())
    golden = _norm(_load())
    assert set(current) == set(golden), "preset set changed"
    for preset in golden:
        cur, gold = current[preset], golden[preset]
        # Same number and order of param blocks (block sequence is stable).
        assert len(cur) == len(gold), (
            f"{preset}: block count changed {len(gold)} -> {len(cur)}"
        )
        for i, (cb, gb) in enumerate(zip(cur, gold)):
            # Each block compared as a dict: key set + values, order-insensitive.
            assert cb == gb, (
                f"{preset}[block {i}, name={gb.get('name')!r}]: fields/values differ "
                f"from golden.\n  added/changed: "
                f"{ {k: cb.get(k) for k in cb if cb.get(k) != gb.get(k)} }\n"
                f"  removed: { {k: gb[k] for k in gb if k not in cb} }"
            )


def _write_golden():
    os.makedirs(os.path.dirname(_GOLDEN), exist_ok=True)
    with open(_GOLDEN, "w") as f:
        json.dump(_emit(), f, indent=1, default=str)
    print(f"wrote {_GOLDEN}")


if __name__ == "__main__":
    import sys
    if "--update" in sys.argv:
        _write_golden()
    else:
        test_preset_param_dicts_unchanged()
        print("OK — matches golden")
