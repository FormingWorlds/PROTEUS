from __future__ import annotations


def none_if_none(val: str) -> str | None:
    """Convert 'none' string into None literal."""
    return None if val == 'none' else val

def zero_if_none(val: str) -> float:
    """Convert 'none' string into float zero."""
    print(val)
    return 0.0 if val == 'none' else val

def dict_replace_none(data):
    """
    Replace all None values with "none" strings.

    Adapted from:
    https://github.com/python-poetry/tomlkit/issues/240#issuecomment-1313283298
    """
    new_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict_replace_none(v)
        if v is not None:
            new_data[k] = v
        else:
            new_data[k] = "none"
    return new_data
