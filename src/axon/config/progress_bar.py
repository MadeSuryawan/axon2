"""Main source of tqdm progress bar."""

from typing import Any

from tqdm import tqdm


def tqdm_args(desc: str = "", total: int = 0, *, scale: bool = True) -> dict[str, Any]:
    bar_format = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}, {elapsed}<{remaining}, "
        "{rate_fmt} {postfix}"
    )

    return {
        "desc": desc,
        "total": total,
        "unit_scale": scale,
        "ncols": 100,
        "colour": "#6bedac",
        "unit": "B",
        "disable": not total,
        "bar_format": bar_format,
        "miniters": 0,
        "mininterval": 0.1,
        "unit_divisor": 1024,
    }


def p_bar(desc: str = "", total: int = 0, *, scale: bool = False) -> tqdm:
    return tqdm(**tqdm_args(desc, total=total, scale=scale))


def reset_pbar(bar: tqdm, count: int, desc: str, *, scale: bool = False) -> tqdm:
    """Reset tqdm progress bar for re-use."""

    bar.reset()
    bar.total = count
    bar.set_description_str(desc)
    bar.unit_scale = scale
    return bar
