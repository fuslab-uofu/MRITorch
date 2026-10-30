"""
Apodization windows for k-space filtering.

1D windows follow the MATLAB definitions used in the legacy pipelines
(`tukeywin`, `fermiwin`, `hann`). `separable_window` builds an N-D window from
1D windows (port of `ndseparablewindow`), and `apply_window` multiplies it
into the trailing dimensions of a k-space tensor.
"""
from __future__ import annotations
from typing import Callable, Sequence, Optional, Union

import math
import torch

WindowFn = Callable[..., torch.Tensor]


def tukey(n: int, r: float=0.5, dtype: torch.dtype=torch.float32, device=None) -> torch.Tensor:
    """
    Symmetric Tukey (tapered cosine) window, as MATLAB `tukeywin(n, r)`.

    Parameters
    ----------
    n : int
        Window length.
    r : float, optional
        Ratio of taper to total length, in [0, 1]. r <= 0 gives a rectangular
        window and r >= 1 gives a Hann window. Defaults to 0.5.
    """
    if n == 1:
        return torch.ones(1, dtype=dtype, device=device)
    if r <= 0:
        return torch.ones(n, dtype=dtype, device=device)
    r = min(r, 1.0)
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    per = r / 2
    tl = math.floor(per * (n - 1)) + 1
    th = n - tl
    w = torch.ones(n, dtype=torch.float64)
    w[:tl] = (1 + torch.cos(math.pi / per * (x[:tl] - per))) / 2
    w[th:] = (1 + torch.cos(math.pi / per * (x[th:] - 1 + per))) / 2
    return w.to(dtype=dtype, device=device)


def hann(n: int, param: Optional[float]=None, dtype: torch.dtype=torch.float32, device=None) -> torch.Tensor:
    """
    Symmetric Hann window, as MATLAB `hann(n)`.

    `param` is accepted for a uniform ``f(n, param)`` window signature and is
    ignored.
    """
    if n == 1:
        return torch.ones(1, dtype=dtype, device=device)
    k = torch.arange(n, dtype=torch.float64)
    w = 0.5 * (1 - torch.cos(2 * math.pi * k / (n - 1)))
    return w.to(dtype=dtype, device=device)


def fermi(n: int, r: float=0.1, dtype: torch.dtype=torch.float32, device=None) -> torch.Tensor:
    """
    Fermi window, port of MATLAB `fermiwin(n, r)`.

    Parameters
    ----------
    n : int
        Window length.
    r : float, optional
        Fraction of the half-width over which the window rolls off. Defaults
        to 0.1.
    """
    halfwidth = n // 2
    r_px = math.floor(halfwidth * r)
    if r_px == 0:
        return torch.ones(n, dtype=dtype, device=device)
    kT = r_px / 9.2  # speed of decay
    ipt = halfwidth - 4.6 * kT  # inflection point
    k = torch.arange(1, halfwidth + 1, dtype=torch.float64)
    w = 1 / (1 + torch.exp((k - ipt) / kT))
    parts = [w.flip(0), w] if 2 * halfwidth == n else [w.flip(0), torch.ones(1, dtype=w.dtype), w]
    return torch.cat(parts).to(dtype=dtype, device=device)


_WINDOWS: dict[str, WindowFn] = {"tukey": tukey, "hann": hann, "fermi": fermi}


def separable_window(
        window: Union[str, WindowFn],
        shape: Sequence[int],
        params: Optional[Sequence[Optional[float]]]=None,
        dtype: torch.dtype=torch.float32,
        device=None
        ) -> torch.Tensor:
    """
    N-D separable window as the outer product of 1D windows.

    Port of MATLAB `ndseparablewindow`. Dimensions of length 1 get a window
    of 1.

    Parameters
    ----------
    window : str or callable
        'tukey', 'hann', 'fermi', or a callable ``f(n, param)`` returning a
        1D tensor of length n.
    shape : sequence of int
        Window shape.
    params : sequence of float or None, optional
        Per-dimension window parameter. A ``None`` entry means no windowing
        along that dimension. If omitted, each window uses its default.

    Returns
    -------
    torch.Tensor
        Window of shape `shape`.
    """
    fn = _WINDOWS[window] if isinstance(window, str) else window
    if params is None:
        params = [...] * len(shape)  # sentinel: use the window's default
    if len(params) != len(shape):
        raise ValueError("params must have one entry per dimension in shape")
    w = torch.ones(shape, dtype=dtype, device=device)
    for d, (n, p) in enumerate(zip(shape, params)):
        if n == 1 or p is None:
            continue
        w1 = fn(n) if p is ... else fn(n, p)
        view = [1] * len(shape)
        view[d] = n
        w = w * w1.to(dtype=dtype, device=device).reshape(view)
    return w


def apply_window(
        ks: torch.Tensor,
        window: Union[str, WindowFn],
        params: Sequence[Optional[float]],
        ) -> torch.Tensor:
    """
    Multiply a separable window into the trailing ``len(params)`` dims of `ks`.

    Parameters
    ----------
    ks : torch.Tensor
        Centered k-space.
    window : str or callable
        See `separable_window`.
    params : sequence of float or None
        Per-dimension window parameter for the trailing dimensions; ``None``
        skips a dimension.
    """
    shape = ks.shape[ks.ndim - len(params):]
    real_dtype = ks.real.dtype if torch.is_complex(ks) else ks.dtype
    w = separable_window(window, shape, params, dtype=real_dtype, device=ks.device)
    return ks * w
