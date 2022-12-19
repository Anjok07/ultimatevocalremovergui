from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

def atan2(y, x):
    r"""Element-wise arctangent function of y/x.
    Returns a new tensor with signed angles in radians.
    It is an alternative implementation of torch.atan2

    Args:
        y (Tensor): First input tensor
        x (Tensor): Second input tensor [shape=y.shape]

    Returns:
        Tensor: [shape=y.shape].
    """
    pi = 2 * torch.asin(torch.tensor(1.0))
    x += ((x == 0) & (y == 0)) * 1.0
    out = torch.atan(y / x)
    out += ((y >= 0) & (x < 0)) * pi
    out -= ((y < 0) & (x < 0)) * pi
    out *= 1 - ((y > 0) & (x == 0)) * 1.0
    out += ((y > 0) & (x == 0)) * (pi / 2)
    out *= 1 - ((y < 0) & (x == 0)) * 1.0
    out += ((y < 0) & (x == 0)) * (-pi / 2)
    return out


# Define basic complex operations on torch.Tensor objects whose last dimension
# consists in the concatenation of the real and imaginary parts.


def _norm(x: torch.Tensor) -> torch.Tensor:
    r"""Computes the norm value of a torch Tensor, assuming that it
    comes as real and imaginary part in its last dimension.

    Args:
        x (Tensor): Input Tensor of shape [shape=(..., 2)]

    Returns:
        Tensor: shape as x excluding the last dimension.
    """
    return torch.abs(x[..., 0]) ** 2 + torch.abs(x[..., 1]) ** 2


def _mul_add(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts.
    The result is added to the `out` tensor"""

    # check `out` and allocate it if needed
    target_shape = torch.Size([max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = out[..., 0] + (real_a * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (real_a * b[..., 1] + a[..., 1] * b[..., 0])
    else:
        out[..., 0] = out[..., 0] + (a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])
    return out


def _mul(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts
    can work in place in case out is a only"""
    target_shape = torch.Size([max(sa, sb) for (sa, sb) in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = real_a * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = real_a * b[..., 1] + a[..., 1] * b[..., 0]
    else:
        out[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        out[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return out


def _inv(z: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise multiplicative inverse of a Tensor with complex
    entries described through their real and imaginary parts.
    can work in place in case out is z"""
    ez = _norm(z)
    if out is None or out.shape != z.shape:
        out = torch.zeros_like(z)
    out[..., 0] = z[..., 0] / ez
    out[..., 1] = -z[..., 1] / ez
    return out


def _conj(z, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Element-wise complex conjugate of a Tensor with complex entries
    described through their real and imaginary parts.
    can work in place in case out is z"""
    if out is None or out.shape != z.shape:
        out = torch.zeros_like(z)
    out[..., 0] = z[..., 0]
    out[..., 1] = -z[..., 1]
    return out


def _invert(M: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Invert 1x1 or 2x2 matrices

    Will generate errors if the matrices are singular: user must handle this
    through his own regularization schemes.

    Args:
        M (Tensor): [shape=(..., nb_channels, nb_channels, 2)]
            matrices to invert: must be square along dimensions -3 and -2

    Returns:
        invM (Tensor): [shape=M.shape]
            inverses of M
    """
    nb_channels = M.shape[-2]

    if out is None or out.shape != M.shape:
        out = torch.empty_like(M)

    if nb_channels == 1:
        # scalar case
        out = _inv(M, out)
    elif nb_channels == 2:
        # two channels case: analytical expression

        # first compute the determinent
        det = _mul(M[..., 0, 0, :], M[..., 1, 1, :])
        det = det - _mul(M[..., 0, 1, :], M[..., 1, 0, :])
        # invert it
        invDet = _inv(det)

        # then fill out the matrix with the inverse
        out[..., 0, 0, :] = _mul(invDet, M[..., 1, 1, :], out[..., 0, 0, :])
        out[..., 1, 0, :] = _mul(-invDet, M[..., 1, 0, :], out[..., 1, 0, :])
        out[..., 0, 1, :] = _mul(-invDet, M[..., 0, 1, :], out[..., 0, 1, :])
        out[..., 1, 1, :] = _mul(invDet, M[..., 0, 0, :], out[..., 1, 1, :])
    else:
        raise Exception("Only 2 channels are supported for the torch version.")
    return out


# Now define the signal-processing low-level functions used by the Separator


def expectation_maximization(
    y: torch.Tensor,
    x: torch.Tensor,
    iterations: int = 2,
    eps: float = 1e-10,
    batch_size: int = 200,
):
    r"""Expectation maximization algorithm, for refining source separation
    estimates.

    This algorithm allows to make source separation results better by
    enforcing multichannel consistency for the estimates. This usually means
    a better perceptual quality in terms of spatial artifacts.

    The implementation follows the details presented in [1]_, taking
    inspiration from the original EM algorithm proposed in [2]_ and its
    weighted refinement proposed in [3]_, [4]_.
    It works by iteratively:

     * Re-estimate source parameters (power spectral densities and spatial
       covariance matrices) through :func:`get_local_gaussian_model`.

     * Separate again the mixture with the new parameters by first computing
       the new modelled mixture covariance matrices with :func:`get_mix_model`,
       prepare the Wiener filters through :func:`wiener_gain` and apply them
       with :func:`apply_filter``.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [4] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [5] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Args:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            initial estimates for the sources
        x (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2)]
            complex STFT of the mixture signal
        iterations (int): [scalar]
            number of iterations for the EM algorithm.
        eps (float or None): [scalar]
            The epsilon value to use for regularization and filters.

    Returns:
        y (Tensor): [shape=(nb_frames, nb_bins, nb_channels, 2, nb_sources)]
            estimated sources after iterations
        v (Tensor): [shape=(nb_frames, nb_bins, nb_sources)]
            estimated power spectral densities
        R (Tensor): [shape=(nb_bins, nb_channels, nb_channels, 2, nb_sources)]
            estimated spatial covariance matrices

    Notes:
        * You need an initial estimate for the sources to apply this
          algorithm. This is precisely what the :func:`wiener` function does.
        * This algorithm *is not* an implementation of the "exact" EM
          proposed in [1]_. In particular, it does compute the posterior
          covariance matrices the same (exact) way. Instead, it uses the
          simplified approximate scheme initially proposed in [5]_ and further
          refined in [3]_, [4]_, that boils down to just take the empirical
          covariance of the recent source estimates, followed by a weighted
          average for the update of the spatial covariance matrix. It has been
          empirically demonstrated that this simplified algorithm is more
          robust for music separation.

    Warning:
        It is *very* important to make sure `x.dtype` is `torch.float64`
        if you want double precision, because this function will **not**
        do such conversion for you from `torch.complex32`, in case you want the
        smaller RAM usage on purpose.

        It is usually always better in terms of quality to have double
        precision, by e.g. calling :func:`expectation_maximization`
        with ``x.to(torch.float64)``.
    """
    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape[:-1]
    nb_sources = y.shape[-1]

    regularization = torch.cat(
        (
            torch.eye(nb_channels, dtype=x.dtype, device=x.device)[..., None],
            torch.zeros((nb_channels, nb_channels, 1), dtype=x.dtype, device=x.device),
        ),
        dim=2,
    )
    regularization = torch.sqrt(torch.as_tensor(eps)) * (
        regularization[None, None, ...].expand((-1, nb_bins, -1, -1, -1))
    )

    # allocate the spatial covariance matrices
    R = [
        torch.zeros((nb_bins, nb_channels, nb_channels, 2), dtype=x.dtype, device=x.device)
        for j in range(nb_sources)
    ]
    weight: torch.Tensor = torch.zeros((nb_bins,), dtype=x.dtype, device=x.device)

    v: torch.Tensor = torch.zeros((nb_frames, nb_bins, nb_sources), dtype=x.dtype, device=x.device)
    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor

        # update the PSD as the average spectrogram over channels
        v = torch.mean(torch.abs(y[..., 0, :]) ** 2 + torch.abs(y[..., 1, :]) ** 2, dim=-2)

        # update spatial covariance matrices (weighted update)
        for j in range(nb_sources):
            R[j] = torch.tensor(0.0, device=x.device)
            weight = torch.tensor(eps, device=x.device)
            pos: int = 0
            batch_size = batch_size if batch_size else nb_frames
            while pos < nb_frames:
                t = torch.arange(pos, min(nb_frames, pos + batch_size))
                pos = int(t[-1]) + 1

                R[j] = R[j] + torch.sum(_covariance(y[t, ..., j]), dim=0)
                weight = weight + torch.sum(v[t, ..., j], dim=0)
            R[j] = R[j] / weight[..., None, None, None]
            weight = torch.zeros_like(weight)

        # cloning y if we track gradient, because we're going to update it
        if y.requires_grad:
            y = y.clone()

        pos = 0
        while pos < nb_frames:
            t = torch.arange(pos, min(nb_frames, pos + batch_size))
            pos = int(t[-1]) + 1

            y[t, ...] = torch.tensor(0.0, device=x.device, dtype=x.dtype)

            # compute mix covariance matrix
            Cxx = regularization
            for j in range(nb_sources):
                Cxx = Cxx + (v[t, ..., j, None, None, None] * R[j][None, ...].clone())

            # invert it
            inv_Cxx = _invert(Cxx)

            # separate the sources
            for j in range(nb_sources):

                # create a wiener gain for this source
                gain = torch.zeros_like(inv_Cxx)

                # computes multichannel Wiener gain as v_j R_j inv_Cxx
                indices = torch.cartesian_prod(
                    torch.arange(nb_channels),
                    torch.arange(nb_channels),
                    torch.arange(nb_channels),
                )
                for index in indices:
                    gain[:, :, index[0], index[1], :] = _mul_add(
                        R[j][None, :, index[0], index[2], :].clone(),
                        inv_Cxx[:, :, index[2], index[1], :],
                        gain[:, :, index[0], index[1], :],
                    )
                gain = gain * v[t, ..., None, None, None, j]

                # apply it to the mixture
                for i in range(nb_channels):
                    y[t, ..., j] = _mul_add(gain[..., i, :], x[t, ..., i, None, :], y[t, ..., j])

    return y, v, R


def wiener(
    targets_spectrograms: torch.Tensor,
    mix_stft: torch.Tensor,
    iterations: int = 1,
    softmask: bool = False,
    residual: bool = False,
    scale_factor: float = 10.0,
    eps: float = 1e-10,
):
    """Wiener-based separation for multichannel audio.

    The method uses the (possibly multichannel) spectrograms  of the
    sources to separate the (complex) Short Term Fourier Transform  of the
    mix. Separation is done in a sequential way by:

    * Getting an initial estimate. This can be done in two ways: either by
      directly using the spectrograms with the mixture phase, or
      by using a softmasking strategy. This initial phase is controlled
      by the `softmask` flag.

    * If required, adding an additional residual target as the mix minus
      all targets.

    * Refinining these initial estimates through a call to
      :func:`expectation_maximization` if the number of iterations is nonzero.

    This implementation also allows to specify the epsilon value used for
    regularization. It is based on [1]_, [2]_, [3]_, [4]_.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [4] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Args:
        targets_spectrograms (Tensor): spectrograms of the sources
            [shape=(nb_frames, nb_bins, nb_channels, nb_sources)].
            This is a nonnegative tensor that is
            usually the output of the actual separation method of the user. The
            spectrograms may be mono, but they need to be 4-dimensional in all
            cases.
        mix_stft (Tensor): [shape=(nb_frames, nb_bins, nb_channels, complex=2)]
            STFT of the mixture signal.
        iterations (int): [scalar]
            number of iterations for the EM algorithm
        softmask (bool): Describes how the initial estimates are obtained.
            * if `False`, then the mixture phase will directly be used with the
            spectrogram as initial estimates.
            * if `True`, initial estimates are obtained by multiplying the
            complex mix element-wise with the ratio of each target spectrogram
            with the sum of them all. This strategy is better if the model are
            not really good, and worse otherwise.
        residual (bool): if `True`, an additional target is created, which is
            equal to the mixture minus the other targets, before application of
            expectation maximization
        eps (float): Epsilon value to use for computing the separations.
            This is used whenever division with a model energy is
            performed, i.e. when softmasking and when iterating the EM.
            It can be understood as the energy of the additional white noise
            that is taken out when separating.

    Returns:
        Tensor: shape=(nb_frames, nb_bins, nb_channels, complex=2, nb_sources)
            STFT of estimated sources

    Notes:
        * Be careful that you need *magnitude spectrogram estimates* for the
        case `softmask==False`.
        * `softmask=False` is recommended
        * The epsilon value will have a huge impact on performance. If it's
        large, only the parts of the signal with a significant energy will
        be kept in the sources. This epsilon then directly controls the
        energy of the reconstruction error.

    Warning:
        As in :func:`expectation_maximization`, we recommend converting the
        mixture `x` to double precision `torch.float64` *before* calling
        :func:`wiener`.
    """
    if softmask:
        # if we use softmask, we compute the ratio mask for all targets and
        # multiply by the mix stft
        y = (
            mix_stft[..., None]
            * (
                targets_spectrograms
                / (eps + torch.sum(targets_spectrograms, dim=-1, keepdim=True).to(mix_stft.dtype))
            )[..., None, :]
        )
    else:
        # otherwise, we just multiply the targets spectrograms with mix phase
        # we tacitly assume that we have magnitude estimates.
        angle = atan2(mix_stft[..., 1], mix_stft[..., 0])[..., None]
        nb_sources = targets_spectrograms.shape[-1]
        y = torch.zeros(
            mix_stft.shape + (nb_sources,), dtype=mix_stft.dtype, device=mix_stft.device
        )
        y[..., 0, :] = targets_spectrograms * torch.cos(angle)
        y[..., 1, :] = targets_spectrograms * torch.sin(angle)

    if residual:
        # if required, adding an additional target as the mix minus
        # available targets
        y = torch.cat([y, mix_stft[..., None] - y.sum(dim=-1, keepdim=True)], dim=-1)

    if iterations == 0:
        return y

    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = torch.max(
        torch.as_tensor(1.0, dtype=mix_stft.dtype, device=mix_stft.device),
        torch.sqrt(_norm(mix_stft)).max() / scale_factor,
    )

    mix_stft = mix_stft / max_abs
    y = y / max_abs

    # call expectation maximization
    y = expectation_maximization(y, mix_stft, iterations, eps=eps)[0]

    # scale estimates up again
    y = y * max_abs
    return y


def _covariance(y_j):
    """
    Compute the empirical covariance for a source.

    Args:
        y_j (Tensor): complex stft of the source.
            [shape=(nb_frames, nb_bins, nb_channels, 2)].

    Returns:
        Cj (Tensor): [shape=(nb_frames, nb_bins, nb_channels, nb_channels, 2)]
            just y_j * conj(y_j.T): empirical covariance for each TF bin.
    """
    (nb_frames, nb_bins, nb_channels) = y_j.shape[:-1]
    Cj = torch.zeros(
        (nb_frames, nb_bins, nb_channels, nb_channels, 2),
        dtype=y_j.dtype,
        device=y_j.device,
    )
    indices = torch.cartesian_prod(torch.arange(nb_channels), torch.arange(nb_channels))
    for index in indices:
        Cj[:, :, index[0], index[1], :] = _mul_add(
            y_j[:, :, index[0], :],
            _conj(y_j[:, :, index[1], :]),
            Cj[:, :, index[0], index[1], :],
        )
    return Cj
