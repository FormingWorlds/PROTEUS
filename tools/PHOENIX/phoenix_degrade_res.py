#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.ndimage import gaussian_filter1d
from spectres import spectres  # Might have to install spectres package


def compute_sigma_and_new_grid(example_file, R_native, R_target, oversamp):
    """
    From spectrum:
    - measure dln(lambda) on the originalgrid
    - compute Gaussian sigma in pixels to degrade R_native to R_target
    - build a new log-lambda grid with desired oversampling rate:
        R_samp_target = oversamp * R_target
        dln(lambda)_new = 1 / R_samp_target
    """
    with open(example_file) as f:
        first = f.readline()
    delim = "\t" if "\t" in first else None
    data = np.loadtxt(example_file, comments="#", delimiter=delim)

    lam = data[:, 0].astype(float)
    lnlam = np.log(lam)
    dln_old = np.median(np.diff(lnlam))

    if R_target >= R_native:
        raise ValueError(f"R_target ({R_target}) must be < R_native ({R_native})")

    #### Compute Gaussian kernel width in pixels for R_native -> R_target
    # Gaussian: FWHM = 2 * sqrt(2 ln 2) * sigma
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

    sigma_ln_native = 1.0 / (R_native * fwhm_factor)
    sigma_ln_target = 1.0 / (R_target * fwhm_factor)

    # Convolution of Gaussians: sigma_target^2 = sigma_native^2 + sigma_kernel^2
    sigma_ln_kernel2 = sigma_ln_target**2 - sigma_ln_native**2
    if sigma_ln_kernel2 <= 0:
        raise ValueError("Computed kernel sigma^2 <= 0; check R_native and R_target")

    sigma_ln_kernel = np.sqrt(sigma_ln_kernel2)
    sigma_pix = sigma_ln_kernel / dln_old  # sigma in pixels on the original grid

    #### Build target log-lambda grid with oversampling = oversamp
    # R_samp_target = oversamp * R_target
    R_samp_target = oversamp * R_target
    dln_new = 1.0 / R_samp_target      # exact dln(lambda) for target sampling R

    ln_min = lnlam.min()
    ln_max = lnlam.max()
    n_new = int(np.floor((ln_max - ln_min) / dln_new)) + 1
    ln_new = ln_min + dln_new * np.arange(n_new)
    lam_new = np.exp(ln_new)

    return sigma_pix, dln_old, dln_new, lam_new


def process_one_file(infile, outdir, sigma_pix, lam_new, R_native, R_target, oversamp, overwrite=False):
    base = os.path.basename(infile)
    root, ext = os.path.splitext(base)
    out_name = f"{root}_R{int(R_target):05d}_os{oversamp:.1f}.txt"
    outfile = os.path.join(outdir, out_name)

    if (not overwrite) and os.path.exists(outfile):
        return f"Skipping {outfile} (exists)"

    # Load data
    with open(infile) as f:
        first = f.readline()
    delim = "\t" if "\t" in first else None
    data = np.loadtxt(infile, comments="#", delimiter=delim)

    lam = data[:, 0].astype(float)
    flux = data[:, 1].astype(float)

    #### Degrade true resolution (Gaussian in ln(lambda) -> Gaussian in pixels here)
    flux_smooth = gaussian_filter1d(flux, sigma_pix, mode="nearest")

    #### Flux-conserving resample onto new log-lambda grid
    lam_min = lam.min()
    lam_max = lam.max()
    mask = (lam_new >= lam_min) & (lam_new <= lam_max)
    lam_new_masked = lam_new[mask]

    flux_rebinned = spectres(lam_new_masked, lam, flux_smooth, fill=0.0)

    header = (
        "wavelength (nm)\tflux (erg/cm^2/s/nm)\n"
        f"Degraded from R_native={R_native:.0f} to R_target={R_target:.0f}, "
        f"Gaussian sigma_pix≈{sigma_pix:.3f}, "
        f"target oversampling={oversamp:.1f} px/FWHM "
        f"(flux-conserving resampling onto log-lambda grid)"
    )
    np.savetxt(
        outfile,
        np.column_stack([lam_new_masked, flux_rebinned]),
        fmt="%.9f\t%.6e",
        header=header,
        comments="# ",
    )
    return f"saved {outfile}"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Degrade true spectral resolution (Gaussian in ln lambda) and then flux-conservingly resample to a new log-lambda grid with a specified oversampling (pixels per FWHM)."
        )
    )
    ap.add_argument(
        "--indir", default="out/",
        help="Input directory with log-lambda spectra (default: out/)",
    )
    ap.add_argument(
        "--outdir", default=None,
        help="Output directory (default: out_R<Rtarget>_os<oversamp>/)",
    )
    ap.add_argument(
        "--Rnative", type=float, default=10000.0,
        help="Native intrinsic resolving power (default: 10000)",
    )
    ap.add_argument(
        "--Rtarget", type=float, required=True,
        help="Target intrinsic resolving power (must be < Rnative)",
    )
    ap.add_argument(
        "--oversamp", type=float, default=3.0,
        help="Desired oversampling in pixels per FWHM at R_target (default: 3.0).",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files in outdir if they exist.",
    )
    ap.add_argument(
        "--nproc", type=int, default=4,
        help="Number of parallel processes (default: 4)",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.txt")))
    if not files:
        print(f"No .txt files found in {args.indir}")
        return

    if args.outdir is None:
        args.outdir = f"out_R{int(args.Rtarget):05d}_os{int(args.oversamp)}"
    os.makedirs(args.outdir, exist_ok=True)

    # Compute sigma_pix and the new wavelength grid once, from the first spectrum
    sigma_pix, dln_old, dln_new, lam_new = compute_sigma_and_new_grid(
        files[0],
        R_native=args.Rnative,
        R_target=args.Rtarget,
        oversamp=args.oversamp,
    )
    print(
        f"Example file: {os.path.basename(files[0])}\n"
        f"  original dln ~ {dln_old:.3e} (R_samp,old ~ {1.0/dln_old:.1f})\n"
        f"  new dln      ~ {dln_new:.3e} (R_samp,new ~ {1.0/dln_new:.1f})\n"
        f"  Gaussian sigma_pix ~ {sigma_pix:.3f} "
        f"(R_native={args.Rnative:.0f} -> R_target={args.Rtarget:.0f}), "
        f"oversampling target = {args.oversamp:.1f} px/FWHM"
    )

    worker = partial(
        process_one_file,
        outdir=args.outdir,
        sigma_pix=sigma_pix,
        lam_new=lam_new,
        R_native=args.Rnative,
        R_target=args.Rtarget,
        oversamp=args.oversamp,
        overwrite=args.overwrite,
    )

    nproc = args.nproc or cpu_count()
    print(f"Processing {len(files)} files using {nproc} processes...\n")

    with Pool(processes=nproc) as pool:
        for msg in pool.imap_unordered(worker, files):
            if msg:
                print(msg)


if __name__ == "__main__":
    main()
