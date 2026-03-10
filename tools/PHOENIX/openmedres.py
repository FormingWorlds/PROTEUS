#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("fits_path")
    p.add_argument("--out")
    args = p.parse_args()

    with fits.open(Path(args.fits_path), memmap=True) as hdul:
        hdu = hdul[0]
        hdr = hdu.header
        data = np.asarray(hdu.data, dtype=np.float64).ravel()

        n = data.size
        i = np.arange(1, n + 1, dtype=np.float64)

        crval1 = float(hdr.get("CRVAL1", 0.0))
        crpix1 = float(hdr.get("CRPIX1", 1.0))
        cdelt1 = float(hdr.get("CDELT1", hdr.get("CD1_1", 1.0)))
        ctype1 = str(hdr.get("CTYPE1", "WAVE")).upper()

        print(f"CRVAL1: {crval1}, CRPIX1:{crpix1}, CRPIX1:{cdelt1}, CRPIX1:{ctype1}")

        if "LOG" in ctype1:
            lam_A = np.exp(crval1 + (i - crpix1) * cdelt1)
        else:
            lam_A = crval1 + (i - crpix1) * cdelt1

        wave_nm = lam_A * 0.1 # angstrom -> nm
        flux_nm = data * 1e-7 # erg/s/cm^2/cm -> per nm

        if args.out:
            np.savetxt(
                Path(args.out),
                np.column_stack([wave_nm, flux_nm]),
                fmt="%.9f\t%.6e",
                header="wavelength (nm)\tflux (erg/cm^2/s/nm)",
                comments="# ",
            )
            print(f'Saved file to {Path(args.out)}.')
        else:
            k = min(5, n)
            idx = np.linspace(0, n - 1, num=k, dtype=int)
            for j in idx:
                print(f"{wave_nm[j]:.6f}\t{flux_nm[j]:.6e}")

if __name__ == "__main__":
    main()
