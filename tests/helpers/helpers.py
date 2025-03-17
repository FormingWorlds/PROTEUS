from __future__ import annotations

from pathlib import Path

from PIL import Image

PROTEUS_ROOT = Path(__file__).parents[2]
NEGLECT = ["N2_mol_atm", "N2_mol_total", "CH4_mol_atm", "CH4_mol_total", "CH4_kg_atm", "CH4_kg_total", "CH4_bar", "runtime"]

def resize_to_match(image1_path, image2_path):
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Resize img2 to match img1's size if they don't match
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    return img1, img2

def df_intersect(df1, df2):
    """
    Filter two dataframes to only the columns which they have in common.
    """

    # Intersection of columns
    inter = set(df1.columns).intersection(set(df2.columns))
    inter = list(inter)

    # Filter
    out1 = df1.loc[:, df1.columns.isin(inter)]
    out2 = df2.loc[:, df2.columns.isin(inter)]

    # Return
    return out1, out2
