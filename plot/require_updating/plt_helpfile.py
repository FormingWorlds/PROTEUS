import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dat_dir = "/Users/tim/runs/coupler_tests/200914/1531/"

df = pd.read_csv(dat_dir+"runtime_helpfile.csv", sep=" ")

print(df)

f, (ax1, ax2) = plt.subplots(2, 1)

df_int = df.loc[df['Input']=='Interior']
df_atm = df.loc[df['Input']=='Atmosphere']

ax1.plot(df_atm["T_surf"], color="blue", label="Atmosphere")
ax1.plot(df_int["T_surf"], color="red", label="Interior")

ax2.plot(df_atm["Heat_flux"], ls="--", color="blue", label="Atmosphere")
ax2.plot(df_int["Heat_flux"], ls="--", color="red", label="Interior")

# print(df_atm["T_surf"].tolist())
# print(df_atm["Heat_flux"].tolist())

# print(df_int["T_surf"].tolist())
# print(df_int["Heat_flux"].tolist())

ax1.set_ylabel("Surface temperature (K)")

ax2.set_yscale("log")
ax2.set_ylabel("Heat flux (W m-2)")

plt.legend()

plt.show()