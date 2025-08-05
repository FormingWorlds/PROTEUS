# Orbit evolution module

from proteus.utils.constants import const_G
import numpy as np

def de_dt(a, e, params):
    Imk2, Mst, G, Rpl, Mpl = params
    return (21/2) * Imk2 * Mst**1.5 * G**0.5 * Rpl**5 / (Mpl * a**6.5) * e

def da_dt(a, e, params):
    return 2 * a * e * de_dt(a, e, params)

def rk4_step_time(a, e, h, params):
    # k1
    k1_e = de_dt(a, e, params)
    k1_a = da_dt(a, e, params)

    # k2
    k2_e = de_dt(a + h*k1_a/2, e + h*k1_e/2, params)
    k2_a = da_dt(a + h*k1_a/2, e + h*k1_e/2, params)

    # k3
    k3_e = de_dt(a + h*k2_a/2, e + h*k2_e/2, params)
    k3_a = da_dt(a + h*k2_a/2, e + h*k2_e/2, params)

    # k4
    k4_e = de_dt(a + h*k3_a, e + h*k3_e, params)
    k4_a = da_dt(a + h*k3_a, e + h*k3_e, params)

    # Combine
    e_next = e + (k1_e + 2*k2_e + 2*k3_e + k4_e) / 6
    a_next = a + (k1_a + 2*k2_a + 2*k3_a + k4_a) / 6

    return a_next, e_next

def find_ae(a0, e0, params, dt, N):
    h = dt / N
    t = h

    a = np.zeros(N)
    e = np.zeros(N)

    a[0] = a0
    e[0] = e0

    while t < dt:
        a, e = rk4_step_time(t, a, e, h, params)
        t += h

    return a[-1], e[-1]

def update_orbit(hf_row:dict, config:Config, dirs:dict):

    Imk2 = hf_row["Imk2"]

    Rpl = hf_row["R_int"]
    Mpl = hf_row["M_int"]
    Mst = hf_row["M_star"]

    sma = hf_row["semimajorax"]
    ecc = hf_row["eccentricity"]

    # Time step
    from proteus.interior.spider import get_all_output_times
    current_time = get_all_output_times(dirs["output"])[-1]

    if current_time <= 1:
        dt = 0

        # Set semimajor axis and eccentricity.
        hf_row["semimajorax"]  = config.orbit.semimajoraxis * AU
        hf_row["eccentricity"] = config.orbit.eccentricity

        return
    else:
        sim_time = get_all_output_times(dirs["output"])[-1]  # yr, as an integer value
        last_sim_time = get_all_output_times(dirs["output"])[-2]  # yr, as an integer value

        dt = sim_time - last_sim_time

    # Number of steps
    N = 10

    # Calculate planet orbit around star
    params = (Imk2, Mst, const_G, Rpl, Mpl)

    # Find new sma and ecc
    a, e = find_ae(sma, ecc, params dt, N)

    # Set semimajor axis and eccentricity.
    hf_row["semimajorax"]  = a
    hf_row["eccentricity"] = e

    return



