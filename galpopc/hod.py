import numpy as np

def populate_galaxies(
    # Halo arrays
    h_mass_log10, h_x, h_y, h_z, h_velocity, h_sigma,
    # Subhalo arrays
    s_mass_log10, s_mass, s_host_velocity, s_n_particles,
    s_x, s_y, s_z, s_velocity,
    # HOD parameters
    lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s,
    # Other parameters
    rsd, Lmin, Lmax
):
    
    from . import galcore  # C extension for centrals + satellites to avoid circular init issues

    Nh = len(h_mass_log10)
    Ns = len(s_mass)

    Mcut = 10.0 ** lnMcut
    M1 = 10.0 ** lnM1

    # --- Allocate outputs for centrals ---
    cen_z = np.empty_like(h_z, dtype=np.float32)
    gal_mask = np.empty(Nh, dtype=np.uint8)

    # --- Compute centrals via C extension ---
    galcore.compute_centrals(
        h_mass_log10.astype(np.float32),
        h_z.astype(np.float32),
        h_velocity.astype(np.float32),
        h_sigma.astype(np.float32),
        float(lnMcut), float(sigma), float(alpha_c), np.random.randint(1 << 30),
        cen_z, gal_mask
    )

    # --- Allocate outputs for satellites ---
    sat_z = np.empty_like(s_z, dtype=np.float32)
    sat_mask = np.empty(Ns, dtype=np.uint8)

    # --- Compute satellites via C extension ---
    galcore.compute_satellites(
        s_mass_log10.astype(np.float32),
        s_mass.astype(np.float32),
        s_n_particles.astype(np.float32),
        s_z.astype(np.float32),
        s_velocity.astype(np.float32),
        s_host_velocity.astype(np.float32),
        float(lnMcut), float(sigma), float(M1), float(kappa),
        float(alpha), float(alpha_s),
        float(Lmin), float(Lmax),
        int(rsd), np.random.randint(1 << 30),
        sat_z, sat_mask
    )

    # --- Return selected galaxy positions ---
    return (
        np.concatenate([h_x[gal_mask], s_x[sat_mask]]),
        np.concatenate([h_y[gal_mask], s_y[sat_mask]]),
        np.concatenate([cen_z[gal_mask], sat_z[sat_mask]])
    )

