using SpecialFunctions
using Random

function populate_galaxies(
    # Halo arrays
    h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
    # Subhalo arrays
    s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
    # HOD parameters
    lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s, 
    # Other parameters 
    rsd, Lmin, Lmax,
)

    # Pre-calculate lengths
    Nh = length(h_mass)
    Ns = length(s_mass)

    # Pre-calculate parameters
    Mcut = 10.0^lnMcut
    M1 = 10.0^lnM1
    Lbox = Lmax - Lmin

    # Halo probability
    p_cen = 0.5.*erfc(lnMcut .- log10.(h_mass)) ./ sqrt(2) ./ sigma
    Rh = rand(Nh)
    gal_mask = Rh .< p_cen

    # Apply RSD
    cen_z = h_z
    if rsd
        cen_z .+= h_velocity .+ alpha_c.*h_sigma.*rand(Nh)
        cen_z[cen_z .> Lmax] .-= Lbox
        cen_z[cen_z .< Lmin] .+= Lbox
    end

    # Satellite probability
    n_cen_sat = 0.5.*erfc(lnMcut .- log10.(s_mass)) ./ sqrt(2) ./ sigma
    n_sat = ((s_mass .- kappa.*Mcut)./M1).^alpha.*n_cen_sat
    p_sat = n_sat./s_n_particles
    Rs = rand(Ns)
    sat_mask = Rs .< p_sat
    
    # Apply RSD to satellites
    sat_z = s_z
    if rsd
        sat_z .+= s_host_velocity .+ alpha_s.*(s_velocity .- s_host_velocity)
        sat_z[sat_z .> Lmax] .-= Lbox
        sat_z[sat_z .< Lmin] .+= Lbox
    end
    
    return (h_x, h_y, gal_z, s_x, s_y, sat_z)
end