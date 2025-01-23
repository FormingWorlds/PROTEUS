include("$(@__DIR__)/../../../lovepy/TidalLoveNumbers.jl")
using .TidalLoveNumbers

# Get precision of Love number module (e.g., Float64, Double64, etc)
prec = TidalLoveNumbers.prec
precc = TidalLoveNumbers.precc

# Calculate heating from interior properties
function calculate_heating( omega::Float64,
                            ecc::Float64,
                            rho::Array{Float64,1},
                            radius::Array{Float64,1},
                            visc::Array{Float64,1},
                            shear::Array{Float64,1},
                            bulk::Array{Float64,1}
                              )::Tuple{Array{Float64,1},Float64,Float64}

    # Internal structure arrays.
    # First element is the innermost layer, last element is the outermost layer
    ρ = convert(Vector{prec}, rho)
    r = convert(Vector{prec}, radius)
    η = convert(Vector{prec}, visc)
    μ = convert(Vector{precc},shear)
    κ = convert(Vector{prec}, bulk)

    # Complex shear modulus for a Maxwell material. Change this for different rheologies.
    μc = 1im * μ*omega ./ (1im*omega .+ μ./η)

    # Outer radius
    R = r[end]

    # Subdivide layers
    rr = expand_layers(r, nr=10)

    # Get gravity at each layer
    g = get_g(rr, ρ);

    # Get y-functions
    tidal_solution = calculate_y(rr, ρ, g, μc, κ)

    # Get k2 tidal Love Number (complex-valued)
    k2 = tidal_solution[5, end, end] - 1

    # Get bulk power output in watts
    power_blk = get_bulk_heating(tidal_solution, omega, R, ecc)

    # Get profile power output (W m-3), converted to W/kg
    power_prf = get_heating_profile(tidal_solution, rr, ρ, g, μc, κ, omega, ecc, res=10.0)
    power_prf = power_prf ./ ρ # Convert to mass heating rate (W/kg)

    # Edot_total = 0.0
    # for i in 1:length(r)-1
    #     layer_mass = 4/3 * π * (r[i+1]^3 - r[i]^3) * ρ[i]
    #     Edot_total += power_prf[i] * layer_mass
    # end
    # println(power_blk, Edot_total)

    return power_prf, power_blk, imag(k2)
end
