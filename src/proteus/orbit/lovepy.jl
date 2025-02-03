include("$(@__DIR__)/../../../lovepy/TidalLoveNumbers.jl")
using .TidalLoveNumbers

# Get precision of Love number module (e.g., Float64, Double64, etc)
prec = TidalLoveNumbers.prec
precc = TidalLoveNumbers.precc
SPATIAL_RES::prec = 10.0

# Calculate heating from interior properties
function calculate_heating( omega::prec,
                            ecc::prec,
                            rho::Array{prec,1},
                            radius::Array{prec,1},
                            visc::Array{prec,1},
                            shear::Array{prec,1},
                            bulk::Array{prec,1};
                            ncalc::Int=2000
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

    # Subdivide input layers such that we have ~ncalc in total
    rr = expand_layers(r, nr=convert(Int,div(ncalc,length(η))))

    # Get gravity at each layer
    g = get_g(rr, ρ);

    # Get y-functions
    tidal_solution = calculate_y(rr, ρ, g, μc, κ)

    # Get k2 tidal Love Number (complex-valued)
    k2 = tidal_solution[5, end, end] - 1

    # Get bulk power output in watts
    power_blk = get_bulk_heating(tidal_solution, omega, R, ecc)

    # Get profile power output (W m-3), converted to W/kg
    power_prf = get_heating_profile(tidal_solution,
                                    rr, ρ, g, μc, κ,
                                    omega, ecc, res=SPATIAL_RES)
    power_prf = power_prf ./ ρ # Convert to mass heating rate (W/kg)

    # Edot_total = 0.0
    # for i in 1:length(r)-1
    #     layer_mass = 4/3 * π * (r[i+1]^3 - r[i]^3) * ρ[i]
    #     Edot_total += power_prf[i] * layer_mass
    # end
    # println(power_blk, Edot_total)

    return power_prf, power_blk, imag(k2)
end
