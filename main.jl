using PyCall
using LinearAlgebra

if !@isdefined pyscf
    pyscf = pyimport("pyscf")
    einsum = pyscf.lib.einsum
end

const F = Float64
const V = Vector{Float64}
const M = Matrix{Float64}
const M4 = Array{Float64,4}

mutable struct QED_CCSD_PARAMS
    mol::PyObject
    nao::Int
    nocc::Int
    nvir::Int

    ω::F
    coup::F
    pol::V

    C::M
    t1::M
    t2::M4
    s1::M
    s2::M4
    γ::F

    u2::M4
    v2::M4

    t1_bar::M
    t2_bar::M4
    s1_bar::M
    s2_bar::M4
    γ_bar::F

    t2_t::M4
    s2_t::M4

    u2_t::M4
    v2_t::M4

    x::M
    y::M

    D_e::M
    D_ep::M
    D_p1::F
    D_p2::F
    d::M4

    κ_bar::M
end

include("integrals.jl")
include("get_matrix.jl")
include("get_vector.jl")
include("run_eT_tor.jl")
include("energy.jl")
include("density.jl")
include("kappa.jl")
include("numgrad.jl")
include("gradient.jl")
include("dipole.jl")
include("get_es.jl")
include("run_eT_clean.jl")

function QED_CCSD_PARAMS(mol, ω, coup, pol, out_name)
    pol *= coup

    C = get_matrix("MO-coeffs", out_name)

    t1 = get_matrix("T1-AMPLITUDES", out_name)

    t2 = unpack_t2(mol, get_vector("T2-AMPLITUDES", out_name))

    u2 = construct_u2(t2)

    s1 = get_matrix("S1-AMPLITUDES", out_name)

    s2 = unpack_t2(mol, get_vector("S2-AMPLITUDES", out_name))

    v2 = construct_u2(s2)

    γ = get_vector("Photon amplitudes", out_name)[1]

    x, y = construct_t1_transformation(mol, t1)

    t1_bar = get_matrix("T1-MULTIPLIERS", out_name)

    t2_bar = unpack_t2(mol, get_vector("T2-MULTIPLIERS", out_name))

    s1_bar = get_matrix("S1-MULTIPLIERS", out_name)

    s2_bar = unpack_t2(mol, get_vector("S2-MULTIPLIERS", out_name))

    γ_bar = get_vector("Photon multipliers", out_name)[1]

    t2_t = t2_tilde(t2_bar)
    s2_t = t2_tilde(s2_bar)

    u2_t = construct_u2(t2_t)
    v2_t = construct_u2(s2_t)

    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    QED_CCSD_PARAMS(
        mol, nao, nocc, nao - nocc, ω, coup, pol,
        C, t1, t2, s1, s2, γ,
        u2, v2,
        t1_bar, t2_bar, s1_bar, s2_bar, γ_bar,
        t2_t, s2_t, u2_t, v2_t,
        x, y,
        zeros(0, 0), zeros(0, 0), 0.0, 0.0, zeros(0, 0, 0, 0),
        zeros(0, 0)
    )
end

function QED_CCSD_PARAMS(mol, ω, coup, pol)
    QED_CCSD_PARAMS(mol, ω, coup, pol, run_eT_tor(mol, ω, coup, pol))
end

function test()
    mol = pyscf.M(atom="""
 H          0.86681        0.60144        5.00000
 H         -0.86681        0.60144        5.00000
 O          0.00000       -0.07579        5.00000
 He         0.10000       -0.02000        7.53000
""", basis="STO-3G")

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)
    # p = @time QED_CCSD_PARAMS(mol, ω, coup, pol, out_name)

    # E_t1 = get_energy_ccsd_t1_transformed(mol, pol, C, t2, s1, γ, ω, x, y)

    # display(D_d - D_e)

    # E_Λ = @time get_energy_t1_density(mol, pol, C, ω, x, y,
    #     D_e, D_ep, D_p1, D_p2, d)

    # E_t1 - E_Λ

    D_eT = get_matrix("DENSITY-1e", out_name)

    @time one_electron_density(p)

    @show maximum(abs, D_eT - p.D_e)

    @show tr(p.D_e)

    @time one_electron_one_photon(p)

    @time photon_density1(p)

    @time photon_density2(p)

    @time two_electron_density(p)

    @show maximum(abs, p.d .- PermutedDimsArray(p.d, (3, 4, 1, 2)))

    D_1e_2e = one_electron_from_two_electron(p.mol, p.d)

    @show maximum(abs, D_1e_2e - p.D_e)

    @time get_energy_ccsd(p)

    E_t1 = @time get_energy_ccsd_t1_transformed(p)

    E_Λ = @time get_energy_t1_density(p)

    @show E_t1 - E_Λ

    @time solve_kappa_bar(p)

    @time get_total_dipole(p)

    # i = 1
    # q = 1

    # g_a = @time get_gradient(p, i, q, 0.00001)

    # g_n = @time numgrad4(mol, ω, coup, pol, i, q, 0.001)

    # @show g_a g_n
end

function test_numgrad()
    mol = pyscf.M(atom="""
 H          0.86681        0.60144        5.00000
 H         -0.86681        0.60144        5.00000
 O          0.00000       -0.07579        5.00000
 He         0.10000       -0.02000        7.53000
""", basis="STO-3G")

    ω = 0.5
    # coup = 0.1
    coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    i = 4
    q = 2

    g2 = @time numgrad2(mol, ω, coup, pol, i, q, 0.001)
    g4 = @time numgrad4(mol, ω, coup, pol, i, q, 0.001)

    @show g2 g4
end
