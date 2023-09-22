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
include("run_eT_new.jl")
include("energy.jl")
include("density.jl")
include("kappa.jl")
include("numgrad.jl")
include("gradient.jl")
include("dipole.jl")
include("get_es.jl")
include("run_eT_clean.jl")
include("cholesky.jl")
include("omega.jl")

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
    QED_CCSD_PARAMS(mol, ω, coup, pol, run_eT_new(mol, ω, coup, pol))
end

function test()
    mol = pyscf.M(atom="""
 H          0.86681        0.60144        5.00000
 H         -0.86681        0.60144        5.00000
 O          0.00000       -0.07579        5.00000
 He         0.10000       -0.02000        7.53000
""", basis="STO-3G")


    mol = pyscf.M(atom="""
H    0.0 0.0 0.0
H    1.0 0.0 0.0
""", basis="cc-pvdz")

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

    D_ep_eT = get_matrix("Density one electron one boson", out_name)

    @time one_electron_density(p)

    @show maximum(abs, D_eT - p.D_e)

    @show tr(p.D_e)

    @time one_electron_one_photon(p)

    @show maximum(abs, D_ep_eT - p.D_ep)

    @time photon_density1(p)

    @time photon_density2(p)

    @show p.D_p2

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

function test_debug_hrsqk()
    mol = pyscf.M(atom="""
    H 0 0 0
    F 1 0 0
""", basis="cc-pvdz")

    ω = 0.5
    coup = 0.05
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    i = 2
    q = 2
    h = 1e-4

    h_ao = numgrad4_int(mol, "int1e_kin", i, q, h) +
           numgrad4_int(mol, "int1e_nuc", i, q, h)

    function make_h_qed(mol)
        d = get_qed_d(mol, p.pol, p.C)
        get_qed_h(mol, p.C, d)
    end

    h_mo = p.C' * h_ao * p.C

    display(h_mo)

    h_qed_mo = numgrad4_f(mol, make_h_qed, i, q, h)

    hder_eT = reshape(get_matrix("h_rs_qk", out_name), p.nao, p.nao, 3, mol.natm)
    hder_qed_eT = reshape(get_matrix("h_qed_rs_qk", out_name), p.nao, p.nao, 3, mol.natm)

    display(hder_eT[:, :, q, i])

    @show maximum(abs, h_mo - hder_eT[:, :, q, i])

    hder = hder_eT + hder_qed_eT

    @show maximum(abs, h_qed_mo - hder[:, :, q, i])
end

function test_debug_1e_grad()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.5 1 0
""", basis="cc-pvdz")

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    @time one_electron_density(p)
    @time one_electron_one_photon(p)
    @time photon_density1(p)
    @time photon_density2(p)

    i = 1
    q = 1
    h = 1e-4

    D_eT = get_matrix("Density", out_name)
    Dep_eT = get_matrix("Density one electron one boson", out_name)

    @show maximum(abs, p.D_e - D_eT)
    @show maximum(abs, p.D_ep - Dep_eT)

    function calc_1e(mol)
        get_1e_energy(mol, p.C, p)
    end

    numgrad4_f(mol, calc_1e, i, q, h)
end

function test_debug_2e_energy()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
""", basis="sto-3g")

    ω = 0.5
    # coup = 0.5
    coup = 0.5
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    one_electron_density(p)
    one_electron_one_photon(p)
    photon_density1(p)
    photon_density2(p)
    two_electron_density(p)

    i = 3
    q = 1
    h = 1e-4

    # function calc_1e(mol)
    #     get_1e_energy(mol, p.C, p)
    # end

    # numgrad4_f(mol, calc_1e, i, q, h)

    E1 = @show get_2e_energy(mol, p.C, p)

    L_J_pq = get_cholesky_vector_blocks("Ld_J", out_name, p)
    W_J_pq = get_cholesky_vector_blocks("W_J", out_name, p)

    d = get_qed_d(mol, p.pol, p.C)

    dt1 = t1_transform_1e(d, p.x, p.y)

    g = get_qed_g(mol, p.C, d)
    # g = get_mo_g(mol, p.C)
    g = t1_transform_2e(g, p.x, p.y)

    W_naive = @time compute_WL(p, L_J_pq)

    @show maximum(abs, W_J_pq - W_naive)

    E2 = @show einsum("Jpq,Jpq->", L_J_pq, W_naive) / 2

    @show E1 - E2

    g_pqrs = einsum("Jpq,Jrs->pqrs", L_J_pq, L_J_pq)

    @show maximum(abs, g - g_pqrs)
end

function test_debug_2e_grad()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.5 1 0
""", basis="cc-pvdz")

    display(mol.ao_labels())

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    one_electron_density(p)
    one_electron_one_photon(p)
    photon_density1(p)
    photon_density2(p)
    two_electron_density(p)

    i = 1
    q = 1
    h = 1e-4

    function calc_2e(mol)
        get_2e_energy(mol, p.C, p)
    end

    @show calc_2e(mol)

    numgrad4_f(mol, calc_2e, i, q, h)
end

function numgrad_LL(mol, ω, coup, pol, i, q, h)
    out_name = "tmp_eT/ccsd"

    function get_LL(mol)
        p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

        Qinv = get_matrix("Q_inv", out_name)

        L_J = get_cholesky_vector_blocks("L_J", out_name, p)

        Q = inv(Qinv)

        L_L = einsum("LJ,Jrs->Lrs", Q, L_J)
    end

    numgrad4_f(mol, get_LL, i, q, h)
end

function test_debug_2e_diff()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.5 1 0
""", basis="sto-3g")

    display(mol.ao_labels())

    ω = 1.0
    coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    i = 1
    q = 1
    h = 1e-4

    LL_grad = numgrad_LL(mol, ω, coup, pol, i, q, h)

    display(LL_grad)

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    LL_grad_eT_full = reshape(get_matrix("g_pqK_1der", out_name),
        size(LL_grad)..., 3, 3)

    LL_grad_eT = LL_grad_eT_full[:, :, :, q, i]

    # function calc_2e_int(mol)
    #     get_qed_g_ao(mol, p.pol)
    #     # mol.intor("int2e")
    # end

    # numgrad4_f(mol, calc_2e_int, i, q, h)[1, 1, 5:7, 5:7]

    # # calc_2e_int(mol)[1, 1, 5:7, 5:7]

    # W_J_pq = get_cholesky_vector_blocks("W_J", out_name, p)

    # g = einsum("Lrs,K")

    maximum(abs, LL_grad - LL_grad_eT ./ 2)
end

function get_omo_C(mol, C)
    S_ao = mol.intor("int1e_ovlp")

    S_mo = C' * S_ao * C

    C * S_mo^(-1 // 2)
end

function test_debug_1e_grad_omo()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.25 1.0 0.0
""", basis="cc-pvdz")

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    @time one_electron_density(p)
    @time one_electron_one_photon(p)
    @time photon_density1(p)
    @time photon_density2(p)

    i = 1
    q = 1
    h = 1e-4

    D_eT = get_matrix("Density", out_name)
    Dep_eT = get_matrix("Density one electron one boson", out_name)

    @show maximum(abs, p.D_e - D_eT)
    @show maximum(abs, p.D_ep - Dep_eT)

    function calc_1e(mol)
        C = get_omo_C(mol, p.C)
        get_1e_energy(mol, C, p)
    end

    function calc_1e_umo(mol)
        get_1e_energy(mol, p.C, p)
    end

    g_umo = numgrad4_f(mol, calc_1e_umo, i, q, h)

    @show g_umo

    numgrad4_f(mol, calc_1e, i, q, h) -
    g_umo
end

function test_debug_2e_grad_omo()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.25 1.0 0.0
""", basis="cc-pvdz")

    display(mol.ao_labels())

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    one_electron_density(p)
    one_electron_one_photon(p)
    photon_density1(p)
    photon_density2(p)
    two_electron_density(p)

    i = 1
    q = 1
    h = 1e-4

    function calc_2e(mol)
        C = get_omo_C(mol, p.C)
        get_2e_energy(mol, C, p)
    end

    function calc_2e_umo(mol)
        get_2e_energy(mol, p.C, p)
    end

    numgrad4_f(mol, calc_2e, i, q, h) -
    numgrad4_f(mol, calc_2e_umo, i, q, h)
end

function test_debug_e_grad_omo()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.25 1.0 0.0
""", basis="cc-pvdz")

    display(mol.ao_labels())

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    one_electron_density(p)
    one_electron_one_photon(p)
    photon_density1(p)
    photon_density2(p)
    two_electron_density(p)

    i = 1
    q = 1
    h = 1e-4

    function calc_e(mol)
        C = get_omo_C(mol, p.C)
        get_1e_energy(mol, C, p) + get_2e_energy(mol, C, p)
    end

    function calc_e_umo(mol)
        get_1e_energy(mol, p.C, p) + get_2e_energy(mol, p.C, p)
    end

    g = numgrad4_f(mol, calc_e, i, q, h)
    g_umo = numgrad4_f(mol, calc_e_umo, i, q, h)

    @show g

    g - g_umo
end

function test_debug_total_grad_omo()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.25 1.0 0.0
""", basis="cc-pvdz")

    display(mol.ao_labels())

    ω = 0.5
    # coup = 0.5
    coup = 0.5
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    one_electron_density(p)
    one_electron_one_photon(p)
    photon_density1(p)
    photon_density2(p)
    two_electron_density(p)

    kappabar = get_matrix("kappa_bar_pq", out_name)

    display(kappabar)

    i = 1
    q = 1
    h = 1e-4

    function calc_e(mol)
        C = get_omo_C(mol, p.C)
        # C = p.C
        get_1e_energy(mol, C, p) + get_2e_energy(mol, C, p) + get_kappa_energy(mol, C, p, kappabar) + mol.energy_nuc()
    end

    # function calc_e_umo(mol)
    #     get_1e_energy(mol, p.C, p) + get_2e_energy(mol, p.C, p)
    # end

    g = numgrad4_f(mol, calc_e, i, q, h)
end

function test_debug_2e_kappa_grad_omo()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.25 1.0 0.0
""", basis="sto-3g")

    display(mol.ao_labels())

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    one_electron_density(p)
    one_electron_one_photon(p)
    photon_density1(p)
    photon_density2(p)
    two_electron_density(p)

    kappabar = get_matrix("kappa_bar_pq", out_name)

    display(kappabar)

    i = 1
    q = 1
    h = 1e-4

    function calc_e(mol)
        C = get_omo_C(mol, p.C)
        # C = p.C
        get_2e_energy(mol, C, p) + get_kappa_2e_energy(mol, C, p, kappabar)
    end

    # function calc_e_umo(mol)
    #     get_1e_energy(mol, p.C, p) + get_2e_energy(mol, p.C, p)
    # end

    g = numgrad4_f(mol, calc_e, i, q, h)
end

function test_debug_1and2e_kappa_grad_omo()
    mol = pyscf.M(atom="""
    H 0 0 0
    H 1 0 0
    O 0.25 1.0 0.0
""", basis="sto-3g")

    display(mol.ao_labels())

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    one_electron_density(p)
    one_electron_one_photon(p)
    photon_density1(p)
    photon_density2(p)
    two_electron_density(p)

    kappabar = get_matrix("kappa_bar_pq", out_name)

    display(kappabar)

    i = 1
    q = 1
    h = 1e-4

    function calc_e(mol)
        C = get_omo_C(mol, p.C)
        # C = p.C
        get_1e_energy(mol, C, p) + get_kappa_1e_energy(mol, C, p, kappabar) +
        get_2e_energy(mol, C, p) + get_kappa_2e_energy(mol, C, p, kappabar)
    end

    # function calc_e_umo(mol)
    #     get_1e_energy(mol, p.C, p) + get_2e_energy(mol, p.C, p)
    # end

    g = numgrad4_f(mol, calc_e, i, q, h)
end

function test_numgrad()
    mol = pyscf.M(atom="""
H    0.0 0.0 0.0
H    1.0 0.0 0.0
O    0.25 1.0 0.0
He   1.0 1.0 1.0
""", basis="cc-pvdz")

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]
    # pol = [0.0, 1.0, 0.0]

    i = 3
    q = 2

    g2 = @time numgrad2(mol, ω, coup, pol, i, q, 0.001)
    g4 = @time numgrad4(mol, ω, coup, pol, i, q, 0.001)

    @show g2 g4
end

function test_omega()
    mol = pyscf.M(atom="""
    H    0.0 0.0 0.0
    H    1.0 0.0 0.0
    O    0.25 1.0 0.0
""", basis="sto-3g")

    ω = 0.5
    coup = 0.5
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    out_name = "tmp_eT/ccsd"

    p = @time QED_CCSD_PARAMS(mol, ω, coup, pol)

    one_electron_density(p)
    one_electron_one_photon(p)
    photon_density1(p)
    photon_density2(p)
    two_electron_density(p)

    D_eT = get_matrix("DENSITY-1e", out_name)
    @show maximum(abs, D_eT - p.D_e)

    D_ep_eT = get_matrix("Density one electron one boson", out_name)

    @show maximum(abs, D_ep_eT - p.D_ep)

    d = get_qed_d(p.mol, p.pol, p.C)

    d_exp = get_qed_dipmom(p.mol, d)

    h = get_qed_h(p.mol, p.C, d)
    g = get_qed_g(p.mol, p.C, d)

    h = t1_transform_1e(h, p.x, p.y)
    g = t1_transform_2e(g, p.x, p.y)
    d = t1_transform_1e(d, p.x, p.y)

    @show √(p.ω / 2)

    # P = 3
    # Q = 2
    Δh = 1e-4

    # D_e:
    # dX = zeros(size(p.D_e))
    # display(p.D_e)
    # D_e_mu = copy(p.D_e)

    # for i in 1:p.nocc
    #     D_e_mu[i, i] -= 2.0
    # end

    # for P in axes(p.D_e, 1), Q in axes(p.D_e, 2)
    #     dX[P, Q] = omega_diff_h(h, g, d, d_exp, p, P, Q, Δh)
    # end

    # display(dX ./ D_e_mu)
    # dX - D_e_mu

    # D_ep
    # dX = zeros(size(p.D_ep))
    # display(p.D_ep)
    # D_ep_mu = copy(p.D_ep)

    # for i in 1:p.nocc
    #     for a in p.nocc+1:size(p.D_ep, 2)
    #         D_ep_mu[i, a] -= 2.0 * p.s1[a - p.nocc, i]
    #     end
    #     D_ep_mu[i, i] -= 2.0 * p.γ
    # end

    # for P in axes(p.D_e, 1), Q in axes(p.D_e, 2)
    #     dX[P, Q] = omega_diff_d(h, g, d, d_exp, p, P, Q, Δh) / √(p.ω / 2)
    # end

    # display(dX)
    # display(dX ./ D_ep_mu)
    # dX - D_ep_mu

    # D_p:
    # dX = omega_diff_d_exp(h, g, d, d_exp, p, Δh)

    # @show dX
    # @show p.D_p2
    # @show dX / p.D_p2

    # d:
    Ps = 1:p.nocc+p.nvir
    Qs = 1:p.nocc+p.nvir
    R = 4
    S = 7

    # dX = zeros(length(Ps), length(Qs))
    dX = zeros(size(p.d))
    d_mu = copy(p.d)

    for i in 1:p.nocc, j in 1:p.nocc
        d_mu[i, i, j, j] -= 4
        d_mu[i, j, j, i] += 2
    end

    d_mu[1:p.nocc, p.nocc+1:p.nao, 1:p.nocc, p.nocc+1:p.nao] .-=
        2 * permutedims(p.u2, (2, 1, 4, 3))

    # d_mu = d_mu[Ps, Qs, R, S]

    # @time for (Pi, P) in enumerate(Ps), (Qi, Q) in enumerate(Qs)
    #     dX[Pi, Qi] = omega_diff_g(h, g, d, d_exp, p, P, Q, R, S, Δh)
    # end

    # display(d_mu)
    # display(dX)

    # display(dX ./ d_mu)
    # dX - d_mu

    for P in axes(d_mu, 1), Q in axes(d_mu, 2), R in axes(d_mu, 3), S in axes(d_mu, 4)
        dX[P, Q, R, S] = omega_diff_g(h, g, d, d_exp, p, P, Q, R, S, Δh)
    end

    @show maximum(abs, dX - d_mu)

    # display(d_mu)
    # display(dX)

    # display(dX ./ d_mu)
    # dX - d_mu

    # o_t1, o_t2, o_s0, o_s1, o_s2 = read_omega(out_name, p)

    # omega_s2(h, g, d, d_exp, p) - o_s2
end
