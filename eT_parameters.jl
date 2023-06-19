using PyCall
using LinearAlgebra

include("get_matrix.jl")
include("get_vector.jl")
include("run_eT_tor.jl")

if !@isdefined pyscf
    pyscf = pyimport("pyscf")
    einsum = pyscf.lib.einsum
end

function get_ao_h(mol)
    mol.intor("int1e_kin") + mol.intor("int1e_nuc")
end

function get_nuc_dipole(mol, pol)
    einsum("i,iq,q->", mol.atom_charges(), mol.atom_coords(), pol)
end

function get_ao_d(mol, pol)
    -einsum("qij,q->ij", mol.intor("int1e_r"), pol)
end

function get_qed_d(mol, pol, C)
    d_nuc = get_nuc_dipole(mol, pol)

    d_el = C' * get_ao_d(mol, pol) * C

    for i in axes(d_el, 1)
        d_el[i, i] += d_nuc / mol.nelectron
    end

    d_el
end

function get_qed_dipmom(mol, d)
    nocc = mol.nelectron ÷ 2

    2sum(d[i, i] for i in 1:nocc)
end

function get_mo_g(mol, C)
    nmo = size(C, 2)
    reshape(pyscf.ao2mo.incore.full(mol.intor("int2e", aosym="s8"), C,
            compact=false),
        (nmo, nmo, nmo, nmo))
end

function get_qed_g(mol, C, d)
    g = get_mo_g(mol, C)

    g + einsum("ij,kl->ijkl", d, d)
end

function get_mo_h(mol, C)
    C' * get_ao_h(mol) * C
end

function get_qed_h(mol, C, d)
    h = get_mo_h(mol, C)

    d_exp = get_qed_dipmom(mol, d)

    h + 1 / 2 * einsum("pr,rq->pq", d, d) -
    d * d_exp +
    1 / (2 * mol.nelectron) * d_exp^2 * I
end

function get_energy_1(mol, h_mo, g_mo, d)
    nocc = mol.nelectron ÷ 2

    2sum(h_mo[i, i] for i in 1:nocc) +
    sum(2g_mo[i, i, j, j] - g_mo[i, j, j, i] for i in 1:nocc, j in 1:nocc) +
    mol.energy_nuc() +
    sum(d[a, i]^2 for i in 1:nocc, a in nocc+1:mol.nao)
end

function get_energy_hf(mol, h, g)
    nocc = mol.nelectron ÷ 2

    2sum(h[i, i] for i in 1:nocc) +
    sum(2g[i, i, j, j] - g[i, j, j, i] for i in 1:nocc, j in 1:nocc) +
    mol.energy_nuc()
end

function make_fock(mol, h, g)
    nocc = mol.nelectron ÷ 2

    h + 2einsum("pqii->pq", g[:, :, 1:nocc, 1:nocc]) -
    einsum("piiq->pq", g[:, 1:nocc, 1:nocc, :])
end

function unpack_t2(mol, t2_packed)
    nocc = mol.nelectron ÷ 2
    nvir = py"int"(mol.nao) - nocc

    np = nocc * nvir

    t2 = zeros(np, np)

    ind = 1
    for p in 1:np, q in 1:p
        t2[p, q] = t2_packed[ind]
        ind += 1
    end

    reshape(Symmetric(t2, :L), nvir, nocc, nvir, nocc)
end

function get_photon_amplitude(out_name)
    reg = r"Photon amplitudes\s+-+\s+\d+\s+(-?\d+\.\d+)"

    parse(Float64, match(reg, read("$out_name.out", String)).captures[1])
end

function get_energy_ccsd(mol, pol, C, t1, t2, s1, γ, ω)
    d = get_qed_d(mol, pol, C)

    d_exp = get_qed_dipmom(mol, d)

    h = get_qed_h(mol, C, d)
    g = get_qed_g(mol, C, d)

    E_hf = get_energy_hf(mol, h, g)

    @show E_hf

    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    o = 1:nocc
    v = nocc+1:nao

    E1 = 2 * einsum("ai,ai->", h[v, o], t1)

    @show E1

    E2 = 4 * einsum("iija,aj->", g[o, o, o, v], t1) -
         2 * einsum("ijja,ai->", g[o, o, o, v], t1) +
         2 * einsum("iajb,ai,bj->", g[o, v, o, v], t1, t1) +
         2 * einsum("iajb,aibj->", g[o, v, o, v], t2) -
         1 * einsum("iajb,aj,bi->", g[o, v, o, v], t1, t1) -
         1 * einsum("iajb,ajbi->", g[o, v, o, v], t2)

    @show E2

    E3 = √(ω / 2) * (2 * einsum("ii->", d[o, o]) * γ +
                     2 * einsum("ia,ai->", d[o, v], s1) +
                     2 * einsum("ia,ai->", d[o, v], t1) * γ)

    @show E3

    E4 = -√(ω / 2) * d_exp * γ

    @show E_hf + E1 + E2 + E3 + E4
end

function get_energy_ccsd_t1_transformed(mol, pol, C, t2, s1, γ, ω, x, y)
    d = get_qed_d(mol, pol, C)

    d_exp = get_qed_dipmom(mol, d)

    h = get_qed_h(mol, C, d)
    g = get_qed_g(mol, C, d)

    h = t1_transform_1e(h, x, y)
    g = t1_transform_2e(g, x, y)
    d = t1_transform_1e(d, x, y)

    E_hf = get_energy_hf(mol, h, g)

    @show E_hf

    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    o = 1:nocc
    v = nocc+1:nao

    E2 = 2 * einsum("iajb,aibj->", g[o, v, o, v], t2) -
         1 * einsum("iajb,ajbi->", g[o, v, o, v], t2)

    @show E2

    E3 = √(ω / 2) * (2 * einsum("ii->", d[o, o]) * γ +
                     2 * einsum("ia,ai->", d[o, v], s1))

    @show E3

    E4 = -√(ω / 2) * d_exp * γ

    @show E_hf + E2 + E3 + E4
end

function construct_t1_transformation(mol, t1)
    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    o = 1:nocc
    v = nocc+1:nao

    x = zeros(nao, nao)
    y = zeros(nao, nao)

    for i in 1:nao
        x[i, i] = y[i, i] = 1.0
    end

    x[v, o] = -t1
    y[o, v] = t1'

    x, y
end

function t1_transform_1e(h, x, y)
    einsum("pr,qs,rs->pq", x, y, h)
end

function t1_transform_2e(g, x, y)
    einsum("pt,qu,rm,sn,tumn->pqrs", x, y, x, y, g)
end

function test()
    mol = pyscf.M(atom="""
 H          0.86681        0.60144        5.00000
 H         -0.86681        0.60144        5.00000
 O          0.00000       -0.07579        5.00000
 He         0.10000       -0.02000        7.53000
""", basis="STO-3G")

    omega = 0.5
    coup = 0.05
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    # out_name = run_eT_tor(mol, omega, coup, pol)
    out_name = "tmp_eT/ccsd"

    pol *= coup

    C = get_matrix("MO-coeffs", out_name)

    # println("C:")
    # display(C)

    t1 = get_matrix("T1-AMPLITUDES", out_name)

    # println("t1:")
    # display(t1)

    t2 = unpack_t2(mol, get_vector("T2-AMPLITUDES", out_name))

    # println("t2:")
    # display(t2)

    s1 = get_matrix("S1-AMPLITUDES", out_name)

    # println("s1:")
    # display(s1)

    s2 = unpack_t2(mol, get_vector("S2-AMPLITUDES", out_name))

    # println("s2:")
    # display(s2)

    γ = get_photon_amplitude(out_name)

    @show γ

    @time get_energy_ccsd(mol, pol, C, t1, t2, s1, γ, omega)

    x, y = construct_t1_transformation(mol, t1)

    # println("x:")
    # display(x)

    # println("y:")
    # display(y)

    @time get_energy_ccsd_t1_transformed(mol, pol, C, t2, s1, γ, omega, x, y)
end
