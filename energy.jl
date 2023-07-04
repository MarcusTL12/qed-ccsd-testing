function get_energy_hf_mo(mol, h_mo, g_mo, d)
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

function get_energy_ccsd(p::QED_CCSD_PARAMS)
    d = get_qed_d(p.mol, p.pol, p.C)

    d_exp = get_qed_dipmom(p.mol, d)

    h = get_qed_h(p.mol, p.C, d)
    g = get_qed_g(p.mol, p.C, d)

    E_hf = get_energy_hf(p.mol, h, g)

    o = 1:p.nocc
    v = p.nocc+1:p.nao

    E1 = 2 * einsum("ai,ai->", h[v, o], p.t1)

    E2 = 4 * einsum("iija,aj->", g[o, o, o, v], p.t1) -
         2 * einsum("ijja,ai->", g[o, o, o, v], p.t1) +
         2 * einsum("iajb,ai,bj->", g[o, v, o, v], p.t1, p.t1) +
         2 * einsum("iajb,aibj->", g[o, v, o, v], p.t2) -
         1 * einsum("iajb,aj,bi->", g[o, v, o, v], p.t1, p.t1) -
         1 * einsum("iajb,ajbi->", g[o, v, o, v], p.t2)

    E3 = √(p.ω / 2) * (2 * einsum("ii->", d[o, o]) * p.γ +
                     2 * einsum("ia,ai->", d[o, v], p.s1) +
                     2 * einsum("ia,ai->", d[o, v], p.t1) * p.γ)

    E4 = -√(p.ω / 2) * d_exp * p.γ

    E = E_hf + E1 + E2 + E3 + E4

    @show E
end

function get_energy_ccsd_t1_transformed(p::QED_CCSD_PARAMS)
    d = get_qed_d(p.mol, p.pol, p.C)

    d_exp = get_qed_dipmom(p.mol, d)

    h = get_qed_h(p.mol, p.C, d)
    g = get_qed_g(p.mol, p.C, d)

    h = t1_transform_1e(h, p.x, p.y)
    g = t1_transform_2e(g, p.x, p.y)
    d = t1_transform_1e(d, p.x, p.y)

    E_hf = get_energy_hf(p.mol, h, g)

    o = 1:p.nocc
    v = p.nocc+1:p.nao

    E2 = 2 * einsum("iajb,aibj->", g[o, v, o, v], p.t2) -
         1 * einsum("iajb,ajbi->", g[o, v, o, v], p.t2)

    E3 = √(p.ω / 2) * (2 * einsum("ii->", d[o, o]) * p.γ +
                     2 * einsum("ia,ai->", d[o, v], p.s1))

    E4 = -√(p.ω / 2) * d_exp * p.γ

    E = E_hf + E2 + E3 + E4

    @show E
end

function get_energy_t1_density(p::QED_CCSD_PARAMS)
    d = get_qed_d(p.mol, p.pol, p.C)

    d_exp = get_qed_dipmom(p.mol, d)

    h = get_qed_h(p.mol, p.C, d)
    g = get_qed_g(p.mol, p.C, d)

    h = t1_transform_1e(h, p.x, p.y)
    g = t1_transform_2e(g, p.x, p.y)
    d = t1_transform_1e(d, p.x, p.y)

    E = p.mol.energy_nuc() +
        einsum("pq,pq->", h, p.D_e) +
        1 / 2 * einsum("pqrs,pqrs->", g, p.d) +
        √(p.ω / 2) * einsum("pq,pq->", d, p.D_ep) -
        √(p.ω / 2) * d_exp * p.D_p2 +
        p.ω * p.D_p1

    @show E
end
