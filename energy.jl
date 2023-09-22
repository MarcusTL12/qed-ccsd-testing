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

    @show E_hf

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

    @show maximum(abs, h .- get_matrix("h_pq_t1", "tmp_eT/ccsd"))

    E = p.mol.energy_nuc() +
        einsum("pq,pq->", h, p.D_e) +
        1 / 2 * einsum("pqrs,pqrs->", g, p.d) +
        √(p.ω / 2) * einsum("pq,pq->", d, p.D_ep) -
        √(p.ω / 2) * d_exp * p.D_p2 +
        p.ω * p.D_p1

    @show E
end

function get_1e_energy(mol, C, p::QED_CCSD_PARAMS)
    d = get_qed_d(mol, p.pol, C)
    d_exp = get_qed_dipmom(mol, d)

    h = get_qed_h(mol, C, d)

    h = t1_transform_1e(h, p.x, p.y)
    d = t1_transform_1e(d, p.x, p.y)

    E = 0.0

    E += einsum("pq,pq->", h, p.D_e)
    E += √(p.ω / 2) * einsum("pq,pq->", d, p.D_ep)
    E -= √(p.ω / 2) * d_exp * p.D_p2
    E += p.ω * p.D_p1

    E
end

function get_2e_energy(mol, C, p::QED_CCSD_PARAMS)
    d = get_qed_d(mol, p.pol, C)

    g = get_qed_g(mol, C, d)

    g = t1_transform_2e(g, p.x, p.y)

    1 / 2 * einsum("pqrs,pqrs->", g, p.d)
end

function get_2e_energy_no_d(mol, C, p::QED_CCSD_PARAMS)
    g = get_mo_g(mol, C)

    g = t1_transform_2e(g, p.x, p.y)

    1 / 2 * einsum("pqrs,pqrs->", g, p.d)
end

function get_kappa_1e_energy(mol, C, p::QED_CCSD_PARAMS, kappabar)
    d = get_qed_d(mol, p.pol, C)

    h = get_qed_h(mol, C, d)

    einsum("pq,pq->", kappabar, h)
end

function get_kappa_2e_energy(mol, C, p::QED_CCSD_PARAMS, kappabar)
    o = 1:p.nocc

    d = get_qed_d(mol, p.pol, C)

    g = get_qed_g(mol, C, d)

    # F = h + 2 * einsum("pqkk->pq", g) - einsum("pkkq->pq", g)
    G = [sum(2 * g[p, q, i, i] - g[p, i, i, q] for i in o) for p in 1:p.nao, q in 1:p.nao]
    # display(F)

    einsum("pq,pq->", kappabar, G)
end
