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

function get_energy_ccsd(mol, pol, C, t1, t2, s1, γ, ω)
    d = get_qed_d(mol, pol, C)

    d_exp = get_qed_dipmom(mol, d)

    h = get_qed_h(mol, C, d)
    g = get_qed_g(mol, C, d)

    E_hf = get_energy_hf(mol, h, g)

    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    o = 1:nocc
    v = nocc+1:nao

    E1 = 2 * einsum("ai,ai->", h[v, o], t1)

    E2 = 4 * einsum("iija,aj->", g[o, o, o, v], t1) -
         2 * einsum("ijja,ai->", g[o, o, o, v], t1) +
         2 * einsum("iajb,ai,bj->", g[o, v, o, v], t1, t1) +
         2 * einsum("iajb,aibj->", g[o, v, o, v], t2) -
         1 * einsum("iajb,aj,bi->", g[o, v, o, v], t1, t1) -
         1 * einsum("iajb,ajbi->", g[o, v, o, v], t2)

    E3 = √(ω / 2) * (2 * einsum("ii->", d[o, o]) * γ +
                     2 * einsum("ia,ai->", d[o, v], s1) +
                     2 * einsum("ia,ai->", d[o, v], t1) * γ)

    E4 = -√(ω / 2) * d_exp * γ

    E = E_hf + E1 + E2 + E3 + E4

    @show E
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

    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    o = 1:nocc
    v = nocc+1:nao

    E2 = 2 * einsum("iajb,aibj->", g[o, v, o, v], t2) -
         1 * einsum("iajb,ajbi->", g[o, v, o, v], t2)

    E3 = √(ω / 2) * (2 * einsum("ii->", d[o, o]) * γ +
                     2 * einsum("ia,ai->", d[o, v], s1))

    E4 = -√(ω / 2) * d_exp * γ

    E = E_hf + E2 + E3 + E4

    @show E
end

function get_energy_t1_density(mol, pol, C, ω, x, y, D_e, D_ep, D_p1, D_p2, d_e)
    d = get_qed_d(mol, pol, C)

    d_exp = get_qed_dipmom(mol, d)

    h = get_qed_h(mol, C, d)
    g = get_qed_g(mol, C, d)

    h = t1_transform_1e(h, x, y)
    g = t1_transform_2e(g, x, y)
    d = t1_transform_1e(d, x, y)

    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    E = mol.energy_nuc() +
        einsum("pq,pq->", h, D_e) +
        1 / 2 * einsum("pqrs,pqrs->", g, d_e) +
        √(ω / 2) * einsum("pq,pq->", d, D_ep) -
        √(ω / 2) * d_exp * D_p2 +
        ω * D_p1

    @show E
end
