function get_ao_h(mol)
    mol.intor("int1e_kin") + mol.intor("int1e_nuc")
end

function get_nuc_dipole(mol, pol)
    einsum("i,iq,q->", mol.atom_charges(), mol.atom_coords(), pol)
end

function get_ao_d(mol, pol)
    d_el = -einsum("qij,q->ij", mol.intor("int1e_r"), pol)

    d_nuc = get_nuc_dipole(mol, pol)

    S = mol.intor("int1e_ovlp")

    d_el + (d_nuc / mol.nelectron) * S
end

function get_qed_d(mol, pol, C)
    C' * get_ao_d(mol, pol) * C
end

function get_qed_dipmom(mol, d)
    nocc = mol.nelectron ÷ 2

    2sum(d[i, i] for i in 1:nocc)
end

function get_mo_g(mol, C, int=mol.intor("int2e", aosym="s8"))
    nmo = size(C, 2)
    reshape(pyscf.ao2mo.incore.full(int, C,
            compact=false),
        (nmo, nmo, nmo, nmo))
end

function get_qed_g(mol, C, d, int=mol.intor("int2e", aosym="s8"))
    g = get_mo_g(mol, C, int)

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

function t2_tilde(t2_bar)
    t2_tilde = Array(t2_bar)

    for a in axes(t2_tilde, 1), i in axes(t2_tilde, 2)
        t2_tilde[a, i, a, i] *= 2
    end

    t2_tilde
end
