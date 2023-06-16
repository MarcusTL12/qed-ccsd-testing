using PyCall

include("get_matrix.jl")

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

function get_energy(mol, h, g)
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
