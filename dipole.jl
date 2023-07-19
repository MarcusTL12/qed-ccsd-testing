
function get_nuc_dipole_full(mol)
    einsum("i,iq->q", mol.atom_charges(), mol.atom_coords())
end

function get_electronic_dipole(p::QED_CCSD_PARAMS)
    d = p.mol.intor("int1e_r")

    d_mo = einsum("mp,nq,kmn->kpq", p.C, p.C, d)

    d_t1 = einsum("pr,qs,krs->kpq", p.x, p.y, d_mo)

    -einsum("kpq,pq->k", d_t1, p.D_e)
end

function get_total_dipole(p::QED_CCSD_PARAMS)
    get_nuc_dipole_full(p.mol) + get_electronic_dipole(p)
end
