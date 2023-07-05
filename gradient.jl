
# Calculate {A, B}_pq
function connection2(A, B)
    einsum("pt,tq->pq", A, B) +
    einsum("qt,pt->pq", A, B)
end

# Calculate {A, B}_pqrs
function connection4(A, B)
    einsum("pt,tqrs->pqrs", A, B) +
    einsum("qt,ptrs->pqrs", A, B) +
    einsum("rt,pqts->pqrs", A, B) +
    einsum("st,pqrt->pqrs", A, B)
end

function get_gradient(p::QED_CCSD_PARAMS, i, q, h)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    S_mo = p.C' * p.mol.intor("int1e_ovlp") * p.C
    dS_ao = numgrad4_int(p.mol, "int1e_ovlp", i, q, h)
    dS_umo = p.C' * dS_ao * p.C

    # Just to check that it becomes zero
    dS_omo = dS_umo - 0.5 * connection2(dS_umo, S_mo)
    @assert maximum(abs, dS_omo) < 1e-10

    d_ao = get_ao_d(p.mol, p.pol)
    d_mo = p.C' * d_ao * p.C
    dd_ao = numgrad4_ao_d(p.mol, p.pol, i, q, h)
    dd_umo = p.C' * dd_ao * p.C

    dd_omo = dd_umo - 0.5 * connection2(dS_umo, d_mo)

    D = p.C[:, o] * p.C[:, o]'

    d_exp = 2 * tr(D * d_ao)
    dd_exp_umo = 2 * tr(D * dd_ao)
    dd_exp_omo = get_qed_dipmom(p.mol, dd_omo)

    h_mo = get_qed_h(p.mol, p.C, d_mo)
    dh_e_ao = numgrad4_int(p.mol, "int1e_kin", i, q, h) +
              numgrad4_int(p.mol, "int1e_nuc", i, q, h)
    dh_umo = p.C' * dh_e_ao * p.C

    dh_umo += 0.5 * (dd_umo * d_mo + d_mo * dd_umo) -
              dd_umo * d_exp - d_mo * dd_exp_umo

    dh_omo = dh_umo - 0.5 * connection2(dS_umo, h_mo)

    g_mo = get_qed_g(p.mol, p.C, d_mo)
    dg_ao = numgrad4_int(p.mol, "int2e", i, q, h)
    dg_e_umo = get_mo_g(p.mol, p.C, dg_ao)
    dg_umo = dg_e_umo +
             einsum("pq,rs->pqrs", dd_umo, d_mo) +
             einsum("pq,rs->pqrs", d_mo, dd_umo)
    dg_omo = dg_umo - 0.5 * connection4(dS_umo, g_mo)

    F_mo = h_mo
    dF_umo = dh_umo

    for i in o
        F_mo .+= 2 * g_mo[:, :, i, i] .- g_mo[:, i, i, :]
        dF_umo .+= 2 * dg_umo[:, :, i, i] .- dg_umo[:, i, i, :]
    end

    dF_omo = dF_umo - 0.5 * connection2(dS_umo, F_mo)

    numgrad4_e_nuc(p.mol, i, q, h) +
    einsum("pq,pq->", dh_omo, p.D_e) +
    0.5 * einsum("pqrs,pqrs->", dg_omo, p.d) +
    √(p.ω / 2) * einsum("pq,pq->", dd_omo, p.D_ep) -
    √(p.ω / 2) * dd_exp_omo * p.D_p2 +
    einsum("ia,ia->", p.κ_bar, dF_omo[o, v])
end
