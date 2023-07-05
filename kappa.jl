
function solve_kappa_bar(p::QED_CCSD_PARAMS)
    # η = zeros(p.nocc, p.nvir)
    # A = zeros(p.nocc, p.nvir, p.nocc, p.nvir)

    o = 1:p.nocc
    v = p.nocc+1:p.nao

    d = get_qed_d(p.mol, p.pol, p.C)

    d_exp = get_qed_dipmom(p.mol, d)

    d_d_exp = 4 * d[v, o]'

    h = get_qed_h(p.mol, p.C, d)
    g = get_qed_g(p.mol, p.C, d)

    h_t1 = t1_transform_1e(h, p.x, p.y)
    g_t1 = t1_transform_2e(g, p.x, p.y)
    d_t1 = t1_transform_1e(d, p.x, p.y)

    η = d_d_exp .* (-einsum("rs,rs->", d_t1, p.D_e) -
                    √(p.ω / 2) * p.D_p2 + d_exp) +
        einsum("as,is->ia", h_t1[v, :], p.D_e[o, :]) +
        einsum("as,si->ia", h_t1[v, :], p.D_e[:, o]) -
        einsum("is,sa->ia", h_t1[o, :], p.D_e[:, v]) +
        einsum("is,as->ia", h_t1[o, :], p.D_e[v, :]) +
        einsum("arst,irst->ia", g_t1[v, :, :, :], p.d[o, :, :, :]) -
        einsum("irst,rast->ia", g_t1[o, :, :, :], p.d[:, v, :, :]) -
        einsum("irst,arst->ia", g_t1[o, :, :, :], p.d[v, :, :, :]) +
        einsum("arst,rist->ia", g_t1[v, :, :, :], p.d[:, o, :, :]) +
        √(p.ω / 2) * (
            einsum("as,is->ia", d_t1[v, :], p.D_ep[o, :]) +
            einsum("as,si->ia", d_t1[v, :], p.D_ep[:, o]) -
            einsum("is,sa->ia", d_t1[o, :], p.D_ep[:, v]) +
            einsum("is,as->ia", d_t1[o, :], p.D_ep[v, :])
        )

    A = -einsum("ia,jb->iajb", d[o, v], d_d_exp) +
        4 * g[o, v, o, v] -
        PermutedDimsArray(g[o, v, o, v], (1, 4, 3, 2)) -
        PermutedDimsArray(g[o, v, v, o], (1, 3, 4, 2))

    Am = reshape(A, p.nocc * p.nvir, p.nocc * p.nvir)
    ηv = reshape(η, p.nocc * p.nvir)

    p.κ_bar = reshape(-Am \ ηv, p.nocc, p.nvir)
end
