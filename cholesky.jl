
function get_cholesky_vector_blocks(name, outfilename, p::QED_CCSD_PARAMS)
    L_oo = get_matrix("$(name)_oo", outfilename)

    n_J = size(L_oo, 1)

    L = zeros(n_J, p.nao, p.nao)

    o = 1:p.nocc
    v = p.nocc+1:p.nao

    L[:, o, o] = reshape(L_oo, n_J, p.nocc, p.nocc)
    L[:, o, v] = reshape(get_matrix("$(name)_ov", outfilename), n_J, p.nocc, p.nvir)
    L[:, v, o] = reshape(get_matrix("$(name)_vo", outfilename), n_J, p.nvir, p.nocc)
    L[:, v, v] = reshape(get_matrix("$(name)_vv", outfilename), n_J, p.nvir, p.nvir)

    L
end

function compute_WL(p::QED_CCSD_PARAMS, L)
    einsum("pqrs,Krs->Kpq", p.d, L)
end
