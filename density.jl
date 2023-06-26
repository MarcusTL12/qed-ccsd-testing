# T1 transformed densities

function one_electron_density(mol, t2, s1, s2, γ,
    t1_bar, t2_bar, s1_bar, s2_bar, γ_bar)
    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    o = 1:nocc
    v = nocc+1:nao

    D = zeros(nao, nao)

    D_oo = @view D[o, o]
    D_ov = @view D[o, v]
    D_vo = @view D[v, o]
    D_vv = @view D[v, v]

    for i in o
        D[i, i] = 2.0
    end

    D_oo .-= 2 * einsum("aibk,ajbk->ij", t2, t2_bar)

    D_oo .-= 1 * einsum("ai,aj->ij", s1, s1_bar) +
             2 * einsum("aibk,ajbk->ij", s2, s2_bar)


    D_ov .+= 2 * einsum("aibj,bj->ia", t2, t1_bar) -
             1 * einsum("ajbi,bj->ia", t2, t1_bar)

    D_ov .+= 2 * s1' * γ_bar +
             2 * einsum("aibj,bj->ia", s2, s1_bar) -
             1 * einsum("ajbi,bj->ia", s2, s1_bar) -
             2 * einsum("aj,bjck,bick->ia", s1, s2_bar, t2) -
             2 * einsum("bi,bjck,ajck->ia", s1, s2_bar, t2)

    D_vo .+= t1_bar

    D_vv .+= 2 * einsum("bicj,aicj->ab", t2, t2_bar)

    D_vv .+= 1 * einsum("bi,ai->ab", s1, s1_bar) +
             2 * einsum("bicj,aicj->ab", s2, s2_bar)

    D
end
