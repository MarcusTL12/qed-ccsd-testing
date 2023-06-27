# T1 transformed densities

function one_electron_density(mol, t2, s1, s2, γ,
    t1_bar, t2_t, s1_bar, s2_t, γ_bar)
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

    D_oo .-= 2 * einsum("aibk,ajbk->ij", t2, t2_t)

    D_oo .-= 1 * einsum("ai,aj->ij", s1, s1_bar) +
             2 * einsum("aibk,ajbk->ij", s2, s2_t)


    D_ov .+= 2 * einsum("aibj,bj->ia", t2, t1_bar) -
             1 * einsum("ajbi,bj->ia", t2, t1_bar)

    D_ov .+= 2 * s1' * γ_bar +
             2 * einsum("aibj,bj->ia", s2, s1_bar) -
             1 * einsum("ajbi,bj->ia", s2, s1_bar) -
             2 * einsum("aj,bjck,bick->ia", s1, s2_t, t2) -
             2 * einsum("bi,bjck,ajck->ia", s1, s2_t, t2)

    D_vo .+= t1_bar

    D_vv .+= 2 * einsum("bicj,aicj->ab", t2, t2_t)

    D_vv .+= 1 * einsum("bi,ai->ab", s1, s1_bar) +
             2 * einsum("bicj,aicj->ab", s2, s2_t)

    D
end

function one_electron_one_photon(mol, t2, s1, s2, γ,
    t1_bar, t2_t, s1_bar, s2_t, γ_bar)
    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    o = 1:nocc
    v = nocc+1:nao

    D = zeros(nao, nao)

    D_oo = @view D[o, o]
    D_ov = @view D[o, v]
    D_vo = @view D[v, o]
    D_vv = @view D[v, v]

    # D0_ij = - ∑_a(s_ai tᴸ_aj)
    # - 2 ∑_abk(s_aibk tᵗ_ajbk)
    # - 2 ∑_abk(t_aibk tᵗ_ajbk γ)
    #
    # + 2 ∑_ak(δ_ij s_ak tᴸ_ak)
    # + 2 ∑_akbl(δ_ij s_akbl tᵗ_akbl)
    # + 2 δ_ij γ

    # D1_ij = - 2 ∑_abk(sᵗ_ajbk t_aibk)
    # + 2 δ_ij γᴸ

    diag_elem = γ + γ_bar +
                einsum("ak,ak->", s1, t1_bar) +
                einsum("akbl,akbl->", s2, t2_t)

    for i in o
        D[i, i] = 2.0 * diag_elem
    end

    D_oo .-= 1 * einsum("ai,aj->ij", s1, t1_bar) +
             2 * einsum("aibk,ajbk->ij", s2, t2_t) +
             2 * einsum("aibk,ajbk->ij", t2, t2_t) * γ

    #  D1_ij = - 2 ∑_abk(s_ai s_bk sᵗ_ajbk)
    #  - 2 ∑_abk(s_aibk sᵗ_ajbk γ)
    #  - ∑_a(s_ai sᴸ_aj γ)

    D_oo .-= 1 * einsum("ai,aj->ij", s1, s1_bar) * γ +
             2 * einsum("ai,bk,ajbk->ij", s1, s1, s2_t) +
             2 * einsum("ai,bk,ajbk->ij", s1, s1, s2_t)

    D_oo .-= 2 * einsum("aibk,ajbk->ij", t2, s2_t)

    #  D0_ia = 2 s_ai
    #  + 2 ∑_bj(s_aibj tᴸ_bj)
    #  -   ∑_jb(s_ajbi tᴸ_bj)
    #
    #  + 4 ∑_bjck(s_bj t_aick tᵗ_bjck)
    #  - 2 ∑_jbck(s_aj t_bick tᵗ_bjck)
    #  - 2 ∑_bjck(s_bi t_ajck tᵗ_bjck)
    #  - 2 ∑_bjkc(s_bj t_akci tᵗ_bjck)
    #
    #  + 2 ∑_bj(t_aibj tᴸ_bj γ)
    #  - ∑_jb(t_ajbi tᴸ_bj γ)

    D_ov .+= 2 * s1' +
             2 * einsum("aibj,bj->ia", s2, t1_bar) -
             1 * einsum("ajbi,bj->ia", s2, t1_bar) +
             4 * einsum("bj,aick,bjck->ia", s1, t2, t2_t) -
             2 * einsum("aj,bick,bjck->ia", s1, t2, t2_t) -
             2 * einsum("bi,ajck,bjck->ia", s1, t2, t2_t) -
             2 * einsum("bj,akci,bjck->ia", s1, t2, t2_t) +
             2 * einsum("aibj,bj->ia", t2, t1_bar) * γ -
             1 * einsum("ajbi,bj->ia", t2, t1_bar) * γ

    #  D1_ia =
    #  + 2 s_ai γ γᴸ
    #
    #  + 2 ∑_bj(s_ai s_bj sᴸ_bj)
    #  - 2 ∑_jb(s_aj s_bi sᴸ_bj)
    #
    #  + 2 ∑_bj(s_aibj sᴸ_bj γ)
    #  -   ∑_jb(s_ajbi sᴸ_bj γ)
    #
    #  + 4 ∑_bjck(s_aibj s_ck sᵗ_bjck)
    #  - 4 ∑_jbkc(s_ajbk s_ci sᵗ_bkcj)
    #  - 4 ∑_jbck(s_bick s_aj sᵗ_bjck)
    #  + 2 ∑_bjck(s_bjck s_ai sᵗ_bjck)
    #  - 2 ∑_jbck(s_ajbi s_ck sᵗ_bjck)
    #
    #  - 2 ∑_jbck(t_bick s_aj sᵗ_bjck γ)
    #  - 2 ∑_bjck(t_ajck s_bi sᵗ_bjck γ)

    D_ov .+= 2 * s1' * γ * γ_bar +
             2 * einsum("ai,bj,bj->ia", s1, s1, s1_bar) -
             2 * einsum("aj,bi,bj->ia", s1, s1, s1_bar) +
             2 * einsum("aibj,bj->ia", s2, s1_bar) * γ -
             1 * einsum("ajbi,bj->ia", s2, s1_bar) * γ +
             4 * einsum("aibj,ck,bjck->ia", s2, s1, s2_t) -
             4 * einsum("ajbk,ci,bkcj->ia", s2, s1, s2_t) -
             4 * einsum("bick,aj,bjck->ia", s2, s1, s2_t) +
             2 * einsum("bjck,ai,bjck->ia", s2, s1, s2_t) -
             2 * einsum("ajbi,ck,bjck->ia", s2, s1, s2_t) -
             2 * einsum("bick,aj,bjck->ia", t2, s1, s2_t) * γ -
             2 * einsum("ajck,bi,bjck->ia", t2, s1, s2_t) * γ

    #  D1_ia = 2 ∑_bj(sᴸ_bj t_aibj)
    #  - ∑_bj(sᴸ_bj t_ajbi)

    D_ov .+= 2 * einsum("aibj,bj->ia", t2, s1_bar) -
             1 * einsum("ajbi,bj->ia", t2, s1_bar)

    #  D0_ai = 2 ∑_bj(s_bj tᵗ_aibj)
    #  + tᴸ_ai γ

    # D1_ai = sᴸ_ai

    D_vo .+= t1_bar * γ + 2 * einsum("bj,aibj->ai", s1, t2_t) + s1_bar

    # D0_ab = ∑_i(s_bi tᴸ_ai)
    #     + 2 ∑_icj(s_bicj tᵗ_aicj)
    #     + 2 ∑_icj(t_bicj tᵗ_aicj γ)

    D_vv .+= 1 * einsum("bj,ai->ab", s1, t1_bar) +
             2 * einsum("bicj,aicj->ab", s2, t2_t) +
             2 * einsum("bicj,aicj->ab", t2, t2_t) * γ

    #  D1_ab =
    #  +   ∑_i(s_bi sᴸ_ai γ)
    #  + 2 ∑_icj(s_bi s_cj sᵗ_aicj)
    #  + 2 ∑_icj(s_bicj sᵗ_aicj γ)

    D_vv .+= 1 * einsum("bi,ai->ab", s1, s1_bar) * γ +
             2 * einsum("bi,cj,aicj->ab", s1, s1, s2_t) +
             2 * einsum("bicj,aicj->ab", s2, s2_t) * γ

    #  D1_ab = 2 ∑_icj(sᵗ_aicj t_bicj)

    D_vv .+= 2 * einsum("aicj,bicj->ab", s2_t, t2)

    D
end
