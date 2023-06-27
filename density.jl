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

function two_electron_density(mol, t2, s1, s2, γ,
    t1_bar, t2_t, s1_bar, s2_t, γ_bar)
    nao = py"int"(mol.nao)
    nocc = mol.nelectron ÷ 2

    o = 1:nocc
    v = nocc+1:nao

    d = zeros(nao, nao, nao, nao)

    d_oooo = @view d[o, o, o, o]
    d_ooov = @view d[o, o, o, v]
    d_oovo = @view d[o, o, v, o]
    d_oovv = @view d[o, o, v, v]
    d_ovoo = @view d[o, v, o, o]
    d_ovov = @view d[o, v, o, v]
    d_ovvo = @view d[o, v, v, o]
    d_ovvv = @view d[o, v, v, v]
    d_vooo = @view d[v, o, o, o]
    d_voov = @view d[v, o, o, v]
    d_vovo = @view d[v, o, v, o]
    d_vovv = @view d[v, o, v, v]
    d_vvoo = @view d[v, v, o, o]
    d_vvov = @view d[v, v, o, v]
    d_vvvo = @view d[v, v, v, o]
    d_vvvv = @view d[v, v, v, v]

    # Λ0:

    # d_ijkl =
    # 4 δ_ij δ_kl
    # - 2 δ_il δ_jk

    for i in o, j in o
        d[i, i, j, j] += 4
        d[i, j, j, i] -= 2
    end

    # - 4 ∑_abm(δ_kl t_aibm tᵗ_ajbm)
    # + 2 ∑_abm(δ_jk t_aibm tᵗ_albm)
    # + 2 ∑_abm(δ_il t_akbm tᵗ_ajbm)
    # - 4 ∑_abm(δ_ij t_akbm tᵗ_albm)

    diag_elem = einsum("aibm,ajbm->ij", t2, t2_t)

    for i in o
        d_oooo[:, :, i, i] .-= 4 * diag_elem
        d_oooo[:, i, i, :] .+= 2 * diag_elem
        d_oooo[i, :, :, i] .+= 2 * diag_elem'
        d_oooo[i, i, :, :] .-= 4 * diag_elem
    end

    # + 2 ∑_ab(t_aibk tᵗ_ajbl)

    d_oooo .+= 2 * einsum("aibk,ajbl->ijkl", t2, t2_t)

    # d_ijka =
    # + 4 ∑_bl(δ_ij t_akbl tᴸ_bl)
    # - 2 ∑_lb(δ_ij t_albk tᴸ_bl)
    # - 2 ∑_bl(δ_jk t_aibl tᴸ_bl)
    # +   ∑_lb(δ_jk t_albi tᴸ_bl)
    #
    # +   ∑_b(t_aibk tᴸ_bj)
    # - 2 ∑_b(t_akbi tᴸ_bj)

    diag_elem1 = einsum("aibl,bl->ia", t2, t1_bar)
    diag_elem2 = einsum("albi,bl->ia", t2, t1_bar)

    diag_elem = 2 * diag_elem1 - diag_elem2

    for i in o
        d_ooov[i, i, :, :] .+= 2 * diag_elem
        d_ooov[:, i, i, :] .-= diag_elem
    end

    d_ooov .+= 1 * einsum("aibk,bj->ijka", t2, t1_bar) -
               2 * einsum("akbi,bj->ijka", t2, t1_bar)

    permutedims!(d_ovoo, d_ooov, (3, 4, 1, 2))

    # d_ijak =
    # + 2 δ_ij tᴸ_ak
    # -   δ_ik tᴸ_aj

    for i in o
        d_oovo[i, i, :, :] .+= 2 * t1_bar
        d_oovo[i, :, :, i] .-= 1 * t1_bar'
    end

    # d_ijab =
    # + 4 ∑_kcl(δ_ij t_bkcl tᵗ_akcl)
    # - 2 ∑_ck(t_bick tᵗ_ajck)
    # - 2 ∑_kc(t_bkci tᵗ_akcj)

    diag_elem = 4 * einsum("akcl,bkcl->ab", t2_t, t2)

    for i in o
        d_oovv[i, i, :, :] .+= diag_elem
    end

    d_oovv .-= 2 * einsum("ajck,bick->ijab", t2_t, t2) +
               2 * einsum("akcj,bkci->ijab", t2_t, t2)

    permutedims!(d_vvoo, d_oovv, (3, 4, 1, 2))

    # d_iajb =
    # 4 t_aibj
    # - 2 t_ajbi
    #
    # + 8 ∑_ckdl(t_aick t_bjdl tᵗ_ckdl)
    # - 4 ∑_kcdl(t_aibk t_cjdl tᵗ_ckdl)
    # - 4 ∑_ckdl(t_aicj t_bkdl tᵗ_ckdl)
    # - 4 ∑_ckld(t_aick t_bldj tᵗ_ckdl)
    # - 4 ∑_ckdl(t_ajck t_bidl tᵗ_ckdl)
    # - 4 ∑_kcdl(t_akbj t_cidl tᵗ_ckdl)
    # - 4 ∑_kcdl(t_akci t_bjdl tᵗ_ckdl)
    # - 4 ∑_kcld(t_alck t_bjdi tᵗ_ckdl)
    # + 2 ∑_kcdl(t_ajbk t_cidl tᵗ_ckdl)
    # + 2 ∑_ckdl(t_ajci t_bkdl tᵗ_ckdl)
    # + 2 ∑_ckld(t_ajck t_bldi tᵗ_ckdl)
    # + 2 ∑_kcdl(t_akbi t_cjdl tᵗ_ckdl)
    # + 2 ∑_klcd(t_akbl t_cidj tᵗ_ckdl)
    # + 2 ∑_kcld(t_akci t_bldj tᵗ_ckdl)
    # + 2 ∑_kcdl(t_akcj t_bidl tᵗ_ckdl)
    # + 2 ∑_kcld(t_alcj t_bkdi tᵗ_ckdl)
    # + 2 ∑_kcld(t_alck t_bidj tᵗ_ckdl)

    d_ovov .+= 4 * permutedims(t2, (2, 1, 4, 3)) -
               2 * permutedims(t2, (2, 3, 4, 1)) +
               8 * einsum("aick,bjdl,ckdl->iajb", t2, t2, t2_t) -
               4 * einsum("aibk,cjdl,ckdl->iajb", t2, t2, t2_t) -
               4 * einsum("aicj,bkdl,ckdl->iajb", t2, t2, t2_t) -
               4 * einsum("aick,bldj,ckdl->iajb", t2, t2, t2_t) -
               4 * einsum("ajck,bidl,ckdl->iajb", t2, t2, t2_t) -
               4 * einsum("akbj,cidl,ckdl->iajb", t2, t2, t2_t) -
               4 * einsum("akci,bjdl,ckdl->iajb", t2, t2, t2_t) -
               4 * einsum("alck,bjdi,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("ajbk,cidl,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("ajci,bkdl,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("ajck,bldi,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("akbi,cjdl,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("akbl,cidj,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("akci,bldj,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("akcj,bidl,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("alcj,bkdi,ckdl->iajb", t2, t2, t2_t) +
               2 * einsum("alck,bidj,ckdl->iajb", t2, t2, t2_t)

    # d_iabj =
    # - 2 ∑_kcl(δ_ij t_akcl tᵗ_bkcl)
    #
    # + 4 ∑_ck(t_aick tᵗ_bjck)
    # - 2 ∑_kc(t_akci tᵗ_bjck)

    diag_elem = 2 * einsum("akcl,bkcl->ab", t2, t2_t)

    for i in o
        d_ovvo[i, :, :, i] .-= diag_elem
    end

    d_ovvo .+= 4 * einsum("aick,bjck->iabj", t2, t2_t) -
               2 * einsum("akci,bjck->iabj", t2, t2_t)

    permutedims!(d_voov, d_ovvo, (3, 4, 1, 2))

    # d_iabc =
    # 2 ∑_j(t_aicj tᴸ_bj)
    # - ∑_j(t_ajci tᴸ_bj)

    d_ovvv .+= 2 * einsum("aicj,bj->iabc", t2, t1_bar) -
               1 * einsum("ajci,bj->iabc", t2, t1_bar)

    permutedims!(d_vvov, d_ovvv, (3, 4, 1, 2))

    # d_aibj =
    # 2 tᵗ_aibj

    d_vovo .+= 2 * t2_t

    # d_aibc =
    # 0

    # d_abcd =
    # 2 ∑_ij(t_bidj tᵗ_aicj)

    d_vvvv .+= 2 * einsum("aicj,bidj->abcd", t2_t, t2)

    d
end
