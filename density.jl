# T1 transformed densities

function one_electron_density(p::QED_CCSD_PARAMS)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    t2 = p.t2
    u2 = p.u2
    s1 = p.s1
    s2 = p.s2
    v2 = p.v2

    t1_bar = p.t1_bar
    t2_t = p.t2_t
    u2_t = p.u2_t
    s1_bar = p.s1_bar
    s2_t = p.s2_t
    v2_t = p.v2_t
    γ = p.γ
    γ_bar = p.γ_bar

    p.D_e = zeros(p.nao, p.nao)

    D = p.D_e

    D_oo = @view D[o, o]
    D_ov = @view D[o, v]
    D_vo = @view D[v, o]
    D_vv = @view D[v, v]

    # D0:

    # D0_ij = 2 δ_ij
    # - ∑_abk(t_aibk ᵗt_ajbk)

    for i in o
        D[i, i] = 2.0
    end

    D_oo .-= 1 * einsum("aibk,ajbk->ij", t2, t2_t)

    # D0_ia = ∑_bj(u_aibj ᴸt_bj)

    D_ov .+= einsum("aibj,bj->ia", u2, t1_bar)

    # D0_ai = ᴸt_ai

    D_vo .+= t1_bar

    # D0_ab = ∑_icj(t_bicj ᵗt_aicj)

    D_vv .+= 1 * einsum("bicj,aicj->ab", t2, t2_t)

    # D1:

    # D1_ij = - ∑_a(s_ai ᴸs_aj)
    # - ∑_abk(s_aibk ᵗs_ajbk)

    D_oo .-= einsum("ai,aj->ij", s1, s1_bar) +
             einsum("aibk,ajbk->ij", s2, s2_t)

    # D1_ia = 2 s_ai ᴸγ
    # + 1 ∑_bj(v_aibj ᴸs_bj)
    # - 1 ∑_jbck(s_aj t_ckbi ᵗs_ckbj)
    # - 1 ∑_bjck(s_bi t_ckaj ᵗs_ckbj)

    D_ov .+= 2 * s1' * γ_bar
    D_ov .+= 1 * einsum("aibj,bj->ia", v2, s1_bar)
    D_ov .-= 1 * einsum("aj,bick,bjck->ia", s1, t2, s2_t)
    D_ov .-= 1 * einsum("bi,ajck,bjck->ia", s1, t2, s2_t)

    # D1_ab =
    #   ∑_i(s_bi ᴸs_ai)
    # + ∑_icj(s_bicj ᵗs_aicj)

    D_vv .+= einsum("bi,ai->ab", s1, s1_bar)
    D_vv .+= einsum("bicj,aicj->ab", s2, s2_t)

    D
end

function one_electron_one_photon(p::QED_CCSD_PARAMS)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    t2 = p.t2
    u2 = p.u2
    s1 = p.s1
    s2 = p.s2
    v2 = p.v2

    t1_bar = p.t1_bar
    t2_t = p.t2_t
    u2_t = p.u2_t
    s1_bar = p.s1_bar
    s2_t = p.s2_t
    v2_t = p.v2_t
    γ = p.γ
    γ_bar = p.γ_bar

    p.D_ep = zeros(p.nao, p.nao)

    D = p.D_ep

    D_oo = @view D[o, o]
    D_ov = @view D[o, v]
    D_vo = @view D[v, o]
    D_vv = @view D[v, v]

    # b:
    # D0_ij = - ∑_a(s_ai tᴸ_aj)
    # - ∑_abk(s_aibk tᵗ_ajbk)
    # - ∑_abk(t_aibk tᵗ_ajbk γ)
    #
    # + 2 ∑_ak(δ_ij s_ak tᴸ_ak)
    # + ∑_akbl(δ_ij s_akbl tᵗ_akbl)
    # + 2 δ_ij γ

    diag_elem = 2 * γ + 2 * γ_bar +
                2 * einsum("ak,ak->", s1, t1_bar) +
                1 * einsum("akbl,akbl->", s2, t2_t)

    for i in o
        D[i, i] += diag_elem
    end

    # These can be fused
    D_oo .-= einsum("ai,aj->ij", s1, t1_bar)
    D_oo .-= einsum("ai,aj->ij", s1, s1_bar) * γ
    D_oo .-= einsum("ai,bk,ajbk->ij", s1, s1, s2_t)

    # These can be fused
    D_oo .-= einsum("aibk,ajbk->ij", s2, t2_t)
    D_oo .-= einsum("aibk,ajbk->ij", t2, t2_t) * γ

    # b:
    #  D1_ij = - ∑_abk(s_ai s_bk sᵗ_ajbk)
    #  - ∑_abk(s_aibk sᵗ_ajbk γ)
    #  - ∑_a(s_ai sᴸ_aj γ)

    # These can be fused
    D_oo .-= einsum("aibk,ajbk->ij", t2, s2_t)
    D_oo .-= einsum("aibk,ajbk->ij", s2, s2_t) * γ

    # b':
    # D1_ij = - ∑_abk(sᵗ_ajbk t_aibk)
    # + 2 δ_ij γᴸ

    # b:
    #  D0_ia = 2 s_ai
    #  + 1 ∑_bj(v_aibj tᴸ_bj)
    #
    #  + 1 ∑_bjck(s_bj u_aick tᵗ_bjck)
    #  - 1 ∑_jbck(s_aj t_bick tᵗ_bjck)
    #  - 1 ∑_bjck(s_bi t_ajck tᵗ_bjck)
    #
    #  + 1 ∑_bj(u_aibj tᴸ_bj γ)

    D_ov .+= s1' * (
        2 +
        2 * γ * γ_bar +
        2 * s1 ⋅ s1_bar +
        1 * s2 ⋅ s2_t
    )

    D_ov .-= 2 * einsum("aj,bi,bj->ia", s1, s1, s1_bar)

    # Can be fused
    D_ov .+= einsum("aibj,bj->ia", v2, t1_bar)
    D_ov .+= einsum("aibj,bj->ia", v2, s1_bar) * γ
    D_ov .+= einsum("aibj,ck,ckbj->ia", v2, s1, s2_t)

    # Can be fused
    D_ov .+= einsum("aibj,bj->ia", u2, s1_bar)
    D_ov .+= einsum("aibj,bj->ia", u2, t1_bar) * γ
    D_ov .+= einsum("aibj,ck,ckbj->ia", u2, s1, t2_t)

    D_ov .-= einsum("aj,bick,bjck->ia", s1, t2, t2_t)
    D_ov .-= einsum("aj,bick,bjck->ia", s1, t2, s2_t) * γ
    D_ov .-= einsum("aj,bick,bjck->ia", s1, s2, s2_t) * 2

    D_ov .-= einsum("bi,ajck,bjck->ia", s1, t2, t2_t)
    D_ov .-= einsum("bi,ajck,bjck->ia", s1, t2, s2_t) * γ
    D_ov .-= einsum("bi,ajck,bjck->ia", s1, s2, s2_t) * 2

    # b:
    #  D1_ia =
    #      s_ai (
    #           + 2 γ γᴸ
    #           + 2 ∑_bj(s_bj sᴸ_bj)
    #           + 1 ∑_bjck(s_bjck sᵗ_bjck)
    #      )
    #
    #  - 2 ∑_jb(s_aj s_bi sᴸ_bj)
    #
    #  + 1 ∑_bj(v_aibj sᴸ_bj γ)
    #
    #  + 1 ∑_bjck(v_aibj sᵗ_bjck s_ck)
    #
    #  - 2 ∑_ckbj(s_akbj s_ci sᵗ_ckbj)
    #  - 2 ∑_ckbj(s_ckbi s_aj sᵗ_ckbj)
    #
    #  - 1 ∑_bjck(t_bick s_aj sᵗ_bjck γ)
    #  - 1 ∑_bjck(t_akbj s_ci sᵗ_bjck γ)

    # b':
    #  D1_ia = ∑_bj(u_aibj sᴸ_bj)

    ###################################
    # recoding ov block

    # D0_ia +=
    # + 2 ∑_bj(Lt_bj s_aibj)
    # - 1 ∑_bj(Lt_bj s_ajbi)
    # + 2 ∑_bj(Lt_bj t_aibj γ)
    # - 1 ∑_bj(Lt_bj t_ajbi γ)
    # - 1 ∑_bjck(Lt_bjck s_aj t_bick)
    # - 1 ∑_bjck(Lt_bjck s_bi t_ajck)
    # + 2 ∑_bjck(Lt_bjck s_bj t_aick)
    # - 1 ∑_bjck(Lt_bjck s_bj t_akci)

    # D_ov .+= 2 * einsum("bj,aibj->ia", p.t1_bar, p.s2)
    # D_ov .-= 1 * einsum("bj,ajbi->ia", p.t1_bar, p.s2)
    # D_ov .+= 2 * einsum("bj,aibj->ia", p.t1_bar, p.t2) * p.γ
    # D_ov .-= 1 * einsum("bj,ajbi->ia", p.t1_bar, p.t2) * p.γ
    # D_ov .-= 1 * einsum("bjck,aj,bick->ia", p.t2_t, p.s1, p.t2)
    # D_ov .-= 1 * einsum("bjck,bi,ajck->ia", p.t2_t, p.s1, p.t2)
    # D_ov .+= 2 * einsum("bjck,bj,aick->ia", p.t2_t, p.s1, p.t2)
    # D_ov .-= 1 * einsum("bjck,bj,akci->ia", p.t2_t, p.s1, p.t2)

    # D1_ia +=
    # + 2 ∑_bj(Ls_bj t_aibj)
    # - 1 ∑_bj(Ls_bj t_ajbi)

    # D_ov .+= 2 * einsum("bj,aibj->ia", p.s1_bar, p.t2)
    # D_ov .-= 1 * einsum("bj,ajbi->ia", p.s1_bar, p.t2)

    # D_ia +=
    # + 2 Ls s_ai γ
    # + 2 ∑_bj(Ls_bj s_ai s_bj)
    # - 2 ∑_bj(Ls_bj s_aj s_bi)
    # + 2 ∑_bj(Ls_bj s_aibj γ)
    # - 1 ∑_bj(Ls_bj s_ajbi γ)
    # + 2 ∑_bjck(Ls_bjck s_aibj s_ck)
    # - 1 ∑_bjck(Ls_bjck s_ajbi s_ck)
    # - 2 ∑_bjck(Ls_bjck s_ajck s_bi)
    # + 1 ∑_bjck(Ls_bjck s_bjck s_ai)
    # - 2 ∑_bjck(Ls_bjck s_bick s_aj)
    # - 1 ∑_bjck(Ls_bjck s_aj t_bick γ)
    # - 1 ∑_bjck(Ls_bjck s_bi t_ajck γ)

    # D_ov .+= 2 * p.γ_bar * p.s1' * p.γ
    # D_ov .+= 2 * einsum("bj,ai,bj->ia", p.s1_bar, p.s1, p.s1)
    # D_ov .-= 2 * einsum("bj,aj,bi->ia", p.s1_bar, p.s1, p.s1)
    # D_ov .+= 2 * einsum("bj,aibj->ia", p.s1_bar, p.s2) * p.γ
    # D_ov .-= 1 * einsum("bj,ajbi->ia", p.s1_bar, p.s2) * p.γ
    # D_ov .+= 2 * einsum("bjck,aibj,ck->ia", p.s2_t, p.s2, p.s1)
    # D_ov .-= 1 * einsum("bjck,ajbi,ck->ia", p.s2_t, p.s2, p.s1)
    # D_ov .-= 2 * einsum("bjck,ajck,bi->ia", p.s2_t, p.s2, p.s1)
    # D_ov .+= 1 * einsum("bjck,bjck,ai->ia", p.s2_t, p.s2, p.s1)
    # D_ov .-= 2 * einsum("bjck,bick,aj->ia", p.s2_t, p.s2, p.s1)
    # D_ov .-= 1 * einsum("bjck,bick,aj->ia", p.s2_t, p.t2, p.s1) * p.γ
    # D_ov .-= 1 * einsum("bjck,ajck,bi->ia", p.s2_t, p.t2, p.s1) * p.γ

    ###################################

    # b:
    #  D0_ai = ∑_bj(s_bj tᵗ_aibj)
    #  + tᴸ_ai γ

    # b':
    # D1_ai = sᴸ_ai

    D_vo .+= t1_bar * γ + einsum("bj,aibj->ai", s1, t2_t) + s1_bar

    # b:
    # D0_ab = ∑_i(s_bi tᴸ_ai)
    #     + ∑_icj(s_bicj tᵗ_aicj)
    #     + ∑_icj(t_bicj tᵗ_aicj γ)

    D_vv .+= einsum("bi,ai->ab", s1, t1_bar)
    D_vv .+= einsum("bi,ai->ab", s1, s1_bar) * γ
    D_vv .+= einsum("bi,cj,aicj->ab", s1, s1, s2_t)

    D_vv .+= einsum("bicj,aicj->ab", s2, t2_t)
    D_vv .+= einsum("bicj,aicj->ab", t2, t2_t) * γ

    # b:
    #  D1_ab =
    #  + ∑_i(s_bi sᴸ_ai γ)
    #  + ∑_icj(s_bi s_cj sᵗ_aicj)
    #  + ∑_icj(s_bicj sᵗ_aicj γ)

    # b':
    #  D1_ab = ∑_icj(sᵗ_aicj t_bicj)

    D_vv .+= einsum("bicj,aicj->ab", t2, s2_t)
    D_vv .+= einsum("bicj,aicj->ab", s2, s2_t) * γ

    D
end

function one_electron_from_two_electron(mol, d)
    einsum("pqrr->pq", d) / (mol.nelectron - 1)
end

# Calculate ⟨Λ| b† b |CC⟩
function photon_density1(p::QED_CCSD_PARAMS)
    p.D_p1 = p.γ * p.γ_bar +
             einsum("ai,ai->", p.s1, p.s1_bar) +
             0.5 * einsum("aibj,aibj->", p.s2, p.s2_t)
end

# Calculate ⟨Λ| b† + b |CC⟩
function photon_density2(p::QED_CCSD_PARAMS)
    p.D_p2 = p.γ + p.γ_bar +
             einsum("ai,ai->", p.s1, p.t1_bar) +
             0.5 * einsum("aibj,aibj->", p.s2, p.t2_t)
end

function two_electron_density(p::QED_CCSD_PARAMS)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    t2 = p.t2
    u2 = p.u2
    s1 = p.s1
    s2 = p.s2
    v2 = p.v2

    t1_bar = p.t1_bar
    t2_t = p.t2_t
    u2_t = p.u2_t
    s1_bar = p.s1_bar
    s2_t = p.s2_t
    v2_t = p.v2_t
    γ = p.γ
    γ_bar = p.γ_bar

    p.d = zeros(p.nao, p.nao, p.nao, p.nao)

    d = p.d

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

    # - 2 ∑_abm(δ_kl t_aibm tᵗ_ajbm)
    # + 1 ∑_abm(δ_jk t_aibm tᵗ_albm)
    # + 1 ∑_abm(δ_il t_akbm tᵗ_ajbm)
    # - 2 ∑_abm(δ_ij t_akbm tᵗ_albm)

    diag_elem = einsum("aibm,ajbm->ij", t2, t2_t)

    for i in o
        d_oooo[:, :, i, i] .-= 2 * diag_elem
        d_oooo[:, i, i, :] .+= 1 * diag_elem
        d_oooo[i, :, :, i] .+= 1 * diag_elem'
        d_oooo[i, i, :, :] .-= 2 * diag_elem
    end

    # + ∑_ab(t_aibk tᵗ_ajbl)

    d_oooo .+= einsum("aibk,ajbl->ijkl", t2, t2_t)

    # d_ijka =
    # + 2 ∑_bl(δ_ij u_akbl tᴸ_bl)
    # -   ∑_bl(δ_jk u_aibl tᴸ_bl)
    #
    # -   ∑_b(u_akbi tᴸ_bj)

    diag_elem = einsum("aibl,bl->ia", u2, t1_bar)

    for i in o
        d_ooov[i, i, :, :] .+= 2 * diag_elem
        d_ooov[:, i, i, :] .-= diag_elem
    end

    d_ooov .-= einsum("akbi,bj->ijka", u2, t1_bar)

    # d_ijak =
    # + 2 δ_ij tᴸ_ak
    # -   δ_ik tᴸ_aj

    for i in o
        d_oovo[i, i, :, :] .+= 2 * t1_bar
        d_oovo[i, :, :, i] .-= 1 * t1_bar'
    end

    # d_ijab =
    # + 2 ∑_ckl(δ_ij t_bkcl tᵗ_akcl)
    # - 1 ∑_ck(t_bick tᵗ_ajck)
    # - 1 ∑_ck(t_bkci tᵗ_akcj)

    diag_elem = 2 * einsum("akcl,bkcl->ab", t2_t, t2)

    for i in o
        d_oovv[i, i, :, :] .+= diag_elem
    end

    d_oovv .-= einsum("ajck,bick->ijab", t2_t, t2) +
               einsum("akcj,bkci->ijab", t2_t, t2)

    # d_iajb =
    #   2 u_aibj
    #
    # + ∑_ckdl(u_aick u_bjdl tᵗ_ckdl)
    # - ∑_ckdl(t_ajck u_bidl tᵗ_ckdl)
    # - ∑_kcld(t_alck u_bjdi tᵗ_ckdl)
    # - ∑_kcdl(t_cjdl u_aibk tᵗ_ckdl)
    # - ∑_ckdl(t_bkdl u_aicj tᵗ_ckdl)
    # - ∑_kcdl(t_cidl u_akbj tᵗ_ckdl)
    # + ∑_klcd(t_akbl t_cidj tᵗ_ckdl)
    # + ∑_kcdl(t_akcj t_bidl tᵗ_ckdl)
    # + ∑_kcld(t_alcj t_bkdi tᵗ_ckdl)

    d_ovov .+= 2 * permutedims(u2, (2, 1, 4, 3)) +
               einsum("aick,bjdl,ckdl->iajb", u2, u2, t2_t) -
               einsum("ajck,bidl,ckdl->iajb", t2, u2, t2_t) -
               einsum("alck,bjdi,ckdl->iajb", t2, u2, t2_t) -
               einsum("cjdl,aibk,ckdl->iajb", t2, u2, t2_t) -
               einsum("bkdl,aicj,ckdl->iajb", t2, u2, t2_t) -
               einsum("cidl,akbj,ckdl->iajb", t2, u2, t2_t) +
               einsum("akbl,cidj,ckdl->iajb", t2, t2, t2_t) +
               einsum("akcj,bidl,ckdl->iajb", t2, t2, t2_t) +
               einsum("alcj,bkdi,ckdl->iajb", t2, t2, t2_t)

    # d_iabj =
    # - ∑_kcl(δ_ij t_akcl tᵗ_bkcl)
    #
    # + ∑_ck(u_aick tᵗ_bjck)

    diag_elem = einsum("akcl,bkcl->ab", t2, t2_t)

    for i in o
        d_ovvo[i, :, :, i] .-= diag_elem
    end

    d_ovvo .+= einsum("aick,bjck->iabj", u2, t2_t)

    # d_iabc =
    #   ∑_j(u_aicj tᴸ_bj)

    d_ovvv .+= einsum("aicj,bj->iabc", u2, t1_bar)

    # d_aibj =
    # tᵗ_aibj

    d_vovo .+= t2_t

    # d_aibc =
    # 0

    # d_abcd =
    # ∑_ij(t_bidj tᵗ_aicj)

    d_vvvv .+= einsum("aicj,bidj->abcd", t2_t, t2)

    # Λ1:

    # d_ijkl =
    # - 2 ∑_a(δ_ij s_ak sᴸ_al)
    # + 1 ∑_a(δ_il s_ak sᴸ_aj)
    #
    # - 2 ∑_a(δ_kl s_ai sᴸ_aj)
    # + 1 ∑_a(δ_jk s_ai sᴸ_al)
    #
    # - 2 ∑_abm(δ_ij s_akbm sᵗ_albm)
    # - 2 ∑_abm(δ_kl s_aibm sᵗ_ajbm)
    # + 1 ∑_abm(δ_il s_akbm sᵗ_ajbm)
    # + 1 ∑_abm(δ_jk s_aibm sᵗ_albm)
    #
    # + 1 ∑_ab(s_aibk sᵗ_ajbl)

    diag_elem = einsum("ak,al->kl", s1, s1_bar) +
                einsum("akbm,albm->kl", s2, s2_t)

    for i in o
        d_oooo[i, i, :, :] .-= 2 * diag_elem
        d_oooo[:, :, i, i] .-= 2 * diag_elem

        d_oooo[i, :, :, i] .+= 1 * diag_elem'
        d_oooo[:, i, i, :] .+= 1 * diag_elem
    end

    d_oooo .+= einsum("aibk,ajbl->ijkl", s2, s2_t)

    # d_ijka =
    # + 4 δ_ij s_ak γᴸ
    # - 2 δ_jk s_ai γᴸ
    #
    # + 2 ∑_bl(δ_ij v_akbl sᴸ_bl)
    # - 1 ∑_bl(δ_jk v_aibl sᴸ_bl)
    #
    # - 2 ∑_lbcm(δ_ij s_al t_bkcm sᵗ_blcm)
    # - 2 ∑_blcm(δ_ij s_bk t_alcm sᵗ_blcm)
    # + 1 ∑_lbcm(δ_jk s_al t_bicm sᵗ_blcm)
    # + 1 ∑_blcm(δ_jk s_bi t_alcm sᵗ_blcm)
    #
    # - 1 ∑_b(v_akbi sᴸ_bj)
    #
    # - 2 s_ak ∑_bcl(t_bicl sᵗ_bjcl)
    # + 1 s_ai ∑_bcl(t_bkcl sᵗ_bjcl)
    #
    # - 1 ∑_bcl(s_bi u_akcl sᵗ_bjcl)
    # + 1 ∑_bcl(s_al t_bick sᵗ_bjcl)
    # + 1 ∑_bcl(s_bk t_aicl sᵗ_bjcl)
    # + 1 ∑_bcl(s_ck t_albi sᵗ_bjcl)

    diag_elem = 2 * s1' * γ_bar
    diag_elem += einsum("akbl,bl->ka", v2, s1_bar)
    diag_elem += -einsum("al,bkcm,blcm->ka", s1, t2, s2_t)
    diag_elem += -einsum("bk,alcm,blcm->ka", s1, t2, s2_t)

    for i in o
        d_ooov[i, i, :, :] .+= 2 * diag_elem
        d_ooov[:, i, i, :] .-= 1 * diag_elem
    end

    d_ooov .-= 2 * einsum("ak,bicl,bjcl->ijka", s1, t2, s2_t)
    d_ooov .+= 1 * einsum("ai,bkcl,bjcl->ijka", s1, t2, s2_t)

    d_ooov .-= 1 * einsum("akbi,bj->ijka", v2, s1_bar)

    d_ooov .-= 1 * einsum("bi,akcl,bjcl->ijka", s1, u2, s2_t)
    d_ooov .+= 1 * einsum("bk,aicl,bjcl->ijka", s1, t2, s2_t)
    d_ooov .+= 1 * einsum("al,bick,bjcl->ijka", s1, t2, s2_t)
    d_ooov .+= 1 * einsum("ck,albi,bjcl->ijka", s1, t2, s2_t)

    # d_ijak =
    # - ∑_b(s_bi sᵗ_bjak)

    d_oovo .-= einsum("bi,bjak->ijak", s1, s2_t)

    # tmp = -einsum("bi,bjak->ijak", s1, s2_t)
    # @show maximum(abs, reshape(tmp, p.nocc^2, p.nocc * p.nvir) .-
    #                    get_matrix("d_vooo", "tmp_eT/ccsd"))

    # d_ijab =
    # + 2 ∑_k(δ_ij s_bk sᴸ_ak)
    # + 2 ∑_kcl(δ_ij s_bkcl sᵗ_akcl)
    #
    # - s_bi sᴸ_aj
    # - ∑_ck(s_bick sᵗ_ajck)
    # - ∑_kc(s_bkci sᵗ_akcj)

    diag_elem = 2 * einsum("ak,bk->ab", s1_bar, s1) +
                2 * einsum("akcl,bkcl->ab", s2_t, s2)

    for i in o
        d_oovv[i, i, :, :] .+= diag_elem
    end

    d_oovv .-= einsum("bi,aj->ijab", s1, s1_bar) +
               einsum("bick,ajck->ijab", s2, s2_t) +
               einsum("bkci,akcj->ijab", s2, s2_t)

    # d_iajb =
    # + 2 v_aibj γᴸ
    #
    # + 2 ∑_ck(s_ai u_bjck sᴸ_ck)
    # + 2 ∑_ck(s_bj u_aick sᴸ_ck)
    # - 1 ∑_ck(s_aj u_bick sᴸ_ck)
    # - 1 ∑_ck(s_ak u_bjci sᴸ_ck)
    # - 1 ∑_ck(s_bi u_ajck sᴸ_ck)
    # - 1 ∑_ck(s_bk u_aicj sᴸ_ck)
    # - 1 ∑_ck(s_ci u_akbj sᴸ_ck)
    # - 1 ∑_ck(s_cj u_aibk sᴸ_ck)
    #
    # + 1 ∑_ckdl(v_aick u_bjdl sᵗ_ckdl)
    # + 1 ∑_ckdl(v_bjck u_aidl sᵗ_ckdl)
    #
    # - 1 ∑_ckdl(v_aibk t_cjdl sᵗ_ckdl)
    # - 1 ∑_ckdl(v_aicj t_bkdl sᵗ_ckdl)
    # - 1 ∑_ckdl(v_ajck t_bidl sᵗ_ckdl)
    # - 1 ∑_ckdl(v_akbj t_cidl sᵗ_ckdl)
    # - 1 ∑_ckdl(v_bjci t_akdl sᵗ_ckdl)
    #
    # - 1 ∑_ckdl(s_alck u_bjdi sᵗ_ckdl)
    # - 1 ∑_ckdl(s_bick u_ajdl sᵗ_ckdl)
    # - 1 ∑_ckdl(s_blck u_aidj sᵗ_ckdl)
    # - 1 ∑_ckdl(s_cidl u_akbj sᵗ_ckdl)
    # - 1 ∑_ckdl(s_cjdl u_aibk sᵗ_ckdl)
    #
    # + 1 ∑_ckdl(s_ajck t_bldi sᵗ_ckdl)
    # + 1 ∑_ckdl(s_akbl t_cidj sᵗ_ckdl)
    # + 1 ∑_ckdl(s_alcj t_bkdi sᵗ_ckdl)
    # + 1 ∑_ckdl(s_bkci t_ajdl sᵗ_ckdl)
    # + 1 ∑_ckdl(s_blci t_akdj sᵗ_ckdl)
    # + 1 ∑_ckdl(s_cidj t_akbl sᵗ_ckdl)

    d_ovov_b = zeros(size(d_ovov))

    d_ovov_b .+= 2 * permutedims(v2, (2, 1, 4, 3)) * γ_bar

    d_ovov_b .+= 2 * einsum("ai,bjck,ck->iajb", s1, u2, s1_bar)
    d_ovov_b .+= 2 * einsum("bj,aick,ck->iajb", s1, u2, s1_bar)
    d_ovov_b .-= 1 * einsum("aj,bick,ck->iajb", s1, u2, s1_bar)
    d_ovov_b .-= 1 * einsum("ak,bjci,ck->iajb", s1, u2, s1_bar)
    d_ovov_b .-= 1 * einsum("bi,ajck,ck->iajb", s1, u2, s1_bar)
    d_ovov_b .-= 1 * einsum("bk,aicj,ck->iajb", s1, u2, s1_bar)
    d_ovov_b .-= 1 * einsum("ci,akbj,ck->iajb", s1, u2, s1_bar)
    d_ovov_b .-= 1 * einsum("cj,aibk,ck->iajb", s1, u2, s1_bar)

    du_ovov = zeros(size(d_ovov))

    du_ovov .+= 1 * einsum("bjck,aidl,ckdl->iajb", v2, u2, s2_t)

    du_ovov .-= 1 * einsum("alck,bjdi,ckdl->iajb", s2, u2, s2_t)
    du_ovov .-= 1 * einsum("bjci,akdl,ckdl->iajb", v2, t2, s2_t)

    du_ovov .-= 1 * einsum("cjdl,aibk,ckdl->iajb", s2, u2, s2_t)
    du_ovov .-= 1 * einsum("aibk,cjdl,ckdl->iajb", v2, t2, s2_t)

    du_ovov .+= 1 * einsum("blci,akdj,ckdl->iajb", s2, t2, s2_t)

    ds_ovov = zeros(size(d_ovov))

    ds_ovov .-= 1 * einsum("bick,ajdl,ckdl->iajb", s2, u2, s2_t)
    ds_ovov .+= 1 * einsum("ajck,bldi,ckdl->iajb", s2, t2, s2_t)

    ds_ovov .-= 1 * einsum("ajck,bidl,ckdl->iajb", v2, t2, s2_t)
    ds_ovov .+= 1 * einsum("bkci,ajdl,ckdl->iajb", s2, t2, s2_t)

    ds_ovov .+= 1 * einsum("akbl,cidj,ckdl->iajb", s2, t2, s2_t)

    ds_ovov .+= 1 * einsum("cidj,akbl,ckdl->iajb", s2, t2, s2_t)

    d_ovov_b .+= du_ovov + permutedims(du_ovov, (3, 4, 1, 2)) + ds_ovov

    d_ovov .+= d_ovov_b

    # d_iabj =
    # 2 s_ai sᴸ_bj
    #
    # - 1 ∑_k(δ_ij s_ak sᴸ_bk)
    # - 1 ∑_kcl(δ_ij s_akcl sᵗ_bkcl)
    #
    # + 1 ∑_ck(v_aick sᵗ_bjck)

    diag_elem = einsum("ak,bk->ab", s1, s1_bar) +
                einsum("akcl,bkcl->ab", s2, s2_t)

    for i in o
        d_ovvo[i, :, :, i] .-= diag_elem
    end

    d_ovvo .+= 2 * einsum("ai,bj->iabj", s1, s1_bar) +
               1 * einsum("aick,bjck->iabj", v2, s2_t)

    # d_iabc =
    # + 1 ∑_j(v_aicj sᴸ_bj)
    #
    # + 1 ∑_jdk(s_cj u_aidk sᵗ_bjdk)
    # + 2 ∑_jdk(s_ai t_cjdk sᵗ_bjdk)
    # - 1 ∑_jdk(s_ci t_ajdk sᵗ_bjdk)
    # - 1 ∑_jdk(s_aj t_cidk sᵗ_bjdk)
    # - 1 ∑_jdk(s_ak t_cjdi sᵗ_bjdk)
    # - 1 ∑_jdk(s_di t_akcj sᵗ_bjdk)

    d_ovvv_b = zeros(size(d_ovvv))

    d_ovvv_b .+= 1 * einsum("aicj,bj->iabc", v2, s1_bar)

    d_ovvv_b .+= 1 * einsum("cj,aidk,bjdk->iabc", s1, u2, s2_t)

    d_ovvv_b .+= 2 * einsum("ai,cjdk,bjdk->iabc", s1, t2, s2_t)
    d_ovvv_b .-= 1 * einsum("ci,ajdk,bjdk->iabc", s1, t2, s2_t)

    d_ovvv_b .-= 1 * einsum("aj,cidk,bjdk->iabc", s1, t2, s2_t)
    d_ovvv_b .-= 1 * einsum("aj,ckdi,bkdj->iabc", s1, t2, s2_t)

    d_ovvv_b .-= 1 * einsum("di,akcj,bjdk->iabc", s1, t2, s2_t)

    d_ovvv .+= d_ovvv_b

    # d_aibc =
    # ∑_j(s_cj sᵗ_aibj)

    d_vovv_b = einsum("cj,aibj->aibc", s1, s2_t)

    d_vovv .+= d_vovv_b

    # d_abcd =
    # ∑_ij(s_bidj sᵗ_aicj)

    d_vvvv_b = einsum("bidj,aicj->abcd", s2, s2_t)

    d_vvvv .+= d_vvvv_b

    permutedims!(d_ovoo, d_ooov, (3, 4, 1, 2))
    permutedims!(d_vooo, d_oovo, (3, 4, 1, 2))
    permutedims!(d_vvoo, d_oovv, (3, 4, 1, 2))
    permutedims!(d_voov, d_ovvo, (3, 4, 1, 2))
    permutedims!(d_vvov, d_ovvv, (3, 4, 1, 2))
    permutedims!(d_vvvo, d_vovv, (3, 4, 1, 2))

    d
end
