
function omega_t1(h, g, d, d_exp, p::QED_CCSD_PARAMS)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    Ω = zeros(p.nvir, p.nocc)

    # h: Ω0_ai +=
    # h_ai
    # + 2 ∑_jb(h_jb t_aibj)
    # - ∑_jb(h_jb t_ajbi)

    Ω += h[v, o]
    Ω += 2 * einsum("jb,aibj->ai", h[o, v], p.t2)
    Ω -= 1 * einsum("jb,ajbi->ai", h[o, v], p.t2)

    # g: Ω0_ai +=
    # 2 ∑_j(g_aijj)
    # - ∑_j(g_ajji)
    # + 2 ∑_bjc(g_abjc t_bicj)
    # - ∑_bjc(g_abjc t_bjci)
    # - 2 ∑_jkb(g_jikb t_ajbk)
    # + ∑_jkb(g_jikb t_akbj)
    # + 4 ∑_jkb(g_jjkb t_aibk)
    # - 2 ∑_jkb(g_jjkb t_akbi)
    # - 2 ∑_jkb(g_jkkb t_aibj)
    # + ∑_jkb(g_jkkb t_ajbi)

    for a in v, i in o, j in o
        Ω[a-p.nocc, i] += 2 * g[a, i, j, j] - g[a, j, j, i]
    end

    Ω += 2 * einsum("abjc,bicj->ai", g[v, v, o, v], p.t2)
    Ω -= 1 * einsum("abjc,bjci->ai", g[v, v, o, v], p.t2)
    Ω -= 2 * einsum("jikb,ajbk->ai", g[o, o, o, v], p.t2)
    Ω += 1 * einsum("jikb,akbj->ai", g[o, o, o, v], p.t2)
    Ω += 4 * einsum("jjkb,aibk->ai", g[o, o, o, v], p.t2)
    Ω -= 2 * einsum("jjkb,akbi->ai", g[o, o, o, v], p.t2)
    Ω -= 2 * einsum("jkkb,aibj->ai", g[o, o, o, v], p.t2)
    Ω += 1 * einsum("jkkb,ajbi->ai", g[o, o, o, v], p.t2)

    # ω: Ω0_ai +=
    # 0


    # ed:Ω0_ai +=
    # d_ai γ √ω/2
    # +   ∑_b(d_ab s_bi √ω/2)
    # -   ∑_j(d_ji s_aj √ω/2)
    # + 2 ∑_j(d_jj s_ai √ω/2)
    # + 2 ∑_jb(d_jb s_aibj √ω/2)
    # -   ∑_jb(d_jb s_ajbi √ω/2)
    # + 2 ∑_jb(d_jb t_aibj γ √ω/2)
    # -   ∑_jb(d_jb t_ajbi γ √ω/2)

    Ω += d[v, o] * p.γ * √(p.ω / 2)
    Ω += 1 * einsum("ab,bi->ai", d[v, v], p.s1) * √(p.ω / 2)
    Ω -= 1 * einsum("ji,aj->ai", d[o, o], p.s1) * √(p.ω / 2)
    Ω += 2 * einsum("jj,ai->ai", d[o, o], p.s1) * √(p.ω / 2)
    # Ω += d_exp * p.s1 * √(p.ω / 2)
    Ω += 2 * einsum("jb,aibj->ai", d[o, v], p.s2) * √(p.ω / 2)
    Ω -= 1 * einsum("jb,ajbi->ai", d[o, v], p.s2) * √(p.ω / 2)
    Ω += 2 * einsum("jb,aibj->ai", d[o, v], p.t2) * p.γ * √(p.ω / 2)
    Ω -= 1 * einsum("jb,ajbi->ai", d[o, v], p.t2) * p.γ * √(p.ω / 2)

    # d: Ω0_ai -=
    # s_ai √ω/2 ⟨d⟩

    Ω -= p.s1 * √(p.ω / 2) * d_exp

    Ω
end

function omega_t2(h, g, d, d_exp, p::QED_CCSD_PARAMS)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    Ω = zeros(p.nvir, p.nocc, p.nvir, p.nocc)

    # h: Ω0_aibj +=
    # + ∑_c(h_ac t_bjci)
    # + ∑_c(h_bc t_aicj)
    # - ∑_k(h_ki t_akbj)
    # - ∑_k(h_kj t_aibk)

    Ω += einsum("ac,bjci->aibj", h[v, v], p.t2)
    Ω += einsum("bc,aicj->aibj", h[v, v], p.t2)
    Ω -= einsum("ki,akbj->aibj", h[o, o], p.t2)
    Ω -= einsum("kj,aibk->aibj", h[o, o], p.t2)

    # g: Ω0_aibj +=
    # g_aibj
    # + 2 ∑_kc(g_aikc t_bjck)
    # - 1 ∑_kc(g_aikc t_bkcj)
    # + 1 ∑_cd(g_acbd t_cidj)
    # - 1 ∑_kc(g_akkc t_bjci)
    # - 1 ∑_ck(g_acki t_bjck)
    # - 1 ∑_ck(g_ackj t_bkci)
    # + 2 ∑_ck(g_ackk t_bjci)
    # + 2 ∑_kc(g_bjkc t_aick)
    # - 1 ∑_kc(g_bjkc t_akci)
    # - 1 ∑_kc(g_bkkc t_aicj)
    # - 1 ∑_ck(g_bcki t_akcj)
    # - 1 ∑_ck(g_bckj t_aick)
    # + 2 ∑_ck(g_bckk t_aicj)
    # + 1 ∑_kl(g_kilj t_akbl)
    # + 1 ∑_kl(g_kilk t_albj)
    # - 2 ∑_kl(g_kill t_akbj)
    # + 1 ∑_kl(g_kjlk t_aibl)
    # - 2 ∑_kl(g_kjll t_aibk)
    # - 2 ∑_kcld(g_kcld t_aibk t_cjdl)
    # + 1 ∑_kcld(g_kcld t_aibk t_cldj)
    # - 2 ∑_kcld(g_kcld t_aicj t_bkdl)
    # + 1 ∑_kcld(g_kcld t_aicj t_bldk)
    # + 4 ∑_kcld(g_kcld t_aick t_bjdl)
    # - 2 ∑_kcld(g_kcld t_aick t_bldj)
    # - 2 ∑_kcld(g_kcld t_aicl t_bjdk)
    # + 1 ∑_kcld(g_kcld t_aicl t_bkdj)
    # - 2 ∑_kcld(g_kcld t_akbj t_cidl)
    # + 1 ∑_kcld(g_kcld t_akbj t_cldi)
    # + 1 ∑_kcld(g_kcld t_akbl t_cidj)
    # - 2 ∑_kcld(g_kcld t_akci t_bjdl)
    # + 1 ∑_kcld(g_kcld t_akci t_bldj)
    # + 1 ∑_kcld(g_kcld t_akcl t_bjdi)
    # + 1 ∑_kcld(g_kcld t_akdi t_bjcl)
    # + 1 ∑_kcld(g_kcld t_akdj t_blci)
    # - 2 ∑_kcld(g_kcld t_akdl t_bjci)

    Ω += g[v, o, v, o]
    Ω += 2 * einsum("aikc,bjck->aibj", g[v, o, o, v], p.t2)
    Ω -= 1 * einsum("aikc,bkcj->aibj", g[v, o, o, v], p.t2)
    Ω += 1 * einsum("acbd,cidj->aibj", g[v, v, v, v], p.t2)
    Ω -= 1 * einsum("akkc,bjci->aibj", g[v, o, o, v], p.t2)
    Ω -= 1 * einsum("acki,bjck->aibj", g[v, v, o, o], p.t2)
    Ω -= 1 * einsum("ackj,bkci->aibj", g[v, v, o, o], p.t2)
    Ω += 2 * einsum("ackk,bjci->aibj", g[v, v, o, o], p.t2)
    Ω += 2 * einsum("bjkc,aick->aibj", g[v, o, o, v], p.t2)
    Ω -= 1 * einsum("bjkc,akci->aibj", g[v, o, o, v], p.t2)
    Ω -= 1 * einsum("bkkc,aicj->aibj", g[v, o, o, v], p.t2)
    Ω -= 1 * einsum("bcki,akcj->aibj", g[v, v, o, o], p.t2)
    Ω -= 1 * einsum("bckj,aick->aibj", g[v, v, o, o], p.t2)
    Ω += 2 * einsum("bckk,aicj->aibj", g[v, v, o, o], p.t2)
    Ω += 1 * einsum("kilj,akbl->aibj", g[o, o, o, o], p.t2)
    Ω += 1 * einsum("kilk,albj->aibj", g[o, o, o, o], p.t2)
    Ω -= 2 * einsum("kill,akbj->aibj", g[o, o, o, o], p.t2)
    Ω += 1 * einsum("kjlk,aibl->aibj", g[o, o, o, o], p.t2)
    Ω -= 2 * einsum("kjll,aibk->aibj", g[o, o, o, o], p.t2)
    Ω -= 2 * einsum("kcld,aibk,cjdl->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,aibk,cldj->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω -= 2 * einsum("kcld,aicj,bkdl->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,aicj,bldk->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 4 * einsum("kcld,aick,bjdl->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω -= 2 * einsum("kcld,aick,bldj->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω -= 2 * einsum("kcld,aicl,bjdk->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,aicl,bkdj->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω -= 2 * einsum("kcld,akbj,cidl->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,akbj,cldi->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,akbl,cidj->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω -= 2 * einsum("kcld,akci,bjdl->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,akci,bldj->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,akcl,bjdi->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,akdi,bjcl->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω += 1 * einsum("kcld,akdj,blci->aibj", g[o, v, o, v], p.t2, p.t2)
    Ω -= 2 * einsum("kcld,akdl,bjci->aibj", g[o, v, o, v], p.t2, p.t2)

    # ω: Ω0_aibj +=
    # 0


    # ed:Ω0_aibj +=
    # d_ai s_bj √ω/2
    # + ∑_c(d_ac s_bjci √ω/2)
    # + d_bj s_ai √ω/2
    # + ∑_c(d_ac t_bjci γ √ω/2)
    # + ∑_c(d_bc s_aicj √ω/2)
    # + ∑_c(d_bc t_aicj γ √ω/2)
    # - ∑_k(d_ki s_akbj √ω/2)
    # - ∑_k(d_ki t_akbj γ √ω/2)
    # - ∑_k(d_kj s_aibk √ω/2)
    # - ∑_k(d_kj t_aibk γ √ω/2)
    # + 2 ∑_k(d_kk s_aibj √ω/2)
    # + 2 ∑_kc(d_kc s_ai t_bjck √ω/2)
    # - ∑_kc(d_kc s_ai t_bkcj √ω/2)
    # - ∑_kc(d_kc s_ak t_bjci √ω/2)
    # + 2 ∑_kc(d_kc s_bj t_aick √ω/2)
    # - ∑_kc(d_kc s_bj t_akci √ω/2)
    # - ∑_kc(d_kc s_bk t_aicj √ω/2)
    # - ∑_kc(d_kc s_ci t_akbj √ω/2)
    # - ∑_kc(d_kc s_cj t_aibk √ω/2)

    Ω += einsum("ai,bj->aibj", d[v, o], p.s1) * √(p.ω / 2)
    Ω += einsum("bj,ai->aibj", d[v, o], p.s1) * √(p.ω / 2)
    Ω += 1einsum("ac,bjci->aibj", d[v, v], p.s2) * √(p.ω / 2)
    Ω += 1einsum("bc,aicj->aibj", d[v, v], p.s2) * √(p.ω / 2)
    Ω -= 1einsum("ki,akbj->aibj", d[o, o], p.s2) * √(p.ω / 2)
    Ω -= 1einsum("kj,aibk->aibj", d[o, o], p.s2) * √(p.ω / 2)
    Ω += 2einsum("kk,aibj->aibj", d[o, o], p.s2) * √(p.ω / 2)
    Ω += 1einsum("ac,bjci->aibj", d[v, v], p.t2) * p.γ * √(p.ω / 2)
    Ω += 1einsum("bc,aicj->aibj", d[v, v], p.t2) * p.γ * √(p.ω / 2)
    Ω -= 1einsum("ki,akbj->aibj", d[o, o], p.t2) * p.γ * √(p.ω / 2)
    Ω -= 1einsum("kj,aibk->aibj", d[o, o], p.t2) * p.γ * √(p.ω / 2)
    Ω += 2einsum("kc,ai,bjck->aibj", d[o, v], p.s1, p.t2) * √(p.ω / 2)
    Ω -= 1einsum("kc,ai,bkcj->aibj", d[o, v], p.s1, p.t2) * √(p.ω / 2)
    Ω -= 1einsum("kc,ak,bjci->aibj", d[o, v], p.s1, p.t2) * √(p.ω / 2)
    Ω += 2einsum("kc,bj,aick->aibj", d[o, v], p.s1, p.t2) * √(p.ω / 2)
    Ω -= 1einsum("kc,bj,akci->aibj", d[o, v], p.s1, p.t2) * √(p.ω / 2)
    Ω -= 1einsum("kc,bk,aicj->aibj", d[o, v], p.s1, p.t2) * √(p.ω / 2)
    Ω -= 1einsum("kc,ci,akbj->aibj", d[o, v], p.s1, p.t2) * √(p.ω / 2)
    Ω -= 1einsum("kc,cj,aibk->aibj", d[o, v], p.s1, p.t2) * √(p.ω / 2)

    # d: Ω0_aibj -=
    # s_aibj √ω/2 ⟨d⟩

    Ω -= p.s2 * √(p.ω / 2) * d_exp

    for a in 1:p.nvir, i in 1:p.nocc
        # Ω[a, i, a, i] *= 0.5
    end

    Ω
end

function omega_s0(h, g, d, d_exp, p::QED_CCSD_PARAMS)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    Ω = 0.0

    # h: Ω1 +=
    # 2 ∑_ia(h_ia s_ai)

    Ω += 2 * einsum("ia,ai->", h[o, v], p.s1)

    # g: Ω1 +=
    # + 4 ∑_ija(g_iija s_aj)
    # - 2 ∑_ija(g_ijja s_ai)
    # + 2 ∑_iajb(g_iajb s_aibj)
    # - 1 ∑_iajb(g_iajb s_ajbi)

    Ω += 4 * einsum("iija,aj->", g[o, o, o, v], p.s1)
    Ω -= 2 * einsum("ijja,ai->", g[o, o, o, v], p.s1)
    Ω += 2 * einsum("iajb,aibj->", g[o, v, o, v], p.s2)
    Ω -= 1 * einsum("iajb,ajbi->", g[o, v, o, v], p.s2)

    # ω: Ω1 +=
    # γ ω

    Ω += p.γ * p.ω

    # ed:Ω1 +=
    # + 2 ∑_i(d_ii √ω/2)
    # + 2 ∑_ia(d_ia s_ai γ √ω/2)

    Ω += 2 * einsum("ii->", d[o, o]) * √(p.ω / 2)
    Ω += 2 * einsum("ia,ai->", d[o, v], p.s1) * p.γ * √(p.ω / 2)

    # d: Ω1 -=
    # √ω/2 ⟨d⟩

    Ω -= √(p.ω / 2) * d_exp

    Ω
end

function omega_s1(h, g, d, d_exp, p::QED_CCSD_PARAMS)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    Ω = zeros(p.nvir, p.nocc)

    # h: Ω1_ai +=
    # + 1 ∑_b(h_ab s_bi)
    # - 1 ∑_j(h_ji s_aj)
    # + 2 ∑_jb(h_jb s_aibj)
    # - 1 ∑_jb(h_jb s_ajbi)

    Ω += 1 * einsum("ab,bi->ai", h[v, v], p.s1)
    Ω -= 1 * einsum("ji,aj->ai", h[o, o], p.s1)
    Ω += 2 * einsum("jb,aibj->ai", h[o, v], p.s2)
    Ω -= 1 * einsum("jb,ajbi->ai", h[o, v], p.s2)

    # g: Ω1_ai +=
    # + 2 ∑_jb(g_aijb s_bj)
    # - 1 ∑_jb(g_ajjb s_bi)
    # - 1 ∑_bj(g_abji s_bj)
    # + 2 ∑_bj(g_abjj s_bi)
    # + 1 ∑_jk(g_jikj s_ak)
    # - 2 ∑_jk(g_jikk s_aj)
    # + 2 ∑_bjc(g_abjc s_bicj)
    # - 1 ∑_bjc(g_abjc s_bjci)
    # - 2 ∑_jkb(g_jikb s_ajbk)
    # + 1 ∑_jkb(g_jikb s_akbj)
    # + 4 ∑_jkb(g_jjkb s_aibk)
    # - 2 ∑_jkb(g_jjkb s_akbi)
    # - 2 ∑_jkb(g_jkkb s_aibj)
    # + 1 ∑_jkb(g_jkkb s_ajbi)
    # - 2 ∑_jbkc(g_jbkc s_aj t_bick)
    # + 1 ∑_jbkc(g_jbkc s_aj t_bkci)
    # - 2 ∑_jbkc(g_jbkc s_bi t_ajck)
    # + 1 ∑_jbkc(g_jbkc s_bi t_akcj)
    # + 4 ∑_jbkc(g_jbkc s_bj t_aick)
    # - 2 ∑_jbkc(g_jbkc s_bj t_akci)
    # - 2 ∑_jbkc(g_jbkc s_bk t_aicj)
    # + 1 ∑_jbkc(g_jbkc s_bk t_ajci)

    Ω += 2 * einsum("aijb,bj->ai", g[v, o, o, v], p.s1)
    Ω -= 1 * einsum("ajjb,bi->ai", g[v, o, o, v], p.s1)
    Ω -= 1 * einsum("abji,bj->ai", g[v, v, o, o], p.s1)
    Ω += 2 * einsum("abjj,bi->ai", g[v, v, o, o], p.s1)
    Ω += 1 * einsum("jikj,ak->ai", g[o, o, o, o], p.s1)
    Ω -= 2 * einsum("jikk,aj->ai", g[o, o, o, o], p.s1)
    Ω += 2 * einsum("abjc,bicj->ai", g[v, v, o, v], p.s2)
    Ω -= 1 * einsum("abjc,bjci->ai", g[v, v, o, v], p.s2)
    Ω -= 2 * einsum("jikb,ajbk->ai", g[o, o, o, v], p.s2)
    Ω += 1 * einsum("jikb,akbj->ai", g[o, o, o, v], p.s2)
    Ω += 4 * einsum("jjkb,aibk->ai", g[o, o, o, v], p.s2)
    Ω -= 2 * einsum("jjkb,akbi->ai", g[o, o, o, v], p.s2)
    Ω -= 2 * einsum("jkkb,aibj->ai", g[o, o, o, v], p.s2)
    Ω += 1 * einsum("jkkb,ajbi->ai", g[o, o, o, v], p.s2)
    Ω -= 2 * einsum("jbkc,aj,bick->ai", g[o, v, o, v], p.s1, p.t2)
    Ω += 1 * einsum("jbkc,aj,bkci->ai", g[o, v, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("jbkc,bi,ajck->ai", g[o, v, o, v], p.s1, p.t2)
    Ω += 1 * einsum("jbkc,bi,akcj->ai", g[o, v, o, v], p.s1, p.t2)
    Ω += 4 * einsum("jbkc,bj,aick->ai", g[o, v, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("jbkc,bj,akci->ai", g[o, v, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("jbkc,bk,aicj->ai", g[o, v, o, v], p.s1, p.t2)
    Ω += 1 * einsum("jbkc,bk,ajci->ai", g[o, v, o, v], p.s1, p.t2)

    # ω: Ω1_ai +=
    # s_ai ω
    Ω += p.ω * p.s1


    # ed:Ω1_ai +=
    # + d_ai √ω/2
    # + 1 ∑_b(d_ab s_bi γ √ω/2)
    # - 1 ∑_j(d_ji s_aj γ √ω/2)
    # + 2 ∑_jb(d_jb s_ai s_bj √ω/2)
    # + 2 ∑_jb(d_jb s_aibj γ √ω/2)
    # - 2 ∑_jb(d_jb s_aj s_bi √ω/2)
    # - 1 ∑_jb(d_jb s_ajbi γ √ω/2)
    # + 2 ∑_jb(d_jb t_aibj √ω/2)
    # - 1 ∑_jb(d_jb t_ajbi √ω/2)

    Ω += d[v, o] * √(p.ω / 2)
    Ω += 1 * einsum("ab,bi->ai", d[v, v], p.s1) * p.γ * √(p.ω / 2)
    Ω -= 1 * einsum("ji,aj->ai", d[o, o], p.s1) * p.γ * √(p.ω / 2)
    Ω += 2 * einsum("jb,aibj->ai", d[o, v], p.s2) * p.γ * √(p.ω / 2)
    Ω -= 1 * einsum("jb,ajbi->ai", d[o, v], p.s2) * p.γ * √(p.ω / 2)
    Ω += 2 * einsum("jb,aibj->ai", d[o, v], p.t2) * √(p.ω / 2)
    Ω -= 1 * einsum("jb,ajbi->ai", d[o, v], p.t2) * √(p.ω / 2)
    Ω += 2 * einsum("jb,ai,bj->ai", d[o, v], p.s1, p.s1) * √(p.ω / 2)
    Ω -= 2 * einsum("jb,aj,bi->ai", d[o, v], p.s1, p.s1) * √(p.ω / 2)

    # d: Ω1_ai +=
    # 0

    Ω
end

function omega_s2(h, g, d, d_exp, p::QED_CCSD_PARAMS)
    o = 1:p.nocc
    v = p.nocc+1:p.nao

    Ω = zeros(p.nvir, p.nocc, p.nvir, p.nocc)

    # h: Ω1_aibj +=
    # + ∑_c(h_ac s_bjci)
    # + ∑_c(h_bc s_aicj)
    # - ∑_k(h_ki s_akbj)
    # - ∑_k(h_kj s_aibk)
    # - ∑_kc(h_kc s_ak t_bjci)
    # - ∑_kc(h_kc s_bk t_aicj)
    # - ∑_kc(h_kc s_ci t_akbj)
    # - ∑_kc(h_kc s_cj t_aibk)

    Ω += einsum("ac,bjci->aibj", h[v, v], p.s2)
    Ω += einsum("bc,aicj->aibj", h[v, v], p.s2)
    Ω -= einsum("ki,akbj->aibj", h[o, o], p.s2)
    Ω -= einsum("kj,aibk->aibj", h[o, o], p.s2)
    Ω -= einsum("kc,ak,bjci->aibj", h[o, v], p.s1, p.t2)
    Ω -= einsum("kc,bk,aicj->aibj", h[o, v], p.s1, p.t2)
    Ω -= einsum("kc,ci,akbj->aibj", h[o, v], p.s1, p.t2)
    Ω -= einsum("kc,cj,aibk->aibj", h[o, v], p.s1, p.t2)

    # g: Ω1_aibj +=
    # + 1 ∑_c(g_aibc s_cj)
    # + 1 ∑_c(g_acbj s_ci)
    # - 1 ∑_k(g_aikj s_bk)
    # - 1 ∑_k(g_bjki s_ak)
    # + 2 ∑_kc(g_aikc s_bjck)
    # - 1 ∑_kc(g_aikc s_bkcj)
    # + 1 ∑_cd(g_acbd s_cidj)
    # - 1 ∑_kc(g_akkc s_bjci)
    # - 1 ∑_ck(g_acki s_bjck)
    # - 1 ∑_ck(g_ackj s_bkci)
    # + 2 ∑_ck(g_ackk s_bjci)
    # + 2 ∑_kc(g_bjkc s_aick)
    # - 1 ∑_kc(g_bjkc s_akci)
    # - 1 ∑_kc(g_bkkc s_aicj)
    # - 1 ∑_ck(g_bcki s_akcj)
    # - 1 ∑_ck(g_bckj s_aick)
    # + 2 ∑_ck(g_bckk s_aicj)
    # + 1 ∑_kl(g_kilj s_akbl)
    # + 1 ∑_kl(g_kilk s_albj)
    # - 2 ∑_kl(g_kill s_akbj)
    # + 1 ∑_kl(g_kjlk s_aibl)
    # - 2 ∑_kl(g_kjll s_aibk)
    # - 1 ∑_ckd(g_ackd s_bk t_cidj)
    # + 2 ∑_ckd(g_ackd s_ci t_bjdk)
    # - 1 ∑_ckd(g_ackd s_ci t_bkdj)
    # - 1 ∑_ckd(g_ackd s_ck t_bjdi)
    # - 1 ∑_ckd(g_ackd s_di t_bjck)
    # - 1 ∑_ckd(g_ackd s_dj t_bkci)
    # + 2 ∑_ckd(g_ackd s_dk t_bjci)
    # - 1 ∑_ckd(g_bckd s_ak t_cjdi)
    # + 2 ∑_ckd(g_bckd s_cj t_aidk)
    # - 1 ∑_ckd(g_bckd s_cj t_akdi)
    # - 1 ∑_ckd(g_bckd s_ck t_aidj)
    # - 1 ∑_ckd(g_bckd s_di t_akcj)
    # - 1 ∑_ckd(g_bckd s_dj t_aick)
    # + 2 ∑_ckd(g_bckd s_dk t_aicj)
    # - 2 ∑_klc(g_kilc s_ak t_bjcl)
    # + 1 ∑_klc(g_kilc s_ak t_blcj)
    # + 1 ∑_klc(g_kilc s_al t_bjck)
    # + 1 ∑_klc(g_kilc s_bl t_akcj)
    # + 1 ∑_klc(g_kilc s_cj t_akbl)
    # + 1 ∑_klc(g_kilc s_ck t_albj)
    # - 2 ∑_klc(g_kilc s_cl t_akbj)
    # + 1 ∑_klc(g_kjlc s_al t_bkci)
    # - 2 ∑_klc(g_kjlc s_bk t_aicl)
    # + 1 ∑_klc(g_kjlc s_bk t_alci)
    # + 1 ∑_klc(g_kjlc s_bl t_aick)
    # + 1 ∑_klc(g_kjlc s_ci t_albk)
    # + 1 ∑_klc(g_kjlc s_ck t_aibl)
    # - 2 ∑_klc(g_kjlc s_cl t_aibk)
    # - 2 ∑_klc(g_kklc s_al t_bjci)
    # - 2 ∑_klc(g_kklc s_bl t_aicj)
    # - 2 ∑_klc(g_kklc s_ci t_albj)
    # - 2 ∑_klc(g_kklc s_cj t_aibl)
    # + 1 ∑_klc(g_kllc s_ak t_bjci)
    # + 1 ∑_klc(g_kllc s_bk t_aicj)
    # + 1 ∑_klc(g_kllc s_ci t_akbj)
    # + 1 ∑_klc(g_kllc s_cj t_aibk)
    # - 2 ∑_kcld(g_kcld s_aibk t_cjdl)
    # + 1 ∑_kcld(g_kcld s_aibk t_cldj)
    # - 2 ∑_kcld(g_kcld s_aicj t_bkdl)
    # + 1 ∑_kcld(g_kcld s_aicj t_bldk)
    # + 4 ∑_kcld(g_kcld s_aick t_bjdl)
    # - 2 ∑_kcld(g_kcld s_aick t_bldj)
    # - 2 ∑_kcld(g_kcld s_aicl t_bjdk)
    # + 1 ∑_kcld(g_kcld s_aicl t_bkdj)
    # - 2 ∑_kcld(g_kcld s_akbj t_cidl)
    # + 1 ∑_kcld(g_kcld s_akbj t_cldi)
    # + 1 ∑_kcld(g_kcld s_akbl t_cidj)
    # - 2 ∑_kcld(g_kcld s_akci t_bjdl)
    # + 1 ∑_kcld(g_kcld s_akci t_bldj)
    # + 1 ∑_kcld(g_kcld s_akcl t_bjdi)
    # + 1 ∑_kcld(g_kcld s_akdi t_bjcl)
    # + 1 ∑_kcld(g_kcld s_akdj t_blci)
    # - 2 ∑_kcld(g_kcld s_akdl t_bjci)
    # - 2 ∑_kcld(g_kcld s_bjci t_akdl)
    # + 1 ∑_kcld(g_kcld s_bjci t_aldk)
    # + 4 ∑_kcld(g_kcld s_bjck t_aidl)
    # - 2 ∑_kcld(g_kcld s_bjck t_aldi)
    # - 2 ∑_kcld(g_kcld s_bjcl t_aidk)
    # + 1 ∑_kcld(g_kcld s_bjcl t_akdi)
    # - 2 ∑_kcld(g_kcld s_bkcj t_aidl)
    # + 1 ∑_kcld(g_kcld s_bkcj t_aldi)
    # + 1 ∑_kcld(g_kcld s_bkcl t_aidj)
    # + 1 ∑_kcld(g_kcld s_bkdi t_alcj)
    # + 1 ∑_kcld(g_kcld s_bkdj t_aicl)
    # - 2 ∑_kcld(g_kcld s_bkdl t_aicj)
    # + 1 ∑_kcld(g_kcld s_cidj t_akbl)
    # + 1 ∑_kcld(g_kcld s_cidk t_albj)
    # - 2 ∑_kcld(g_kcld s_cidl t_akbj)
    # + 1 ∑_kcld(g_kcld s_cjdk t_aibl)
    # - 2 ∑_kcld(g_kcld s_cjdl t_aibk)

    Ω -= 1 * einsum("aikj,bk->aibj", g[v, o, o, o], p.s1)
    Ω -= 1 * einsum("bjki,ak->aibj", g[v, o, o, o], p.s1)
    Ω += 1 * einsum("aibc,cj->aibj", g[v, o, v, v], p.s1)
    Ω += 1 * einsum("acbj,ci->aibj", g[v, v, v, o], p.s1)
    Ω += 2 * einsum("aikc,bjck->aibj", g[v, o, o, v], p.s2)
    Ω -= 1 * einsum("aikc,bkcj->aibj", g[v, o, o, v], p.s2)
    Ω += 1 * einsum("acbd,cidj->aibj", g[v, v, v, v], p.s2)
    Ω -= 1 * einsum("akkc,bjci->aibj", g[v, o, o, v], p.s2)
    Ω -= 1 * einsum("acki,bjck->aibj", g[v, v, o, o], p.s2)
    Ω -= 1 * einsum("ackj,bkci->aibj", g[v, v, o, o], p.s2)
    Ω += 2 * einsum("ackk,bjci->aibj", g[v, v, o, o], p.s2)
    Ω += 2 * einsum("bjkc,aick->aibj", g[v, o, o, v], p.s2)
    Ω -= 1 * einsum("bjkc,akci->aibj", g[v, o, o, v], p.s2)
    Ω -= 1 * einsum("bkkc,aicj->aibj", g[v, o, o, v], p.s2)
    Ω -= 1 * einsum("bcki,akcj->aibj", g[v, v, o, o], p.s2)
    Ω -= 1 * einsum("bckj,aick->aibj", g[v, v, o, o], p.s2)
    Ω += 2 * einsum("bckk,aicj->aibj", g[v, v, o, o], p.s2)
    Ω += 1 * einsum("kilj,akbl->aibj", g[o, o, o, o], p.s2)
    Ω += 1 * einsum("kilk,albj->aibj", g[o, o, o, o], p.s2)
    Ω -= 2 * einsum("kill,akbj->aibj", g[o, o, o, o], p.s2)
    Ω += 1 * einsum("kjlk,aibl->aibj", g[o, o, o, o], p.s2)
    Ω -= 2 * einsum("kjll,aibk->aibj", g[o, o, o, o], p.s2)
    Ω -= 1 * einsum("ackd,bk,cidj->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω += 2 * einsum("ackd,ci,bjdk->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("ackd,ci,bkdj->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("ackd,ck,bjdi->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("ackd,di,bjck->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("ackd,dj,bkci->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω += 2 * einsum("ackd,dk,bjci->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("bckd,ak,cjdi->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω += 2 * einsum("bckd,cj,aidk->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("bckd,cj,akdi->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("bckd,ck,aidj->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("bckd,di,akcj->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 1 * einsum("bckd,dj,aick->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω += 2 * einsum("bckd,dk,aicj->aibj", g[v, v, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kilc,ak,bjcl->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kilc,ak,blcj->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kilc,al,bjck->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kilc,bl,akcj->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kilc,cj,akbl->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kilc,ck,albj->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kilc,cl,akbj->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kjlc,al,bkci->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kjlc,bk,aicl->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kjlc,bk,alci->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kjlc,bl,aick->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kjlc,ci,albk->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kjlc,ck,aibl->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kjlc,cl,aibk->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kklc,al,bjci->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kklc,bl,aicj->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kklc,ci,albj->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kklc,cj,aibl->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kllc,ak,bjci->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kllc,bk,aicj->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kllc,ci,akbj->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω += 1 * einsum("kllc,cj,aibk->aibj", g[o, o, o, v], p.s1, p.t2)
    Ω -= 2 * einsum("kcld,aibk,cjdl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,aibk,cldj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,aicj,bkdl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,aicj,bldk->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 4 * einsum("kcld,aick,bjdl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,aick,bldj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,aicl,bjdk->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,aicl,bkdj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,akbj,cidl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,akbj,cldi->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,akbl,cidj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,akci,bjdl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,akci,bldj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,akcl,bjdi->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,akdi,bjcl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,akdj,blci->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,akdl,bjci->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,bjci,akdl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,bjci,aldk->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 4 * einsum("kcld,bjck,aidl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,bjck,aldi->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,bjcl,aidk->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,bjcl,akdi->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,bkcj,aidl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,bkcj,aldi->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,bkcl,aidj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,bkdi,alcj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,bkdj,aicl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,bkdl,aicj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,cidj,akbl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,cidk,albj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,cidl,akbj->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω += 1 * einsum("kcld,cjdk,aibl->aibj", g[o, v, o, v], p.s2, p.t2)
    Ω -= 2 * einsum("kcld,cjdl,aibk->aibj", g[o, v, o, v], p.s2, p.t2)

    # ω: Ω1_aibj +=
    # s_aibj ω

    Ω += p.s2 * p.ω

    # ed:Ω1_aibj +=
    # + 1 ∑_c(d_ac s_bj s_ci √ω/2)
    # + 1 ∑_c(d_ac s_bjci γ √ω/2)
    # + 1 ∑_c(d_ac t_bjci √ω/2)
    # + 1 ∑_c(d_bc s_ai s_cj √ω/2)
    # + 1 ∑_c(d_bc s_aicj γ √ω/2)
    # + 1 ∑_c(d_bc t_aicj √ω/2)
    # - 1 ∑_k(d_ki s_ak s_bj √ω/2)
    # - 1 ∑_k(d_ki s_akbj γ √ω/2)
    # - 1 ∑_k(d_ki t_akbj √ω/2)
    # - 1 ∑_k(d_kj s_ai s_bk √ω/2)
    # - 1 ∑_k(d_kj s_aibk γ √ω/2)
    # - 1 ∑_k(d_kj t_aibk √ω/2)
    # + 2 ∑_kc(d_kc s_ai s_bjck √ω/2)
    # - 1 ∑_kc(d_kc s_ai s_bkcj √ω/2)
    # + 2 ∑_kc(d_kc s_aibj s_ck √ω/2)
    # - 2 ∑_kc(d_kc s_aibk s_cj √ω/2)
    # - 2 ∑_kc(d_kc s_aicj s_bk √ω/2)
    # + 2 ∑_kc(d_kc s_aick s_bj √ω/2)
    # - 2 ∑_kc(d_kc s_ak s_bjci √ω/2)
    # - 1 ∑_kc(d_kc s_ak t_bjci γ √ω/2)
    # - 2 ∑_kc(d_kc s_akbj s_ci √ω/2)
    # - 1 ∑_kc(d_kc s_akci s_bj √ω/2)
    # - 1 ∑_kc(d_kc s_bk t_aicj γ √ω/2)
    # - 1 ∑_kc(d_kc s_ci t_akbj γ √ω/2)
    # - 1 ∑_kc(d_kc s_cj t_aibk γ √ω/2)

    Ω += 1 * einsum("ac,bj,ci->aibj", d[v, v], p.s1, p.s1) * √(p.ω / 2)
    Ω += 1 * einsum("bc,ai,cj->aibj", d[v, v], p.s1, p.s1) * √(p.ω / 2)
    Ω -= 1 * einsum("ki,ak,bj->aibj", d[o, o], p.s1, p.s1) * √(p.ω / 2)
    Ω -= 1 * einsum("kj,ai,bk->aibj", d[o, o], p.s1, p.s1) * √(p.ω / 2)
    Ω += 1 * einsum("ac,bjci->aibj", d[v, v], p.t2) * √(p.ω / 2)
    Ω += 1 * einsum("bc,aicj->aibj", d[v, v], p.t2) * √(p.ω / 2)
    Ω -= 1 * einsum("kj,aibk->aibj", d[o, o], p.t2) * √(p.ω / 2)
    Ω -= 1 * einsum("ki,akbj->aibj", d[o, o], p.t2) * √(p.ω / 2)
    Ω += 1 * einsum("ac,bjci->aibj", d[v, v], p.s2) * √(p.ω / 2) * p.γ
    Ω += 1 * einsum("bc,aicj->aibj", d[v, v], p.s2) * √(p.ω / 2) * p.γ
    Ω -= 1 * einsum("ki,akbj->aibj", d[o, o], p.s2) * √(p.ω / 2) * p.γ
    Ω -= 1 * einsum("kj,aibk->aibj", d[o, o], p.s2) * √(p.ω / 2) * p.γ
    Ω += 2 * einsum("kc,bjck,ai->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω -= 1 * einsum("kc,bkcj,ai->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω += 2 * einsum("kc,aibj,ck->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω -= 2 * einsum("kc,aibk,cj->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω -= 2 * einsum("kc,aicj,bk->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω += 2 * einsum("kc,aick,bj->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω -= 2 * einsum("kc,bjci,ak->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω -= 2 * einsum("kc,akbj,ci->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω -= 1 * einsum("kc,akci,bj->aibj", d[o, v], p.s2, p.s1) * √(p.ω / 2)
    Ω -= 1 * einsum("kc,bjci,ak->aibj", d[o, v], p.t2, p.s1) * √(p.ω / 2) * p.γ
    Ω -= 1 * einsum("kc,aicj,bk->aibj", d[o, v], p.t2, p.s1) * √(p.ω / 2) * p.γ
    Ω -= 1 * einsum("kc,akbj,ci->aibj", d[o, v], p.t2, p.s1) * √(p.ω / 2) * p.γ
    Ω -= 1 * einsum("kc,aibk,cj->aibj", d[o, v], p.t2, p.s1) * √(p.ω / 2) * p.γ

    # d: Ω1_aibj +=
    # 0

    # for a in 1:p.nvir, i in 1:p.nocc
    #     Ω[a, i, a, i] *= 0.5
    # end

    Ω
end

function contract_omega(h, g, d, d_exp, p::QED_CCSD_PARAMS)
    x = 0.0

    x += einsum("ai,ai->", p.t1_bar, omega_t1(h, g, d, d_exp, p))
    x += 0.5 * einsum("aibj,aibj->", p.t2_t, omega_t2(h, g, d, d_exp, p))
    x += p.γ_bar * omega_s0(h, g, d, d_exp, p)
    x += einsum("ai,ai->", p.s1_bar, omega_s1(h, g, d, d_exp, p))
    x += 0.5 * einsum("aibj,aibj->", p.s2_t, omega_s2(h, g, d, d_exp, p))

    x
end

function read_omega(out_name, p::QED_CCSD_PARAMS)
    omega = get_vector("omega_print", out_name)

    nt1 = length(p.t1)
    nt2 = ((nt1 + 1) * nt1) ÷ 2

    omega_t1 = reshape((@view omega[1:nt1]), size(p.t1))
    omega = @view omega[nt1+1:end]
    omega_t2 = unpack_t2(p.mol, (@view omega[1:nt2]))
    omega = @view omega[nt2+1:end]
    omega_s0 = omega[1]
    omega = @view omega[2:end]
    omega_s1 = reshape((@view omega[1:nt1]), size(p.t1))
    omega = @view omega[nt1+1:end]
    omega_s2 = unpack_t2(p.mol, (@view omega[1:nt2]))

    omega_t1, omega_t2, omega_s0, omega_s1, omega_s2
end

function omega_diff_h(h, g, d, d_exp, p, P, Q, Δh)
    h_cpy = copy(h)

    dX = 0.0

    h_cpy[P, Q] = h[P, Q] + 2Δh
    dX -= contract_omega(h_cpy, g, d, d_exp, p)

    h_cpy[P, Q] = h[P, Q] + Δh
    dX += 8contract_omega(h_cpy, g, d, d_exp, p)

    h_cpy[P, Q] = h[P, Q] - Δh
    dX -= 8contract_omega(h_cpy, g, d, d_exp, p)

    h_cpy[P, Q] = h[P, Q] - 2Δh
    dX += contract_omega(h_cpy, g, d, d_exp, p)

    dX / 12Δh
end

function omega_diff_d(h, g, d, d_exp, p, P, Q, Δh)
    d_cpy = copy(d)

    dX = 0.0

    d_cpy[P, Q] = d[P, Q] + 2Δh
    dX -= contract_omega(h, g, d_cpy, d_exp, p)

    d_cpy[P, Q] = d[P, Q] + Δh
    dX += 8contract_omega(h, g, d_cpy, d_exp, p)

    d_cpy[P, Q] = d[P, Q] - Δh
    dX -= 8contract_omega(h, g, d_cpy, d_exp, p)

    d_cpy[P, Q] = d[P, Q] - 2Δh
    dX += contract_omega(h, g, d_cpy, d_exp, p)

    dX / 12Δh
end

function omega_diff_d_exp(h, g, d, d_exp, p, Δh)
    dX = 0.0

    dX -= contract_omega(h, g, d, d_exp + 2Δh, p)
    dX += 8contract_omega(h, g, d, d_exp + Δh, p)
    dX -= 8contract_omega(h, g, d, d_exp - Δh, p)
    dX += contract_omega(h, g, d, d_exp - 2Δh, p)

    dX / 12Δh
end

function omega_diff_g(h, g, d, d_exp, p, P, Q, R, S, Δh)
    g_cpy = copy(g)

    dX = 0.0

    g_cpy[P, Q, R, S] = g[P, Q, R, S] + 2Δh
    g_cpy[R, S, P, Q] = g[R, S, P, Q] + 2Δh
    dX -= contract_omega(h, g_cpy, d, d_exp, p)

    g_cpy[P, Q, R, S] = g[P, Q, R, S] + Δh
    g_cpy[R, S, P, Q] = g[R, S, P, Q] + Δh
    dX += 8contract_omega(h, g_cpy, d, d_exp, p)

    g_cpy[P, Q, R, S] = g[P, Q, R, S] - Δh
    g_cpy[R, S, P, Q] = g[R, S, P, Q] - Δh
    dX -= 8contract_omega(h, g_cpy, d, d_exp, p)

    g_cpy[P, Q, R, S] = g[P, Q, R, S] - 2Δh
    g_cpy[R, S, P, Q] = g[R, S, P, Q] - 2Δh
    dX += contract_omega(h, g_cpy, d, d_exp, p)

    if (P, Q) == (R, S)
        dX *= 2
    end

    dX / 12Δh
end
