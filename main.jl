using PyCall
using LinearAlgebra

if !@isdefined pyscf
    pyscf = pyimport("pyscf")
    einsum = pyscf.lib.einsum
end

include("integrals.jl")
include("get_matrix.jl")
include("get_vector.jl")
include("run_eT_tor.jl")
include("energy.jl")
include("density.jl")

function test()
    mol = pyscf.M(atom="""
 H          0.86681        0.60144        5.00000
 H         -0.86681        0.60144        5.00000
 O          0.00000       -0.07579        5.00000
 He         0.10000       -0.02000        7.53000
""", basis="STO-3G")

    omega = 0.5
    coup = 0.05
    # coup = 0.0
    pol = [0.577350, 0.577350, 0.577350]

    # out_name = run_eT_tor(mol, omega, coup, pol)
    out_name = "tmp_eT/ccsd"

    pol *= coup

    C = get_matrix("MO-coeffs", out_name)

    # println("C:")
    # display(C)

    t1 = get_matrix("T1-AMPLITUDES", out_name)

    # println("t1:")
    # display(t1)

    t2 = unpack_t2(mol, get_vector("T2-AMPLITUDES", out_name))

    # println("t2:")
    # display(t2)

    s1 = get_matrix("S1-AMPLITUDES", out_name)

    # println("s1:")
    # display(s1)

    s2 = unpack_t2(mol, get_vector("S2-AMPLITUDES", out_name))

    # println("s2:")
    # display(s2)

    γ = get_vector("Photon amplitudes", out_name)[1]

    get_energy_ccsd(mol, pol, C, t1, t2, s1, γ, omega)

    x, y = construct_t1_transformation(mol, t1)

    # println("x:")
    # display(x)

    # println("y:")
    # display(y)

    get_energy_ccsd_t1_transformed(mol, pol, C, t2, s1, γ, omega, x, y)

    t1_bar = get_matrix("T1-MULTIPLIERS", out_name)
end
