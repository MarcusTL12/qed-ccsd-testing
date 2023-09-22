function make_inp_new(mol, omega, coup, pol, multipliers)
    buf = IOBuffer()

    for i in 1:mol.natm
        atom = mol.atom_symbol(i - 1)
        coords = mol.atom_coord(i - 1) / 1.8897261245650618

        println(buf, atom, "    ", coords[1], " ", coords[2], " ", coords[3])
    end

    geom = String(take!(buf)[1:end-1])

    multipliers_code = if multipliers
        """
- do
    mean value

- cc mean value
    dipole
    molecular gradient
"""
    else
        """
- do
    ground state
"""
    end

    """
- system
    charge: 0

- method
    qed-hf
    qed-ccsd

- memory
    available: 8

- solver cholesky
    threshold: 1.0d-12

- solver scf
    algorithm:          scf-diis
    energy threshold:   1.0d-10
    gradient threshold: 1.0d-10

- solver cc gs
    omega threshold:  1.0d-10
    energy threshold: 1.0d-10

$multipliers_code

- solver cc multipliers
   threshold: 1.0d-11

- boson
    modes:          1
    boson states:   {1}
    frequency:      {$omega}
    coupling:       {$coup}
    polarization:   {$(pol[1]), $(pol[2]), $(pol[3])}

- geometry
basis: $(mol.basis)
$geom
"""
end

function run_eT_new(mol, omega, coup, pol, outdir="tmp_eT/", multipliers=true)
    inp = make_inp_new(mol, omega, coup, pol, multipliers)

    inp_file = joinpath(outdir, "ccsd.inp")

    open(inp_file, "w") do io
        print(io, inp)
    end

    run(`$(homedir())/eT_dev/build/eT_launch.py $inp_file --omp 24 -s`)

    joinpath(outdir, "ccsd")
end
