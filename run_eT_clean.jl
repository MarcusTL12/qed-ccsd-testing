function make_inp_es(mol, omega, coup, pol, states)
    buf = IOBuffer()

    for i in 1:mol.natm
        atom = mol.atom_symbol(i - 1)
        coords = mol.atom_coord(i - 1) / 1.8897261245650618

        println(buf, atom, "    ", coords[1], " ", coords[2], " ", coords[3])
    end

    geom = String(take!(buf)[1:end-1])

"""
- system
   charge: 0

- do
   excited state

- memory
   available: 8

- solver cholesky
   threshold: 1.0d-12

- solver scf
   algorithm:          scf-diis
   energy threshold:   1.0d-10
   gradient threshold: 1.0d-10

- method
   qed-hf
   qed-ccsd

- solver cc gs
   omega threshold:  1.0d-10
   energy threshold: 1.0d-10

- solver cc es
   algorithm:          davidson
   singlet states:     $states
   residual threshold: 1.0d-10
   energy threshold:   1.0d-10
   left eigenvectors

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

function run_eT_clean(mol, omega, coup, pol, states=1, outdir="tmp_eT/")
    inp = make_inp_es(mol, omega, coup, pol, states)

    inp_file = joinpath(outdir, "ccsd.inp")

    open(inp_file, "w") do io
        print(io, inp)
    end

    run(`$(homedir())/eT_clean/build/eT_launch.py $inp_file --omp 24`)

    joinpath(outdir, "ccsd")
end
