function make_inp(mol, omega, coup, pol, multipliers)
    buf = IOBuffer()

    for i in 1:mol.natm
        atom = mol.atom_symbol(i - 1)
        coords = mol.atom_coord(i - 1) / 1.8897261245650618

        println(buf, atom, "    ", coords[1], " ", coords[2], " ", coords[3])
    end

    geom = String(take!(buf)[1:end-1])

    multipliers_code = if multipliers
        """
do
    ground state
    mean value
end do
    
hf mean value
    dipole
end hf mean value
    
cc mean value
    dipole
end cc mean value"""
    else
        """
do
    ground state
end do"""
    end

    """
system
    name: H2O
    charge: 0
end system

method
    qed-hf
    qed-ccsd-sd
end method

memory
    available: 8
end memory

solver cholesky
    threshold: 1.0d-12
end solver cholesky
 
solver scf
    algorithm:          scf-diis
    energy threshold:   1.0d-10
    gradient threshold: 1.0d-10
end solver scf

solver cc gs
    omega threshold:  1.0d-10
    energy threshold: 1.0d-10
end solver cc gs

$multipliers_code

qed
    photons:        1
    omega cavity:   $omega
    coupling:       $coup
    x polarization: $(pol[1])
    y polarization: $(pol[2])
    z polarization: $(pol[3])
end qed

geometry
basis: $(mol.basis)
$geom
end geometry
"""
end

function run_eT_tor(mol, omega, coup, pol, outdir="tmp_eT/", multipliers=true)
    inp = make_inp(mol, omega, coup, pol, multipliers)

    inp_file = joinpath(outdir, "ccsd.inp")

    open(inp_file, "w") do io
        print(io, inp)
    end

    run(`$(homedir())/eT_Tor/build/eT_launch $inp_file --omp 48`)

    joinpath(outdir, "ccsd")
end
