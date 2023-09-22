
function get_energy(out_dir::AbstractString)
    parse(Float64,
        match(r"Final ground state energy \(a\.u\.\):\s+(-?\d+\.\d+)",
            read("$out_dir.out", String)).captures[1]
    )
end

function get_energy(mol, omega, coup, pol)
    # run_eT_tor(mol, omega, coup, pol, "tmp_eT/grad", false) |> get_energy
    run_eT_new(mol, omega, coup, pol, "tmp_eT/grad", false) |> get_energy
end

function perturb_geometry(mol, i, q, dx)
    coords = mol.atom_coords()

    coords[i, q] += dx

    atom = [
        (mol.atom_symbol(j - 1), (coords[j, 1], coords[j, 2], coords[j, 3]))
        for j in axes(coords, 1)
    ]

    pyscf.M(atom=atom, basis=mol.basis, unit="bohr")
end

function numgrad2(mol, omega, coup, pol, i, q, h)
    function ef(n)
        get_energy(perturb_geometry(mol, i, q, n * h), omega, coup, pol)
    end

    (ef(1) - ef(-1)) / 2h
end

function numgrad4(mol, omega, coup, pol, i, q, h)
    function ef(n)
        get_energy(perturb_geometry(mol, i, q, n * h), omega, coup, pol)
    end

    (-ef(2) + 8ef(1) - 8ef(-1) + ef(-2)) / 12h
end

function numgrad2_int(mol, intname, i, q, h)
    function int_func(n)
        perturb_geometry(mol, i, q, n * h).intor(intname, aosym="s8")
    end

    (int_func(1) - int_func(-1)) ./ 2h
end

function numgrad4_int(mol, intname, i, q, h)
    function int_func(n)
        perturb_geometry(mol, i, q, n * h).intor(intname, aosym="s8")
    end

    (-int_func(2) + 8int_func(1) - 8int_func(-1) + int_func(-2)) ./ 12h
end

function numgrad4_ao_d(mol, pol, i, q, h)
    function int_func(n)
        get_ao_d(perturb_geometry(mol, i, q, n * h), pol)
    end

    (-int_func(2) + 8int_func(1) - 8int_func(-1) + int_func(-2)) ./ 12h
end

function numgrad4_f(mol, f, i, q, h)
    function int_func(n)
        f(perturb_geometry(mol, i, q, n * h))
    end

    (-int_func(2) + 8int_func(1) - 8int_func(-1) + int_func(-2)) ./ 12h
end

function numgrad4_e_nuc(mol, i, q, h)
    function ef(n)
        perturb_geometry(mol, i, q, n * h).energy_nuc()
    end

    (-ef(2) + 8ef(1) - 8ef(-1) + ef(-2)) / 12h
end

function numgrad2_es(mol, omega, coup, pol, states, i, q, h)
    function ef(n)
        get_es(perturb_geometry(mol, i, q, n * h), omega, coup, pol, states)
    end

    (ef(1) - ef(-1)) / 2h
end

function numgrad4_es(mol, omega, coup, pol, states, i, q, h)
    function ef(n)
        get_es(perturb_geometry(mol, i, q, n * h), omega, coup, pol, states)
    end

    (-ef(2) + 8ef(1) - 8ef(-1) + ef(-2)) / 12h
end
