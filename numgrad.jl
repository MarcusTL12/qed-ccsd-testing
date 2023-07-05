
function get_energy(out_dir::AbstractString)
    parse(Float64,
        match(r"Final ground state energy \(a\.u\.\):\s+(-?\d+\.\d+)",
            read("$out_dir.out", String)).captures[1]
    )
end

function get_energy(mol, omega, coup, pol)
    run_eT_tor(mol, omega, coup, pol, "tmp_eT/grad", false) |> get_energy
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
