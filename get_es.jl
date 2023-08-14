function get_es_string(outfilename)
    reg = r"CCSD excitation energies:.+?-+.+?-+\n(.+?)\n +-+"s

    match(
        reg, read("$outfilename.out", String)
    ).captures[1]
end

function get_es(outfilename)
    e_gs = get_energy(outfilename)

    [parse(Float64, first(Iterators.drop(eachsplit(l), 1))) + e_gs
     for l in eachsplit(get_es_string(outfilename), '\n')]
end

function get_es(mol, omega, coup, pol, states)
    run_eT_clean(mol, omega, coup, pol, states, "tmp_eT/es") |> get_es
end
