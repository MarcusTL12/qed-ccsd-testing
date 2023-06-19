function make_find_vector_reg(name)
    Regex("$(replace(name, '(' => "\\(", ')' => "\\)"))\\s+-+(.+?)--", "s")
end

function get_vector_string(name, outfilename)
    match(
        make_find_vector_reg(name), read("$outfilename.out", String)
    ).captures[1]
end

function parse_vector(vecstring)
    reg = r"(\d+) +(-?\d+\.\d+)"

    unsorted = Tuple{Int,Float64}[]

    for m in eachmatch(reg, vecstring)
        push!(unsorted,
            (parse(Int, m.captures[1]), parse(Float64, m.captures[2]))
        )
    end

    sorted = zeros(length(unsorted))

    for (i, x) in unsorted
        sorted[i] = x
    end

    sorted
end

function get_vector(name, outfilename)
    parse_vector(get_vector_string(name, outfilename))
end
