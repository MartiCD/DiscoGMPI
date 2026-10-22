using Test
using DiscoGMPI

function ffd_test_text(header, blocks)
    io = IOBuffer()
    write_ffd(io, header, blocks)
    return String(take!(io))
end

function ffd_test_parse(text; lazy=true)
    return mktempdir() do directory
        path = joinpath(directory, "test.ffd")
        write(path, text)
        file = read_ffd(path; lazy=lazy)
        # Collect while the backing file still exists, also testing lazy reads.
        return file.header, [collect(each_ffd_point(block)) for block in file.blocks]
    end
end

function ffd_test_signature(sample)
    return (sample.freq, sample.pol, sample.theta_inc, sample.phi_inc,
            sample.theta_obs, sample.phi_obs, sample.Etheta, sample.Ephi)
end

@testset "FFD v1 metadata and explicit round trips" begin
    header = FFDHeader(; solver="DiscoGMPI 0.1.0", problem="synthetic complex data",
                       date="2026-10-22T12:34:56Z",
                       extras=Dict("phase_convention" => "exp(+i omega t)", "encoding" => "ascii"))
    samples = [FFDSample(1e9, FFD_V, 30, 0, 45, 90, 1 + 2im, 3 - 4im),
               FFDSample(2e9, "H", 45, 90, 60, 135, -1e-20 + 1e20im, -0.0 + 0.0im)]
    block = FFDExplicitBlock(samples)
    @test validate_ffd(header) === header
    @test validate_ffd(block) === block
    @test validate_ffd(header, [block]) === nothing
    text = ffd_test_text(header, [block])
    @test occursin("1.0000000000000000e+09 V", text)
    @test text == ffd_test_text(header, [block])
    @test occursin("solver DiscoGMPI 0.1.0", text)
    for lazy in (true, false)
        parsed_header, points = ffd_test_parse(text; lazy=lazy)
        @test parsed_header.solver == header.solver
        @test parsed_header.problem == header.problem
        @test parsed_header.date == header.date
        @test parsed_header.field_unit == "m"
        @test parsed_header.extras == header.extras
        @test ffd_test_signature.(only(points)) == ffd_test_signature.(samples)
    end
    # Whitespace, full-line comments (including inside DATA), and blank lines.
    decorated = replace(text, "[DATA]\n" => "[DATA]\n  # a column header follows\n\n",
                        "1.0000000000000000e+09 V" => "\n\t# first sample\n1.0000000000000000e+09\tV")
    @test ffd_test_signature.(only(last(ffd_test_parse(decorated)))) == ffd_test_signature.(samples)
    @test_throws ArgumentError FFDHeader(; version="2.0")
    @test_throws ArgumentError FFDHeader(; extras=Dict("encoding" => "binary_float64_le"))
    @test_throws ArgumentError FFDHeader(; extras=Dict("freq_unit" => "GHz"))
    @test_throws ArgumentError FFDHeader(; extras=Dict("bad key" => "value"))
    @test_throws ArgumentError FFDHeader(; solver="x\n[END_FILE_HEADER]")
    @test_throws ArgumentError FFDHeader(; field_unit="")
    @test_throws ArgumentError FFDHeader(; freq_unit=" Hz")
    @test_throws ArgumentError FFDHeader(; freq_unit="bad unit")
    @test_throws ArgumentError FFDHeader(; angle_unit="de g")
    @test_throws ArgumentError FFDHeader(; field_unit="V / m")
    @test_throws ArgumentError FFDHeader(; extras=Dict("empty" => ""))
    @test_throws ArgumentError FFDSample(1e9, "v", 0, 0, 0, 0, 0im, 0im)
    for invalid in (NaN, Inf, -Inf)
        @test_throws ArgumentError FFDAxis(:freq, [invalid])
        @test_throws ArgumentError validate_ffd(FFDExplicitBlock([
            FFDSample(1e9, FFD_V, 0, 0, 0, 0, complex(invalid), 0im)]))
    end
    @test_throws ArgumentError FFDAxis(:freq, Float64[])
    @test_throws ArgumentError FFDAxis(:pol, ["X"])
    @test_throws ArgumentError FFDAxis(:pol, [1.0])
    @test_throws ArgumentError FFDAxis(:time, [1.0])
    @test_throws ArgumentError FFDExplicitBlock(samples; n_points=-1)
    @test_throws ArgumentError ffd_test_text(header, AbstractFFDBlock[])
    @test_throws ArgumentError validate_ffd(header, AbstractFFDBlock[])
end

@testset "FFD ISO 8601 date profile" begin
    sample = FFDSample(1e9, FFD_V, 0, 0, 0, 0, 1im, 0im)
    valid = ("2026-10-22T12:34:56Z", "2024-02-29T00:00:00.123456789Z",
             "2026-10-22T12:34:56+02:30", "2026-10-22T12:34:56.1-05:00",
             "2026-10-22T12:34:56", "0000-01-01T00:00:00Z")
    for date in valid
        header = FFDHeader(; date=date)
        @test header.date == date
        parsed, _ = ffd_test_parse(ffd_test_text(header, [FFDExplicitBlock([sample])]))
        @test parsed.date == date
    end
    invalid = ("not an ISO date", "2026-10-22", "2026-02-29T12:34:56Z",
               "2024-02-30T12:34:56Z", "2026-13-01T12:34:56Z", "2026-00-01T12:34:56Z",
               "2026-01-00T12:34:56Z", "2026-10-22T24:00:00Z", "2026-10-22T12:60:00Z",
               "2026-10-22T12:34:60Z", "2026-10-22T12:34:56+24:00",
               "2026-10-22T12:34:56+02:60", "2026-10-22T12:34:56+0200",
               "2026-10-22T12:34:56.Z", "2026-10-22t12:34:56z", "2026-10-22T12:34:56Z\nother")
    template = ffd_test_text(FFDHeader(; date=valid[1]), [FFDExplicitBlock([sample])])
    for date in invalid
        @test_throws ArgumentError FFDHeader(; date=date)
        @test_throws ArgumentError ffd_test_parse(replace(template, valid[1] => date))
    end
    # User-defined metadata may use arbitrary single-line values, including text
    # resembling a date; only the standard date field receives date validation.
    @test FFDHeader(; extras=Dict("run_date_description" => "sometime next week")).extras["run_date_description"] == "sometime next week"
end

@testset "FFD grid striding and equivalent encodings" begin
    header = FFDHeader()
    # Two outer axes, two inner rows per cell: the final axis varies fastest.
    axes = [FFDAxis(:freq, [1e9, 2e9]), FFDAxis(:pol, [FFD_V, FFD_H])]
    rows = [(30.0 + i, 0.0, 45.0, 90.0, Float64(i), -Float64(i), 0.0, 1.0) for i in 1:8]
    block = FFDGriddedBlock(axes, rows)
    points = collect(each_ffd_point(block))
    @test [(p.freq, p.pol) for p in points] == [
        (1e9, FFD_V), (1e9, FFD_V), (1e9, FFD_H), (1e9, FFD_H),
        (2e9, FFD_V), (2e9, FFD_V), (2e9, FFD_H), (2e9, FFD_H)]
    @test [p.theta_inc for p in points] == collect(31.0:38.0)
    @test [p.Etheta for p in points] == [i - im*i for i in 1:8]
    # Canonical samples can be supplied for a grid only when factored-out
    # coordinates agree with the cell encoded by their position.
    @test ffd_test_signature.(collect(each_ffd_point(FFDGriddedBlock(axes, points)))) == ffd_test_signature.(points)
    bad_points = copy(points)
    bad_points[1] = FFDSample(3e9, FFD_V, 31, 0, 45, 90, 1-im, 1im)
    @test_throws ArgumentError validate_ffd(FFDGriddedBlock(axes, bad_points))
    @test ffd_test_signature.(collect(expand_block(block))) == ffd_test_signature.(points)
    explicit = FFDExplicitBlock(points)
    for lazy in (true, false)
        _, expanded = ffd_test_parse(ffd_test_text(header, [explicit, block]); lazy=lazy)
        @test ffd_test_signature.(expanded[1]) == ffd_test_signature.(expanded[2])
    end
    # The standard production layout, with all coordinates as axes and M=1.
    fully_axes = [FFDAxis(:freq, [1e9]), FFDAxis(:pol, ["V"]),
                  FFDAxis(:theta_inc, [180.0]), FFDAxis(:phi_inc, [180.0]),
                  FFDAxis(:phi_obs, [0.0, 90.0]), FFDAxis(:theta_obs, [0.0, 45.0, 90.0])]
    fully_rows = [(Float64(i), 0.0, -Float64(i), 1.0) for i in 1:6]
    fully = FFDGriddedBlock(fully_axes, fully_rows)
    full_points = collect(each_ffd_point(fully))
    @test [(p.phi_obs, p.theta_obs) for p in full_points] == [
        (0.0, 0.0), (0.0, 45.0), (0.0, 90.0), (90.0, 0.0), (90.0, 45.0), (90.0, 90.0)]
    @test ffd_test_signature.(only(last(ffd_test_parse(ffd_test_text(header, [fully]))))) == ffd_test_signature.(full_points)
    # Empty blocks have no samples and remain serializable, including a declared grid.
    empty = FFDGriddedBlock(fully_axes, Tuple[]; n_points=0)
    @test isempty(collect(each_ffd_point(empty)))
    @test isempty(only(last(ffd_test_parse(ffd_test_text(header, [empty])))))
    @test_throws ArgumentError FFDGriddedBlock(axes, rows; sweep_order=[:pol, :freq])
    @test_throws ArgumentError FFDGriddedBlock(axes, rows; n_points=7)
    @test_throws ArgumentError FFDGriddedBlock(FFDAxis[], rows)
    @test_throws ArgumentError FFDGriddedBlock([axes[1], axes[1]], rows)
    @test_throws ArgumentError FFDGriddedBlock(axes, rows; columns=[
        :freq, :phi_inc, :theta_obs, :phi_obs, :Etheta_re, :Etheta_im, :Ephi_re, :Ephi_im])
    @test_throws ArgumentError FFDGriddedBlock(axes, rows; columns=[
        :theta_inc, :phi_inc, :theta_obs, :phi_obs, :Ephi_re, :Etheta_im, :Etheta_re, :Ephi_im])
    @test_throws ArgumentError validate_ffd(FFDExplicitBlock(points; n_points=7))
    @test_throws ArgumentError collect(each_ffd_point(FFDExplicitBlock(points; n_points=9)))
    # Malformed positional structs and mutation of the public axis arrays fail
    # as format errors, including when mutation happens after iterator creation.
    with_empty_count = FFDGriddedBlock(axes, points, 0, [:freq, :pol], block.columns)
    @test_throws ArgumentError first(each_ffd_point(with_empty_count))
    tuple_rows = [(p.pol, p.theta_inc, p.phi_inc, p.theta_obs, p.phi_obs,
                   real(p.Etheta), imag(p.Etheta), real(p.Ephi), imag(p.Ephi)) for p in points]
    for supplied_rows in (points, tuple_rows)
        mutable_axis = FFDAxis(:freq, [1e9])
        mutable_grid = FFDGriddedBlock([mutable_axis], supplied_rows;
            n_points=8, columns=[:pol, :theta_inc, :phi_inc, :theta_obs, :phi_obs,
                                :Etheta_re, :Etheta_im, :Ephi_re, :Ephi_im])
        iterator = each_ffd_point(mutable_grid)
        empty!(mutable_axis.values)
        @test_throws ArgumentError first(iterator)
    end
    resized_axis = FFDAxis(:freq, [1e9])
    resized_grid = FFDGriddedBlock([resized_axis], points;
        n_points=8, columns=[:pol, :theta_inc, :phi_inc, :theta_obs, :phi_obs,
                            :Etheta_re, :Etheta_im, :Ephi_re, :Ephi_im])
    resized_iterator = each_ffd_point(resized_grid)
    append!(resized_axis.values, [2e9, 3e9])
    @test_throws ArgumentError first(resized_iterator)
end

@testset "FFD canonical interface is independent of sample producer" begin
    # This tests the serialization boundary, not a mixed-order near-to-far
    # extraction. Two storage layouts produce the same completed fields.
    equal_layout = (Etheta=ComplexF64[1+2im, 3-4im], Ephi=ComplexF64[5-6im, 7+8im])
    mixed_layout = (real_theta=[1.0, 3.0], imag_theta=[2.0, -4.0],
                    real_phi=[5.0, 7.0], imag_phi=[-6.0, 8.0])
    from_equal = [FFDSample(1e9, FFD_V, 30, 0, 45, 90i, equal_layout.Etheta[i], equal_layout.Ephi[i]) for i in 1:2]
    from_mixed = (FFDSample(1e9, FFD_V, 30, 0, 45, 90i,
                           complex(mixed_layout.real_theta[i], mixed_layout.imag_theta[i]),
                           complex(mixed_layout.real_phi[i], mixed_layout.imag_phi[i])) for i in 1:2)
    @test ffd_test_text(FFDHeader(), [FFDExplicitBlock(from_equal)]) ==
          ffd_test_text(FFDHeader(), [FFDExplicitBlock(from_mixed; n_points=2)])
end

@testset "FFD specification examples" begin
    path = joinpath(@__DIR__, "data", "ffd", "spec_examples.ffd")
    for lazy in (true, false)
        file = read_ffd(path; lazy=lazy)
        @test file.header.solver == "DiscoG 1.0"
        @test file.header.field_unit == "V/m"
        @test file.header.extras["campaign"] == "specification examples"
        @test length(file.blocks) == 3
        @test [b.n_points for b in file.blocks] == [8, 16, 8]
        explicit, intermediate, fully = [collect(each_ffd_point(b)) for b in file.blocks]
        @test explicit[1].Etheta == 9.87654321e-4 + 2.34567890e-4im
        @test explicit[end].Ephi == 2.40000000e-3 - 1.30000000e-3im
        @test intermediate[5].freq == 2e9
        @test intermediate[5].pol == FFD_V
        @test intermediate[5].theta_inc == 900.0
        @test intermediate[9].freq == 1e9
        @test intermediate[9].pol == FFD_H
        @test fully[3].theta_inc == 45.0
        @test fully[3].phi_obs == 90.0
        @test fully[5].freq == 2e9
        @test validate_ffd(file) === nothing
        rewritten = ffd_test_text(file.header, file.blocks)
        _, reparsed = ffd_test_parse(rewritten; lazy=lazy)
        @test ffd_test_signature.(reparsed[2]) == ffd_test_signature.(intermediate)
    end
end

@testset "FFD streaming and atomic output" begin
    mktempdir() do directory
        path = joinpath(directory, "stream.ffd")
        seen = Ref(0)
        samples = (begin
            seen[] += 1
            FFDSample(1e9, FFD_V, 30, 0, 45, i, complex(i, -i), complex(2i, 3i))
        end for i in 1:777)
        block = FFDExplicitBlock(samples; n_points=777)
        @test seen[] == 0
        @test write_ffd(path, FFDHeader(), [block]) == path
        @test seen[] == 777
        file = read_ffd(path)
        @test !(file.blocks[1].rows isa AbstractVector)
        points = collect(each_ffd_point(file.blocks[1]))
        @test length(points) == 777
        @test points[257].Etheta == 257 - 257im
        @test points[end].Ephi == 1554 + 2331im
        @test length(collect(each_ffd_row(file.blocks[1]))) == 777
        # Partial consumption does not leave a descriptor open, even before GC.
        if isdir("/proc/self/fd")
            descriptors = length(readdir("/proc/self/fd"))
            for _ in 1:100
                @test first(each_ffd_point(file.blocks[1])).freq == 1e9
            end
            @test length(readdir("/proc/self/fd")) == descriptors
        end
        @test first(each_ffd_point(file.blocks[1])).Etheta == 1 - 1im
        # Rewriting the same path from lazy source data is safe: rename is last.
        write_ffd(path, file)
        @test ffd_test_signature.(collect(each_ffd_point(read_ffd(path).blocks[1]))) == ffd_test_signature.(points)
        old_text = read(path, String)
        old_files = Set(readdir(directory))
        for bad in (FFDExplicitBlock(points; n_points=776),
                    FFDExplicitBlock(points; n_points=778),
                    FFDExplicitBlock([(1.0,)]),
                    FFDExplicitBlock([FFDSample(1e9, FFD_V, 0, 0, 0, 0, NaN + 0im, 0im)]))
            @test_throws ArgumentError write_ffd(path, FFDHeader(), [bad])
            @test read(path, String) == old_text
            @test Set(readdir(directory)) == old_files
        end
        @test_throws ArgumentError write_ffd(joinpath(directory, "absent.ffd"), FFDHeader(), AbstractFFDBlock[])
        @test Set(readdir(directory)) == old_files
        broken_rows = (i == 2 ? error("sample producer failed") : points[i] for i in 1:3)
        @test_throws ErrorException write_ffd(path, FFDHeader(), [FFDExplicitBlock(broken_rows; n_points=3)])
        @test read(path, String) == old_text
        @test Set(readdir(directory)) == old_files
        # An OS publication failure is propagated without a copy/remove fallback.
        occupied = joinpath(directory, "occupied.ffd")
        mkdir(occupied)
        marker = joinpath(occupied, "keep.txt")
        write(marker, "keep the destination directory")
        with_directory = Set(readdir(directory))
        @test_throws Base.IOError write_ffd(occupied, FFDHeader(), [FFDExplicitBlock(points)])
        @test read(marker, String) == "keep the destination directory"
        @test Set(readdir(directory)) == with_directory
        # A lazy view detects stale source metadata instead of silently using its offsets.
        lazy = read_ffd(path)
        open(path, "a") do io
            write(io, "\n# source changed\n")
        end
        @test_throws ArgumentError first(each_ffd_row(lazy.blocks[1]))
    end
end

@testset "FFD malformed file rejection" begin
    sample = FFDSample(1e9, FFD_V, 30, 0, 45, 90, 1 + 2im, 3 - 4im)
    text = ffd_test_text(FFDHeader(), [FFDExplicitBlock([sample])])
    invalid = [
        replace(text, "version 1.0\n" => ""),
        replace(text, "freq_unit Hz\n" => ""),
        replace(text, "angle_unit deg\n" => ""),
        replace(text, "field_unit m\n" => ""),
        replace(text, "version 1.0" => "version 2.0"),
        replace(text, "version 1.0" => "version 1.0\nencoding binary_float64_le"),
        replace(text, "version 1.0" => "version 1.0\nversion 1.0"),
        replace(text, "[FILE_HEADER]" => "[FILE_HEADER] extra"),
        replace(text, "[END_FILE_HEADER]" => "[END_FILE_HEADER] # inline"),
        replace(text, "storage explicit" => "storage unknown"),
        replace(text, "storage explicit" => "storage explicit\nfreq_unit GHz"),
        replace(text, "storage explicit" => "storage explicit\nstorage explicit"),
        replace(text, "storage explicit" => "storage explicit\nsweep_order freq"),
        replace(text, "storage explicit\n" => ""),
        replace(text, "n_points 1" => "n_points 1e0"),
        replace(text, "n_points 1" => "n_points -1"),
        replace(text, "n_points 1" => "n_points 9999999999999999999999999999999"),
        replace(text, "n_points 1" => "n_points 2"),
        replace(text, "n_points 1" => "n_points 0"),
        replace(text, "[DATA]" => "[DATA] trailing"),
        replace(text, "[END_DATA]" => "[END_DATA] # inline"),
        replace(text, "[END_BLOCK]" => ""),
        replace(text, "[END_BLOCK]" => "[END_BLOCK]\nextra"),
        replace(text, "Etheta_re Etheta_im Ephi_re Ephi_im" => "Ephi_re Etheta_im Etheta_re Ephi_im"),
        replace(text, "freq pol theta_inc" => "freq freq theta_inc"),
        replace(text, "freq pol theta_inc" => "freq pol bogus"),
        replace(text, "1.0000000000000000e+09 V" => "1000000000 V"),
        replace(text, "1.0000000000000000e+09 V" => "NaN V"),
        replace(text, "1.0000000000000000e+09 V" => "1e999 V"),
        replace(text, "1.0000000000000000e+09 V" => "1e9 h"),
        replace(text, "[END_DATA]" => "1e9 V 3e1 0e0 4.5e1 9e1 1e0 2e0 3e0 4e0\n[END_DATA]"),
        replace(text, "[END_DATA]" => "# standalone comments allowed\n[END_DATA] extra"),
    ]
    for bad_text in invalid
        @test_throws ArgumentError ffd_test_parse(bad_text)
    end
    # Valid header-only input still fails because files require one or more blocks.
    @test_throws ArgumentError ffd_test_parse(first(split(text, "[BLOCK]")))
    axes = [FFDAxis(:freq, [1e9, 2e9])]
    rows = [(FFD_V, 30.0, 0.0, 45.0, 90.0, 1.0, 2.0, 3.0, 4.0) for _ in 1:2]
    grid_text = ffd_test_text(FFDHeader(), [FFDGriddedBlock(axes, rows)])
    for bad_text in (
        replace(grid_text, "sweep_order freq\n" => ""),
        replace(grid_text, "sweep_order freq" => "sweep_order pol"),
        replace(grid_text, "[AXIS freq]" => "[AXIS unknown]"),
        replace(grid_text, "[AXIS freq]" => "[AXIS freq] extra"),
        replace(grid_text, "1.0000000000000000e+09 2.0000000000000000e+09" => ""),
        replace(grid_text, "n_points 2" => "n_points 3"),
        replace(grid_text, "[DATA]" => "[AXIS pol]\nV\n[END_AXIS]\n[DATA]"),
        replace(grid_text, "storage gridded" => "storage explicit"),
    )
        @test_throws ArgumentError ffd_test_parse(bad_text)
    end
    # An inline comment in a data row is not silently stripped.
    @test_throws ArgumentError ffd_test_parse(replace(text, "[END_DATA]" => "# harmless\n[END_DATA]") |> t -> replace(t,
        "4.0000000000000000e+00" => "4.0000000000000000e+00 # no inline comments"))
end
