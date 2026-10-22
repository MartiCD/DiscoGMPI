using Dates

"""Incident polarization supported by the FFD v1.0 text format."""
@enum FFDPolarization FFD_V FFD_H

const _FFD_COORDINATES = (:freq, :pol, :theta_inc, :phi_inc, :theta_obs, :phi_obs)
const _FFD_FIELDS = (:Etheta_re, :Etheta_im, :Ephi_re, :Ephi_im)
const _FFD_HEADER_KEYS = ("version", "solver", "problem", "date", "freq_unit", "angle_unit", "field_unit")
const _FFD_SCIENTIFIC = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$"
const _FFD_DATETIME = r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(?:Z|([+-])(\d{2}):(\d{2}))?$"

_ffd_error(message) = throw(ArgumentError("FFD: " * message))
_ffd_pol(value::FFDPolarization) = value
function _ffd_pol(value)
    value == "V" && return FFD_V
    value == "H" && return FFD_H
    _ffd_error("polarization must be V or H; got $(repr(value))")
end
_ffd_pol_token(value::FFDPolarization) = value == FFD_V ? "V" : "H"

"""
    FFDSample(freq, pol, theta_inc, phi_inc, theta_obs, phi_obs, Etheta, Ephi)

One canonical far-field sample. Units and physical conventions belong to the
file header and producing application; the codec applies no field transforms.
`pol` is `FFD_V`, `FFD_H`, `"V"`, or `"H"`.
"""
struct FFDSample
    freq::Float64
    pol::FFDPolarization
    theta_inc::Float64
    phi_inc::Float64
    theta_obs::Float64
    phi_obs::Float64
    Etheta::ComplexF64
    Ephi::ComplexF64
end

function FFDSample(freq, pol::AbstractString, theta_inc, phi_inc, theta_obs, phi_obs, Etheta, Ephi)
    return FFDSample(freq, _ffd_pol(pol), theta_inc, phi_inc, theta_obs, phi_obs, Etheta, Ephi)
end

"""
    FFDHeader(; version="1.0", freq_unit="Hz", angle_unit="deg", field_unit="m",
              solver=nothing, problem=nothing, date=nothing, extras=Dict())

Global metadata. Additional fields are retained in `extras`. FFD v1 supports
only ASCII text (`encoding` absent or `ascii`). Unit labels do not convert data.
Optional dates use the ISO 8601 extended calendar profile
`YYYY-MM-DDTHH:MM:SS[.fraction][Z|±HH:MM]`; a timezone may be omitted. Calendar
dates are validated, hours are 00–23, and minutes/seconds are 00–59. Leap seconds
and 24:00 are outside this profile. Offset hours are 00–23 and minutes 00–59.
"""
struct FFDHeader
    version::String
    freq_unit::String
    angle_unit::String
    field_unit::String
    solver::Union{Nothing,String}
    problem::Union{Nothing,String}
    date::Union{Nothing,String}
    extras::Dict{String,String}
end

function FFDHeader(; version="1.0", freq_unit="Hz", angle_unit="deg", field_unit="m",
                   solver=nothing, problem=nothing, date=nothing, extras=Dict())
    header = FFDHeader(string(version), string(freq_unit), string(angle_unit), string(field_unit),
                       isnothing(solver) ? nothing : string(solver),
                       isnothing(problem) ? nothing : string(problem),
                       isnothing(date) ? nothing : string(date),
                       Dict(string(k) => string(v) for (k, v) in pairs(extras)))
    validate_ffd(header)
    return header
end

function _ffd_validate_datetime(value)
    parts = match(_FFD_DATETIME, value)
    isnothing(parts) && _ffd_error("date must be an ISO 8601 extended calendar datetime")
    year, month, day, hour, minute, second = parse.(Int, parts.captures[1:6])
    try
        Dates.Date(year, month, day)
    catch error
        error isa ArgumentError || rethrow()
        _ffd_error("date has an invalid calendar day")
    end
    (0 <= hour <= 23 && 0 <= minute <= 59 && 0 <= second <= 59) ||
        _ffd_error("date has an invalid clock time")
    if !isnothing(parts[8])
        offset_hour = parse(Int, parts[9])
        offset_minute = parse(Int, parts[10])
        (0 <= offset_hour <= 23 && 0 <= offset_minute <= 59) ||
            _ffd_error("date has an invalid UTC offset")
    end
    return nothing
end

function validate_ffd(header::FFDHeader)
    header.version == "1.0" || _ffd_error("unsupported version $(repr(header.version)); expected 1.0")
    fields = Pair{String,String}[
        "version" => header.version, "freq_unit" => header.freq_unit,
        "angle_unit" => header.angle_unit, "field_unit" => header.field_unit,
    ]
    for name in (:solver, :problem, :date)
        value = getproperty(header, name)
        isnothing(value) || push!(fields, string(name) => value)
    end
    for (key, value) in header.extras
        key in _FFD_HEADER_KEYS && _ffd_error("extra header field $key overrides a standard field")
        push!(fields, key => value)
    end
    for (key, value) in fields
        occursin(r"^[A-Za-z_][A-Za-z_0-9]*$", key) || _ffd_error("invalid header key $(repr(key))")
        isempty(strip(value)) && _ffd_error("header field $key cannot be empty")
        value == strip(value) || _ffd_error("header value $key has surrounding whitespace")
        occursin(r"[\r\n]", value) && _ffd_error("header value $key must fit on one line")
    end
    for name in (:freq_unit, :angle_unit, :field_unit)
        occursin(r"\s", getproperty(header, name)) && _ffd_error("$name must be a single unit token")
    end
    isnothing(header.date) || _ffd_validate_datetime(header.date)
    get(header.extras, "encoding", "ascii") == "ascii" ||
        _ffd_error("unsupported encoding; FFD v1.0 supports ASCII text only")
    return header
end

"""A declared FFD coordinate axis; values must be finite and the axis nonempty."""
struct FFDAxis
    name::Symbol
    values::Union{Vector{Float64},Vector{FFDPolarization}}
    function FFDAxis(name::Symbol, values::Union{Vector{Float64},Vector{FFDPolarization}})
        name in _FFD_COORDINATES || _ffd_error("unknown coordinate axis $name")
        isempty(values) && _ffd_error("axis $name cannot be empty")
        if name == :pol
            values isa Vector{FFDPolarization} || _ffd_error("pol axis requires V/H values")
        else
            values isa Vector{Float64} || _ffd_error("numeric axis $name requires real values")
            all(isfinite, values) || _ffd_error("axis $name contains nonfinite values")
        end
        return new(name, values)
    end
end

function FFDAxis(name::Union{Symbol,AbstractString}, values)
    key = Symbol(name)
    key in _FFD_COORDINATES || _ffd_error("unknown coordinate axis $key")
    data = key == :pol ? FFDPolarization[_ffd_pol(v) for v in values] : Float64[v for v in values]
    isempty(data) && _ffd_error("axis $key cannot be empty")
    key == :pol || all(isfinite, data) || _ffd_error("axis $key contains nonfinite values")
    return FFDAxis(key, data)
end

abstract type AbstractFFDBlock end

"""
    FFDExplicitBlock(rows; n_points=length(rows), columns=...)

Explicit rows may be `FFDSample`s or tuples in column order. Supply `n_points`
for a generator without a known length. Rows are consumed only when iterated.
"""
struct FFDExplicitBlock{R} <: AbstractFFDBlock
    rows::R
    n_points::Int
    columns::Vector{Symbol}
end

function FFDExplicitBlock(rows; n_points=length(rows), columns=(_FFD_COORDINATES..., _FFD_FIELDS...))
    block = FFDExplicitBlock(rows, Int(n_points), Symbol[Symbol(c) for c in columns])
    _ffd_validate_layout(block)
    return block
end

"""
    FFDGriddedBlock(axes, rows; n_points=length(rows), sweep_order=..., columns=...)

Axes run from outermost to innermost. The remaining coordinates precede the
four field columns in each row. `n_points / prod(length(axis.values))` rows
belong to each grid cell; this may exceed one.
"""
struct FFDGriddedBlock{R} <: AbstractFFDBlock
    axes::Vector{FFDAxis}
    rows::R
    n_points::Int
    sweep_order::Vector{Symbol}
    columns::Vector{Symbol}
end

function FFDGriddedBlock(axes, rows; n_points=length(rows),
                         sweep_order=[axis.name for axis in axes], columns=nothing)
    axis_list = FFDAxis[axis for axis in axes]
    names = [axis.name for axis in axis_list]
    cols = isnothing(columns) ? Symbol[c for c in _FFD_COORDINATES if !(c in names)] : Symbol[Symbol(c) for c in columns]
    isnothing(columns) && append!(cols, _FFD_FIELDS)
    block = FFDGriddedBlock(axis_list, rows, Int(n_points), Symbol[Symbol(c) for c in sweep_order], cols)
    _ffd_validate_layout(block)
    return block
end

_ffd_axes(::FFDExplicitBlock) = FFDAxis[]
_ffd_axes(block::FFDGriddedBlock) = block.axes

function _ffd_validate_layout(block::AbstractFFDBlock)
    block.n_points >= 0 || _ffd_error("n_points must be a nonnegative integer")
    cols = block.columns
    length(cols) >= 4 && Tuple(cols[end-3:end]) == _FFD_FIELDS ||
        _ffd_error("field columns must end with Etheta_re Etheta_im Ephi_re Ephi_im")
    coordinate_cols = cols[1:end-4]
    all(c -> c in _FFD_COORDINATES, coordinate_cols) || _ffd_error("unknown data coordinate column")
    names = [axis.name for axis in _ffd_axes(block)]
    all(c -> c in _FFD_COORDINATES, names) || _ffd_error("unknown coordinate axis")
    combined = vcat(names, coordinate_cols)
    length(combined) == 6 && Set(combined) == Set(_FFD_COORDINATES) ||
        _ffd_error("each coordinate must occur exactly once, as an axis or a column")
    if block isa FFDGriddedBlock
        isempty(block.axes) && _ffd_error("gridded storage requires at least one axis")
        names == block.sweep_order || _ffd_error("axis declarations must exactly match sweep_order")
        product = 1
        for axis in block.axes
            isempty(axis.values) && _ffd_error("axis $(axis.name) cannot be empty")
            if axis.name == :pol
                all(v -> v isa FFDPolarization, axis.values) || _ffd_error("pol axis requires V/H values")
            else
                all(isfinite, axis.values) || _ffd_error("axis $(axis.name) contains nonfinite values")
            end
            length(axis.values) <= div(typemax(Int), product) || _ffd_error("axis product overflows Int")
            product *= length(axis.values)
        end
        block.n_points % product == 0 || _ffd_error("n_points must be divisible by the axis-length product")
    end
    return block
end

function _ffd_sample_row(sample::FFDSample, columns)
    return Tuple(c == :Etheta_re ? real(sample.Etheta) :
                 c == :Etheta_im ? imag(sample.Etheta) :
                 c == :Ephi_re ? real(sample.Ephi) :
                 c == :Ephi_im ? imag(sample.Ephi) : getproperty(sample, c) for c in columns)
end

function _ffd_checked_row(row, columns)
    data = row isa FFDSample ? _ffd_sample_row(row, columns) : Tuple(row)
    length(data) == length(columns) || _ffd_error("row has $(length(data)) values; expected $(length(columns))")
    return Tuple(if col == :pol
                     _ffd_pol(value)
                 else
                     value isa Real || _ffd_error("column $col requires a real numeric value")
                     number = Float64(value)
                     isfinite(number) || _ffd_error("column $col contains a nonfinite value")
                     number
                 end for (col, value) in zip(columns, data))
end

function _ffd_checked_block_row(row, block, number)
    number <= block.n_points || _ffd_error("more rows than n_points")
    if row isa FFDSample && block isa FFDGriddedBlock
        cell = div(number - 1, _ffd_rows_per_cell(block))
        for axis in Iterators.reverse(block.axes)
            cell, index = divrem(cell, length(axis.values))
            getproperty(row, axis.name) == axis.values[index + 1] ||
                _ffd_error("sample coordinate $(axis.name) disagrees with its gridded axis")
        end
    end
    return _ffd_checked_row(row, block.columns)
end

function _ffd_rows_per_cell(block::FFDGriddedBlock)
    # Axis vectors are mutable. Recheck the arithmetic invariants in O(number
    # of axes), so a malformed or subsequently mutated layout cannot divide by
    # zero or overflow while expanding a row.
    isempty(block.axes) && _ffd_error("gridded storage requires at least one axis")
    product = 1
    for axis in block.axes
        count = length(axis.values)
        count > 0 || _ffd_error("axis $(axis.name) cannot be empty")
        count <= div(typemax(Int), product) || _ffd_error("axis product overflows Int")
        product *= count
    end
    block.n_points > 0 && block.n_points % product == 0 ||
        _ffd_error("data rows require a positive n_points divisible by the axis-length product")
    return div(block.n_points, product)
end

"""Iterate stored rows as validated tuples in `block.columns` order."""
function each_ffd_row(block::AbstractFFDBlock)
    _ffd_validate_layout(block)
    return (_ffd_checked_block_row(row, block, number) for (number, row) in enumerate(block.rows))
end

struct FFDPointIterator{B,R}
    block::B
    rows::R
end
Base.IteratorSize(::Type{<:FFDPointIterator}) = Base.SizeUnknown()
Base.eltype(::Type{<:FFDPointIterator}) = FFDSample

function _ffd_point(block, row, number)
    values = Dict{Symbol,Any}(zip(block.columns, row))
    if block isa FFDGriddedBlock
        cell = div(number - 1, _ffd_rows_per_cell(block))
        for axis in Iterators.reverse(block.axes)
            cell, index = divrem(cell, length(axis.values))
            values[axis.name] = axis.values[index + 1]
        end
    end
    return FFDSample(values[:freq], values[:pol], values[:theta_inc], values[:phi_inc],
                     values[:theta_obs], values[:phi_obs],
                     complex(values[:Etheta_re], values[:Etheta_im]),
                     complex(values[:Ephi_re], values[:Ephi_im]))
end

function Base.iterate(points::FFDPointIterator, state=(nothing, 1, true))
    row_state, number, initial = state
    result = initial ? iterate(points.rows) : iterate(points.rows, row_state)
    if isnothing(result)
        number == points.block.n_points + 1 || _ffd_error("row count does not match n_points")
        return nothing
    end
    number <= points.block.n_points || _ffd_error("more rows than n_points")
    row, next_state = result
    return _ffd_point(points.block, row, number), (next_state, number + 1, false)
end

"""Lazily expand stored rows into canonical `FFDSample`s, in file order."""
each_ffd_point(block::AbstractFFDBlock) = FFDPointIterator(block, each_ffd_row(block))
"""Alias for `each_ffd_point`; expansion does not allocate the full dataset."""
expand_block(block::AbstractFFDBlock) = each_ffd_point(block)

function validate_ffd(block::AbstractFFDBlock; check_rows=true)
    _ffd_validate_layout(block)
    if check_rows
        count = 0
        for row in each_ffd_row(block)
            count += 1
            count <= block.n_points || _ffd_error("more rows than n_points")
        end
        count == block.n_points || _ffd_error("read $count rows; n_points declares $(block.n_points)")
    end
    return block
end

"""FFD global metadata and blocks; lazy file-backed rows remain valid while the source file is unchanged."""
struct FFDFile
    header::FFDHeader
    blocks::Vector{AbstractFFDBlock}
end
FFDFile(header::FFDHeader, blocks) = FFDFile(header, AbstractFFDBlock[block for block in blocks])

function validate_ffd(header::FFDHeader, blocks; check_rows=true)
    validate_ffd(header)
    count = 0
    for block in blocks
        validate_ffd(block; check_rows=check_rows)
        count += 1
    end
    count > 0 || _ffd_error("a file must contain at least one block")
    return nothing
end
validate_ffd(file::FFDFile; kwargs...) = validate_ffd(file.header, file.blocks; kwargs...)

function _ffd_write_value(io, value)
    if value isa FFDPolarization
        print(io, _ffd_pol_token(value))
    else
        @printf(io, "%.16e", value)
    end
end

function _ffd_write_values(io, values)
    for (index, value) in enumerate(values)
        index == 1 || print(io, ' ')
        _ffd_write_value(io, value)
    end
    println(io)
end

"""Write FFD text sequentially to an open stream. The caller owns the stream."""
function write_ffd(io::IO, header::FFDHeader, blocks)
    validate_ffd(header)
    println(io, "[FILE_HEADER]")
    for name in _FFD_HEADER_KEYS
        value = getproperty(header, Symbol(name))
        isnothing(value) || println(io, name, ' ', value)
    end
    for name in sort!(collect(keys(header.extras)))
        println(io, name, ' ', header.extras[name])
    end
    println(io, "[END_FILE_HEADER]")
    block_count = 0
    for block in blocks
        _ffd_validate_layout(block)
        block_count += 1
        println(io, "\n[BLOCK]")
        println(io, "storage ", block isa FFDExplicitBlock ? "explicit" : "gridded")
        if block isa FFDGriddedBlock
            println(io, "sweep_order ", join(block.sweep_order, ' '))
        end
        println(io, "n_points ", block.n_points)
        for axis in _ffd_axes(block)
            println(io, "[AXIS ", axis.name, ']')
            _ffd_write_values(io, axis.values)
            println(io, "[END_AXIS]")
        end
        println(io, "[DATA]")
        println(io, join(block.columns, ' '))
        count = 0
        for row in each_ffd_row(block)
            count += 1
            count <= block.n_points || _ffd_error("more rows than n_points")
            _ffd_write_values(io, row)
        end
        count == block.n_points || _ffd_error("wrote $count rows; n_points declares $(block.n_points)")
        println(io, "[END_DATA]\n[END_BLOCK]")
    end
    block_count > 0 || _ffd_error("a file must contain at least one block")
    return nothing
end

"""
    write_ffd(path, header, blocks)
    write_ffd(path, file::FFDFile)

Consume row iterators sequentially and atomically rename a temporary sibling
file after successful validation. On failure an existing destination is kept
and the temporary file is removed. The destination directory must exist.
"""
function write_ffd(path::AbstractString, header::FFDHeader, blocks)
    destination = abspath(path)
    temporary, io = mktemp(dirname(destination); cleanup=false)
    try
        write_ffd(io, header, blocks)
        close(io)
        # Base.Filesystem.rename falls back to copy-and-remove after an OS
        # rename error. Call the same runtime primitive without that fallback:
        # publication must either be an atomic rename or leave the old file.
        error = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), temporary, destination)
        error < 0 && Base.uv_error("atomic FFD rename", error)
    finally
        isopen(io) && close(io)
        ispath(temporary) && rm(temporary)
    end
    return destination
end
write_ffd(destination, file::FFDFile) = write_ffd(destination, file.header, file.blocks)

mutable struct _FFDReader
    io::IO
    line::Int
end

function _ffd_line(reader::_FFDReader)
    while !eof(reader.io)
        reader.line += 1
        line = strip(readline(reader.io))
        (isempty(line) || startswith(line, '#')) && continue
        return String(line)
    end
    return nothing
end

function _ffd_required_line(reader::_FFDReader)
    line = _ffd_line(reader)
    isnothing(line) && _ffd_error("unexpected end of file after line $(reader.line)")
    return line
end

function _ffd_expect(reader::_FFDReader, expected)
    line = _ffd_required_line(reader)
    line == expected || _ffd_error("line $(reader.line): expected $expected, got $(repr(line))")
end

function _ffd_pair(line, reader)
    match_value = match(r"^([A-Za-z_][A-Za-z_0-9]*)[ \t]+(.+)$", line)
    isnothing(match_value) && _ffd_error("line $(reader.line): malformed header field $(repr(line))")
    return String(match_value[1]), String(match_value[2])
end

function _ffd_number(token, context)
    occursin(_FFD_SCIENTIFIC, token) || _ffd_error("$context: expected a scientific-notation number, got $(repr(token))")
    value = tryparse(Float64, token)
    (isnothing(value) || !isfinite(value)) && _ffd_error("$context: nonfinite or invalid number $(repr(token))")
    return value
end

function _ffd_parse_row(line, columns, context)
    tokens = split(line)
    length(tokens) == length(columns) || _ffd_error("$context: expected $(length(columns)) columns; got $(length(tokens))")
    return Tuple(col == :pol ? _ffd_pol(token) : _ffd_number(token, context) for (col, token) in zip(columns, tokens))
end

function _ffd_parse_header(reader)
    _ffd_expect(reader, "[FILE_HEADER]")
    fields = Dict{String,String}()
    while true
        line = _ffd_required_line(reader)
        line == "[END_FILE_HEADER]" && break
        key, value = _ffd_pair(line, reader)
        haskey(fields, key) && _ffd_error("line $(reader.line): duplicate header field $key")
        fields[key] = value
    end
    for key in ("version", "freq_unit", "angle_unit", "field_unit")
        haskey(fields, key) || _ffd_error("missing mandatory file-header field $key")
    end
    extras = Dict(key => value for (key, value) in fields if !(key in _FFD_HEADER_KEYS))
    return FFDHeader(; version=fields["version"], freq_unit=fields["freq_unit"],
                     angle_unit=fields["angle_unit"], field_unit=fields["field_unit"],
                     solver=get(fields, "solver", nothing), problem=get(fields, "problem", nothing),
                     date=get(fields, "date", nothing), extras=extras)
end

# A chunk iterator closes its stream before returning each buffer, including on
# parse errors or partial consumption. It holds O(chunk_size * columns) memory.
struct _FFDDiskRows
    path::String
    offset::Int64
    n_points::Int
    columns::Vector{Symbol}
    file_size::Int64
    file_mtime::Float64
end
Base.IteratorSize(::Type{_FFDDiskRows}) = Base.HasLength()
Base.length(rows::_FFDDiskRows) = rows.n_points
Base.eltype(::Type{_FFDDiskRows}) = Tuple

function _ffd_read_chunk(rows::_FFDDiskRows, offset, remaining)
    metadata = stat(rows.path)
    (metadata.size == rows.file_size && metadata.mtime == rows.file_mtime) ||
        _ffd_error("source file changed after read_ffd: $(rows.path)")
    return open(rows.path, "r") do io
        seek(io, offset)
        reader = _FFDReader(io, 0)
        buffer = Tuple[]
        for _ in 1:min(256, remaining)
            line = _ffd_required_line(reader)
            push!(buffer, _ffd_parse_row(line, rows.columns, "file-backed data"))
        end
        return buffer, position(io)
    end
end

function Base.iterate(rows::_FFDDiskRows, state=(rows.offset, 0, Tuple[], 1))
    offset, consumed, buffer, index = state
    consumed >= rows.n_points && return nothing
    if index > length(buffer)
        buffer, offset = _ffd_read_chunk(rows, offset, rows.n_points - consumed)
        index = 1
    end
    return buffer[index], (offset, consumed + 1, buffer, index + 1)
end

function _ffd_parse_axis(reader, first_line)
    declaration = match(r"^\[AXIS ([A-Za-z_][A-Za-z_0-9]*)\]$", first_line)
    isnothing(declaration) && _ffd_error("line $(reader.line): malformed axis delimiter")
    name = Symbol(declaration[1])
    name in _FFD_COORDINATES || _ffd_error("unknown coordinate axis $name")
    data = name == :pol ? FFDPolarization[] : Float64[]
    while true
        line = _ffd_required_line(reader)
        line == "[END_AXIS]" && break
        for token in split(line)
            push!(data, name == :pol ? _ffd_pol(token) : _ffd_number(token, "line $(reader.line), axis $name"))
        end
    end
    return FFDAxis(name, data)
end

function _ffd_parse_block(reader, path, metadata; lazy)
    fields = Dict{String,String}()
    line = _ffd_required_line(reader)
    while !startswith(line, '[')
        key, value = _ffd_pair(line, reader)
        key in ("storage", "sweep_order", "n_points") || _ffd_error("unknown block-header field $key")
        haskey(fields, key) && _ffd_error("duplicate block-header field $key")
        fields[key] = value
        line = _ffd_required_line(reader)
    end
    all(key -> haskey(fields, key), ("storage", "n_points")) || _ffd_error("block requires storage and n_points")
    storage = fields["storage"]
    storage in ("explicit", "gridded") || _ffd_error("unsupported storage mode $(repr(storage))")
    occursin(r"^\d+$", fields["n_points"]) || _ffd_error("n_points must be a decimal nonnegative integer")
    count = tryparse(Int, fields["n_points"])
    isnothing(count) && _ffd_error("n_points overflows Int")
    axes = FFDAxis[]
    while startswith(line, "[AXIS")
        push!(axes, _ffd_parse_axis(reader, line))
        line = _ffd_required_line(reader)
    end
    line == "[DATA]" || _ffd_error("line $(reader.line): expected [DATA]")
    columns = Symbol.(split(_ffd_required_line(reader)))
    offset = position(reader.io)
    rows = lazy ? _FFDDiskRows(path, offset, count, columns, metadata.size, metadata.mtime) : Tuple[]
    if storage == "explicit"
        isempty(axes) || _ffd_error("explicit storage cannot declare axes")
        haskey(fields, "sweep_order") && _ffd_error("explicit storage cannot declare sweep_order")
        block = FFDExplicitBlock(rows; n_points=count, columns=columns)
    else
        haskey(fields, "sweep_order") || _ffd_error("gridded storage requires sweep_order")
        block = FFDGriddedBlock(axes, rows; n_points=count,
                                sweep_order=Symbol.(split(fields["sweep_order"])), columns=columns)
    end
    actual = 0
    while true
        line = _ffd_required_line(reader)
        line == "[END_DATA]" && break
        actual += 1
        actual <= count || _ffd_error("line $(reader.line): more rows than n_points")
        row = _ffd_parse_row(line, columns, "line $(reader.line)")
        lazy || push!(rows, row)
    end
    actual == count || _ffd_error("read $actual rows; n_points declares $count")
    _ffd_expect(reader, "[END_BLOCK]")
    return block
end

"""
    read_ffd(path; lazy=true) -> FFDFile

Parse and validate a complete FFD v1.0 text file. The default scans rows without
retaining them, storing file offsets instead. Subsequent row/point iteration
reads bounded chunks, closing every stream before yielding; stopping early
therefore does not leak file descriptors. Keep the source file unchanged while
using these blocks. Axis vectors and block metadata remain in memory.

Use `lazy=false` to load all stored rows for small files. Both modes validate
scientific-notation numbers, metadata, layouts, and exact row counts.
"""
function read_ffd(path::AbstractString; lazy=true)
    source = abspath(path)
    metadata = stat(source)
    return open(source, "r") do io
        reader = _FFDReader(io, 0)
        header = _ffd_parse_header(reader)
        blocks = AbstractFFDBlock[]
        while true
            line = _ffd_line(reader)
            isnothing(line) && break
            line == "[BLOCK]" || _ffd_error("line $(reader.line): expected [BLOCK]")
            push!(blocks, _ffd_parse_block(reader, source, metadata; lazy=lazy))
        end
        isempty(blocks) && _ffd_error("a file must contain at least one block")
        return FFDFile(header, blocks)
    end
end
