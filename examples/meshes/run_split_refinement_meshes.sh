#!/bin/bash
set -euo pipefail

# --------------------------------------------------------------------
# User settings
# --------------------------------------------------------------------

GEO_FILE="periodic_box_1x025x025_split_refined.geo"
OUT_DIR="refined_meshes"

NxTarget=2
NRefineValues=(0 1 2 3)

COARSE_VTK="periodic_box_coarse.vtk"

# --------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------

if ! command -v gmsh > /dev/null 2>&1; then
    echo "Error: gmsh was not found in PATH."
    exit 1
fi

if [ ! -f "$GEO_FILE" ]; then
    echo "Error: GEO file not found: $GEO_FILE"
    exit 1
fi

mkdir -p "$OUT_DIR"

# --------------------------------------------------------------------
# Compute maximum requested refinement level
# --------------------------------------------------------------------

MAX_REFINE=0
for NRefine in "${NRefineValues[@]}"; do
    if [ "$NRefine" -gt "$MAX_REFINE" ]; then
        MAX_REFINE="$NRefine"
    fi
done

# --------------------------------------------------------------------
# Generate coarse periodic mesh once
# --------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Generating coarse periodic mesh with NxTarget=${NxTarget}"
echo "This corresponds to NRefine=0"
echo "------------------------------------------------------------"

rm -f "$COARSE_VTK"

gmsh "$GEO_FILE" \
    -setnumber NxTarget "$NxTarget" \
    -nopopup \
    -parse_and_exit

if [ ! -f "$COARSE_VTK" ]; then
    echo "Error: expected coarse mesh was not created: $COARSE_VTK"
    exit 1
fi

# Store level 0: no split refinement
cp "$COARSE_VTK" "${OUT_DIR}/periodic_box_Nx${NxTarget}_split_refine0.vtk"

echo "Saved: ${OUT_DIR}/periodic_box_Nx${NxTarget}_split_refine0.vtk"

# --------------------------------------------------------------------
# Split-refine incrementally from the mesh file
# --------------------------------------------------------------------

CURRENT_MESH="$COARSE_VTK"

if [ "$MAX_REFINE" -ge 1 ]; then
    for level in $(seq 1 "$MAX_REFINE"); do
        echo "------------------------------------------------------------"
        echo "Applying split refinement level ${level}"
        echo "------------------------------------------------------------"

        NEXT_MESH="${OUT_DIR}/periodic_box_Nx${NxTarget}_split_refine${level}.vtk"

        gmsh "$CURRENT_MESH" \
            -refine \
            -format vtk \
            -o "$NEXT_MESH" \
            -nopopup

        if [ ! -f "$NEXT_MESH" ]; then
            echo "Error: expected refined mesh was not created: $NEXT_MESH"
            exit 1
        fi

        echo "Saved: $NEXT_MESH"

        CURRENT_MESH="$NEXT_MESH"
    done
fi

# --------------------------------------------------------------------
# Check requested meshes
# --------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Requested meshes generated:"
echo "------------------------------------------------------------"

for NRefine in "${NRefineValues[@]}"; do
    FILE="${OUT_DIR}/periodic_box_Nx${NxTarget}_split_refine${NRefine}.vtk"
    if [ -f "$FILE" ]; then
        echo "  $FILE"
    else
        echo "  Missing: $FILE"
        exit 1
    fi
done

echo "------------------------------------------------------------"
echo "All requested split-refined meshes generated successfully."
echo "Output directory: $OUT_DIR"
echo "------------------------------------------------------------"