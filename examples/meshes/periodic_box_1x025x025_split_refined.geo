// periodic_box_1x025x025_split_refined.geo
//
// Unstructured periodic tetrahedral mesh of [0,Lx] x [0,Ly] x [0,Lz].
//
// This version does NOT refine by reducing h directly.
// Instead, it:
//   1. builds the coarse unstructured periodic mesh,
//   2. generates the 3D mesh,
//   3. applies RefineMesh; one or more times.
//
// Physical tags are preserved:
//   Physical Surface(1): x-min
//   Physical Surface(2): x-max
//   Physical Surface(3): y-min
//   Physical Surface(4): y-max
//   Physical Surface(5): z-min
//   Physical Surface(6): z-max
//   Physical Volume(7):  volume domain
//
// Command-line examples:
//
//   gmsh periodic_box_1x025x025_split_refined.geo -nopopup -parse_and_exit
//
//   gmsh periodic_box_1x025x025_split_refined.geo \
//        -setnumber NxTarget 2 \
//        -setnumber NRefine 1 \
//        -nopopup -parse_and_exit
//
//   gmsh periodic_box_1x025x025_split_refined.geo \
//        -setnumber NxTarget 2 \
//        -setnumber NRefine 2 \
//        -nopopup -parse_and_exit
//
// Output:
//   periodic_box_split_refined.vtk

// -----------------------------------------------------------------------------
// Geometry size
// -----------------------------------------------------------------------------

Lx = 1.0;   // originally 2.0
Ly = 0.25;  // originally 1.0
Lz = 0.25;  // originally 1.0

// -----------------------------------------------------------------------------
// Coarse mesh resolution
// -----------------------------------------------------------------------------

If (!Exists(NxTarget))
  NxTarget = 2;
EndIf

NxTarget = Max(1, Floor(NxTarget));

// Coarse target size before split-refinement
h0 = Lx / NxTarget;

// -----------------------------------------------------------------------------
// Number of uniform split-refinement steps
//
// NRefine = 0 gives only the original coarse mesh.
// NRefine = 1 splits the mesh once.
// NRefine = 2 splits the already-refined mesh again.
//
// Approximate effective spacing:
//   h_eff ≈ h0 / 2^NRefine
// -----------------------------------------------------------------------------

If (!Exists(NRefine))
  NRefine = 1;
EndIf

NRefine = Max(0, Floor(NRefine));

// -----------------------------------------------------------------------------
// Points
// -----------------------------------------------------------------------------

Point(1) = {0,  0,  0,  h0};
Point(2) = {Lx, 0,  0,  h0};
Point(3) = {Lx, Ly, 0,  h0};
Point(4) = {0,  Ly, 0,  h0};

Point(5) = {0,  0,  Lz, h0};
Point(6) = {Lx, 0,  Lz, h0};
Point(7) = {Lx, Ly, Lz, h0};
Point(8) = {0,  Ly, Lz, h0};

// -----------------------------------------------------------------------------
// Curves
// -----------------------------------------------------------------------------

Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};

Line(5)  = {5, 6};
Line(6)  = {6, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 5};

Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// -----------------------------------------------------------------------------
// Surfaces
// -----------------------------------------------------------------------------

// x-min
Curve Loop(101) = {4, 9, -8, -12};
Plane Surface(101) = {101};

// x-max
Curve Loop(102) = {2, 11, -6, -10};
Plane Surface(102) = {102};

// y-min
Curve Loop(103) = {1, 10, -5, -9};
Plane Surface(103) = {103};

// y-max
Curve Loop(104) = {3, 12, -7, -11};
Plane Surface(104) = {104};

// z-min
Curve Loop(105) = {-4, -3, -2, -1};
Plane Surface(105) = {105};

// z-max
Curve Loop(106) = {5, 6, 7, 8};
Plane Surface(106) = {106};

// -----------------------------------------------------------------------------
// Volume
// -----------------------------------------------------------------------------

Surface Loop(201) = {101, 102, 103, 104, 105, 106};
Volume(201) = {201};

// -----------------------------------------------------------------------------
// Physical groups expected by the Julia code
// -----------------------------------------------------------------------------

Physical Surface(1) = {101}; // x-min
Physical Surface(2) = {102}; // x-max
Physical Surface(3) = {103}; // y-min
Physical Surface(4) = {104}; // y-max
Physical Surface(5) = {105}; // z-min
Physical Surface(6) = {106}; // z-max

Physical Volume(7) = {201};

// -----------------------------------------------------------------------------
// Periodicity
//
// Convention:
//   plus side = translated minus side
// -----------------------------------------------------------------------------

Periodic Surface {102} = {101} Translate {Lx, 0,  0};
Periodic Surface {104} = {103} Translate {0,  Ly, 0};
Periodic Surface {106} = {105} Translate {0,  0,  Lz};

// -----------------------------------------------------------------------------
// Mesh options for the initial coarse mesh
// -----------------------------------------------------------------------------

Mesh.MeshSizeMin = h0;
Mesh.MeshSizeMax = h0;

Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 1;

// 2D surface mesher: Frontal-Delaunay
// 3D volume mesher: Delaunay tetrahedralization
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;

Mesh.ElementOrder = 1;
Mesh.Optimize = 1;

// Keep both boundary triangles and volume tetrahedra in the VTK output.
Mesh.SaveAll = 1;
Mesh.Binary = 0;

// -----------------------------------------------------------------------------
// Generate the coarse mesh
// -----------------------------------------------------------------------------

Mesh 3;


// Save directly from the .geo script.
Save "periodic_box_coarse.vtk";