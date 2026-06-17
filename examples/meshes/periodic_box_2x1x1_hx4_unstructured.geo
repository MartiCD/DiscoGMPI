// periodic_box_2x1x1_hx4_unstructured.geo
//
// Unstructured periodic tetrahedral mesh of [0,2] x [0,1] x [0,1].
//
// This file is designed to match the conventions used by the Julia script:
//   Physical Surface(1): x-min
//   Physical Surface(2): x-max
//   Physical Surface(3): y-min
//   Physical Surface(4): y-max
//   Physical Surface(5): z-min
//   Physical Surface(6): z-max
//   Physical Volume(7):  material/volume domain
//
// The target mesh size is chosen from "4 elements along x":
//   h = Lx / 4 = 0.5
//
// The mesh remains unstructured: no Transfinite Curve, no Transfinite Surface,
// no Recombine. Gmsh will generate an unstructured triangular surface mesh and
// an unstructured tetrahedral volume mesh.

Lx = 2.0;
Ly = 1.0;
Lz = 1.0;

NxTarget = 4;
h = Lx / NxTarget;

// -----------------------------------------------------------------------------
// Points
// -----------------------------------------------------------------------------

Point(1) = {0,  0,  0,  h};
Point(2) = {Lx, 0,  0,  h};
Point(3) = {Lx, Ly, 0,  h};
Point(4) = {0,  Ly, 0,  h};

Point(5) = {0,  0,  Lz, h};
Point(6) = {Lx, 0,  Lz, h};
Point(7) = {Lx, Ly, Lz, h};
Point(8) = {0,  Ly, Lz, h};

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
//
// Surface IDs are explicit and stable.
// Physical tags below are the ones read by the Julia pipeline.
// -----------------------------------------------------------------------------

// x-min, outward normal approximately (-1,0,0)
Curve Loop(101) = {4, 9, -8, -12};
Plane Surface(101) = {101};

// x-max, outward normal approximately (+1,0,0)
Curve Loop(102) = {2, 11, -6, -10};
Plane Surface(102) = {102};

// y-min, outward normal approximately (0,-1,0)
Curve Loop(103) = {1, 10, -5, -9};
Plane Surface(103) = {103};

// y-max, outward normal approximately (0,+1,0)
Curve Loop(104) = {3, 12, -7, -11};
Plane Surface(104) = {104};

// z-min, outward normal approximately (0,0,-1)
Curve Loop(105) = {-4, -3, -2, -1};
Plane Surface(105) = {105};

// z-max, outward normal approximately (0,0,+1)
Curve Loop(106) = {5, 6, 7, 8};
Plane Surface(106) = {106};

// -----------------------------------------------------------------------------
// Volume
// -----------------------------------------------------------------------------

Surface Loop(201) = {101, 102, 103, 104, 105, 106};
Volume(201) = {201};

// -----------------------------------------------------------------------------
// Physical groups expected by the code
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
// Important convention:
//   plus side = translated minus side
// -----------------------------------------------------------------------------

Periodic Surface {102} = {101} Translate {Lx, 0,  0};
Periodic Surface {104} = {103} Translate {0,  Ly, 0};
Periodic Surface {106} = {105} Translate {0,  0,  Lz};

// -----------------------------------------------------------------------------
// Mesh options
// -----------------------------------------------------------------------------

Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;

Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 1;

// 2D surface mesher: Frontal-Delaunay.
// 3D volume mesher: Delaunay tetrahedralization.
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;

Mesh.ElementOrder = 1;
Mesh.Optimize = 1;

// Keep both boundary triangles and volume tetrahedra in the VTK output.
Mesh.SaveAll = 1;
Mesh.Binary = 0;
