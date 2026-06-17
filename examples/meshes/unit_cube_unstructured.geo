// Unstructured tetrahedral mesh of the unit cube [0,1]^3.
//
// Used by examples/convergence_distributed_poisson_bracket_maxwell.jl when
// --mesh-family=unstructured is selected.
//
// Override the target resolution from the command line with:
//   gmsh unit_cube_unstructured.geo -3 -setnumber NxTarget 8

Lx = 1.0;
Ly = 1.0;
Lz = 1.0;

If (!Exists(NxTarget))
  NxTarget = 2;
EndIf
NxTarget = Max(1, Floor(NxTarget));
h = Lx / NxTarget;

Point(1) = {0,  0,  0,  h};
Point(2) = {Lx, 0,  0,  h};
Point(3) = {Lx, Ly, 0,  h};
Point(4) = {0,  Ly, 0,  h};

Point(5) = {0,  0,  Lz, h};
Point(6) = {Lx, 0,  Lz, h};
Point(7) = {Lx, Ly, Lz, h};
Point(8) = {0,  Ly, Lz, h};

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

Curve Loop(101) = {4, 9, -8, -12};
Plane Surface(101) = {101}; // x-min

Curve Loop(102) = {2, 11, -6, -10};
Plane Surface(102) = {102}; // x-max

Curve Loop(103) = {1, 10, -5, -9};
Plane Surface(103) = {103}; // y-min

Curve Loop(104) = {3, 12, -7, -11};
Plane Surface(104) = {104}; // y-max

Curve Loop(105) = {-4, -3, -2, -1};
Plane Surface(105) = {105}; // z-min

Curve Loop(106) = {5, 6, 7, 8};
Plane Surface(106) = {106}; // z-max

Surface Loop(201) = {101, 102, 103, 104, 105, 106};
Volume(201) = {201};

// The convergence driver rebuilds boundary triangles and assigns the solver
// boundary tag itself, but physical groups keep the generated VTK readable.
Physical Surface(1) = {101};
Physical Surface(2) = {102};
Physical Surface(3) = {103};
Physical Surface(4) = {104};
Physical Surface(5) = {105};
Physical Surface(6) = {106};
Physical Volume(7) = {201};

Mesh.MeshSizeMin = h;
Mesh.MeshSizeMax = h;
Mesh.MeshSizeFromPoints = 1;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 1;

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.ElementOrder = 1;
Mesh.Optimize = 1;

Mesh.SaveAll = 1;
Mesh.Binary = 0;
