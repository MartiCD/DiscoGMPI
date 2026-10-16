// pec_almond_scattering.geo
//
// Tetrahedral mesh generator for the Austin CEM Benchmark Suite
// Problem IIIA PEC almond target.
//
// Geometry:
//   computational domain = Cartesian box minus imported closed almond surface
//   target default length = 9.936 in = 0.2523744 m
//
// Physical groups expected by DiscoGMPI:
//   Physical Surface(1):  x-min outer PML boundary
//   Physical Surface(2):  x-max outer PML boundary
//   Physical Surface(3):  y-min outer PML boundary
//   Physical Surface(4):  y-max outer PML boundary
//   Physical Surface(5):  z-min outer PML boundary
//   Physical Surface(6):  z-max outer PML boundary
//   Physical Surface(10): inner PEC almond boundary
//   Physical Volume(7):   air + PML computational volume
//
// This file intentionally imports a closed triangulated almond surface instead
// of rebuilding the almond from an approximate analytic formula. The Austin
// benchmark package available in this workspace contains a repaired STL:
//
//   /home/mcircuns/CODE/AustinCEMBenchmarks/NASA-Almond_trimesh_repaired.stl
//
// The recommended generator is:
//
//   bash generate_pec_almond_scattering_mesh.sh
//
// It creates a temporary import file containing the required `Merge "file.stl"`
// command, then includes this reusable geometry body. This avoids depending on
// Gmsh-version-specific string handling for `Merge`.
//
// If you prefer to call Gmsh manually, use:
//
//   printf 'Merge "/path/to/NASA-Almond_trimesh_repaired.stl";\nInclude "%s";\n' \
//        "$PWD/pec_almond_scattering.geo" > /tmp/pec_almond_import.geo
//   gmsh /tmp/pec_almond_import.geo -3 -format vtk \
//        -o pec_almond_scattering_fx1_epw4.vtk \
//        -setnumber FrequencyHz 3.5e9 \
//        -setnumber ElementsPerWavelength 4 \
//        -setnumber AirBufferWavelengths 1.0 \
//        -setnumber PMLWidthWavelengths 1.0
//
// Gmsh keeps the imported almond triangles as the exact PEC boundary. Therefore
// the target surface resolution is controlled primarily by the STL quality; the
// volume mesh size fields below control the tetrahedra away from the surface.

// -----------------------------------------------------------------------------
// User parameters
// -----------------------------------------------------------------------------

If (!Exists(C0))
  C0 = 299792458.0;
EndIf

// Default to the first Austin IIIA reference frequency.
If (!Exists(FrequencyHz))
  FrequencyHz = 3.5e9;
EndIf

If (!Exists(Lambda0))
  Lambda0 = C0 / FrequencyHz;
EndIf

// Target length in meters. The imported STL in this workspace already has
// length 9.936 in, but this scale parameter allows size-sweep meshes.
If (!Exists(AlmondLength))
  AlmondLength = 9.936 * 0.0254;
EndIf

// Target number of first-order elements per free-space wavelength in the
// near-target region.
If (!Exists(ElementsPerWavelength))
  ElementsPerWavelength = 4.0;
EndIf

// Air gap and PML thickness are specified in wavelengths by default, because
// the Austin problem is dimensional.
If (!Exists(AirBufferWavelengths))
  AirBufferWavelengths = 1.0;
EndIf

If (!Exists(PMLWidthWavelengths))
  PMLWidthWavelengths = 1.0;
EndIf

If (!Exists(AirBuffer))
  AirBuffer = AirBufferWavelengths * Lambda0;
EndIf

If (!Exists(PMLWidth))
  PMLWidth = PMLWidthWavelengths * Lambda0;
EndIf

If (!Exists(AlmondMeshSizeFactor))
  AlmondMeshSizeFactor = 0.75;
EndIf

If (!Exists(PMLMeshSizeFactor))
  PMLMeshSizeFactor = 3.0;
EndIf

If (!Exists(FarMeshSizeFactor))
  FarMeshSizeFactor = 4.0;
EndIf

If (!Exists(AlmondRefinementDistance))
  AlmondRefinementDistance = Min(0.5 * Lambda0, AirBuffer);
EndIf

// Optional azimuthal rotation of the imported almond about z, in degrees.
// The benchmark aspect sweep is normally handled by rotating the incident
// direction, so the default mesh orientation is left unchanged.
If (!Exists(RotationZDegrees))
  RotationZDegrees = 0.0;
EndIf

If (!Exists(OptimizeMesh))
  OptimizeMesh = 0;
EndIf

// 1 = Delaunay. If a local Gmsh build has HXT support, 10 can be faster for
// large meshes.
If (!Exists(VolumeAlgorithm3D))
  VolumeAlgorithm3D = 1;
EndIf

If (!Exists(MeshOrder))
  MeshOrder = 1;
EndIf

If (!Exists(CurvatureSamples))
  CurvatureSamples = 0;
EndIf

C0 = Max(C0, 1.0e-12);
FrequencyHz = Max(FrequencyHz, 1.0e-12);
Lambda0 = Max(Lambda0, 1.0e-12);
AlmondLength = Max(AlmondLength, 1.0e-12);
ElementsPerWavelength = Max(ElementsPerWavelength, 1.0);
AirBuffer = Max(AirBuffer, 1.0e-12);
PMLWidth = Max(PMLWidth, 1.0e-12);
AlmondRefinementDistance = Max(AlmondRefinementDistance, 1.0e-12);
OptimizeMesh = Floor(OptimizeMesh);
VolumeAlgorithm3D = Floor(VolumeAlgorithm3D);
MeshOrder = Floor(Max(MeshOrder, 1));
CurvatureSamples = Floor(Max(CurvatureSamples, 0));

// Bounds of NASA-Almond_trimesh_repaired.stl before centering/scaling. They are
// used only for deterministic normalization and box sizing.
RawMinX = -0.10515683889389038;
RawMaxX =  0.14721755683422089;
RawMinY = -0.04878604784607887;
RawMaxY =  0.04877427592873573;
RawMinZ = -0.01626094989478588;
RawMaxZ =  0.01626094989478588;

RawLength = RawMaxX - RawMinX;
RawWidth = RawMaxY - RawMinY;
RawThickness = RawMaxZ - RawMinZ;

RawCenterX = 0.5 * (RawMinX + RawMaxX);
RawCenterY = 0.5 * (RawMinY + RawMaxY);
RawCenterZ = 0.5 * (RawMinZ + RawMaxZ);

TargetScale = AlmondLength / RawLength;
RotationZ = RotationZDegrees * Pi / 180.0;

TargetHalfX =
  0.5 * TargetScale *
  (RawLength * Abs(Cos(RotationZ)) + RawWidth * Abs(Sin(RotationZ)));
TargetHalfY =
  0.5 * TargetScale *
  (RawLength * Abs(Sin(RotationZ)) + RawWidth * Abs(Cos(RotationZ)));
TargetHalfZ = 0.5 * TargetScale * RawThickness;

h0 = Lambda0 / ElementsPerWavelength;
hAlmond = AlmondMeshSizeFactor * h0;
hPML = PMLMeshSizeFactor * h0;
hFar = FarMeshSizeFactor * h0;

HalfX = TargetHalfX + AirBuffer + PMLWidth;
HalfY = TargetHalfY + AirBuffer + PMLWidth;
HalfZ = TargetHalfZ + AirBuffer + PMLWidth;

eps = 1.0e-7 * Max(HalfX, Max(HalfY, HalfZ));

// -----------------------------------------------------------------------------
// Imported PEC almond surface
// -----------------------------------------------------------------------------

almond[] = Surface{:};

// Center the target in the computational box, scale it to AlmondLength, then
// apply an optional z-rotation.
Translate {-RawCenterX, -RawCenterY, -RawCenterZ} {
  Surface{almond[]};
}

Dilate {{0.0, 0.0, 0.0}, TargetScale} {
  Surface{almond[]};
}

Rotate {{0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, RotationZ} {
  Surface{almond[]};
}

// -----------------------------------------------------------------------------
// Outer Cartesian box
// -----------------------------------------------------------------------------

Point(101) = {-HalfX, -HalfY, -HalfZ, hFar};
Point(102) = { HalfX, -HalfY, -HalfZ, hFar};
Point(103) = { HalfX,  HalfY, -HalfZ, hFar};
Point(104) = {-HalfX,  HalfY, -HalfZ, hFar};
Point(105) = {-HalfX, -HalfY,  HalfZ, hFar};
Point(106) = { HalfX, -HalfY,  HalfZ, hFar};
Point(107) = { HalfX,  HalfY,  HalfZ, hFar};
Point(108) = {-HalfX,  HalfY,  HalfZ, hFar};

Line(101) = {101, 102};
Line(102) = {102, 103};
Line(103) = {103, 104};
Line(104) = {104, 101};
Line(105) = {105, 106};
Line(106) = {106, 107};
Line(107) = {107, 108};
Line(108) = {108, 105};
Line(109) = {101, 105};
Line(110) = {102, 106};
Line(111) = {103, 107};
Line(112) = {104, 108};

Curve Loop(101) = {104, 109, -108, -112};
Plane Surface(101) = {101}; // x-min

Curve Loop(102) = {102, 111, -106, -110};
Plane Surface(102) = {102}; // x-max

Curve Loop(103) = {101, 110, -105, -109};
Plane Surface(103) = {103}; // y-min

Curve Loop(104) = {103, 112, -107, -111};
Plane Surface(104) = {104}; // y-max

Curve Loop(105) = {101, 102, 103, 104};
Plane Surface(105) = {105}; // z-min

Curve Loop(106) = {105, 106, 107, 108};
Plane Surface(106) = {106}; // z-max

Surface Loop(201) = {101, 102, 103, 104, 105, 106};
Surface Loop(202) = {almond[]};

// The second surface loop is an internal closed cavity, so the almond interior
// is excluded from the computational volume.
Volume(301) = {201, 202};

// -----------------------------------------------------------------------------
// Physical tags
// -----------------------------------------------------------------------------

Physical Surface(1) = {101};       // x-min outer PML boundary
Physical Surface(2) = {102};       // x-max outer PML boundary
Physical Surface(3) = {103};       // y-min outer PML boundary
Physical Surface(4) = {104};       // y-max outer PML boundary
Physical Surface(5) = {105};       // z-min outer PML boundary
Physical Surface(6) = {106};       // z-max outer PML boundary
Physical Surface(10) = {almond[]}; // inner PEC almond boundary

Physical Volume(7) = {301};        // air + PML volume

// -----------------------------------------------------------------------------
// Mesh-size control
// -----------------------------------------------------------------------------

Field[1] = Distance;
Field[1].SurfacesList = {almond[]};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = hAlmond;
Field[2].SizeMax = hFar;
Field[2].DistMin = 0.0;
Field[2].DistMax = AlmondRefinementDistance;

Field[3] = Distance;
Field[3].SurfacesList = {101, 102, 103, 104, 105, 106};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = hPML;
Field[4].SizeMax = hFar;
Field[4].DistMin = 0.0;
Field[4].DistMax = PMLWidth;

Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

Mesh.MeshSizeMin = Min(hAlmond, Min(hPML, hFar));
Mesh.MeshSizeMax = Max(hAlmond, Max(hPML, hFar));

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = CurvatureSamples;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.Algorithm = 6;
Mesh.Algorithm3D = VolumeAlgorithm3D;
Mesh.ElementOrder = MeshOrder;
Mesh.Optimize = OptimizeMesh;
Mesh.OptimizeNetgen = 0;

// Keep both boundary triangles and volume tetrahedra in the VTK output.
Mesh.SaveAll = 1;
Mesh.Binary = 0;
