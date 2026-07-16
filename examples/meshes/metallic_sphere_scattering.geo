// metallic_sphere_scattering.geo
//
// Unstructured tetrahedral mesh for electromagnetic scattering by a perfectly
// conducting metallic sphere embedded in a Cartesian box with nonlinear PML
// slabs near the six outer faces.
//
// Geometry:
//   computational domain = Cartesian box minus inner sphere
//   sphere center        = (0, 0, 0)
//   default radius       = 1
//
// Physical groups expected by DiscoGMPI:
//   Physical Surface(1):  x-min outer PML boundary
//   Physical Surface(2):  x-max outer PML boundary
//   Physical Surface(3):  y-min outer PML boundary
//   Physical Surface(4):  y-max outer PML boundary
//   Physical Surface(5):  z-min outer PML boundary
//   Physical Surface(6):  z-max outer PML boundary
//   Physical Surface(10): inner metallic sphere, PEC boundary
//   Physical Volume(7):   air + PML computational volume
//
// The PML is not a separate material volume in this mesh. DiscoGMPI applies the
// Abarbanel--Gottlieb--Hesthaven nonlinear PML through Cartesian coordinate
// profiles, so the solver should use the same PMLWidth value given here:
//
//   x PML slabs: |x| >= HalfX - PMLWidth
//   y PML slabs: |y| >= HalfY - PMLWidth
//   z PML slabs: |z| >= HalfZ - PMLWidth
//
// Example:
//   gmsh metallic_sphere_scattering.geo -3 -format vtk \
//        -o metallic_sphere_scattering.vtk \
//        -setnumber ElementsPerWavelength 13 \
//        -setnumber PMLWidth 1.0 \
//        -setnumber AirBuffer 1.0
//
// `ElementsPerWavelength` controls the sphere and near-field resolution. The
// far-field and PML mesh sizes are intentionally coarser by default; otherwise a
// Cartesian box with one wavelength of air and one wavelength of PML on every
// side creates tens of millions of tetrahedra.
//
// For the Fezoui et al. Section 4.3 normalized benchmark:
//   Radius = 1, Lambda0 = 1, k0 = 2*pi, ka = 2*pi.
//
// A cheaper validation-gate mesh can be generated with:
//   gmsh metallic_sphere_scattering.geo -3 -format vtk \
//        -o metallic_sphere_scattering_validation.vtk \
//        -setnumber ElementsPerWavelength 5 \
//        -setnumber SphereMeshSizeFactor 1.0 \
//        -setnumber PMLMeshSizeFactor 3.0 \
//        -setnumber FarMeshSizeFactor 4.0 \
//        -setnumber CurvatureSamples 8 \
//        -setnumber PMLWidth 1.0 \
//        -setnumber AirBuffer 1.0
//
// A still lighter high-order RCS smoke mesh with four first-order elements per
// wavelength near the sphere is:
//   gmsh metallic_sphere_scattering.geo -3 -format vtk \
//        -o metallic_sphere_scattering_epw4.vtk \
//        -setnumber ElementsPerWavelength 4 \
//        -setnumber SphereMeshSizeFactor 1.0 \
//        -setnumber PMLMeshSizeFactor 3.0 \
//        -setnumber FarMeshSizeFactor 4.0 \
//        -setnumber CurvatureSamples 8 \
//        -setnumber PMLWidth 1.0 \
//        -setnumber AirBuffer 1.0 \
//        -setnumber OptimizeMesh 0

SetFactory("OpenCASCADE");

// -----------------------------------------------------------------------------
// User parameters
// -----------------------------------------------------------------------------

If (!Exists(Radius))
  Radius = 1.0;
EndIf

If (!Exists(Lambda0))
  Lambda0 = 1.0;
EndIf

// Target number of first-order elements per central wavelength near the sphere.
If (!Exists(ElementsPerWavelength))
  ElementsPerWavelength = 13.0;
EndIf

// Physical air gap from the sphere surface to the PML interface along each
// coordinate direction. The default is one central wavelength.
If (!Exists(AirBuffer))
  AirBuffer = Lambda0;
EndIf

// Cartesian PML thickness at each side of the box. The default is one central
// wavelength; use the same value in the DiscoGMPI scattering driver.
If (!Exists(PMLWidth))
  PMLWidth = Lambda0;
EndIf

If (!Exists(SphereMeshSizeFactor))
  SphereMeshSizeFactor = 0.75;
EndIf

If (!Exists(PMLMeshSizeFactor))
  PMLMeshSizeFactor = 4.0;
EndIf

If (!Exists(FarMeshSizeFactor))
  FarMeshSizeFactor = 4.0;
EndIf

// Distance from the metallic sphere over which the mesh transitions from
// hSphere to hFar. Keeping this smaller than AirBuffer avoids filling the whole
// air region with the fine sphere resolution.
If (!Exists(SphereRefinementDistance))
  SphereRefinementDistance = Min(0.5 * Lambda0, AirBuffer);
EndIf

// Leave optimization disabled by default. For large scattering meshes the
// optimizer can require more memory than the meshing step itself.
If (!Exists(OptimizeMesh))
  OptimizeMesh = 0;
EndIf

// Gmsh 3D algorithm. 1 = Delaunay. On builds with HXT support, try
// `-setnumber VolumeAlgorithm3D 10` for large meshes.
If (!Exists(VolumeAlgorithm3D))
  VolumeAlgorithm3D = 1;
EndIf

If (!Exists(MeshOrder))
  MeshOrder = 1;
EndIf

If (!Exists(CurvatureSamples))
  CurvatureSamples = 24;
EndIf

Radius = Max(Radius, 1.0e-12);
Lambda0 = Max(Lambda0, 1.0e-12);
ElementsPerWavelength = Max(ElementsPerWavelength, 1.0);
AirBuffer = Max(AirBuffer, 1.0e-12);
PMLWidth = Max(PMLWidth, 1.0e-12);
SphereRefinementDistance = Max(SphereRefinementDistance, 1.0e-12);
OptimizeMesh = Floor(OptimizeMesh);
VolumeAlgorithm3D = Floor(VolumeAlgorithm3D);
CurvatureSamples = Floor(Max(CurvatureSamples, 0));

h0 = Lambda0 / ElementsPerWavelength;
hSphere = SphereMeshSizeFactor * h0;
hPML = PMLMeshSizeFactor * h0;
hFar = FarMeshSizeFactor * h0;

HalfX = Radius + AirBuffer + PMLWidth;
HalfY = Radius + AirBuffer + PMLWidth;
HalfZ = Radius + AirBuffer + PMLWidth;

eps = 1.0e-7 * Max(HalfX, Max(HalfY, HalfZ));

// -----------------------------------------------------------------------------
// CAD geometry: Cartesian box minus metallic sphere
// -----------------------------------------------------------------------------

Box(1) = {-HalfX, -HalfY, -HalfZ, 2.0 * HalfX, 2.0 * HalfY, 2.0 * HalfZ};
Sphere(2) = {0.0, 0.0, 0.0, Radius};

domain[] = BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; };

// Surface selections after the Boolean operation. These stable physical tags are
// what the Julia side should use for the boundary registry.
xmin[] = Surface In BoundingBox {
  -HalfX - eps, -HalfY - eps, -HalfZ - eps,
  -HalfX + eps,  HalfY + eps,  HalfZ + eps
};
xmax[] = Surface In BoundingBox {
   HalfX - eps, -HalfY - eps, -HalfZ - eps,
   HalfX + eps,  HalfY + eps,  HalfZ + eps
};
ymin[] = Surface In BoundingBox {
  -HalfX - eps, -HalfY - eps, -HalfZ - eps,
   HalfX + eps, -HalfY + eps,  HalfZ + eps
};
ymax[] = Surface In BoundingBox {
  -HalfX - eps,  HalfY - eps, -HalfZ - eps,
   HalfX + eps,  HalfY + eps,  HalfZ + eps
};
zmin[] = Surface In BoundingBox {
  -HalfX - eps, -HalfY - eps, -HalfZ - eps,
   HalfX + eps,  HalfY + eps, -HalfZ + eps
};
zmax[] = Surface In BoundingBox {
  -HalfX - eps, -HalfY - eps,  HalfZ - eps,
   HalfX + eps,  HalfY + eps,  HalfZ + eps
};
sphere[] = Surface In BoundingBox {
  -Radius - eps, -Radius - eps, -Radius - eps,
   Radius + eps,  Radius + eps,  Radius + eps
};

// -----------------------------------------------------------------------------
// Physical tags
// -----------------------------------------------------------------------------

Physical Surface(1) = {xmin[]};    // x-min outer PML boundary
Physical Surface(2) = {xmax[]};    // x-max outer PML boundary
Physical Surface(3) = {ymin[]};    // y-min outer PML boundary
Physical Surface(4) = {ymax[]};    // y-max outer PML boundary
Physical Surface(5) = {zmin[]};    // z-min outer PML boundary
Physical Surface(6) = {zmax[]};    // z-max outer PML boundary
Physical Surface(10) = {sphere[]}; // inner PEC metallic sphere

Physical Volume(7) = {domain[]};   // air + PML volume

// -----------------------------------------------------------------------------
// Mesh-size control
// -----------------------------------------------------------------------------

// Resolve the curved PEC scatterer accurately, then transition to the target
// wavelength-based volume size before entering the PML.
Field[1] = Distance;
Field[1].SurfacesList = {sphere[]};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = hSphere;
Field[2].SizeMax = hFar;
Field[2].DistMin = 0.0;
Field[2].DistMax = SphereRefinementDistance;

// Keep outer-box/PML faces from becoming coarser than the wavelength target.
Field[3] = Distance;
Field[3].SurfacesList = {xmin[], xmax[], ymin[], ymax[], zmin[], zmax[]};

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = hPML;
Field[4].SizeMax = hFar;
Field[4].DistMin = 0.0;
Field[4].DistMax = PMLWidth;

Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

Mesh.MeshSizeMin = Min(hSphere, Min(hPML, hFar));
Mesh.MeshSizeMax = Max(hSphere, Max(hPML, hFar));

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = CurvatureSamples;
Mesh.MeshSizeExtendFromBoundary = 0;

// Frontal-Delaunay for the surfaces and the selected 3D algorithm in volume.
Mesh.Algorithm = 6;
Mesh.Algorithm3D = VolumeAlgorithm3D;

Mesh.ElementOrder = MeshOrder;
// Mesh.Optimize = OptimizeMesh;
Mesh.OptimizeNetgen = 0;

// Keep both boundary triangles and volume tetrahedra in the VTK output.
Mesh.SaveAll = 1;
Mesh.Binary = 0;
