//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 2, 0.5, 0.5};
//+
Physical Surface("PBC", 1) = {1};
//+
Physical Surface("PBC", 2) = {2};
//+
Physical Surface("PBC", 2) = {2};
