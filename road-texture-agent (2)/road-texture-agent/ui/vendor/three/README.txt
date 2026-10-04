three.js r186 (MIT licence, see LICENSE), bundled so the 3D tab works offline:
three.module.min.js and three.core.min.js minified from build/three.module.js
and build/three.core.js (the module's import of ./three.core.js pointed at
./three.core.min.js), and under addons/ the files the 3D tab uses from
examples/jsm, unchanged. ui/index.html maps "three" and "three/addons/" to
these in its import map.

Also bundled, each with its own MIT licence:
  ../three-mesh-bvh/index.module.js        three-mesh-bvh 0.9.15 (build/)
  ../three-gpu-pathtracer/index.module.js  three-gpu-pathtracer 0.0.26 (build/), with a light
                                           tree added for the photo render (LightTreeUniform and
                                           the GLSL marked "light tree"; see its header)
