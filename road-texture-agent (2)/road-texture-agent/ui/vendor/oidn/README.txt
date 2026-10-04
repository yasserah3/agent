Intel Open Image Denoise, for Render photo in the 3D tab (ui/denoise.js).

rt_hdr_calb_cnrm.tza  the trained weights of OIDN's RT filter for HDR pictures
                      with clean (noise-free) albedo and normal, base model.
                      From https://github.com/RenderKit/oidn-weights (the
                      weights of Open Image Denoise 2.4/2.5), unchanged.

ui/denoise.js runs the network (OIDN 2.4.1's U-Net, input and output
processing, autoexposure and tiles), ported to WebGL2. Tested against
oidnDenoise 2.5 with these weights: the same picture to within 0.1%.

Copyright 2018 Intel Corporation. Apache License 2.0: LICENSE.txt.
