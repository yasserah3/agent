// Blender's sky (the Sky Texture of Blender 5, type Multiple Scattering, its
// default), ported to JavaScript so the 3D tab has the same sky as Blender and
// Cycles: sunlight scattered by the air and by aerosols (dust, haze), absorbed by
// ozone, with the light scattered many times over approximated, and the sun's
// disc seen through the air.
//
// Ported from Blender 5.0/5.1: intern/sky/source/sky_multiple_scattering.cpp
// (Copyright 2022 Fernando Garcia Linan, 2011-2025 Blender Authors, MIT licence)
// and the sky lookup in intern/cycles/kernel/svm/sky.h (Copyright 2011-2025
// Blender Authors, Apache License 2.0); see vendor/blender-sky/LICENSES.txt.
// Changes: written in JavaScript, the result in linear Rec.709 RGB as Cycles
// shows it. Directions in Blender's axes: x east, y north, z up.

// Earth's atmosphere (lengths in km)
const GROUND_ALBEDO = 0.3;
const PHASE_ISOTROPIC = 1 / (4 * Math.PI);
const RAYLEIGH_PHASE_SCALE = (3 / 16) / Math.PI;
const G = 0.8, SQR_G = G * G;                       // aerosols anisotropy
const EARTH_RADIUS = 6371.0;
const ATMOSPHERE_THICKNESS = 100.0;
const ATMOSPHERE_RADIUS = EARTH_RADIUS + ATMOSPHERE_THICKNESS;
const TRANSMITTANCE_STEPS = 64, IN_SCATTERING_STEPS = 64;
const TRANSMITTANCE_RES_X = 256, TRANSMITTANCE_RES_Y = 64;
// spectral data sampled at 630, 560, 490, 430 nm for an urban area
const SUN_SPECTRAL_IRRADIANCE = [1.679, 1.828, 1.986, 1.307];
const MOLECULAR_SCATTERING_COEFFICIENT_BASE = [6.605e-3, 1.067e-2, 1.842e-2, 3.156e-2];
const OZONE_ABSORPTION_CROSS_SECTION = [3.472e-25, 3.914e-25, 1.349e-25, 11.03e-27];
const OZONE_MEAN_DOBSON = 334.5;                   // average ozone of monthly mean values
const AEROSOL_ABSORPTION_CROSS_SECTION = [2.8722e-24, 4.6168e-24, 7.9706e-24, 1.3578e-23];
const AEROSOL_SCATTERING_CROSS_SECTION = [1.5908e-22, 1.7711e-22, 2.0942e-22, 2.4033e-22];
const AEROSOL_BASE_DENSITY = 1.3681e20;
const AEROSOL_BACKGROUND_DENSITY = 2e6;
const AEROSOL_HEIGHT_SCALE = 0.73;
const MS_FIT = [0.217, 0.347, 0.594, 1.0];   // the multiple-scattering fit's colour
// spectral to XYZ
const SPECTRAL_XYZ = [
  [53.386917738564668023, 22.981337506691024754, 0.0],
  [43.904844466369358263, 71.347795700053393866, 0.102506867965741307],
  [1.6137278251608962005, 18.422960591455485011, 31.742921188390805758],
  [20.762668673810577145, 2.3614213523314368527, 110.48009643252140334]];

const sqr = x => x * x;
const mix = (a, b, t) => a + (b - a) * t;
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const molecular_phase = c => RAYLEIGH_PHASE_SCALE * (1 + sqr(c));
function aerosol_phase(c){
  const den = 1 + SQR_G + 2 * G * c;
  return (1 / (4 * Math.PI)) * (1 - SQR_G) / (den * Math.sqrt(den));
}
const sun_direction = c => [-Math.sqrt(Math.max(1 - c * c, 0)), 0, c];

function ray_sphere_intersection(pos, dir, radius){
  const b = dot3(pos, dir), c = dot3(pos, pos) - radius * radius;
  if(c > 0 && b > 0) return -1;
  const d = b * b - c;
  if(d < 0) return -1;
  if(d >= b * b) return -b + Math.sqrt(d);
  return -b - Math.sqrt(d);
}

function spectral_to_xyz(L){
  const xyz = [0, 0, 0];
  for(let i = 0; i < 4; i++) for(let k = 0; k < 3; k++) xyz[k] += SPECTRAL_XYZ[i][k] * L[i];
  return xyz;
}

class SkyMultipleScattering {
  constructor(air, aerosol, ozone){
    this.air = air; this.aerosol = aerosol; this.ozone = ozone;
    this.lut = null;
    this.tg = new Float64Array(4); this.t0 = new Float64Array(4); this.th = new Float64Array(4);   // scratch
  }

  // absorption and scattering at an altitude (km): aerosol absorption, aerosol
  // scattering, molecular absorption (ozone), molecular scattering, each per wavelength
  coefficients(h, out){
    const aer = AEROSOL_BASE_DENSITY * (Math.exp(-h / AEROSOL_HEIGHT_SCALE) + AEROSOL_BACKGROUND_DENSITY / AEROSOL_BASE_DENSITY) * this.aerosol;
    const log_h = Math.log(Math.max(h, 1e-4));
    const oz = 3.78547397e20 * Math.exp(-sqr(log_h - 3.22261) * 5.55555555 - log_h) * OZONE_MEAN_DOBSON * this.ozone;
    const mol = Math.exp(-0.07771971 * Math.pow(h, 1.16364243)) * this.air;
    for(let i = 0; i < 4; i++){
      out.aa[i] = AEROSOL_ABSORPTION_CROSS_SECTION[i] * aer;
      out.as[i] = AEROSOL_SCATTERING_CROSS_SECTION[i] * aer;
      out.ma[i] = OZONE_ABSORPTION_CROSS_SECTION[i] * oz;
      out.ms[i] = MOLECULAR_SCATTERING_COEFFICIENT_BASE[i] * mol;
    }
  }

  // the atmosphere's transmittance from an altitude (0 ground, 1 top) to the sun
  transmittance(cos_theta, normalized_altitude){
    const sd = sun_direction(cos_theta);
    const o = [0, 0, mix(EARTH_RADIUS, ATMOSPHERE_RADIUS, normalized_altitude)];
    const t_d = ray_sphere_intersection(o, sd, ATMOSPHERE_RADIUS);
    const dt = t_d / TRANSMITTANCE_STEPS;
    const sum = [0, 0, 0, 0], c = { aa: [0, 0, 0, 0], as: [0, 0, 0, 0], ma: [0, 0, 0, 0], ms: [0, 0, 0, 0] };
    for(let s = 0; s < TRANSMITTANCE_STEPS; s++){
      const t = (s + 0.5) * dt;
      const h = Math.max(Math.hypot(o[0] + sd[0] * t, o[1] + sd[1] * t, o[2] + sd[2] * t) - EARTH_RADIUS, 0);
      this.coefficients(h, c);
      for(let i = 0; i < 4; i++) sum[i] += (c.aa[i] + c.as[i] + c.ma[i] + c.ms[i]) * dt;
    }
    return sum.map(v => Math.exp(-v));
  }

  precompute_lut(){
    this.lut = new Float64Array(TRANSMITTANCE_RES_X * TRANSMITTANCE_RES_Y * 4);
    for(let y = 0; y < TRANSMITTANCE_RES_Y; y++){
      for(let x = 0; x < TRANSMITTANCE_RES_X; x++){
        const T = this.transmittance((x / (TRANSMITTANCE_RES_X - 1)) * 2 - 1, y / (TRANSMITTANCE_RES_Y - 1));
        this.lut.set(T, (y * TRANSMITTANCE_RES_X + x) * 4);
      }
    }
  }

  // bilinear lookups in the LUT (written into out, which they return)
  lookup_transmittance(cos_theta, normalized_altitude, out){
    const u = Math.min(Math.max(cos_theta * 0.5 + 0.5, 0), 1), v = Math.min(Math.max(normalized_altitude, 0), 1);
    const x = (TRANSMITTANCE_RES_X - 1) * u, y = (TRANSMITTANCE_RES_Y - 1) * v;
    const x1 = Math.floor(x), y1 = Math.floor(y);
    const x2 = Math.min(x1 + 1, TRANSMITTANCE_RES_X - 1), y2 = Math.min(y1 + 1, TRANSMITTANCE_RES_Y - 1);
    const fx = x - x1, fy = y - y1, lut = this.lut;
    const a = (y1 * TRANSMITTANCE_RES_X + x1) * 4, b = (y1 * TRANSMITTANCE_RES_X + x2) * 4;
    const c = (y2 * TRANSMITTANCE_RES_X + x1) * 4, d = (y2 * TRANSMITTANCE_RES_X + x2) * 4;
    for(let i = 0; i < 4; i++){
      const bottom = lut[a + i] + (lut[b + i] - lut[a + i]) * fx, top = lut[c + i] + (lut[d + i] - lut[c + i]) * fx;
      out[i] = bottom + (top - bottom) * fy;
    }
    return out;
  }

  lookup_transmittance_at_ground(cos_theta, out){
    const u = Math.min(Math.max(cos_theta * 0.5 + 0.5, 0), 1);
    const x = (TRANSMITTANCE_RES_X - 1) * u, x1 = Math.floor(x), x2 = Math.min(x1 + 1, TRANSMITTANCE_RES_X - 1), fx = x - x1;
    const lut = this.lut, a = x1 * 4, b = x2 * 4;
    for(let i = 0; i < 4; i++) out[i] = lut[a + i] + (lut[b + i] - lut[a + i]) * fx;
    return out;
  }

  lookup_transmittance_to_sun(normalized_altitude, out){
    const v = Math.min(Math.max(normalized_altitude, 0), 1);
    const y = (TRANSMITTANCE_RES_Y - 1) * v, y1 = Math.floor(y), y2 = Math.min(y1 + 1, TRANSMITTANCE_RES_Y - 1), fy = y - y1;
    const x = TRANSMITTANCE_RES_X - 1, lut = this.lut;
    const a = (y1 * TRANSMITTANCE_RES_X + x) * 4, b = (y2 * TRANSMITTANCE_RES_X + x) * 4;
    for(let i = 0; i < 4; i++) out[i] = lut[a + i] + (lut[b + i] - lut[a + i]) * fy;
    return out;
  }

  // light scattered more than once: from the ground, and a fit of the rest of the sky
  lookup_multiscattering(cos_theta, normalized_height, d, out){
    const omega = 2 * Math.PI * (1 - Math.sqrt(Math.max(1 - sqr(EARTH_RADIUS / d), 0)));
    const Tg = this.lookup_transmittance_at_ground(cos_theta, this.tg);
    const T0 = this.lookup_transmittance_to_sun(0, this.t0), Th = this.lookup_transmittance_to_sun(normalized_height, this.th);
    const fit = 0.02 * (1 / (1 + 5 * Math.exp(-17.92 * cos_theta)));
    for(let i = 0; i < 4; i++){
      const L_ground = PHASE_ISOTROPIC * omega * (GROUND_ALBEDO / Math.PI) * Tg[i] * (T0[i] / Th[i]) * cos_theta;
      out[i] = fit * MS_FIT[i] + L_ground;
    }
    return out;
  }

  // in-scattered radiance along a ray
  inscattering(sun_dir, o, dir, t_d){
    const cos_theta = -dot3(dir, sun_dir);
    const mp = molecular_phase(cos_theta), ap = aerosol_phase(cos_theta);
    const dt = t_d / IN_SCATTERING_STEPS;
    const L = [0, 0, 0, 0], T = [1, 1, 1, 1];
    const c = { aa: [0, 0, 0, 0], as: [0, 0, 0, 0], ma: [0, 0, 0, 0], ms: [0, 0, 0, 0] };
    const Ts = [0, 0, 0, 0], ms = [0, 0, 0, 0];
    for(let s = 0; s < IN_SCATTERING_STEPS; s++){
      const t = (s + 0.5) * dt;
      const x0 = o[0] + dir[0] * t, x1 = o[1] + dir[1] * t, x2 = o[2] + dir[2] * t;
      const d = Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
      const altitude = Math.max(d - EARTH_RADIUS, 0), nh = altitude / ATMOSPHERE_THICKNESS;
      const sample_cos = (x0 * sun_dir[0] + x1 * sun_dir[1] + x2 * sun_dir[2]) / d;
      this.coefficients(altitude, c);
      this.lookup_transmittance(sample_cos, nh, Ts);
      this.lookup_multiscattering(sample_cos, nh, d, ms);
      for(let i = 0; i < 4; i++){
        const ext = c.aa[i] + c.as[i] + c.ma[i] + c.ms[i];
        const S = SUN_SPECTRAL_IRRADIANCE[i] * (c.ms[i] * (mp * Ts[i] + ms[i]) + c.as[i] * (ap * Ts[i] + ms[i]));
        const st = Math.exp(-dt * ext);
        // energy-conserving analytical integration (Hillaire, Frostbite)
        L[i] += T[i] * (S - S * st) / Math.max(ext, 1e-7);
        T[i] *= st;
      }
    }
    return L;
  }
}

const clampAltitude = m => Math.min(Math.max(m, 1), 99999) / 1000;     // metres to km

/**
 * The sky texture Cycles precomputes (512 x 256, CIE XYZ): elevation from straight
 * down to straight up (more rows toward the horizon), azimuth relative to the sun.
 */
export function precomputeTexture(sun_elevation, { altitude = 100, air = 1, aerosol = 1, ozone = 1, width = 512, height = 256 } = {}){
  const sms = new SkyMultipleScattering(air, aerosol, ozone);
  sms.precompute_lut();
  const alt = clampAltitude(altitude), half_width = width / 2;
  const sun_dir = sun_direction(Math.cos(Math.PI / 2 - sun_elevation));
  const o = [0, 0, EARTH_RADIUS + alt];
  const px = new Float32Array(width * height * 3);
  for(let y = 0; y < height; y++){
    for(let x = 0; x < half_width; x++){
      const az = 2 * Math.PI * (x + 0.5) / width;
      const l = (y + 0.5) / height * 2 - 1;
      const el = Math.sign(l) * l * l * Math.PI / 2;               // more texels toward the horizon
      const dir = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
      const atmos = ray_sphere_intersection(o, dir, ATMOSPHERE_RADIUS), ground = ray_sphere_intersection(o, dir, EARTH_RADIUS);
      const xyz = spectral_to_xyz(sms.inscattering(sun_dir, o, dir, ground < 0 ? atmos : ground));
      const a = (y * width + x) * 3, b = (y * width + (width - x - 1)) * 3;           // mirrored
      px[a] = px[b] = xyz[0]; px[a + 1] = px[b + 1] = xyz[1]; px[a + 2] = px[b + 2] = xyz[2];
    }
  }
  return { px, width, height };
}

/** The sun's disc: its radiance (XYZ) at the disc's bottom and top edges, through the air. */
export function precomputeSun(sun_elevation, angular_diameter, { altitude = 100, air = 1, aerosol = 1, ozone = 1 } = {}){
  const sms = new SkyMultipleScattering(air, aerosol, ozone);
  const alt = clampAltitude(altitude), half = angular_diameter / 2;
  const solid_angle = 2 * Math.PI * (1 - Math.cos(half));
  const at = el => {
    const T = sms.transmittance(Math.cos(Math.PI / 2 - el), alt / ATMOSPHERE_THICKNESS);
    return spectral_to_xyz(SUN_SPECTRAL_IRRADIANCE.map((s, i) => s * T[i] / solid_angle));
  };
  return { bottom: at(sun_elevation - half), top: at(sun_elevation + half), solid_angle };
}

// CIE XYZ to linear Rec.709 (Blender's scene linear), negatives clamped as Cycles does
export function xyzToRgb(x, y, z){
  return [Math.max(0, 3.2404542 * x - 1.5371385 * y - 0.4985314 * z),
          Math.max(0, -0.9692660 * x + 1.8760108 * y + 0.0415560 * z),
          Math.max(0, 0.0556434 * x - 0.2040259 * y + 1.0572252 * z)];
}

function texel(tex, x, y){
  // bilinear, clamped at the edges (Cycles' linear image lookup)
  const { px, width, height } = tex;
  const fx = Math.min(Math.max(x * width - 0.5, 0), width - 1), fy = Math.min(Math.max(y * height - 0.5, 0), height - 1);
  const x0 = Math.floor(fx), y0 = Math.floor(fy), tx = fx - x0, ty = fy - y0;
  const x1 = Math.min(x0 + 1, width - 1), y1 = Math.min(y0 + 1, height - 1);
  const out = [0, 0, 0];
  for(let c = 0; c < 3; c++){
    const v00 = px[(y0 * width + x0) * 3 + c], v10 = px[(y0 * width + x1) * 3 + c];
    const v01 = px[(y1 * width + x0) * 3 + c], v11 = px[(y1 * width + x1) * 3 + c];
    out[c] = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
  }
  return out;
}

/**
 * The sky's radiance in a direction (Blender's axes: x east, y north, z up),
 * linear Rec.709 RGB, as Cycles' Sky Texture gives it. sky: from makeSky.
 * disc: whether to show the sun's disc.
 */
export function radiance(sky, dir, disc = true){
  const { tex, sun, elevation, size, intensity, earthAngle } = sky;
  const kr = sky.kernelRotation;
  const theta = Math.acos(Math.min(Math.max(dir[2], -1), 1)), phi = Math.atan2(dir[1], dir[0]);
  const dir_el = Math.PI / 2 - theta;
  // spherical_to_direction(elevation - pi/2, rotation - pi/2)
  const st = elevation - Math.PI / 2, sp = kr - Math.PI / 2;
  const sd = [Math.sin(st) * Math.cos(sp), Math.sin(st) * Math.sin(sp), Math.cos(st)];
  const ang = Math.acos(Math.min(Math.max(dot3(dir, sd), -1), 1)), half = size * 0.5;
  let xyz = [0, 0, 0];
  if(disc && ang < half && dir_el > -earthAngle){
    const y = (dir_el - elevation) / size + 0.5;
    const limb = 1 - 0.6 * (1 - Math.sqrt(Math.max(0, 1 - sqr(ang / half))));
    xyz = sun.bottom.map((b, i) => mix(b, sun.top[i], y) * intensity * limb);
  }
  let x = (-phi - Math.PI / 2 + kr) / (2 * Math.PI);
  x -= Math.floor(x);
  const y = Math.sign(dir_el) * Math.sqrt(Math.abs(dir_el) * 2 / Math.PI) * 0.5 + 0.5;
  const s = texel(tex, x, y);
  return xyzToRgb(xyz[0] + s[0], xyz[1] + s[1], xyz[2] + s[2]);
}

/**
 * A sky: the sun's elevation and rotation (radians, Blender's Sun Rotation: the
 * sun's compass bearing, 0 north, clockwise seen from above), its angular size
 * and intensity, the altitude (m) and the air, aerosol and ozone densities.
 */
export function makeSky({ elevation, rotation = 0, size = 0.009512, intensity = 1, altitude = 100, air = 1, aerosol = 1, ozone = 1 }){
  // Cycles turns the Sun Rotation the user sets into 2 pi minus it (SkyTextureNode::simplify_settings)
  const kr = (2 * Math.PI - (((rotation % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI))) % (2 * Math.PI);
  const alt = clampAltitude(altitude);
  return { tex: precomputeTexture(elevation, { altitude, air, aerosol, ozone }),
           sun: precomputeSun(elevation, size, { altitude, air, aerosol, ozone }),
           elevation, rotation, kernelRotation: kr, size, intensity,
           earthAngle: Math.PI / 2 - Math.asin(EARTH_RADIUS / (EARTH_RADIUS + alt)) };
}

/**
 * The sun as a light: its colour (linear RGB, brightest channel 1) and its
 * irradiance on a surface facing it (the disc's radiance times its solid
 * angle), in the sky's own units.
 */
export function sunLight(sky){
  const mid = sky.sun.bottom.map((b, i) => (b + sky.sun.top[i]) / 2 * sky.intensity);
  const rgb = xyzToRgb(mid[0], mid[1], mid[2]).map(v => v * sky.sun.solid_angle * 0.8);  // limb darkening averages to 0.8
  const m = Math.max(...rgb, 1e-9);
  return { color: rgb.map(v => v / m), irradiance: m };
}

// ----------------------------------------------------------------- for the 3D tab
// The sky as all-round pictures (equirectangular, in three.js's layout, half
// floats): once with the sun's disc for the background (dimmed by discScale, so
// it reads white but its glow stays a halo) and once without, for the light the
// sky gives; the sky's colour just above the horizon all round; its light on a
// level surface; and the sun as a light. Elevation and azimuth in radians, the
// azimuth a compass bearing (0 north, east a quarter turn clockwise).

// float to half float, as three.js's DataUtils.toHalfFloat (clamped to the largest half)
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
function toHalf(v){
  f32[0] = Math.min(v, 65504);
  const x = u32[0], s = (x >> 16) & 0x8000, e = ((x >> 23) & 0xff) - 112, m = x & 0x7fffff;
  if(e <= 0) return s | ((m | 0x800000) >> (1 - e + 13)) * (e > -11);
  if(e >= 31) return s | 0x7c00;
  return s | (e << 10) | (m >> 13);
}

// three.js's direction for pixel (i, j) of a width x height equirect picture, in Blender's axes
function direction(i, j, width, height){
  const a = ((i + 0.5) / width - 0.5) * 2 * Math.PI, lat = ((j + 0.5) / height - 0.5) * Math.PI;
  const x = Math.cos(a) * Math.cos(lat), y = Math.sin(lat), z = Math.sin(a) * Math.cos(lat);
  return [x, -z, y];                                   // three.js (x east, y up, z south) to Blender (x east, y north, z up)
}

/**
 * Everything the 3D tab needs of a sky, for the sun at this elevation and bearing.
 */
export function buildSkyMaps({ elevation, azimuth, width = 2048, height = 1024, discScale = 1e-3 }){
  const sky = makeSky({ elevation, rotation: azimuth });
  const plain = new Uint16Array(width * height * 4);
  for(let j = 0; j < height; j++){
    for(let i = 0; i < width; i++){
      const rgb = radiance(sky, direction(i, j, width, height), false);
      const k = (j * width + i) * 4;
      plain[k] = toHalf(rgb[0]); plain[k + 1] = toHalf(rgb[1]); plain[k + 2] = toHalf(rgb[2]); plain[k + 3] = 0x3c00;
    }
  }
  // the background: the same, with the sun's disc (dimmed by discScale, so it reads
  // white but its glow stays a halo)
  const disc = plain.slice();
  const sd = [Math.sin(azimuth) * Math.cos(elevation), Math.cos(azimuth) * Math.cos(elevation), Math.sin(elevation)];
  const reach = Math.cos(sky.size * 2);
  for(let j = 0; j < height; j++){
    for(let i = 0; i < width; i++){
      const d = direction(i, j, width, height);
      if(d[0] * sd[0] + d[1] * sd[1] + d[2] * sd[2] < reach) continue;
      // the disc is smaller than a pixel or two: averaged over the pixel
      let r = 0, g = 0, b = 0;
      for(let s = 0; s < 16; s++){
        const p = radiance(sky, direction(i + (s % 4) / 4 - 0.375, j + Math.floor(s / 4) / 4 - 0.375, width, height), true);
        r += p[0]; g += p[1]; b += p[2];
      }
      const k = (j * width + i) * 4, base = radiance(sky, d, false);
      disc[k] = toHalf(base[0] + (r / 16 - base[0]) * discScale);
      disc[k + 1] = toHalf(base[1] + (g / 16 - base[1]) * discScale);
      disc[k + 2] = toHalf(base[2] + (b / 16 - base[2]) * discScale);
    }
  }
  // the sky's colour just above the horizon all round (the middle of 16 bearings), its
  // light on a level surface (radiance times the cosine, over the upper half), and the sun
  const ring = [];
  for(let k = 0; k < 16; k++){
    const a = k / 16 * 2 * Math.PI, el = 1.5 * Math.PI / 180;
    ring.push(radiance(sky, [Math.sin(a) * Math.cos(el), Math.cos(a) * Math.cos(el), Math.sin(el)], false));
  }
  const mid = c => { const v = ring.map(x => x[c]).sort((a, b) => a - b); return (v[7] + v[8]) / 2; };
  let E = [0, 0, 0];
  const n = 48;
  for(let a = 0; a < n * 2; a++){
    for(let b = 0; b < n; b++){
      const el = (b + 0.5) / n * Math.PI / 2, az = (a + 0.5) / (n * 2) * 2 * Math.PI;
      const p = radiance(sky, [Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el)], false);
      const w = Math.sin(el) * Math.cos(el) * (Math.PI / 2 / n) * (2 * Math.PI / (n * 2));   // cos x solid angle
      E = E.map((v, c) => v + p[c] * w);
    }
  }
  return { width, height, plain, disc, horizon: [mid(0), mid(1), mid(2)], skyIrradiance: E, sun: sunLight(sky) };
}
