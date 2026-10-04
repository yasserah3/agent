// Works out Blender's sky (ui/sky_blender.js) away from the page, so the 3D
// view does not stall while it is computed.
import { buildSkyMaps } from './sky_blender.js';

self.onmessage = e => {
  const { id, ...args } = e.data;
  const maps = buildSkyMaps(args);
  self.postMessage({ id, ...maps }, [maps.plain.buffer, maps.disc.buffer]);
};
