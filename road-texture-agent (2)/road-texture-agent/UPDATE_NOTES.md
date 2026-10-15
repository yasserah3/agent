# Road Texture Agent — focused update

Version: 2026.10.12-texture-updates1
Base version: 2026.10.11-estimate1

## Install

1. Close the running Road Texture Agent server.
2. Extract this ZIP directly into the existing `road-texture-agent` folder, where `server.py`, `app`, `ui`, and `workspace` already exist. Allow replacement of the included program files.
3. Start `python server.py` as usual and refresh the browser.

The ZIP contains only the changed program files and this note. It contains no workspace, training data, uploads, materials library replacement, or generated results.

Generate the texture once after installing so it has the saved geometry, source layers, settings snapshot, and update history. Subsequent edits use Update 2D texture.

## Highway groups

In Road markings, use the Highway centre-line groups controls:

- Group 1: double white lines.
- Group 2: single yellow centre lines.
- Press Pick to highlight detected highways. Click a highway to assign it to that group; clicking it again removes the assignment. Assigning it to the other group moves it there.
- Press Done or Escape to stop picking. Apply with Update 2D texture.
- Picks are saved per uploaded mask. Turning off Use selected highway groups restores automatic line styles without deleting the picks.

On divided highways, the selected style follows the two sides of the existing median; the median itself is preserved. Other lane and outer edge lines keep their existing roles. Highway identification uses the existing Highway width setting.

## Pedestrian crossings

Each Pedestrian crossing preset has a Scale and rotate to each road and junction checkbox. When enabled, its dimensions scale relative to a 6 m reference road, and placement follows the local kerb/approach frame. When disabled, its existing fixed measurements and placement behavior apply. Update 2D texture or Update decals applies the change to that layer's crossings.

## Tracked updates

- Generate texture saves the current texture, marking, decal, material, island, and model settings.
- Update 2D texture compares current controls with the saved values, then updates the affected layers and saves the new snapshot and an update entry containing the old/new values.
- No changed values means no texture update and no additional history entry.
- Saved values and Update history links download the logs. The files are `workspace/artifacts/gen_<id>_settings.json` and `gen_<id>_updates.jsonl`.
- Update view first commits pending 2D edits, then uses the history and the existing model-part cache to replace changed road surfaces, paving, markings, or decals while keeping the camera and unrelated parts.
- Changes to actual geometry settings, such as mask scale or output resolution, rebuild the affected geometry. They are recorded explicitly rather than treated as a material-only edit.
- Large-map patch signatures include highway group assignments along their whole highway chains, so unaffected patches can be reused.

## Verification

Processing/export checks cover material and wear preservation during marking updates, unchanged-value no-ops, highway group matching and persistence, crossing fit, surface geometry reuse, partial 3D marking/surface/paving updates, painted markings, highway continuity past side roads, highway setting changes, resolution changes, and settings/history contents. Python and JavaScript syntax checks pass.

The automated browser connection was unavailable in this session; map picking and other browser interactions still need a live check after installation.
