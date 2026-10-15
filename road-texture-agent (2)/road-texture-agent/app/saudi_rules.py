"""Verified SHC 602 Chapter 4 rules; geometry choices remain editable assumptions.

References are printed page numbers in the supplied 575-page English manual.
Exact 80/100 km/h boundaries overlap or are omitted in some tables. We use
the higher-speed category at a boundary and disclose that implementation choice.
"""
REFERENCES = {
    'centre': 'SHC 602 4.1.3, 4.2.1, Table 4-1 (pp. 205-206)',
    'lane': 'SHC 602 4.2.2, Table 4-2 (pp. 206-207)',
    'double': 'SHC 602 4.2.3, Table 4-3 (pp. 207-208)',
    'edge': 'SHC 602 4.2.5, Table 4-6 (pp. 213-214)',
    'zebra': 'SHC 602 4.4.1 (p. 227); 4.5.3 (p. 257)',
    'stop': 'SHC 602 4.4.3 (p. 235)',
    'giveway': 'SHC 602 4.4.4 (pp. 235-236)',
    'stopword': 'SHC 602 4.4.2 (p. 228)',
    'arrows': 'SHC 602 4.4.2 (p. 228)',
    'box': 'SHC 602 4.4.7 (p. 251)',
}


def line_values(speed, posted_speed=None):
    speed = float(speed)
    posted_speed = speed if posted_speed is None else float(posted_speed)
    dash, gap = (6.0, 12.0) if speed >= 100 else (3.0, 6.0) if speed >= 80 else (2.0, 4.0)
    return {'centre.dash_m': dash, 'centre.gap_m': gap,
            'centre.width_m': .15, 'lane.dash_m': dash, 'lane.gap_m': gap,
            'lane.width_m': .20 if speed >= 100 else .15,
            'double.width_m': .15 if speed >= 80 else .10,
            'edge.width_m': .20 if posted_speed > 100 else .15}


def preset_values(preset, values, speed):
    """Standard dimensions over user values; retain unspecified artistic parameters."""
    v = dict(values)
    if preset == 'zebra':
        v.update(stripe_m=.5, gap_m=.5, length_m=max(2.0, v['length_m']), rb_back_m=max(6.0, v['rb_back_m']))
    elif preset == 'stop':
        v.update(width_m=.3, dist_m=1.0)
    elif preset == 'giveway':
        v.update(seg_m=.5, gap_m=.2)
    elif preset == 'stopword' and float(speed) > 50:
        v['height_m'] = max(2.5, v['height_m'])
    return v
