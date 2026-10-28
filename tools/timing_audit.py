# Timing audit: action-to-response lag in zone temperatures.
# For sharp damper/reheat steps, measure mean temp change at lags 1..6.
import json
import sys
import numpy as np

SC, DUMP = sys.argv[1], sys.argv[2]
meta = json.load(open(f'smart_buildings/results/{SC}/scenario_meta.json'))
d = np.load(f'smart_buildings/results/{SC}/{DUMP}/obs_dump.npz')
obs, acts = d['obs'], d['action_continuous']
fields = meta['observation_field_names']

# zone temp stats for denormalization
zt = meta['zone_temp_indices']


def denorm_temp(col):
    name = fields[col]
    st = meta['observation_normalization_metadata'].get(name, {})
    return obs[:, col] * float(st.get('sample_std', 1.0)) + float(st.get('sample_mean', 0.0))


za = meta['zone_action_indices']
resp_d = {k: [] for k in range(1, 7)}
resp_r = {k: [] for k in range(1, 7)}
n_d = n_r = 0
for vav, amap in za.items():
    # find this zone's temp column: vav id maps via ahu assignments? use zone index order
    pass
# Simpler: pair zone_temp_indices order with zone_action order (both zone-ordered)
zts = [denorm_temp(c) for c in zt]
vavs = list(za.keys())
for zi, vav in enumerate(vavs):
    if zi >= len(zts):
        break
    t = zts[zi]
    amap = za[vav]
    di = amap.get('supply_air_damper_percentage_command')
    ri = amap.get('reheat_valve_percentage_command')
    def isolated(series, thr=0.3, gap=8):
        idx = np.where(np.abs(series) > thr)[0]
        keep = []
        for s in idx:
            if s + 6 >= len(t) or s < gap:
                continue
            if np.any(np.abs(series[s - gap:s]) > thr):
                continue
            if np.any(np.abs(series[s + 1:s + 7]) > thr):
                continue
            keep.append(s)
        return keep

    if di is not None:
        dd = np.diff(acts[:, di], prepend=acts[0, di])
        for s in isolated(dd):
            n_d += 1
            for k in range(1, 7):
                resp_d[k].append(t[s + k] - t[s])
    if ri is not None:
        dr = np.diff(acts[:, ri], prepend=acts[0, ri])
        for s in isolated(dr):
            n_r += 1
            for k in range(1, 7):
                resp_r[k].append(t[s + k] - t[s])
print(f'{SC} damper steps={n_d} reheat steps={n_r}')
for k in range(1, 7):
    a = np.array(resp_d[k])
    print(f'damper lag{k}: mean={a.mean():+.5f}K std={a.std():.5f}')
for k in range(1, 7):
    a = np.array(resp_r[k])
    print(f'reheat lag{k}: mean={a.mean():+.5f}K std={a.std():.5f}')
