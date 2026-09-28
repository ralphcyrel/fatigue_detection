# Backend changes required by the 2026-09-28 Pi build

**Ship all of these with the `GET /drivers/{id}/calibration` endpoint, before
the Pi build is deployed — not after.** Each item lists what goes wrong if
the Pi goes out first.

## 1. New endpoint: calibration keyed by driver and device

```
GET /drivers/{driver_id}/calibration?device_id=pi-01

200 {"status": "ok", "data": {
       "driver_id": 6, "calibration_id": 12,
       "captured_on_device_id": "pi-01", "captured_at": "...",
       "baselines":  {"ear", "perclos", "blink_duration", "blink_frequency", "mar"},
       "thresholds": {"ear_threshold", "yawn_threshold"}}}
    -> this driver's active calibration captured on device_id if one exists,
       else their most recent active calibration from any device.
       Never another driver's.

404 {"lock_reason": "no_baseline", "status": "..."}   -> the driver has none
```

The Pi classifies the record itself from `captured_on_device_id`, and rejects
any record whose `driver_id` is not the driver it asked for.

*If the Pi ships first:* it falls back to `GET /devices/{id}/calibration`,
accepting it only for the same driver. That endpoint does not say where the
calibration was captured, so the Pi flags it `foreign_device`: **every
pre-drive locks with `foreign_device_baseline`** until the new endpoint is
live, unless the legacy endpoint already returns `captured_on_device_id`.

## 2. New lock reason: `foreign_device_baseline`

Must be an accepted value of:

- `reason` on `POST /override-requests`
- `lock_reason` on `POST /assessments`

It means the driver is enrolled, but only on another unit. Pre-drive locks
without running an assessment. The operator action differs from
`no_baseline`: re-enrol on this unit, versus enrol at all.

*If the Pi ships first:* an override request carrying it is rejected. The
starter stays locked (fail-secure), but the request never reaches the portal,
so **the operator never sees the driver waiting**.

## 3. Calibration provenance fields

Accept and store these on `POST /fatigue-events`, `POST /override-requests`
and `POST /assessments`:

| field | type | values |
|---|---|---|
| `calibration_source` | string | `device`, `foreign_device`, `self_seeded` |
| `calibration_id` | int / null | the calibration record used (or refused) |
| `calibration_device_id` | string / null | the unit it was captured on |
| `calibration_repairs` | string[] / null | monitoring-only repairs, see below |

`calibration_repairs` codes: `ear_threshold_recomputed`,
`ear_pair_self_seeded`, `perclos_baseline_floored`, `<field>_defaulted`
(e.g. `blink_duration_baseline_defaulted`). Only monitoring repairs a broken
record, and only when its EAR baseline is plausible (0.18–0.35); pre-drive
never does. If repairs show up often, the enrollment path has a problem.

Reading an override request: `reason: no_baseline` *with* a `calibration_id`
means a record existed but was refused as broken. The provenance is there as
context; approving does not make that record usable.

*If the Pi ships first:* a Laravel `FormRequest` normally drops fields it
does not validate, so the provenance would be **silently lost**, not rejected.
Portal users would have no way to see which DANGER events were scored on a
foreign, repaired or self-seeded baseline.

## 4. Assessment samples: new per-frame key

Each entry of `samples` on `POST /assessments` gains `ear_norm_frs` (the
0.5 s rolling median the FRS used). Needed only if the backend validates or
stores sample keys individually.
