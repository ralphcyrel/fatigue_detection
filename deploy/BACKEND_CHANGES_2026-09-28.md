# Backend changes required by the 2026-09-28 Pi build

**Ship all of these with the `GET /drivers/{id}/calibration` endpoint, before
the Pi build is deployed — not after.** Each item lists what goes wrong if
the Pi goes out first.

> **Deployment order (items 6 and 7, added 2026-09-30):**
>
> 1. Backend: store and return `ear_closed_baseline` (item 6) and accept the
>    `no_ear_baseline` monitoring fault (item 7). Deploy.
> 2. Check it: `GET /drivers/{id}/calibration` for a test record returns
>    `baselines.ear_closed`.
> 3. **Only then** deploy the Pi build with the closed-eye enrollment, and
>    re-enrol drivers.
>
> Laravel drops fields it does not validate **without an error**. If the Pi goes
> first, every enrollment POST succeeds, `ear_closed_baseline` is thrown away,
> and every new record reads back as a pre-2026-09-30 record: the closed-eye
> contrast check (the new safety check on every calibration) silently never
> applies. The Pi reads the record back after enrolling and exits with code 7
> ("backend DROPPED ear_closed_baseline") when this happens, but the record has
> already been saved by then.

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
record, and only when its EAR baseline is credible; pre-drive never does. If
repairs show up often, the enrollment path has a problem. (Credible was a fixed
0.18–0.35 until 2026-09-30; it is now the closed-eye contrast check, see item 6.)

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

## 5. Missing route: `GET /ping` (found 2026-09-29)

The Pi and the touchscreen launcher probe reachability with `GET /api/ping`
(README, API table). The backend does not define it:

```
GET http://192.168.1.200/api/ping  ->  404 {"message": "The route api/ping could not be found.", ...}
```

Add it outside the auth middleware (it is a liveness check, not data):

```php
// routes/api.php
Route::get('/ping', fn () => response()->json(['status' => 'ok']));
```

*Until it ships:* since 2026-09-29 the Pi counts any answer below HTTP 500 as
reachable and warns once that `/ping` returned 404. Before that change every
start-up logged "Backend NOT reachable" and the launcher's pill showed
OFFLINE for a backend that was serving every real endpoint.

The same 404 body carried `exception` and `file` (a server path): the backend
appears to run with `APP_DEBUG=true`, which exposes stack traces to any
client. Turn it off outside development.

## 6. Calibration field: `ear_closed_baseline` (added 2026-09-30) — BEFORE the Pi

Enrollment now also measures the driver's EAR with the eyes deliberately shut,
and the Pi judges a calibration by the contrast `ear_closed / ear` (valid at
or below 0.60), not by a fixed EAR range. The fixed 0.18–0.35 range locked out
driver 8, whose EAR (0.353–0.368) is valid.

```
POST /drivers/{driver_id}/enroll
  ... existing top-level fields ...
  "ear_closed_baseline": 0.1812        <- new, float, required from this build

GET /drivers/{driver_id}/calibration?device_id=pi-01
  "baselines": {"ear", "ear_closed", "perclos", "blink_duration",
                "blink_frequency", "mar"}    <- "ear_closed" new
```

Existing records have no value: keep the column nullable, return `null` (or
omit it) for them, and don't backfill. The Pi grandfathers records without it
if their EAR baseline is within 0.18–0.50. That covers drivers 5, 6 and 8.

**Order: this ships and is verified before the Pi build that sends it.** See
the deployment-order box at the top: a `FormRequest` that does not list
`ear_closed_baseline` drops it silently, and the contrast check then never
runs for any newly enrolled driver.

*If the Pi ships first:* enrollment exits with code 7 after saving; the
driver's record is usable (grandfathered) but unchecked; re-enrol every
driver enrolled in between once the backend is fixed.

## 7. Monitoring fault type `no_ear_baseline` (added 2026-09-30)

`POST /monitoring-faults` gains a third `fault_type`, same body shape:

| field | value |
|---|---|
| `fault_type` | `no_ear_baseline` |
| `severity` | `warning` |
| `gap_s` | seconds since the fault opened (not a no-face gap) |
| `last_known` | `null` (nothing was ever scored) |
| `resolution` | `driver_identified` (new) or `ignition_off` |

Opened when monitoring has no calibration for the driver and the EAR
self-seed discards 6 attempts in a row: the face is visible but the eyes
cannot be scored. The unit shows FAULT and only the head-pose override
runs. Until 2026-09-30 the seed restarted forever, with no report.

*If the Pi ships first:* the POST is rejected (`fault_type` / `resolution`
fail validation), the Pi logs the rejection, and the operator never sees
that a driver is being monitored with no EAR baseline.
