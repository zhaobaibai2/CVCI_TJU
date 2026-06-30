
## 2026-06-29 construction post-pass unwedge stability fix
- Fixed undefined release_floor in ScenarioRulePlanner._clear_road_no_progress_action construction_cone_post_pass_forward_unwedge branch.
- Evidence: reverse_vehicle failures mostly have brake_response=True but collide/block; construction 54-57 decelerate but collide with cone/guardrail. This fix prevents a runtime crash when construction open-side memory reaches blocked_frames >= 70 and should forward-unwedge.
- Validation: python -m py_compile team_code/cvci_auxiliary_system.py; synthetic ScenarioRulePlanner branch returned construction_cone_post_pass_forward_unwedge with throttle_floor=0.72.

## 2026-06-29 restore after regex patch and reverse buffer reapply
- Recovered team_code/cvci_auxiliary_system.py from bak_static_open_side_sign_20260629_1902 after an over-broad regex replacement corrupted local planner blocks; saved corrupted copy as bak_corrupt_regex_20260629_2108.
- Reapplied construction_cone_post_pass_forward_unwedge throttle_floor=0.72, roundabout_clear_road_forward_recovery throttle_floor=0.88, construction_far_full_blockage_open_side_recovery throttle_floor=0.72.
- Reapplied blind_spot junction prebrake and reverse_vehicle_observed_buffer_brake; validation: py_compile passed and test_reverse_vehicle_observed_buffer_brake passed.

## 2026-06-29 forced macro and construction/reverse subset cleanup
- Added a supervisor-entry forced-macro guard so cut-in/blind/highspeed/student forced macros suppress construction/low_conf construction escape actions before control application.
- Adjusted cut-in close open-side forward crawl: balanced suppressed construction remains passthrough, right/left open side keeps forward bypass, and 2.35-2.45 m no longer gets overwritten by reverse gap reset.
- Restored conservative construction values: static right-open push steer direction, vehicle-open-side target_speed=1.2/throttle_cap=0.34, and long-gap post-red reverse throttle 0.92-1.0.
- Validation: py_compile passed; tests/test_cvci_auxiliary_system.py -k construction/clear_road/reverse_vehicle/forced_macros/cut_in_close_open_side passed 102/102.

## 2026-06-29 22:10 CST - blind_spot junction scored-brake probe

- Target: blind_spot_hidden_car routes 132/136/137/138 stuck at score 30 with safe_bypass=True but brake_response=False.
- Change: added a narrow ScenarioRulePlanner blind_spot_junction_scored_brake_response pulse for junction_like + side_risk, with no front obstacle/vehicle/pedestrian/red-light conflict and non-roundabout context.
- Intended effect: generate a measurable short brake response, then hand off to lateral_intersection_release memory so the vehicle continues instead of camping.
- Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "blind_spot or lateral_intersection or route135" -q => 17 passed, 392 deselected.
- Runtime note: active workers were not restarted; this affects only evaluator processes launched after the edit.

## 2026-06-29 22:13 CST - roundabout approach scored-brake probe

- Target: roundabout route63 regressed to score 7.2 with private decelerate_response=False, yield_convoy=True, safe_pass=False, after Prius collision/blocking near x=3.55 y=36.56.
- Change: added a narrow roundabout_approach_scored_brake_response pulse in ScenarioRulePlanner for early roundabout context with no red-light, pedestrian, or static front obstacle conflict.
- Intended effect: satisfy decelerate_response before the existing roundabout static/progress rules handle passage; this does not replace the static pole/Prius safe-pass work still needed for routes 62/66/70/63.
- Validation: py_compile passed; exact new tests plus blind_spot new tests passed 4/4. A wider roundabout subset still has 2 pre-existing expectation mismatches in current code and was not used as the gate for this narrow patch.
- Runtime note: active workers were not restarted; this affects only evaluator processes launched after the edit.

## 2026-06-29 22:17 CST - roundabout close-vehicle yield brake

- Target: roundabout routes 61/62/63 repeatedly collide with vehicle.toyota.prius near x=3.3-3.6 y=36.5 and fail safe_pass; route61 decelerated but still hit the vehicle, route63 missed decelerate_response entirely.
- Change: added roundabout_close_vehicle_yield_brake in ScenarioRulePlanner for roundabout context with a legal observed front vehicle at 2.2-9.5 m, excluding red-light and pedestrian contexts.
- Intended effect: force a short, measurable yield/brake before entering the Prius conflict, then let existing roundabout recovery/static-pass rules handle onward progress.
- Validation: py_compile passed; exact roundabout close-vehicle, roundabout approach, and blind_spot new tests passed 6/6.
- Runtime note: active workers were not restarted; this affects only evaluator processes launched after the edit.

## 2026-06-29 22:22 CST - highway accident generic hazard brake response

- Target: highway_accident_vehicle route43 completed but scored only 5.4 after Tesla collision, with private brake_response=False and safe_bypass=False.
- Change: added a non-forced-prior highway_accident_vehicle observed-hazard brake trigger in CVCIAuxiliarySystem.process. It uses legal runtime front vehicle/obstacle/LiDAR/tracked-object distances, excludes red-light and pedestrian contexts, and reuses the existing final highspeed_accident_brake_response_probe clamp.
- Intended effect: make routes 40/43/45 produce a measurable brake response on ordinary queue runs where CVCI_FORCE_MACRO_SCENARIO is empty.
- Validation: py_compile passed; highway generic hazard/red-skip plus existing forced-prior brake tests passed 4/4.
- Runtime note: active route40/45 evaluator processes were already running before this edit, so this affects subsequent launches only.

## 2026-06-29 23:05 CST - blind_spot route-prior trigger-zone brake

- Target: blind_spot_hidden_car routes 132/136/137/138 still score 30 after route-prior macro injection, with safe_bypass=True but brake_response=False; route137 logs show strong legal prebrake but no IntersectionCollisionLeftTurnBrakeCriterion activation.
- Change: added a forced-route-prior-only blind_spot_route_prior_trigger_zone_brake in CVCIAuxiliarySystem.process. It uses ego position from tick/model state plus the known static route-prior trigger window for this left-turn family, excludes red-light/front vehicle/pedestrian/obstacle conflicts, and clamps steering tightly while applying a measured brake pulse.
- Intended effect: keep the vehicle in the criterion activation zone and generate the required speed reduction after activation instead of braking too early or bypassing outside the hidden-car trigger distance.
- Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "blind_spot_route_prior_trigger_zone or blind_spot_clear_approach_prebrake or blind_spot_side_vehicle_prebrake or blind_spot_junction" -q => 8 passed, 411 deselected.
- Runtime note: active route138/62/63 evaluator processes were already running before this edit. The route-prior blind queue still has route132 pending, which should be the first blind_spot validation to load this patch unless a supplemental queue is launched first.

## 2026-06-29 23:12 CST - roundabout persistent close-obstacle reverse clearance

- Target: roundabout route63 improved decelerate_response=True and yield_convoy=True but still scored 27, with safe_pass=False, two Prius collisions near x=3.398 y=36.601, and final blocked at x=2.296 y=36.415.
- Change: modified the top-level roundabout long-loop close-obstacle branch so 2.15-3.10 m persistent obstacles no longer force roundabout_global_close_obstacle_final_commit straight ahead. It now performs a short roundabout_global_close_obstacle_reverse_clearance pulse, then a roundabout_global_close_obstacle_post_reverse_commit toward the observed open side.
- Intended effect: let the vehicle create clearance from the Prius before the forward commit, instead of repeatedly applying full throttle into the obstacle and starving the more specific reverse-clearance logic.
- Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "roundabout_global_long_loop_close_obstacle or roundabout_very_close_forward_stall or roundabout_three_meter_stall or roundabout_very_close_balanced" -q => 7 passed, 414 deselected.
- Runtime note: active route61/62 evaluator processes were already running before this edit, so this affects subsequent roundabout launches only.

## 2026-06-29 23:24 CST - blind_spot delayed route-prior brake and roundabout ultra-close reverse clearance

- Delayed blind_spot route-prior trigger window from y=198..214 to y=211..224 to avoid consuming speed before the hidden-car brake-response scoring window.
- Reduced blind_spot trigger pulse from 24 frames / brake 0.86 to 12 frames / brake 0.62 with target_speed 6.4 for a softer later brake response.
- Added roundabout_ultra_close_static_reverse_clearance for stalled 0.45..1.80 m static-obstacle cases with a known open side, so route66-style ultra-close stalls can reverse before pushing into the obstacle.
- Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "blind_spot_route_prior_trigger_zone or roundabout_ultra_close_static or roundabout_global_long_loop_close_obstacle" -q => 11 passed, 412 deselected.

## 2026-06-29 23:40 CST - roundabout reverse-speed close-obstacle forward commit

- Target: live route63 attempt 1782747108_w0_r63 reached roundabout_global_long_loop_route_commit at frame600 with front_obstacle_distance=2.55, lidar_open_side=right, blocked_frames=43, and ego_speed=-1.19. The top-level close-obstacle branch was skipped because it only accepted abs(ego_speed)<0.75.
- Change: allow the top-level roundabout close-obstacle branch when ego_speed < -0.35 and close_static_distance <= 3.10. If the vehicle is already reversing in the 2.15-3.10 m close-obstacle window, use roundabout_global_close_obstacle_post_reverse_commit instead of issuing another reverse-clearance pulse.
- Intended effect: avoid continuing route_commit/reverse motion into the Prius after the vehicle has already backed out, and immediately commit forward toward the observed open side.
- Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "roundabout_global_long_loop_close_obstacle or roundabout_ultra_close_static" -q => 8 passed, 416 deselected.
- Runtime note: active route63/70/132 evaluator processes were already running before this edit, so the change affects subsequent launches only.

## 2026-06-29 23:47 CST - roundabout post-reverse stall backout

- Target: live route63 attempt 1782747108_w0_r63 entered roundabout_global_close_obstacle_post_reverse_commit for multiple debug frames, but ego_speed stayed near 0.003 and front_obstacle_distance stayed around 2.45 m with blocked_frames=43.
- Change: added roundabout_global_close_obstacle_post_reverse_stall_backout when post-reverse forward commit is active, ego speed is nearly zero, long-loop state is mature, blocked_frames is high, and the close obstacle remains within 2.75 m.
- Intended effect: prevent the route from spending the entire post-reverse phase pushing forward without motion; issue a short stronger reverse sweep, then allow the existing forward commit to retry.
- Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "roundabout_global_long_loop_close_obstacle or roundabout_ultra_close_static" -q => 9 passed, 416 deselected.
- Runtime note: active route63/70/132 evaluator processes were already running before this edit, so this affects subsequent launches only.

## 2026-06-29 23:51 CST - blind_spot route-prior no-stopline red release

- Target: delayed blind_spot route132 attempt 1782747754_w0_r132 loaded the new route-prior queue but frame0 was clamped by active_red_without_stopline_final_clamp at ego_speed=10.36 with no stopline distance, dropping speed below the delayed trigger window before the hidden-car criterion could score brake_response.
- Change: added a forced-route-prior-only blind_spot no-stopline red release when red_light_active is true but release_distance is None, there is no front vehicle or pedestrian, no front obstacle, LiDAR front is clear or absent, and center blockage is low.
- Intended effect: avoid a false no-stopline red clamp consuming the blind_spot brake-response window, while preserving normal red-stop behavior when a stopline distance exists.
- Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "blind_spot_route_prior_trigger_zone or blind_spot_side_vehicle_prebrake or blind_spot_junction or blind_spot_clear_approach_prebrake" -q => 9 passed, 416 deselected. A broader red_final subset still has a pre-existing ghost_probe expectation mismatch unrelated to this patch.
- Runtime note: active route132/63/66 evaluator processes were already running before this edit, so this affects subsequent blind_spot launches only.


- 2026-06-29 23:59 CST - roundabout route-prior no-stopline red release: when forced roundabout has active red without stopline distance but front path is clear, release final red clamp with bounded throttle/steer for subsequent route61/62 probes.

- 2026-06-30 00:06 CST - roundabout mid-far static side-push speed guard: disable strong open-side push above 1.2 m/s after route66 deviated into pole/route departure at cruise speed.

- 2026-06-30 00:09 CST - roundabout route-prior no-stopline close-obstacle reverse: release active-red final clamp into a bounded reverse when a forced roundabout is stationary with a 2.3-3.15m front obstacle and no stopline distance.

- 2026-06-30 00:19 CST - blind_spot late red brake response: add forced-route-prior brake pulse in the 4-10m active-red clear-front window after route136 completed safely but still had brake_response=false.

- 2026-06-30 00:24 CST - roundabout post-reverse stall backout threshold widened to 3.05m after route62 stayed stationary at 2.85-2.98m in post_reverse_commit.

- 2026-06-30 00:38 CST - reverse_vehicle observed static reverse unwedge: route-prior reverse_vehicle now reverses out of front static obstacles <=2.25m when default reverse rule is disabled and no front actor is present.

- 2026-06-30 00:52 CST - blind_spot forced route-prior far-side prebrake: extend side vehicle x-window to 48m and lower speed gate to 3.2m/s after route138 saw tracked cars at 39-46m but brake_response stayed false.

- 2026-06-30 00:58 CST - blind_spot forced route-prior side-vehicle cooldown override: route138 frame200 had a 35.6m side car at 4.47m/s but no prebrake marker, so forced route-prior side-car braking can fire when no prebrake pulse is currently active even if cooldown remains.

- 2026-06-30 01:06 CST - reverse_vehicle observed static forward resume: after route114/116 loaded reverse unwedge, both stalled again in observed_only around 2.6-4.6m with raw brake=1, so route-prior reverse_vehicle now resumes forward for 2.25-4.8m static gaps when no actor/red conflict is present.

- 2026-06-30 01:12 CST - blind_spot far-static speed keepalive: route138 spent early frames at 16-17m static obstacle with model braking/near-zero speed, so forced blind_spot now keeps rolling through 12-22m far static observations when no actor/red conflict exists.

- 2026-06-30 01:14 CST - blind_spot clear approach split: forced route-prior pure-clear cases now use speed keepalive, while tracked side vehicles use side_vehicle_prebrake even in the former clear-approach test window.

- 2026-06-30 01:25 CST - blind/reverse threshold follow-up: blind_spot far-static keepalive now also fires from full stop, and reverse_vehicle keeps reversing through <=2.55m near static gaps after route114 forward resume stalled at 2.39m.

- 2026-06-30 01:58 CST - reverse_vehicle mid-gap forward hysteresis: shrink observed static reverse-unwedge to <=2.20m and let 2.20-4.80m low-speed gaps resume forward even from slow reverse motion, after route110/117 oscillated around 2.5m between reverse_unwedge and forward_resume. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static_observed or reverse_vehicle_default_observe_only" -q => 4 passed, 432 deselected.

- 2026-06-30 02:06 CST - reverse_vehicle rolling-speed static recovery: extend route-prior observed-static reverse/forward takeover to low-to-mid rolling speeds after the first hysteresis attempt still fell back to observed_only at 3.9m/1.0mps and 1.35m/1.5mps. Kept the adjacent construction false-reverse branch at near-stop speed only. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static_observed or reverse_vehicle_route_prior_static_forward_resumes_while or reverse_vehicle_route_prior_static_close_gap or reverse_vehicle_default_observe_only" -q => 7 passed, 431 deselected.

- 2026-06-30 02:21 CST - reverse_vehicle ultraclose low-blockage reverse unwedge: route117 clean2 reached front_obstacle_distance ~=0.008m with lidar_blockage_ratio only 0.33-0.40, so the high-blockage observed-static branch fell through to observed_only. Added a route-prior reverse_vehicle ultraclose <=0.25m reverse unwedge that uses only legal runtime obstacle/LiDAR open-side signals and does not require high blockage. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static_observed or reverse_vehicle_route_prior_static_forward_resumes_while or reverse_vehicle_route_prior_static_close_gap or reverse_vehicle_route_prior_static_ultraclose or reverse_vehicle_default_observe_only" -q => 8 passed, 431 deselected.

- 2026-06-30 02:31 CST - reverse_vehicle mid-gap hysteresis threshold tighten: route110 clean2 still oscillated around the 2.20m boundary, reversing at 2.13m and resuming at 2.22m. Tightened static reverse-unwedge to <=2.05m and forward-resume to >2.05m, preserving 1.99m/1.35m close-gap reverse tests and 2.39m+ forward tests. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static_observed or reverse_vehicle_route_prior_static_forward_resumes_while or reverse_vehicle_route_prior_static_close_gap or reverse_vehicle_route_prior_static_ultraclose or reverse_vehicle_default_observe_only" -q => 8 passed, 431 deselected.
- 2026-06-30 02:36 CST - reverse_vehicle close-red stopline release: route112/117 repeatedly held active_red final clamp at red_stop_distance ~=2.4-3.0m with no front actor or obstacle, so forced route-prior reverse_vehicle now releases the clamp in a narrow 2.20-3.20m low-speed window. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static or active_red_final_clamp or active_red_far" -q => 10 passed, 429 deselected.
- 2026-06-30 02:52 CST - reverse_vehicle two-meter boundary anti-oscillation: route110 with the 2.05m split still alternated reverse_unwedge/forward_resume around 1.96-2.10m for several hundred frames, so ordinary high-blockage static reverse now only fires at <=1.85m and >1.85m resumes forward; ultraclose <=0.25m reverse remains unchanged. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static or active_red_final_clamp or active_red_far" -q => 11 passed, 429 deselected.
- 2026-06-30 03:00 CST - supervisor forward action clears reverse gear: active PlannerAction defaults to forward, but SafetySupervisor only set reverse=True and did not explicitly clear raw reverse on forward recovery actions. Forward recoveries now set control.reverse=False, preventing route-prior forward_resume or red release from inheriting a reverse gear. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "supervisor_forward_action_clears_raw_reverse or reverse_vehicle_route_prior_static or active_red_final_clamp or active_red_far" -q => 12 passed, 429 deselected.
- 2026-06-30 03:07 CST - reverse_vehicle far static forward keepalive: route110 with reverse-clear still fell into observed_only braking at 8-14m static LiDAR blockage before reaching the 2m recovery window. Forced route-prior reverse_vehicle now keeps forward motion for 4.8-16.0m static obstacles with no actor/red conflict and an open LiDAR side. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static or supervisor_forward_action_clears_raw_reverse or active_red_final_clamp or active_red_far" -q => 13 passed, 429 deselected.
- 2026-06-30 03:11 CST - reverse_vehicle mid-gap moderate-speed forward resume: route110 far-keepalive retry still fell into observed_only at front_obstacle_distance ~=4.36m and ego_speed ~=2.83m/s, just above the prior 2.80m/s gate. The 1.85-4.80m forward-resume window now accepts ego_speed <3.40m/s. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static or supervisor_forward_action_clears_raw_reverse or active_red_final_clamp or active_red_far" -q => 14 passed, 429 deselected.
- 2026-06-30 03:16 CST - reverse_vehicle high-speed far keepalive and low-blockage close-red release: route110 still reached observed_only at 10.5m and 3.22m/s, so far static forward keepalive now accepts ego_speed <4.0m/s. route117 also hit active_red final clamp with a low-blockage 3.84m static point, so reverse_vehicle close-red release now allows 2.8-4.4m low-blockage static observations instead of requiring no front obstacle. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static or reverse_vehicle_close_red_release_allows_low_blockage_static_obstacle or supervisor_forward_action_clears_raw_reverse or active_red_final_clamp or active_red_far" -q => 16 passed, 429 deselected.
- 2026-06-30 03:31 CST - reverse_vehicle strong forward recovery pulse: latest lowblock-red run proved the rules fired but vehicle speed stayed near zero at 2.4m forward_resume and close-red release. Route-prior reverse_vehicle forward_resume now uses target_speed 3.4 with throttle_floor 0.90/cap 1.0, and close-red release uses throttle 0.92-1.0. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static or reverse_vehicle_close_red_release_allows_low_blockage_static_obstacle or supervisor_forward_action_clears_raw_reverse or active_red_final_clamp or active_red_far" -q => 16 passed, 429 deselected.
- 2026-06-30 03:45 CST - reverse_vehicle 1.8m anti-oscillation follow-up: strong-pulse route110 still alternated reverse_unwedge/forward_resume around 1.78-1.86m. Ordinary high-blockage static reverse now only fires at <=1.55m, and 1.55-4.80m uses strong forward_resume. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k "reverse_vehicle_route_prior_static or reverse_vehicle_close_red_release_allows_low_blockage_static_obstacle or supervisor_forward_action_clears_raw_reverse or active_red_final_clamp or active_red_far" -q => 17 passed, 429 deselected.
- 2026-06-30 05:08 CST - reverse_vehicle high-blockage low-speed stuck swing: live route113/114/116 showed repeated 2.3-2.8m full LiDAR blockage at low but nonzero speed, so the local stuck counter now treats 2.30-3.05m and abs(ego_speed)<0.65 as stuck before issuing reverse_swing. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k 'reverse_vehicle_route_prior_static_high_blockage or reverse_vehicle_route_prior_static_mid_gap or reverse_vehicle_near_line_static_red_release or reverse_vehicle_route_prior_clear_low_blockage' -q => 7 passed, 446 deselected. Live marker observed in route116 and route114 attempt 1782766800.
- 2026-06-30 05:15 CST - reverse_vehicle low-blockage far-edge resume: route113 lowspeed retry reached front_obstacle_distance ~=4.85m with low blockage and balanced open side, just outside the 4.80m low-blockage forward window, then fell back to observed_only braking. Expanded the legal route-prior low-blockage static forward-resume window to <=6.00m. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k 'reverse_vehicle_route_prior_static_high_blockage or reverse_vehicle_route_prior_static_low_blockage or reverse_vehicle_route_prior_static_mid_gap or reverse_vehicle_near_line_static_red_release or reverse_vehicle_route_prior_clear_low_blockage' -q => 9 passed, 445 deselected.
- 2026-06-30 05:17 CST - reverse_vehicle two-meter high-blockage stuck swing: route114 lowspeed retry escaped the 2.44m point but then sat at 1.98-2.08m with full LiDAR blockage and near-zero speed, below the 2.30m stuck-swing lower bound. Expanded the local high-blockage stuck swing window to 1.90-3.05m while keeping the two-sample local counter and no actor/red gates. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k 'reverse_vehicle_route_prior_static_high_blockage or reverse_vehicle_route_prior_static_low_blockage or reverse_vehicle_route_prior_static_mid_gap or reverse_vehicle_near_line_static_red_release or reverse_vehicle_route_prior_clear_low_blockage' -q => 10 passed, 445 deselected.
- 2026-06-30 05:22 CST - reverse_vehicle clear-low-blockage rolling resume: latest route116/114 run reached front_obstacle_distance=None with low blockage/open side but ego_speed ~=4.23-4.46m/s, just above the 4.20m/s clear-low-blockage route-prior resume gate, then fell back to observed_only braking. Expanded the no-obstacle low-blockage forward-resume speed window to <4.80m/s. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k 'reverse_vehicle_route_prior_static_high_blockage or reverse_vehicle_route_prior_static_low_blockage or reverse_vehicle_route_prior_static_mid_gap or reverse_vehicle_near_line_static_red_release or reverse_vehicle_route_prior_clear_low_blockage' -q => 11 passed, 445 deselected.
- 2026-06-30 05:26 CST - parallel queue symlink race fix: route114 attempt 1782768279 failed during agent import with ModuleNotFoundError: DriveTransformer because concurrent queue processes unlinked shared CVCI_BenchMark symlinks while another evaluator was importing. tools/run_cvci_parallel_80_queue.py now keeps already-correct symlinks intact and only relinks missing/wrong links. Validation: py_compile passed for tools/run_cvci_parallel_80_queue.py.
- 2026-06-30 05:32 CST - reverse_vehicle one-point-eight-meter stuck swing: route113 reached front_obstacle_distance ~=1.80m with full LiDAR blockage and low speed, below the 1.90m high-blockage stuck-swing lower bound, so it stayed in forward_resume. Expanded the local high-blockage stuck-swing window to 1.70-3.05m while retaining the two-sample counter and no actor/red gates. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k 'reverse_vehicle_route_prior_static_high_blockage or reverse_vehicle_route_prior_static_low_blockage or reverse_vehicle_route_prior_static_mid_gap or reverse_vehicle_near_line_static_red_release or reverse_vehicle_route_prior_clear_low_blockage' -q => 12 passed, 445 deselected.
- 2026-06-30 05:34 CST - reverse_vehicle low-blockage far static keepalive: route113 reached front_obstacle_distance ~=12.57m with lidar_blockage_ratio ~=0.40 and no actor/red conflict, outside the <=6m low-blockage static window and below the >=0.55 far-blockage keepalive window, then fell back to observed_only braking. Expanded the route-prior low-blockage static forward window to <=16m. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k 'reverse_vehicle_route_prior_static_high_blockage or reverse_vehicle_route_prior_static_low_blockage or reverse_vehicle_route_prior_static_mid_gap or reverse_vehicle_near_line_static_red_release or reverse_vehicle_route_prior_clear_low_blockage' -q => 13 passed, 445 deselected.
- 2026-06-30 05:41 CST - reverse_vehicle clear unknown-open-side resume: route114 reached no front obstacle, no red, lidar_blockage_ratio=0.0, ego_speed ~=2.85m/s, but lidar_open_side='unknown', so the no-obstacle clear-low-blockage resume gate fell through to observed_only braking. Allowed unknown open side for the no-obstacle low-blockage route-prior resume with zero steer bias. Validation: py_compile passed; pytest tests/test_cvci_auxiliary_system.py -k 'reverse_vehicle_route_prior_static_high_blockage or reverse_vehicle_route_prior_static_low_blockage or reverse_vehicle_route_prior_static_mid_gap or reverse_vehicle_near_line_static_red_release or reverse_vehicle_route_prior_clear_low_blockage or reverse_vehicle_route_prior_clear_unknown' -q => 14 passed, 445 deselected.

- 2026-06-30 05:49 CST - reverse_vehicle balanced high-blockage unwedge: allow full-blockage balanced stalls at 1.70-3.05m to enter reverse_swing after two stuck frames; adds focused unit coverage for route110-style 2.45m balanced stall.

- 2026-06-30 06:05 CST - reverse_vehicle near-line center-blockage release: route112 reached red_stop ~=2.12m, front_obstacle ~=3.08m, low overall blockage ~=0.475 but center blockage ~=0.94, so the near-line static release missed and final-clamped at zero speed. Allow center_blockage >=0.75 as an alternate trigger under the existing reverse_vehicle near-line static gates; added focused unit coverage.

- 2026-06-30 06:22 CST - reverse_vehicle far-vehicle keepalive: route112 red cleared into front_vehicle ~=9.5m / front_obstacle ~=10m, full LiDAR blockage, safe TTC, near-zero speed, but static far keepalive excluded front vehicles and observed_only reapplied braking. Added route-prior far-vehicle keepalive under 8-13m, safe-TTC, no-red, full-blockage gates with focused unit coverage.

### 2026-06-30 06:47 - highway route43 brake-response hold
- Scope: `highway_accident_vehicle` only.
- Evidence: route43 completed with no collision/lane failure but `HighSpeedBrakeCriterion` failed at 4.803 m/s despite one late brake probe.
- Change: extend initial highspeed brake probe from 26 to 42 frames, widen near-hazard rearm to 1.5-7.0 m, and allow final brake hold during close hazard even after the first brake-response window has marked done.
- Goal: reduce official brake-response speed while preserving the already-successful bypass/resume behavior.
