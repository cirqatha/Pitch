from dataclasses import dataclass, field


@dataclass
class Geometry:
    spk_left: tuple = (-0.13, 0.12, 0.0)
    spk_right: tuple = (0.13, 0.12, 0.0)
    mic: tuple = (0.0, -0.01, 0.20)
    hand_z: float = 0.12
    screen_width: float = 0.31
    deck_depth: float = 0.22


@dataclass
class Config:
    n_fft: int = 1024
    hop: int = 256
    c: float = 346.0
    probe_level: float = 0.30
    level_target_peak: float = 0.5

    sweep_lo: float = 9000.0
    sweep_hi: float = 23500.0
    band_floor: float = 9500.0
    band_target: float = 7000.0
    band_min: float = 1800.0
    snr_band_req: float = 18.0
    quiet_above: float = 15000.0
    hiss_penalty_db: float = 1.5

    cir_len: int = 1024
    kaiser_beta: float = 4.5
    gate_min: float = 0.08
    gate_max: float = 1.00
    det_acq_db: float = 13.0
    det_track_db: float = 8.0
    sidelobe_guard_db: float = 22.0
    speckle_smooth_m: float = 0.025

    timing_lag_win: float = 6.0
    timing_tau: float = 0.03
    glitch_samples: float = 8.0
    ref_tau: float = 60.0
    bg_tau_track: float = 3.0
    bg_tau_far: float = 0.4
    bg_tau_idle: float = 0.4
    bg_quiet_s: float = 0.5
    protect_behind_m: float = 0.55
    cfar_kappa: float = 0.25
    floor_down_tau: float = 0.5
    floor_up_tau: float = 8.0
    det_time_tau: float = 0.025

    dop_win: int = 32
    dop_win_fast: int = 16
    dop_ref_tau: float = 5.0
    dop_kappa: float = 0.25
    dop_match_tol: float = 2.0

    t_noise: float = 0.45
    t_sweep: float = 0.80
    t_background: float = 1.40
    t_anchor_max: float = 9.0
    latency_max: float = 1.6
    anchor_pos: tuple = (0.0, 0.20)
    anchor_det_db: float = 7.0

    tbd_cell_m: float = 0.015
    tbd_xmax: float = 0.35
    tbd_ymin: float = 0.12
    tbd_ymax: float = 0.65
    tbd_clip: float = 6.0
    tbd_clip_lo: float = -1.5
    tbd_bias: float = 0.7
    tbd_near_prior: float = 0.6
    tbd_lambda: float = 0.985
    tbd_on: float = 12.0
    occ_tau: float = 6.0
    occ_protect_m: float = 0.08
    refine_win_m: float = 0.03
    refine_agree_m: float = 0.04

    kf_accel: float = 6.0
    kf_vmax: float = 1.5
    kf_amax: float = 30.0
    kf_meas_x: float = 0.06
    kf_meas_y: float = 0.04
    kf_gate: float = 3.0
    kf_gate_min: float = 0.025
    kf_gate_max: float = 0.08
    kf_prior_w: float = 1.0
    kf_det: float = 0.3
    kf_ev_ref: float = 2.0
    kf_coast_tau: float = 0.2
    kf_init_pos: float = 0.03
    kf_init_vel: float = 0.3
    kf_refine_scale: float = 0.6
    kf_dop_sx: float = 0.25
    kf_dop_sy: float = 0.10
    kf_dop_q: float = 20.0

    mht_lambda: float = 0.985
    mht_ev_lo: float = -1.5
    mht_show: float = 20.0
    mht_min_score: float = -10.0
    mht_merge_m: float = 0.05
    mht_max: int = 5
    mht_switch: float = 1.5
    mht_switch_s: float = 0.2
    mht_switch_far: float = 2.5
    mht_switch_s_far: float = 0.3
    mht_vote_tau: float = 0.3
    mht_vote_m: float = 0.06
    mht_vote_on: float = 0.6
    mht_vote_off: float = 0.2
    mht_edge_s: float = 0.3
    mht_edge_frac: float = 0.7
    mht_recent_tau: float = 0.15
    mht_weak: float = 0.8
    mht_strong: float = 2.0
    mht_weak_s: float = 0.1

    n_cands: int = 3
    cand_sep_m: float = 0.10
    conf_m: float = 0.07
    conf_s: float = 1.5
    conf_min_s: float = 0.2
    conf_win_s: float = 0.7
    stale_m: float = 0.03
    stale_s: float = 1.5
    coast_s: float = 0.45
    reacq_s: float = 1.5
    reacq_m: float = 0.10
    hold_s: float = 2.5
    hold_lost_s: float = 0.6

    arm_same_m: float = 0.03
    arm_front_m: float = 0.06
    arm_max_m: float = 0.35
    arm_behind_m: float = 0.10
    arm_ev_frac: float = 0.4
    arm_switch_s: float = 0.2

    far_y: float = 0.45
    far_conf: float = 1.6
    reach_r0: float = 0.12
    reach_v: float = 0.3
    reach_s: float = 3.0
    reach_conf: float = 1.6

    x_limits: tuple = (-0.45, 0.45)
    y_limits: tuple = (0.03, 0.85)

    geometry: Geometry = field(default_factory=Geometry)
