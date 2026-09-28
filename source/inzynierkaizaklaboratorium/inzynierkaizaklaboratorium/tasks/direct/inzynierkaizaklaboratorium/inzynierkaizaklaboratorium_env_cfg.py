# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab_assets.robots.inzynierka import PORZADKI_ROBOT_CFG


@configclass
class EventCfg:
    """Minimalne, deterministyczne środowisko na CLEAN WALK V1."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (0.8, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


@configclass
class InzynierkaizaklaboratoriumEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 4
    action_space = 10

    # 3 ang vel + 3 gravity + 3 commands
    # + 10 joint pos + 10 joint vel + 10 actions = 39
    observation_space = 39
    state_space = 0

    # Pełny zakres ruchu potrzebny do zginania kolan.
    action_scale = 0.5

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
    )

    events: EventCfg = EventCfg()
    robot_cfg: ArticulationCfg = PORZADKI_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Kontakt stóp tylko do rewardu, NIE do obserwacji policy.
    left_foot_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/porzadki_skrecajacy_urdf_export/stopaL_1",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    right_foot_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/porzadki_skrecajacy_urdf_export/stopaR_1",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    # Reset / terminacja.
    base_height_target = 0.2
    base_height_fall = 0.110
    max_tilt_error = 0.75
    reset_joint_noise = 0.04
    target_filter_alpha = 0.20

    # Startujemy od prostego chodu do przodu.
    # Cały reward i obserwacje są już 2D+yaw, więc później zmieniasz tylko
    # command_mode na "omni" bez przebudowy sieci.
    command_mode = "forward_backward"
    forward_command_x = -0.02 #docelowo 0.12

    # TRAINING: losowana prędkość do przodu.
    # forward_command_min = 0.08
    # forward_command_max = 0.15
    #
    # # 15% komend = stanie w miejscu.
    # standing_command_probability = 0.15
    # Zakres prędkości podczas treningu.
    forward_command_min = 0.08
    forward_command_max = 0.15

    # 15% wylosowanych komend = stanie w miejscu.
    standing_command_probability = 0.10
    command_resample_time_s = 10

    # Gotowe zakresy na późniejszy etap omnidirectional.
    omni_min_speed = 0.08
    omni_max_speed = 0.22
    omni_max_yaw_rate = 0.70

    gamepad_deadzone = 0.15
    gamepad_max_lin_speed = 0.25
    gamepad_max_yaw_rate = 0.80

    # Reward CLEAN WALK V1.
    lin_vel_tracking_sigma = 0.02
    yaw_vel_tracking_sigma = 0.100

    rew_scale_alive = 0.20
    rew_scale_terminated = -20.0

    rew_scale_track_lin_vel_xy = 12.0
    rew_scale_track_yaw_vel = 3.0

    rew_scale_orientation = -1.0
    rew_scale_lin_vel_z = -0.25
    rew_scale_ang_vel_xy = -0.15

    # Delikatne anty-exploity, bez narzucania fazy chodu.
    rew_scale_both_feet_air = -5
    rew_scale_foot_slip = -0.15

    # Minimalna regularizacja; slew limiter robi większość ochrony serw.
    rew_scale_action_rate = -0.01
    # --- NATURALNY KROK / ZGINANIE NOGI ---
    swing_clearance_target = 0.04  # 4 cm wystarczy; nie robimy bociana było 0,025
    swing_min_time = 0.12  # krótsze oderwanie = raczej drganie # z 0,08 na 0,12
    swing_max_time = 0.45  # nie opłaca się wisieć na jednej nodze

    # Zgięcie "duckowe" kolana nogi przenoszonej.
    # 0.35 rad ~= 20 stopni.
    swing_knee_flex_target = 0.15
    rew_scale_swing_knee_flex = 1.0

    rew_scale_swing_clearance = 4.0
    rew_scale_step_complete = 8.0
    rew_scale_forward_progress = 14.0

    # Kara za "jazdę" przez długi czas z obiema stopami przyklejonymi.
    rew_scale_double_support_stuck = -4.0
    # Normalny chód może mieć krótki double-support.
    # Karzemy dopiero przedłużone stanie na obu stopach.
    double_support_grace_time = 0.20
    double_support_full_penalty_time = 0.50

    # Nie pozwalamy też wisieć wiecznie na jednej nodze.
    rew_scale_overlong_swing = -2.0
    rew_scale_lateral_vel = -25.0
    lateral_vel_sigma = 0.01
    lateral_vel_deadzone = 0.03
    rew_scale_yaw_rate = -15.0
    # Kara za długookresowe odpływanie w bok.
    # rew_scale_lateral_bias = -200.0

    # Filtr EMA bocznej prędkości.
    # lateral_bias_alpha = 0.05
    swing_balance_alpha = 0.03
    rew_scale_swing_balance = -40.0
    straightness_lateral_sigma = 0.035
    straightness_yaw_sigma = 0.12

    step_straightness_floor = 0.20
    progress_straightness_floor = 0.35

    lateral_bias_alpha = 0.03
    lateral_bias_deadband = 0.015
    rew_scale_lateral_bias = -50.0
    knee_balance_alpha = 0.03
    rew_scale_knee_flex_balance = -4.0

    rew_scale_knee_pair_use = 4.0

    roll_bias_alpha = 0.03
    roll_bias_deadband = 0.010
    roll_bias_sigma = 0.06
    rew_scale_roll_bias = -40.0

    # --- STANDING ---
    # Miękkie trzymanie pozycji neutralnej tylko dla cmd = 0.
    stand_pose_sigma = 0.025
    stand_height_sigma = 0.025

    rew_scale_stand_pose = 0.0
    rew_scale_stand_height = 0.0
    rew_scale_stand_joint_vel = 0.0
    rew_scale_stand_double_contact = 2.0

    rew_scale_stand_lin_vel = -10.0
    rew_scale_stand_yaw_vel = -4.0
    rew_scale_stand_upright = -10.0
    rew_scale_stand_action = 0.0
    # CHCEMY DWIE STOPY NA ZIEMI.
    rew_scale_stand_single_support = -2.0
    # --- STANDING ---

    # stand_height_sigma = 0.025
    #
    # rew_scale_stand_pose = 0.0
    # rew_scale_stand_height = 0.0
    #
    # # Może aktywnie pracować nogami żeby nie upaść.
    # rew_scale_stand_joint_vel = 0.0
    # rew_scale_stand_action = 0.0
    #
    # # Dwie stopy mają być na ziemi.
    # rew_scale_stand_double_contact = 2.0
    # rew_scale_stand_single_support = -2.0
    #
    # # Nie odjeżdżaj i nie obracaj się.
    # rew_scale_stand_lin_vel = -10.0
    # rew_scale_stand_yaw_vel = -4.0
    #
    # # Baza pionowo.
    # rew_scale_stand_upright = -10.0

    # Obie nogi mają mieć podobne zgięcie kolana.
    stand_knee_sym_deadband = 0.08
    rew_scale_stand_knee_sym = -6.0
    # rew_scale_stand_obrot10 = -8.0

    stand_ankle_sym_deadband = 0.02

    rew_scale_stand_ankle_sym = 0.0
    # rew_scale_stand_obrot9_extra = -10.0

    # --- PŁASKIE STOPY PODCZAS STANIA ---
    stand_foot_flat_deadband_deg = 4.0
    rew_scale_stand_foot_flat = -15.0

    # backward_command_min = 0.04
    # backward_command_max = 0.04
    #
    # # Spośród KOMEND RUCHU 1/3 będzie do tyłu.
    # # Przy stand=25% daje około:
    # # 25% stand
    # # 25% backward
    # # 50% forward
    # backward_probability = 0.90
    # rew_scale_backward_discovery = 0.0
    # # rew_scale_backward_track = 10.0
    # # backward_tracking_sigma = 0.003
    #
    # command_move_threshold = 0.01
    #
    # rew_scale_backward_signed_progress = 0.0
    #
    # backward_velocity_ema_alpha = 0.05
    #
    # rew_scale_backward_track = 0.0
    # backward_tracking_sigma = 0.0006
    #
    # backward_foot_placement_sigma = 0.015
    # backward_foot_placement_floor = 0.25
    # # NOWE: prawdziwy krok.
    # rew_scale_backward_real_step = 30.0
    #
    # # NOWE: nie wolno zdobywać cofania przez telepanie na 2 stopach.
    # rew_scale_backward_shuffle = -8.0
    #
    # backward_double_support_grace = 0.10
    # backward_double_support_full = 0.35

    # # ============================================================
    # # BACKWARD - STAGE 1: NAUKA PRAWDZIWEGO KROKU
    # # ============================================================
    #
    # backward_command_min = 0.04
    # backward_command_max = 0.04
    #
    # backward_probability = 0.80
    #
    # command_move_threshold = 0.01
    #
    # # Na STAGE 1 NIE płacimy za prędkość do tyłu.
    # rew_scale_backward_discovery = 0.0
    # rew_scale_backward_signed_progress = 0.0
    # rew_scale_backward_track = 0.0
    #
    # backward_velocity_ema_alpha = 0.05
    # backward_tracking_sigma = 0.0006
    #
    # # Prawdziwy swing:
    # # brak kontaktu + druga noga podpiera + min. 4 mm fizycznego liftu.
    # backward_real_lift_min = 0.004
    #
    # # Małe kroki do tyłu.
    # backward_step_min_time = 0.08
    # backward_step_max_time = 0.25
    #
    # # Ile nagradzamy sam REALNY single support.
    # rew_scale_backward_single_support = 8
    #
    # # Główna nagroda za prawidłowy naprzemienny krok.
    # rew_scale_backward_alt_step = 12.0
    #
    # # Kara za realny, ale za krótki mikrokrok.
    # rew_scale_backward_premature_touchdown = -0.5
    #
    # # Kara za przesuwanie stopy będącej na ziemi.
    # rew_scale_backward_contact_slip = -4.0
    #
    # # Kara za jazdę bazą do tyłu przy długim double-support.
    # rew_scale_backward_shuffle = -4.0
    #
    # backward_shuffle_grace = 0.10
    # backward_shuffle_full = 0.25
    #
    # # ============================================================
    # # BACKWARD STAGE 1 - MAŁE NORMALNE KROKI
    # # ============================================================
    #
    # # Oficjalny-style biped single stance.
    # backward_air_time_target = 0.10
    #
    # # Mała marchewka za odciążenie jednej nogi.
    # rew_scale_backward_load_transfer = 0.75
    #
    # ============================================================
    # BACKWARD CURRICULUM V1
    # Zachowujemy działający gait forward i uczymy go
    # stopniowo dla małych ujemnych komend.
    # ============================================================

    # Pierwszy etap: naprawdę małe cofanie.
    # command_move_threshold = 0.01, więc nie schodzimy do 0.01.
    backward_command_min = 0.015
    backward_command_max = 0.025

    # 10% stand zostaje.
    # Z pozostałych 90%:
    # ~45% forward
    # ~45% backward
    backward_probability = 0.50

    command_move_threshold = 0.01

    # Stare backward velocity rewards na razie wyłączone.
    # Kierunek daje nam command-relative progress.
    rew_scale_backward_discovery = 0.0
    rew_scale_backward_signed_progress = 0.0
    rew_scale_backward_track = 0.0

    backward_velocity_ema_alpha = 0.05
    backward_tracking_sigma = 0.0006

    # Nadal używane przez istniejące obliczenia/debug.
    backward_foot_placement_sigma = 0.015
    backward_foot_placement_floor = 0.25

    # Stare eksperymentalne rewardy WYŁĄCZONE.
    # Kod może sobie na razie istnieć, ale nie steruje uczeniem.
    backward_real_lift_min = 0.004
    backward_step_min_time = 0.08
    backward_step_max_time = 0.25
    backward_air_time_target = 0.10

    rew_scale_backward_single_support = 0.0
    rew_scale_backward_alt_step = 0.0
    rew_scale_backward_premature_touchdown = 0.0
    rew_scale_backward_load_transfer = 0.0

    # ------------------------------------------------------------
    # NOWY CURRICULUM
    # ------------------------------------------------------------

    # Backward dostaje na początek tylko 25% normalnego
    # rewardu za postęp bazy.
    # Forward nadal ma 100%.
    backward_progress_scale = 0.05

    # Małe kroczki są OK.
    backward_step_air_target = 0.10

    # 2 cm = pełna jakość placementu.
    # 1-5 mm nadal daje częściowy reward.
    backward_step_placement_target = 0.015

    # Dense: noga podczas swingu idzie w stronę komendy.
    rew_scale_backward_swing_direction = 4.0

    # Event: po swingu stopa ląduje ZA podporową
    # w kierunku cofania.
    rew_scale_backward_step_direction = 6.0

    # Anty-szuranie zostaje.
    rew_scale_backward_contact_slip = -6.0
    rew_scale_backward_shuffle = -8.0

    backward_shuffle_grace = 0.10
    backward_shuffle_full = 0.25

    # Małe kroczki są OK.
    backward_air_deadband = 0.04

    # ============================================================
    # BACKWARD STRAIGHTENING
    # Nie tworzymy osobnego rewardu za "stanie prosto".
    # Prostota BRAMKUJE tylko istniejące rewardy backward gait.
    # ============================================================

    # Boczne przesuwanie podczas backward.
    # 0.00 m/s -> factor 1.0
    # 0.01 m/s -> ~0.78
    # 0.02 m/s -> ~0.37
    backward_straight_lateral_sigma = 0.020

    # Yaw podczas backward.
    # 0.00 rad/s -> factor 1.0
    # 0.05 rad/s -> ~0.78
    # 0.10 rad/s -> ~0.37
    backward_straight_yaw_sigma = 0.10

    # Nie zerujemy całkowicie rewardu przy krzywej próbie,
    # bo nie chcemy zabić dopiero co odkrytego backward gait.
    backward_swing_straightness_floor = 0.25
    backward_step_straightness_floor = 0.10

    # ============================================================
    # BACKWARD - WYMUSZENIE NAPRZEMIENNEJ PRACY NÓG
    # ============================================================

    # Gdy jest kolej konkretnej nogi, jej prawdziwy swing
    # jest bardziej wartościowy.
    backward_expected_swing_scale = 1.50

    # Jeśli agent próbuje ponownie użyć tej samej nogi,
    # dense swing reward prawie znika.
    backward_repeat_swing_scale = 0.15

    # Powtórzony touchdown tej samej nogi nadal dostaje
    # minimalny reward, żeby nie zrobić twardej ściany.
    backward_repeat_step_scale = 0.10

    # Po kroku jednej nogi nagradzamy PRAWDZIWE oderwanie
    # oczekiwanej przeciwnej nogi.
    rew_scale_backward_expected_swing_lift = 2.5

    # Jeżeli oczekiwana noga zamiast się oderwać szura po ziemi.
    rew_scale_backward_expected_foot_slip = -8.0

    # ============================================================
    # BACKWARD - PRAWDZIWE UŻYWANIE OBU NÓG
    # ============================================================

    backward_real_pair_alpha = 0.03
    backward_real_pair_target = 0.08

    rew_scale_backward_real_pair_use = 8.0
    rew_scale_backward_real_swing_balance = -6.0



