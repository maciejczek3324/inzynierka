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

    # Górne przeguby obrotowe obrot1/obrot2 były celowo ograniczone do 0.08 rad.
    # Zachowujemy tę ochronę dla starego forward/backward/side, żeby nie
    # rozwalić wyuczonych gaitów. Przy PURE YAW odblokowujemy je mocniej.
    turn_action_scale_locomotion = 0.08
    turn_action_scale_yaw = 0.50

    # Przy PURE YAW górne przeguby dostają też szybszą odpowiedź celu.
    # Reszta chodu nadal używa starego target_filter_alpha = 0.20.
    turn_target_filter_alpha_yaw = 0.40

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
    command_mode = "forward_backward_side_yaw"

    # Komenda używana przy num_envs == 1 (PLAY / szybki test checkpointu).
    # STAGE TURN: czysty obrót w miejscu, bez translacji.
    play_command_x = 0.0
    play_command_y = 0.0
    play_command_yaw = 0.45

    # Zostawione dla kompatybilności ze starszym kodem/testami.
    forward_command_x = -0.02  # docelowo 0.12

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
    # SIDE OVERNIGHT V4: trochę więcej stand + zmiana komendy w trakcie epizodu.
    # 5 s daje trening przejść move<->stand bez zmiany 39D obserwacji.
    standing_command_probability = 0.08
    command_resample_time_s = 5.0

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

    # Legacy heading-relative forward_progress is disabled for backward.
    # Backward now has its own reward for WORLD-axis velocity.
    # Forward retains 100% of its existing progress reward.
    backward_progress_scale = 0.0

    # Małe kroczki są OK.
    backward_step_air_target = 0.10

    # 2 cm = pełna jakość placementu.
    # 1-5 mm nadal daje częściowy reward.
    backward_step_placement_target = 0.015

    # Dense: noga podczas swingu idzie w stronę komendy.
    rew_scale_backward_swing_direction = 4.0

    # Event: po swingu stopa ląduje ZA podporową
    # w kierunku cofania.
    rew_scale_backward_step_direction = 2.0  # bootstrap; real cycle pays more

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
    backward_repeat_swing_scale = 0.0

    # Powtórzony touchdown tej samej nogi nadal dostaje
    # minimalny reward, żeby nie zrobić twardej ściany.
    backward_repeat_step_scale = 0.0

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

    rew_scale_backward_real_pair_use = 3.0  # support shaping, not main goal
    rew_scale_backward_real_swing_balance = -6.0

    backward_real_lift_target = 0.025  # 25 mm = pełna jakość

    # NOWE: ukończony prawdziwy NAPRZEMIENNY krok.
    rew_scale_backward_real_alt_step = 8.0

    # NOWE: jeśli robot jedzie dzięki "pchaniu" przy słabym użyciu obu nóg.
    rew_scale_backward_push_glide = -6.0

    # Oczekiwana noga ma zginać kolano podczas prawdziwego swingu.
    rew_scale_backward_expected_knee_flex = 2.0

    # Backward-specific actual knee flex -> partial extension.
    # Left flex = +delta, right flex = -delta.
    backward_knee_flex_target = 0.12        # rad, extra flex after liftoff
    backward_knee_extension_target = 0.08  # rad, extension from swing peak
    backward_knee_min_flex = 0.035         # rad, cycle event minimum
    backward_knee_min_extension = 0.020   # rad, cycle event minimum
    backward_cycle_lift_min = 0.010        # m, only accepted step
    rew_scale_backward_knee_extension = 3.0







    # ============================================================
    # BACKWARD: FIXED WORLD HEADING + WORLD-AXIS VELOCITY
    # Keeps observation_space=39: correction uses existing cmd yaw channel.
    # ============================================================
    backward_heading_kp = 1.5                  # 1/s, yaw-rate = kp * heading error
    backward_heading_max_yaw_rate = 0.40       # rad/s, only during backward
    backward_heading_deadband = 0.035          # rad (~2 deg)
    backward_heading_penalty_span = 0.50       # rad after deadband -> full loss
    backward_heading_quality_sigma = 0.60     # rad, attenuate gait rewards on turns
    backward_heading_gait_floor = 0.15        # still enough gradient at bad yaw
    backward_heading_fail_angle = 1.00        # rad (~57 deg), backward only
    rew_scale_backward_heading_hold = -3.0

    # Move in the requested direction in the ORIGINAL world heading,
    # not in the robot's rotated frame. Zero for no backward motion.
    backward_axis_speed_sigma = 0.070         # m/s, gradual overspeed reduction
    backward_axis_lateral_sigma = 0.08        # m/s, gradual lateral reduction
    rew_scale_backward_axis_track = 5.0

    # ============================================================
    # BACKWARD WORLD-LINE HOLD
    # Robot ma nie tylko patrzeć w dobrym kierunku,
    # ale również pozostać na linii, z której rozpoczął cofanie.
    # ============================================================

    # PD -> korekcyjna komenda cmd_y.
    backward_line_kp = 0.80
    backward_line_kd = 0.35

    backward_line_pos_deadband = 0.010  # 1 cm
    backward_line_vel_deadband = 0.020  # 2 cm/s

    # Nie każemy mu gwałtownie wracać na linię.
    backward_line_max_lateral_cmd = 0.08  # m/s

    # Jakość toru używana do bramkowania dużych rewardów kroku.
    backward_line_velocity_scale = 0.08  # m/s
    backward_line_position_scale = 0.06  # m
    backward_line_reward_floor = 0.15

    # Bezpośrednia kara za duży world-lateral drift.
    backward_world_lateral_deadband = 0.03
    backward_world_lateral_full = 0.20
    rew_scale_backward_world_lateral = -2.0

    # ============================================================
    # SIDE WALK - STAGE 1
    #
    # Zachowujemy wyuczony forward / backward / stand i dokładamy
    # małe, czyste komendy +/-Y bez obracania kadłuba.
    # ============================================================

    # Spośród komend RUCHU 35% to side.
    # Przy standing_command_probability = 0.10 daje około:
    # 31.5% side, 29.25% forward, 29.25% backward, 10% stand.
    # W STAGE TURN side zajmuje 30% komend ruchu; yaw ma osobne 35%.
    # YAW + REPAIR V2: na ten krótki etap wyłączamy SIDE.
    # SIDE OVERNIGHT V4: główny cel runu, ale zostawiamy miejsce na YAW/F/B/stand.
    side_probability = 0.60

    # Na start małe prędkości, żeby wykorzystać transfer istniejącego gaitu.
    side_command_min = 0.040
    side_command_max = 0.040

    # ============================================================
    # SIDE STEP-TOGETHER ("ODSTAWNO-DOSTAWNY")
    #
    # +Y: lewa noga odstawia -> prawa dostawia
    # -Y: prawa noga odstawia -> lewa dostawia
    #
    # Pełna nagroda wymaga:
    # - ruchu stopy W BOK zgodnie z komendą,
    # - realnego liftu,
    # - braku krzyżowania nóg,
    # - małego rozjazdu przód/tył stóp,
    # - po "odstawieniu" zwiększenia rozstawu,
    # - po "dostawieniu" powrotu do normalnego rozstawu.
    # ============================================================

    side_air_deadband = 0.04
    side_air_target = 0.12

    side_step_displacement_min = 0.010       # 1 cm: minimalny realny krok w bok
    side_step_displacement_target = 0.035    # 3.5 cm: pełna jakość przesunięcia stopy

    # OVERNIGHT bootstrap: 6 mm wystarcza do przejścia state-machine z ODSTAW do DOSTAW.
    # Pełna jakość nadal wymaga wyraźnego liftu (18 mm).
    side_min_clearance = 0.006
    side_clearance_target = 0.018

    side_open_width_min = 0.005              # REPAIR: 5 mm bootstrap; wcześniej 10 mm blokowało 100% lead-eventów
    side_open_width_target = 0.025           # 2.5 cm dodatkowego rozstawu = pełna jakość

    side_close_width_sigma = 0.018           # trailing foot ma wrócić do rozstawu bazowego
    side_close_accept_error = 0.025          # max błąd rozstawu przy zaakceptowanym "dostaw"

    side_min_width_ratio = 0.55              # nogi nie mogą się skrzyżować / zapaść do środka

    # Stopy mają być obok siebie w osi X, nie jedna wyraźnie "z przodu".
    side_fore_aft_deadband = 0.015           # 1.5 cm bez kary
    side_fore_aft_full = 0.060               # 6 cm = pełna kara
    side_fore_aft_quality_scale = 0.035

    # Generic step_complete zostawiamy jako bootstrap, ale główną nagrodą
    # ma być teraz poprawna sekwencja ODSTAW -> DOSTAW.
    side_generic_step_scale = 0.20

    rew_scale_side_expected_swing = 3.0
    # W ENV ten reward jest teraz aktywny tylko przy ROBUST single-support,
    # więc nie płaci już za zwykły double-support.
    rew_scale_side_expected_support = 1.5

    # PLUS REPAIR V1:
    # +Y ma odciążyć i realnie oderwać LEWĄ nogę prowadzącą.
    rew_scale_side_plus_left_lift = 4.0

    rew_scale_side_open_progress = 0.0        # REPAIR: dense sygnał do realnego otwierania rozstawu
    rew_scale_side_lead_step = 8.0
    rew_scale_side_trail_step = 10.0
    rew_scale_side_fore_aft = -6.0
    rew_scale_side_narrow_stance = -6.0

    # Heading hold: side ma być translacją +/-Y bez obracania robota.
    # Używa tego samego world-heading reference zapisanego na resecie.
    side_heading_kp = 1.50
    side_heading_max_yaw_rate = 0.30
    side_heading_deadband = 0.035
    side_heading_penalty_span = 0.50
    side_heading_fail_angle = 1.00
    rew_scale_side_heading_hold = -2.0

    # Chód boczny wymaga świadomego transferu ciężaru w roll.
    # Nie wyłączamy ochrony roll, tylko osłabiamy ją podczas pure-side.
    side_roll_penalty_scale = 0.35

    # Dodatkowa kara za przesuwanie stopy po ziemi podczas side.
    # Bazowy foot_slip nadal działa.
    rew_scale_side_contact_slip = -6.0

    # ============================================================
    # PURE YAW TURN - STAGE 1
    #
    # Uczymy najpierw obrotu CAŁEJ BAZY w miejscu:
    # cmd_x = 0, cmd_y = 0, cmd_yaw != 0.
    # Dopiero po opanowaniu tego etapu połączymy yaw z translacją i padem.
    # ============================================================

    # Spośród wszystkich komend RUCHU 35% to czysty obrót.
    # Przy stand=10%, side=30% daje w przybliżeniu:
    # 31.5% yaw, 27% side, 15.75% forward, 15.75% backward, 10% stand.
    # YAW + REPAIR V2:
    # przy stand=5% daje ok. 76% epizodów PURE YAW,
    # reszta zostaje na forward/backward dla podtrzymania pamięci.
    yaw_turn_probability = 0.15

    # FINAL TURN STAGE: szybszy, ale nadal rozsądny zakres.
    # 0.25-0.55 rad/s ~= 14.3-31.5 deg/s. PLAY = 0.45 rad/s ~= 25.8 deg/s.
    # PLUS REPAIR V1: stała komenda ułatwia odkrycie brakującego mirrored skill.
    yaw_turn_command_min = 0.45
    yaw_turn_command_max = 0.45

    # ------------------------------------------------------------
    # PRAWDZIWY KROK SKRĘTNY
    # ------------------------------------------------------------
    # Minimalny lokalny fore-aft arc swing-foot. Dla +yaw lewa idzie
    # lekko w tył, prawa w przód; dla -yaw odwrotnie.
    yaw_turn_arc_deadband = 0.006
    yaw_turn_arc_target = 0.035

    # Realny liftoff/air-time - nie nagradzamy szurania i mikro-touchdownów.
    yaw_turn_lift_target = 0.018
    yaw_turn_air_deadband = 0.045
    yaw_turn_air_target = 0.14

    # Górny yaw-joint może wyraźnie pracować w SWINGU.
    # Po kontakcie duże wykręcenie staje się drogie, dzięki czemu noga
    # podporowa prostuje obrot1/obrot2 i obraca nad nią bazę.
    yaw_turn_hip_twist_deadband = 0.08
    yaw_turn_hip_twist_target = 0.30
    yaw_turn_stance_twist_deadband = 0.12
    yaw_turn_stance_twist_full = 0.35

    # Same-side repeat zachowuje mały discovery signal, pełna kasa jest
    # za naprzemienny krok.
    yaw_turn_repeat_scale = 0.02             # REPAIR: prawie zerowy reward za kolejny krok tą samą nogą
    yaw_turn_dense_repeat_scale = 0.10       # REPAIR: dense arc/twist też preferuje nogę przeciwną do poprzedniej

    yaw_turn_right_arc_boost = 5.0
    yaw_turn_right_step_boost = 3.0

    # PLUS REPAIR V1:
    # +yaw ma odciążyć i realnie oderwać PRAWĄ nogę.
    # Istniejący right_arc_boost przejmie prowadzenie, gdy noga już swingnie.
    rew_scale_yaw_plus_right_lift = 1.0

    # YAW + REPAIR V2:
    # signed dense arc od 0 mm. 2 cm = pełny sygnał discovery.
    yaw_plus_right_arc_discovery_target = 0.015
    rew_scale_yaw_plus_right_arc_discovery = 8.0

    # YAW + REPAIR V3:
    # kara tylko za KOLEJNY zaakceptowany krok lewej nogi przy +yaw.
    # Ma złamać lokalne optimum "kręcę się głównie lewą".
    rew_scale_yaw_plus_left_repeat_penalty = -1.0

    # Dense discovery + real touchdown event + cykl swing/recenter.
    rew_scale_yaw_turn_swing = 2.0
    rew_scale_yaw_turn_swing_twist = 1.5
    rew_scale_yaw_turn_step = 8.0
    rew_scale_yaw_turn_stance_twist = -3.0

    # Anty-exploity dla obrotu w miejscu.
    rew_scale_yaw_turn_contact_slip = -8.0
    rew_scale_yaw_turn_translation = -8.0

    # ============================================================
    # SIDE +Y OVERNIGHT V4 -- reward odporny na contact-flicker
    # ============================================================

    # Ground drag jest liczony z rzeczywistej prędkości stopy przy ROBUST kontakcie,
    # a nie z displacementu liczonego od potencjalnie fałszywego liftoff.
    side_plus_left_ground_drag_speed_target = 0.05
    rew_scale_side_plus_left_ground_drag = -1.5

    # Absolutny clearance jest tylko małą marchewką; nie chcemy płacić dużo
    # za samo wiszenie nogą w górze.
    side_plus_left_clearance_discovery_target = 0.012
    rew_scale_side_plus_left_clearance_discovery = 6.0

    # Stary boolean ~left_contact został wyłączony: log pokazał reward ~5 mimo
    # praktycznie zerowego realnego kroku. Zostaje tylko jako debug.
    rew_scale_side_plus_left_liftoff_discovery = 0.0

    # Główny discovery: dodatnia prędkość Z LEWEJ stopy względem PRAWEJ podpory.
    # Reward zanika po osiągnięciu 15 mm, żeby nie promować wiecznego unoszenia.
    side_plus_left_up_velocity_target = 0.08
    side_plus_left_up_velocity_fade_clearance = 0.015
    rew_scale_side_plus_left_up_velocity = 8.0

    # Po prawdziwym, filtrowanym single-support przejmują clearance + air-time.
    side_plus_left_real_swing_target = 0.015
    rew_scale_side_plus_left_real_swing = 10.0

    # Przy +Y translacja nadal ma gradient, ale pełny tracking/progress dostaje
    # dopiero wtedy, gdy aktualnie oczekiwana stopa faktycznie się unosi.
    side_plus_velocity_reward_floor = 0.35


