# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.devices import Se3Gamepad
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .inzynierkaizaklaboratorium_env_cfg import InzynierkaizaklaboratoriumEnvCfg


class InzynierkaizaklaboratoriumEnv(DirectRLEnv):
    """CLEAN WALK V1: velocity tracking bez ręcznie narzuconej trajektorii stóp."""

    cfg: InzynierkaizaklaboratoriumEnvCfg

    def __init__(
        self,
        cfg: InzynierkaizaklaboratoriumEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        print("\n===================================")
        print("ROBOT IS FIXED BASE:", self._robot.is_fixed_base)
        print("===================================\n")

        self._actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_timer = torch.zeros(self.num_envs, device=self.device)

        self._current_targets = self._robot.data.default_joint_pos.clone()

        self.stopa_l_idx = self._robot.data.body_names.index("stopaL_1")
        self.stopa_r_idx = self._robot.data.body_names.index("stopaR_1")

        self.knee_l_idx = self._robot.data.joint_names.index("obrot8")
        self.knee_r_idx = self._robot.data.joint_names.index("obrot7")

        self.gamepad = None

        if self.num_envs == 1:
            self.gamepad = Se3Gamepad()

        # ============================================================
        # PAMIĘĆ NATURALNEGO KROKU
        # MUSI ISTNIEĆ DLA KAŻDEJ LICZBY ŚRODOWISK
        # ============================================================

        self._left_swing_time = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._right_swing_time = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._left_swing_ema = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._left_knee_flex_ema = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._right_knee_flex_ema = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._right_swing_ema = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._double_support_time = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._left_max_clearance = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._right_max_clearance = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._prev_left_contact = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self._prev_right_contact = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        # self._last_rewarded_step = torch.zeros(
        #     self.num_envs,
        #     dtype=torch.int8,
        #     device=self.device,
        # )
        self._prev_root_pos = self._robot.data.root_pos_w.clone()
        self._lateral_vel_ema = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._roll_ema = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.turn_l_idx = self._robot.data.joint_names.index("obrot1")
        self.turn_r_idx = self._robot.data.joint_names.index("obrot2")

        # ============================================================
        # DEBUG WSZYSTKICH STAWÓW
        # ============================================================

        self.debug_joint_indices = {
            f"obrot{i}": self._robot.data.joint_names.index(f"obrot{i}")
            for i in range(1, 11)
        }

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "alive",
                "terminated",
                "track_lin_vel_xy",
                "track_yaw_vel",
                "orientation",
                "lin_vel_z",
                "ang_vel_xy",
                "both_feet_air",
                "foot_slip",
                "action_rate",
                "swing_clearance",
                "step_complete",
                "forward_progress",
                "single_support_debug",
                "double_support_debug",
                "foot_height_diff_debug",
                "load_transfer_debug",
                "swing_knee_flex",
                "swing_knee_flex_debug",
                "knee_l_pos_debug",
                "knee_r_pos_debug",
                "knee_l_abs_debug",
                "knee_r_abs_debug",
                "left_contact_debug",
                "right_contact_debug",
                "double_support_stuck",
                "overlong_swing",
                "lateral_vel",
                "lateral_speed_debug",
                "yaw_rate",
                "yaw_rate_debug",
                "lateral_signed_debug",
                "yaw_rate_signed_debug",
                # "lateral_speed_debug",
                "lateral_bias_debug",
                "lateral_bias",
                "left_step_debug",
                "right_step_debug",
                "left_swing_debug",
                "right_swing_debug",
                "swing_balance",
                "straightness_gate_debug",
                "step_gate_debug",
                "progress_gate_debug",
                "bias_progress_gate_debug",
                "knee_flex_balance",
                "left_knee_flex_ema_debug",
                "right_knee_flex_ema_debug",
                "knee_pair_use",
                "knee_pair_use_debug",
                "roll_signed_debug",
                "roll_abs_debug",
                "roll_ema_debug",
                "roll_bias",
                "roll_progress_gate_debug",
                "obrot1_delta_debug",
                "obrot1_abs_debug",

                "obrot2_delta_debug",
                "obrot2_abs_debug",

                "obrot3_delta_debug",
                "obrot3_abs_debug",

                "obrot4_delta_debug",
                "obrot4_abs_debug",

                "obrot5_delta_debug",
                "obrot5_abs_debug",

                "obrot6_delta_debug",
                "obrot6_abs_debug",

                "obrot7_delta_debug",
                "obrot7_abs_debug",

                "obrot8_delta_debug",
                "obrot8_abs_debug",

                "obrot9_delta_debug",
                "obrot9_abs_debug",

                "obrot10_delta_debug",
                "obrot10_abs_debug",

                "forward_vel_debug",
                "forward_command_debug",
                "forward_error_debug",

                "stand_pose",
                "stand_height",
                "stand_joint_vel",
                "stand_double_contact",

                "stand_fraction_debug",
                "stand_forward_speed_debug",
                "stand_lateral_speed_debug",
                "stand_yaw_rate_debug",
                "stand_pose_error_debug",
                "stand_height_error_debug",

                "stand_lin_vel",
                "stand_yaw_vel",
                "stand_upright",
                "stand_tilt_debug",
                "stand_action",
                "stand_double_contact_debug",
                "stand_single_support_debug",
                "stand_single_support",
                "stand_knee_sym",
                "stand_knee_sym_debug",

                # "stand_obrot9_debug",
                # "stand_obrot10_debug",
                "stand_obrot9_abs_debug",
                "stand_obrot10_abs_debug",
                # "stand_obrot10",
                "stand_ankle_sym",
                # "stand_obrot9_extra",
                # "stand_obrot9_abs_debug",
                # "stand_obrot10_abs_debug",
                "stand_ankle_sym_debug",
                "stand_foot_flat",
                "stand_left_foot_tilt_deg_debug",
                "stand_right_foot_tilt_deg_debug",
                "backward_fraction_debug",
                "backward_command_debug",
                "backward_vel_debug",
                "backward_error_debug",
                "backward_cross_speed_debug",

                "forward_fraction_debug",
                "forward_only_command_debug",
                "forward_only_vel_debug",
                "forward_only_error_debug",
                "backward_discovery",
                "backward_track",
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        self._left_foot_contact = ContactSensor(self.cfg.left_foot_contact)
        self._right_foot_contact = ContactSensor(self.cfg.right_foot_contact)
        self.scene.sensors["left_foot_contact"] = self._left_foot_contact
        self.scene.sensors["right_foot_contact"] = self._right_foot_contact

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(
                global_prim_paths=[self.cfg.terrain.prim_path]
            )

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    def _sample_training_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        self._commands[env_ids] = 0.0

        # if self.cfg.command_mode == "forward":
        #     self._commands[env_ids, 0] = self.cfg.forward_command_x

        if self.cfg.command_mode == "forward":
            n = len(env_ids)

            # _commands są już wyzerowane wyżej.
            # Część środowisk dostaje dokładnie 0.0 -> nauka stania.
            selector = torch.rand(n, device=self.device)

            walking_mask = (
                    selector >= self.cfg.standing_command_probability
            )

            walking_ids = torch.nonzero(
                walking_mask,
                as_tuple=False,
            ).flatten()

            if len(walking_ids) > 0:
                target_env_ids = env_ids[walking_ids]

                forward_speed = (
                        self.cfg.forward_command_min
                        + (
                                self.cfg.forward_command_max
                                - self.cfg.forward_command_min
                        )
                        * torch.rand(
                    len(walking_ids),
                    device=self.device,
                )
                )

                self._commands[target_env_ids, 0] = forward_speed
        elif self.cfg.command_mode == "forward_backward":
            n = len(env_ids)

            selector = torch.rand(
                n,
                device=self.device,
            )

            stand_mask = (
                    selector < self.cfg.standing_command_probability
            )

            moving_mask = ~stand_mask

            direction_selector = torch.rand(
                n,
                device=self.device,
            )

            backward_mask = (
                    moving_mask
                    & (
                            direction_selector
                            < self.cfg.backward_probability
                    )
            )

            forward_mask = (
                    moving_mask
                    & (~backward_mask)
            )

            # --------------------------------------------------------
            # FORWARD
            # --------------------------------------------------------

            forward_local_ids = torch.nonzero(
                forward_mask,
                as_tuple=False,
            ).flatten()

            if len(forward_local_ids) > 0:
                target_env_ids = env_ids[forward_local_ids]

                speed = (
                        self.cfg.forward_command_min
                        + (
                                self.cfg.forward_command_max
                                - self.cfg.forward_command_min
                        )
                        * torch.rand(
                    len(forward_local_ids),
                    device=self.device,
                )
                )

                self._commands[target_env_ids, 0] = speed

            # --------------------------------------------------------
            # BACKWARD
            # --------------------------------------------------------

            backward_local_ids = torch.nonzero(
                backward_mask,
                as_tuple=False,
            ).flatten()

            if len(backward_local_ids) > 0:
                target_env_ids = env_ids[backward_local_ids]

                speed = (
                        self.cfg.backward_command_min
                        + (
                                self.cfg.backward_command_max
                                - self.cfg.backward_command_min
                        )
                        * torch.rand(
                    len(backward_local_ids),
                    device=self.device,
                )
                )

                self._commands[target_env_ids, 0] = -speed

        elif self.cfg.command_mode == "omni":
            n = len(env_ids)
            selector = torch.rand(n, device=self.device)

            ids_translation = selector < 0.55
            ids_yaw = (selector >= 0.55) & (selector < 0.75)
            ids_combined = (selector >= 0.75) & (selector < 0.95)
            ids_stop = selector >= 0.95

            moving = ids_translation | ids_combined
            moving_ids = torch.nonzero(moving, as_tuple=False).flatten()

            if len(moving_ids) > 0:
                angle = (
                    2.0 * math.pi * torch.rand(len(moving_ids), device=self.device)
                    - math.pi
                )
                speed = (
                    self.cfg.omni_min_speed
                    + (self.cfg.omni_max_speed - self.cfg.omni_min_speed)
                    * torch.rand(len(moving_ids), device=self.device)
                )
                target_env_ids = env_ids[moving_ids]
                self._commands[target_env_ids, 0] = speed * torch.cos(angle)
                self._commands[target_env_ids, 1] = speed * torch.sin(angle)

            yaw_or_combined = ids_yaw | ids_combined
            yaw_ids = torch.nonzero(yaw_or_combined, as_tuple=False).flatten()

            if len(yaw_ids) > 0:
                target_env_ids = env_ids[yaw_ids]
                yaw = (
                    2.0 * torch.rand(len(yaw_ids), device=self.device) - 1.0
                ) * self.cfg.omni_max_yaw_rate
                self._commands[target_env_ids, 2] = yaw

            stop_ids = torch.nonzero(ids_stop, as_tuple=False).flatten()
            if len(stop_ids) > 0:
                self._commands[env_ids[stop_ids]] = 0.0

        else:
            raise ValueError(
                f"Nieznany command_mode: {self.cfg.command_mode!r}. "
                "Użyj 'forward', 'forward_backward' albo 'omni'."
            )

        self._command_timer[env_ids] = self.cfg.command_resample_time_s

    def _read_gamepad_commands(self):
        pad_commands, _ = self.gamepad.advance()

        x_input = float(pad_commands[0])
        y_input = float(pad_commands[1])
        yaw_input = float(pad_commands[5])

        dz = self.cfg.gamepad_deadzone

        if abs(x_input) < dz:
            x_input = 0.0
        if abs(y_input) < dz:
            y_input = 0.0
        if abs(yaw_input) < dz:
            yaw_input = 0.0

        self._commands[:, 0] = x_input * self.cfg.gamepad_max_lin_speed
        self._commands[:, 1] = y_input * self.cfg.gamepad_max_lin_speed
        self._commands[:, 2] = yaw_input * self.cfg.gamepad_max_yaw_rate

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = torch.clamp(actions.clone(), min=-1.0, max=1.0)

        action_scale = torch.full_like(
            self._actions,
            self.cfg.action_scale,
        )

        action_scale[:, self.turn_l_idx] = 0.08
        action_scale[:, self.turn_r_idx] = 0.08

        desired_targets = (
                self._robot.data.default_joint_pos
                + action_scale * self._actions
        )

        filtered_targets = (
                self.cfg.target_filter_alpha * desired_targets
                + (1.0 - self.cfg.target_filter_alpha) * self._current_targets
        )

        max_target_delta = 5.24 * self.step_dt

        target_delta = torch.clamp(
            filtered_targets - self._current_targets,
            min=-max_target_delta,
            max=max_target_delta,
        )

        self._current_targets = self._current_targets + target_delta

        if self.num_envs == 1:
            # TEST CHECKPOINTU - dokładnie ta sama komenda co podczas treningu
            self._commands[:, 0] = self.cfg.forward_command_x
            self._commands[:, 1] = 0.0
            self._commands[:, 2] = 0.0
        else:
            self._command_timer -= self.step_dt

            resample_ids = torch.nonzero(
                self._command_timer <= 0.0,
                as_tuple=False,
            ).flatten()

            if len(resample_ids) > 0:
                self._sample_training_commands(resample_ids)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._current_targets)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos
                - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
            ],
            dim=-1,
        )

        obs = torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        obs = torch.clamp(obs, min=-100.0, max=100.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Główny cel: 2D velocity tracking w układzie robota.
        forward_vel = self._robot.data.root_lin_vel_b[:, 0]
        lateral_vel = self._robot.data.root_lin_vel_b[:, 1]
        yaw_rate = self._robot.data.root_ang_vel_b[:, 2]

        # ============================================================
        # RUCH WZGLĘDEM KIERUNKU KOMENDY
        # Działa dla przód / tył / bok / skos.
        # ============================================================

        vel_xy = self._robot.data.root_lin_vel_b[:, :2]
        cmd_xy = self._commands[:, :2]

        command_speed = torch.norm(
            cmd_xy,
            dim=1,
        )

        move_gate = (
                command_speed > 0.05
        ).float()

        stand_gate = (
                (command_speed <= 0.05)
                & (torch.abs(self._commands[:, 2]) <= 0.05)
        ).float()

        forward_cmd_gate = (
                self._commands[:, 0] > 0.05
        ).float()

        backward_cmd_gate = (
                self._commands[:, 0] < -0.05
        ).float()

        safe_command_speed = torch.clamp(
            command_speed,
            min=0.01,
        )

        cmd_dir = (
                cmd_xy
                / safe_command_speed.unsqueeze(1)
        )

        cmd_dir *= move_gate.unsqueeze(1)

        # Prędkość dokładnie W KIERUNKU zadanej komendy.
        vel_along_cmd = torch.sum(
            vel_xy * cmd_dir,
            dim=1,
        )

        # Prędkość PROSTOPADŁA do zadanej komendy.
        # Forward -> to jest vy.
        # Backward -> -vy.
        # Bok -> głównie vx.
        vel_cross_cmd = (
                cmd_dir[:, 0] * vel_xy[:, 1]
                - cmd_dir[:, 1] * vel_xy[:, 0]
        )

        # ============================================================
        # DEBUG POZYCJI STAWÓW WZGLĘDEM POZYCJI NEUTRALNEJ
        # ============================================================

        joint_delta_all = (
                self._robot.data.joint_pos
                - self._robot.data.default_joint_pos
        )
        # lateral_factor = torch.exp(
        #     -torch.square(lateral_vel / self.cfg.straightness_lateral_sigma)
        # )
        #
        # yaw_factor = torch.exp(
        #     -torch.square(yaw_rate / self.cfg.straightness_yaw_sigma)
        # )
        #
        # straightness_gate = lateral_factor * yaw_factor
        #
        # step_gate = (
        #         self.cfg.step_straightness_floor
        #         + (1.0 - self.cfg.step_straightness_floor) * straightness_gate
        # )
        #
        # progress_gate = (
        #         self.cfg.progress_straightness_floor
        #         + (1.0 - self.cfg.progress_straightness_floor) * straightness_gate
        # )




        # alpha = self.cfg.lateral_bias_alpha
        #
        # self._lateral_vel_ema = (
        #         (1.0 - alpha) * self._lateral_vel_ema
        #         + alpha * lateral_vel
        # )
        # self._lateral_vel_ema = torch.zeros(
        #     self.num_envs,
        #     device=self.device,
        # )
        # forward_vel_error = torch.square(
        #     self.cfg.forward_command_x - forward_vel
        # )
        #
        # track_lin_vel_xy = torch.exp(
        #     -forward_vel_error / self.cfg.lin_vel_tracking_sigma
        # )

        lin_vel_error = (
                torch.square(self._commands[:, 0] - forward_vel)
                + torch.square(self._commands[:, 1] - lateral_vel)
        )

        track_lin_vel_xy = torch.exp(
            -lin_vel_error / self.cfg.lin_vel_tracking_sigma
        )

        yaw_error = torch.square(
            self._commands[:, 2]
            - self._robot.data.root_ang_vel_b[:, 2]
        )
        track_yaw_vel = torch.exp(
            -yaw_error / self.cfg.yaw_vel_tracking_sigma
        )
        # lateral_factor = torch.exp(
        #     -torch.square(
        #         lateral_vel / self.cfg.straightness_lateral_sigma
        #     )
        # )

        lateral_factor = torch.exp(
            -torch.square(
                vel_cross_cmd
                / self.cfg.straightness_lateral_sigma
            )
        )

        # yaw jest bezpośrednio obserwowalny przez policy z IMU,
        # więc krok może być mocno uzależniony od poprawnego yaw-rate.
        yaw_factor = track_yaw_vel

        # Debug ogólnej jakości kierunku.
        straightness_gate = lateral_factor * yaw_factor

        # KROK:
        # nie blokujemy kroku przez vy,
        # ale krok przy skręcaniu bez komendy yaw przestaje być opłacalny.
        step_gate = yaw_factor

        # PROGRESS:
        # lateral obcina nagrodę za postęp,
        # ale zostawiamy 50% floor, żeby nie zabić chodu.
        progress_gate = (
                0.50
                + 0.50 * lateral_factor
        )
        # ============================================================
        # KOMENDA RUCHU / JAKOŚĆ RUCHU W KIERUNKU KOMENDY
        # ============================================================

        # command_speed = torch.norm(
        #     self._commands[:, :2],
        #     dim=1,
        # )
        #
        # move_gate = (
        #         command_speed > 0.05
        # ).float()
        #
        # stand_gate = (
        #         (command_speed <= 0.05)
        #         & (torch.abs(self._commands[:, 2]) <= 0.05)
        # ).float()
        # ============================================================
        # STANDING
        # Aktywne wyłącznie dla zerowej komendy.
        # ============================================================

        stand_pose_excess = torch.clamp(
            torch.abs(joint_delta_all) - 0.10,
            min=0.0,
        )

        stand_pose_error = torch.mean(
            torch.square(stand_pose_excess),
            dim=1,
        )
        stand_tilt_error = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]),
            dim=1,
        )
        stand_action_error = torch.mean(
            torch.square(self._actions),
            dim=1,
        )

        # stand_pose_quality = torch.exp(
        #     -stand_pose_error / self.cfg.stand_pose_sigma
        # )

        base_height = self._robot.data.root_pos_w[:, 2]

        stand_height_quality = torch.exp(
            -torch.square(
                (base_height - self.cfg.base_height_target)
                / self.cfg.stand_height_sigma
            )
        )

        stand_joint_vel_error = torch.mean(
            torch.square(self._robot.data.joint_vel),
            dim=1,
        )
        stand_lin_vel_error = torch.sum(
            torch.square(self._robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )

        stand_yaw_vel_error = torch.square(
            self._robot.data.root_ang_vel_b[:, 2]
        )
        # ============================================================
        # DŁUGOOKRESOWY DRYF BOCZNY
        # Szybkie kołysanie L/P się wygładza.
        # Stały dryf w jedną stronę pozostaje.
        # ============================================================

        # lateral_error = (
        #         lateral_vel
        #         - self._commands[:, 1]
        # )

        lateral_error = vel_cross_cmd

        alpha = self.cfg.lateral_bias_alpha

        self._lateral_vel_ema = (
                (1.0 - alpha) * self._lateral_vel_ema
                + alpha * lateral_error
        )

        lateral_bias_error = torch.clamp(
            torch.abs(self._lateral_vel_ema)
            - self.cfg.lateral_bias_deadband,
            min=0.0,
        )
        bias_quality = torch.exp(
            -torch.square(
                self._lateral_vel_ema / 0.04
            )
        )

        bias_progress_gate = (
                0.35
                + 0.65 * bias_quality
        )

        # forward_quality = torch.clamp(
        #     forward_vel / self.cfg.forward_command_x,
        #     min=0.0,
        #     max=1.0,
        # )
        # safe_forward_command = torch.clamp(
        #     self._commands[:, 0],
        #     min=0.01,
        # )
        #
        # forward_quality = torch.clamp(
        #     forward_vel / safe_forward_command,
        #     min=0.0,
        #     max=1.0,
        # )
        #
        # # Przy komendzie 0.0 nie ma jakości "chodu".
        # forward_quality *= move_gate

        forward_quality = torch.clamp(
            vel_along_cmd / safe_command_speed,
            min=0.0,
            max=1.0,
        )

        forward_quality *= move_gate

        # ============================================================
        # BOOTSTRAP CHODU DO TYŁU
        # ============================================================

        backward_cmd_gate = (
                self._commands[:, 0] < -0.05
        ).float()

        # 0 = brak cofania
        # 1 = osiągnięta lub przekroczona zadana prędkość cofania
        backward_progress_quality = (
                forward_quality
                * backward_cmd_gate
        )

        # Podczas nauki tyłu nie płacimy pełnej nagrody
        # za samo dreptanie w miejscu.
        backward_motion_gate = (
                1.0
                - backward_cmd_gate
                + backward_cmd_gate
                * (
                        0.25
                        + 0.75 * backward_progress_quality
                )
        )

        motion_quality = forward_quality

        # straightness_gate = (
        #         0.25
        #         + 0.75 * forward_quality
        # )

        # lateral_quality = torch.exp(
        #     -torch.square(lateral_vel)
        #     / self.cfg.lateral_vel_sigma
        # )

        # motion_quality = (
        #         forward_quality
        #         * lateral_quality
        # )
        motion_quality = forward_quality

        gait_gate = (
                0.30
                + 0.70 * motion_quality
        )

        orientation_error = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]),
            dim=1,
        )
        # ============================================================
        # STAŁY PRZECHYŁ BAZY W ROLL
        # projected_gravity Y ~ boczny przechył bazy
        # Szybkie kołysanie podczas chodu wygładza EMA.
        # ============================================================

        roll_signal = self._robot.data.projected_gravity_b[:, 1]

        roll_alpha = self.cfg.roll_bias_alpha

        self._roll_ema = (
                (1.0 - roll_alpha) * self._roll_ema
                + roll_alpha * roll_signal
        )

        roll_bias_error = torch.clamp(
            torch.abs(self._roll_ema)
            - self.cfg.roll_bias_deadband,
            min=0.0,
        )

        roll_quality = torch.exp(
            -torch.square(
                self._roll_ema / self.cfg.roll_bias_sigma
            )
        )

        roll_progress_gate = (
                0.50
                + 0.50 * roll_quality
        )
        step_drift_gate = (
                0.60
                + 0.40 * bias_quality
        )

        step_roll_gate = (
                0.65
                + 0.35 * roll_quality
        )

        step_direction_gate = (
                yaw_factor
                * step_drift_gate
                * step_roll_gate
        )

        lin_vel_z_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]),
            dim=1,
        )

        left_contact = (
            self._left_foot_contact.data.current_contact_time[:, 0] > 0.0
        )
        right_contact = (
            self._right_foot_contact.data.current_contact_time[:, 0] > 0.0
        )
        # ============================================================
        # OBCIĄŻENIE STÓP
        # ============================================================

        left_contact_force = torch.abs(
            self._left_foot_contact.data.net_forces_w[:, 0, 2]
        )

        right_contact_force = torch.abs(
            self._right_foot_contact.data.net_forces_w[:, 0, 2]
        )

        total_contact_force = (
                left_contact_force
                + right_contact_force
                + 1.0e-6
        )

        # 0.0 = obie nogi obciążone podobnie
        # 1.0 = praktycznie cały ciężar na jednej nodze
        load_transfer_quality = torch.abs(
            left_contact_force - right_contact_force
        ) / total_contact_force

        load_transfer_shaped = torch.clamp(
            (load_transfer_quality - 0.05) / 0.65,
            min=0.0,
            max=1.0,
        )
        # ============================================================
        # NATURALNY SWING NOGI - BEZ METRONOMU
        # ============================================================

        pos_l = self._robot.data.body_pos_w[
            :, self.stopa_l_idx, :
        ]

        pos_r = self._robot.data.body_pos_w[
            :, self.stopa_r_idx, :
        ]
        # ============================================================
        # PŁASKOŚĆ STÓP PODCZAS STANIA
        #
        # Zakładamy, że lokalna oś Z bryły stopy jest normalną
        # do podeszwy.
        #
        # Dla płaskiej stopy lokalne Z po obrocie powinno być
        # równoległe do światowego Z.
        # ============================================================

        quat_l = self._robot.data.body_quat_w[:, self.stopa_l_idx, :]
        quat_r = self._robot.data.body_quat_w[:, self.stopa_r_idx, :]

        # Isaac Lab: quaternion = [w, x, y, z]
        wl, xl, yl, zl = quat_l.unbind(dim=1)
        wr, xr, yr, zr = quat_r.unbind(dim=1)

        # Składowa Z lokalnej osi Z stopy po obrocie do świata.
        left_foot_normal_z = (
                1.0 - 2.0 * (xl * xl + yl * yl)
        )

        right_foot_normal_z = (
                1.0 - 2.0 * (xr * xr + yr * yr)
        )

        left_foot_normal_z = torch.clamp(
            left_foot_normal_z,
            min=-1.0,
            max=1.0,
        )

        right_foot_normal_z = torch.clamp(
            right_foot_normal_z,
            min=-1.0,
            max=1.0,
        )

        # 0.0 = stopa idealnie poziomo
        # 1.0 = stopa obrócona o 90 stopni.
        left_foot_flat_raw = (
                1.0 - torch.square(left_foot_normal_z)
        )

        right_foot_flat_raw = (
                1.0 - torch.square(right_foot_normal_z)
        )

        # Mały deadband: nie wymagamy laboratoryjnego 0.000 stopnia.
        flat_deadband = math.sin(
            math.radians(self.cfg.stand_foot_flat_deadband_deg)
        ) ** 2

        left_foot_flat_error = torch.clamp(
            left_foot_flat_raw - flat_deadband,
            min=0.0,
        )

        right_foot_flat_error = torch.clamp(
            right_foot_flat_raw - flat_deadband,
            min=0.0,
        )

        # Obie stopy są równie ważne.
        stand_foot_flat_error = 0.5 * (
                left_foot_flat_error
                + right_foot_flat_error
        )
        # Swing istnieje TYLKO wtedy, gdy:
        #
        # lewa nie dotyka + prawa naprawdę podpiera
        # albo
        # prawa nie dotyka + lewa naprawdę podpiera
        #
        # Bunny-hop obu nóg NIE dostaje tutaj rewardu.

        left_swing = (~left_contact) & right_contact
        right_swing = (~right_contact) & left_contact
        # ============================================================
        # DŁUGOOKRESOWY BALANS UŻYWANIA LEWEJ / PRAWEJ NOGI
        # Nie narzuca fazy ani częstotliwości chodu.
        # ============================================================

        alpha = self.cfg.swing_balance_alpha

        self._left_swing_ema = (
                (1.0 - alpha) * self._left_swing_ema
                + alpha * left_swing.float()
        )

        self._right_swing_ema = (
                (1.0 - alpha) * self._right_swing_ema
                + alpha * right_swing.float()
        )

        swing_balance_error = torch.abs(
            self._left_swing_ema
            - self._right_swing_ema
        )
        overlong_swing = (
                (
                        left_swing
                        & (self._left_swing_time > self.cfg.swing_max_time)
                )
                |
                (
                        right_swing
                        & (self._right_swing_time > self.cfg.swing_max_time)
                )
        ).float()

        overlong_swing *= move_gate
        # ============================================================
        # DUCK KNEE FLEX
        #
        # Poprawne zgięcie konstrukcji:
        # LEWE  = minus
        # PRAWE = plus
        #
        # obrot8 -> lewe kolano
        # obrot7 -> prawe kolano
        # ============================================================

        knee_l = self._robot.data.joint_pos[:, self.knee_l_idx]
        knee_r = self._robot.data.joint_pos[:, self.knee_r_idx]

        default_knee_l = self._robot.data.default_joint_pos[:, self.knee_l_idx]
        default_knee_r = self._robot.data.default_joint_pos[:, self.knee_r_idx]
        left_knee_delta = knee_l - default_knee_l
        right_knee_delta = knee_r - default_knee_r

        # ============================================================
        # SYMETRIA KOLAN PODCZAS STANIA
        #
        # Kierunki zgięcia kolan są przeciwne:
        # lewe  -> dodatni delta
        # prawe -> ujemny delta
        # dlatego dla porównania odwracamy znak prawego.
        # ============================================================

        stand_left_knee_flex = left_knee_delta
        stand_right_knee_flex = -right_knee_delta

        stand_knee_sym_diff = torch.abs(
            stand_left_knee_flex - stand_right_knee_flex
        )

        # Małe różnice są dozwolone, żeby robot mógł balansować.
        stand_knee_sym_excess = torch.clamp(
            stand_knee_sym_diff - self.cfg.stand_knee_sym_deadband,
            min=0.0,
        )

        stand_knee_sym_error = torch.square(
            stand_knee_sym_excess
        )

        left_knee_flex = torch.clamp(
            left_knee_delta,
            min=0.0,
        )

        right_knee_flex = torch.clamp(
            -right_knee_delta,
            min=0.0,
        )

        # DODATKOWE zgięcie względem crouch neutral.
        #
        # LEWE:
        # +0.85 -> +1.00 = dodatkowe +0.15 rad zgięcia
        # left_knee_flex = torch.clamp(
        #     knee_l - default_knee_l,
        #     min=0.0,
        # )
        #
        # # PRAWE:
        # # -0.85 -> -1.00 = dodatkowe +0.15 rad zgięcia
        # right_knee_flex = torch.clamp(
        #     default_knee_r - knee_r,
        #     min=0.0,
        # )

        left_knee_quality = torch.clamp(
            left_knee_flex / self.cfg.swing_knee_flex_target,
            0.0,
            1.0,
        )

        right_knee_quality = torch.clamp(
            right_knee_flex / self.cfg.swing_knee_flex_target,
            0.0,
            1.0,
        )
        knee_alpha = self.cfg.knee_balance_alpha

        self._left_knee_flex_ema = torch.where(
            left_swing,
            (1.0 - knee_alpha) * self._left_knee_flex_ema
            + knee_alpha * left_knee_quality,
            self._left_knee_flex_ema,
        )

        self._right_knee_flex_ema = torch.where(
            right_swing,
            (1.0 - knee_alpha) * self._right_knee_flex_ema
            + knee_alpha * right_knee_quality,
            self._right_knee_flex_ema,
        )
        knee_pair_quality = torch.minimum(
            self._left_knee_flex_ema,
            self._right_knee_flex_ema,
        )
        knee_flex_balance_error = torch.abs(
            self._left_knee_flex_ema
            - self._right_knee_flex_ema
        )
        swing_knee_flex = (
                left_swing.float() * left_knee_quality
                + right_swing.float() * right_knee_quality
        )

        single_support_debug = (
                left_swing | right_swing
        ).float()

        double_support_debug = (
                left_contact & right_contact
        ).float()

        foot_height_diff_debug = torch.abs(
            pos_l[:, 2] - pos_r[:, 2]
        )
        knee_step_gate = (
                0.70
                + 0.30 * knee_pair_quality
        )
        # ------------------------------------------------------------
        # WYSOKOŚĆ NOGI SWING WZGLĘDEM NOGI PODPOROWEJ
        #
        # Nie używamy absolutnego Z świata.
        # Jeśli prawa stoi na ziemi, patrzymy ile wyżej jest lewa.
        # ------------------------------------------------------------

        left_clearance = torch.clamp(
            pos_l[:, 2] - pos_r[:, 2],
            min=0.0,
        )

        right_clearance = torch.clamp(
            pos_r[:, 2] - pos_l[:, 2],
            min=0.0,
        )

        # ------------------------------------------------------------
        # CZAS SWING
        # ------------------------------------------------------------

        self._left_swing_time = torch.where(
            left_swing,
            self._left_swing_time + self.step_dt,
            self._left_swing_time,
        )

        self._right_swing_time = torch.where(
            right_swing,
            self._right_swing_time + self.step_dt,
            self._right_swing_time,
        )

        # ------------------------------------------------------------
        # NAJWIĘKSZY CLEARANCE PODCZAS TEGO SWING
        # ------------------------------------------------------------

        self._left_max_clearance = torch.where(
            left_swing,
            torch.maximum(
                self._left_max_clearance,
                left_clearance,
            ),
            self._left_max_clearance,
        )

        self._right_max_clearance = torch.where(
            right_swing,
            torch.maximum(
                self._right_max_clearance,
                right_clearance,
            ),
            self._right_max_clearance,
        )
        # ============================================================
        # DENSE DISCOVERY REWARD
        #
        # Daje policy gradient zanim nauczy się pełnego kroku.
        # Reward rośnie do 4 cm, a POTEM SIĘ SATURUJE.
        #
        # Nie ma żadnej korzyści z unoszenia nogi na 8 czy 12 cm.
        # ============================================================

        # ============================================================
        # PRE-LIFT -> SWING
        #
        # Robot dostaje małą wskazówkę jeszcze zanim całkowicie
        # oderwie stopę od ziemi.
        #
        # Po prawdziwym przejściu na jedną nogę reward robi się
        # dużo większy.
        # ============================================================

        foot_height_diff = torch.abs(
            pos_l[:, 2] - pos_r[:, 2]
        )

        double_support = left_contact & right_contact

        stand_double_contact = double_support.float()

        stand_single_support = (
                left_contact ^ right_contact
        ).float()
        # ============================================================
        # PRZEDŁUŻONY DOUBLE SUPPORT
        #
        # Krótki double support jest normalny w chodzie.
        # Karzemy tylko sytuację:
        # "jadę do przodu i od dawna obie stopy są na ziemi".
        # ============================================================

        self._double_support_time = torch.where(
            double_support,
            self._double_support_time + self.step_dt,
            torch.zeros_like(self._double_support_time),
        )

        double_support_stuck = torch.clamp(
            (
                    self._double_support_time
                    - self.cfg.double_support_grace_time
            )
            / (
                    self.cfg.double_support_full_penalty_time
                    - self.cfg.double_support_grace_time
            ),
            min=0.0,
            max=1.0,
        )
        # 1.0 przez normalny, krótki double-support.
        # Przy ciągłym staniu > 0.5 s spada do 0.10.
        support_reward_gate = (
                1.0
                - 0.90 * double_support_stuck * move_gate
        )
        double_support_gate = (
                0.25
                + 0.75 * forward_quality
        )

        double_support_stuck = (
                double_support_stuck
                * move_gate
                * double_support_gate
        )

        # Nie karzemy zwykłego stania.
        # Kara rośnie tylko wtedy, gdy naprawdę próbuje jechać.
        # double_support_stuck = (
        #         double_support_stuck
        #         * move_gate
        #         * motion_quality
        # )
        # double_support_stuck = (
        #         double_support_stuck
        #         * move_gate
        # )
        single_support = left_swing | right_swing

        # ------------------------------------------------------------
        # ETAP 1: PRE-LIFT
        #
        # Już pierwsze milimetry różnicy wysokości dają sygnał.
        # Maksimum pre-lift osiąga przy 1 cm.
        #
        # Ten reward jest celowo słaby.
        # ------------------------------------------------------------

        # ============================================================
        # ETAP 1: TRANSFER CIĘŻARU
        #
        # Gdy obie stopy są jeszcze na ziemi, nagradzamy odciążanie
        # jednej z nich zamiast sztucznego podnoszenia originu stopy.
        #
        # Maksymalnie jest to mała nagroda - samo stanie przechylonym
        # na jedną nogę nie może wygrać zadania.
        # ============================================================

        # prelift_reward = (
        #         double_support.float()
        #         * 0.15
        #         * load_transfer_shaped
        # )
        prelift_reward = torch.zeros_like(load_transfer_shaped)
        # ------------------------------------------------------------
        # ETAP 2: PRAWDZIWY SWING
        #
        # Po utracie kontaktu jednej nogi celem staje się 4 cm.
        # ------------------------------------------------------------

        # swing_height_quality = torch.clamp(
        #     foot_height_diff / self.cfg.swing_clearance_target,
        #     min=0.0,
        #     max=1.0,
        # )
        #
        # left_time_gate = (
        #         self._left_swing_time <= self.cfg.swing_max_time
        # ).float()
        #
        # right_time_gate = (
        #         self._right_swing_time <= self.cfg.swing_max_time
        # ).float()

        # ============================================================
        # ETAP 2: PRAWDZIWY SINGLE SUPPORT
        #
        # Sam fakt oderwania jednej nogi daje już 50% jakości.
        # Kolejne 50% zależy od wysokości stopy.
        # ============================================================

        swing_height_quality = torch.clamp(
            foot_height_diff / self.cfg.swing_clearance_target,
            min=0.0,
            max=1.0,
        )

        left_time_gate = (
                self._left_swing_time <= self.cfg.swing_max_time
        ).float()

        right_time_gate = (
                self._right_swing_time <= self.cfg.swing_max_time
        ).float()

        left_swing_quality = (
                1.00
                + 1.00 * swing_height_quality
        )

        right_swing_quality = (
                1.00
                + 1.00 * swing_height_quality
        )
        # ============================================================
        # SWING REWARD Z WYMUSZENIEM NAPRZEMIENNOŚCI
        #
        # _last_rewarded_step:
        #  0  -> jeszcze nie ustalono strony, może zacząć dowolną nogą
        # +1  -> ostatnia była PRAWA, teraz nagradzamy LEWĄ
        # -1  -> ostatnia była LEWA, teraz nagradzamy PRAWĄ
        # ============================================================

        # no_step_yet = self._last_rewarded_step == 0
        #
        # allow_left_swing = (
        #         no_step_yet
        #         | (self._last_rewarded_step == 1)
        # )
        #
        # allow_right_swing = (
        #         no_step_yet
        #         | (self._last_rewarded_step == -1)
        # )
        #
        # # swing_reward = (
        # #         left_swing.float()
        # #         * left_swing_quality
        # #         * left_time_gate
        # #         * allow_left_swing.float()
        # #
        # #         +
        # #
        # #         right_swing.float()
        # #         * right_swing_quality
        # #         * right_time_gate
        # #         * allow_right_swing.float()
        # # )
        # swing_reward = (
        #         left_swing.float()
        #         * left_swing_quality
        #         * left_time_gate
        #
        #         +
        #
        #         right_swing.float()
        #         * right_swing_quality
        #         * right_time_gate
        # )
        swing_reward = (
                left_swing.float()
                * left_swing_quality
                * left_time_gate

                +

                right_swing.float()
                * right_swing_quality
                * right_time_gate
        )

        # ------------------------------------------------------------
        # Razem:
        #
        # obie stopy na ziemi -> mała marchewka za rozpoczęcie lift
        # jedna noga w powietrzu -> pełna marchewka za clearance
        # ------------------------------------------------------------

        # swing_clearance_quality = (
        #         prelift_reward
        #         + swing_reward
        # )
        swing_clearance_quality = swing_reward
        # Reward aktywny tylko wtedy, kiedy użytkownik chce ruchu.
        # command_speed = torch.norm(
        #     self._commands[:, :2],
        #     dim=1,
        # )
        #
        # move_gate = (
        #         command_speed > 0.05
        # ).float()
        #
        # # ============================================================
        # # CZY ROBOT FAKTYCZNIE JEDZIE W KIERUNKU KOMENDY
        # # ============================================================
        #
        # velocity_along_command = torch.sum(
        #     self._robot.data.root_lin_vel_b[:, :2]
        #     * self._commands[:, :2],
        #     dim=1,
        # ) / (command_speed + 1.0e-6)
        #
        # motion_quality = torch.clamp(
        #     velocity_along_command / (command_speed + 1.0e-6),
        #     min=0.0,
        #     max=1.0,
        # )
        #
        # # Nawet stojąc dostaje 10% sygnału discovery,
        # # ale pełny gait reward dostaje dopiero gdy naprawdę jedzie.
        # gait_gate = (
        #         0.30
        #         + 0.70 * motion_quality
        # )

        swing_clearance_quality *= move_gate * gait_gate
        # Jakość faktycznej realizacji zadanej prędkości.
        # Używana tylko przy nagrodzie za UKOŃCZONY krok.
        locomotion_gate = track_lin_vel_xy
        # ============================================================
        # UKOŃCZONY KROK
        # ============================================================

        left_touchdown = (
                left_contact
                & (~self._prev_left_contact)
        )

        right_touchdown = (
                right_contact
                & (~self._prev_right_contact)
        )
        left_valid_time = (
                (self._left_swing_time >= self.cfg.swing_min_time)
                &
                (self._left_swing_time <= self.cfg.swing_max_time)
        )

        right_valid_time = (
                (self._right_swing_time >= self.cfg.swing_min_time)
                &
                (self._right_swing_time <= self.cfg.swing_max_time)
        )
        left_step_clearance_quality = torch.clamp(
            self._left_max_clearance
            / self.cfg.swing_clearance_target,
            min=0.0,
            max=1.0,
        )

        right_step_clearance_quality = torch.clamp(
            self._right_max_clearance
            / self.cfg.swing_clearance_target,
            min=0.0,
            max=1.0,
        )

        left_step_quality = (
                0.50
                + 0.50 * left_step_clearance_quality
        )

        right_step_quality = (
                0.50
                + 0.50 * right_step_clearance_quality
        )

        # left_step_complete = (
        #         left_touchdown.float()
        #         * left_valid_time.float()
        #         * left_step_quality
        # )
        #
        # right_step_complete = (
        #         right_touchdown.float()
        #         * right_valid_time.float()
        #         * right_step_quality
        # )
        # ============================================================
        # TOUCHDOWN REWARD - GĘSTY, BEZ PROGU ZERO/JEDEN
        #
        # 0.033 s swing -> część nagrody
        # 0.066 s swing -> większa część
        # >= 0.08 s     -> pełna jakość czasu
        # ============================================================

        left_step_time_quality = torch.clamp(
            self._left_swing_time / self.cfg.swing_min_time,
            min=0.0,
            max=1.0,
        )

        right_step_time_quality = torch.clamp(
            self._right_swing_time / self.cfg.swing_min_time,
            min=0.0,
            max=1.0,
        )

        # Nie nagradzamy wiszenia wiecznie w powietrzu.
        left_step_time_quality *= (
                self._left_swing_time <= self.cfg.swing_max_time
        ).float()

        right_step_time_quality *= (
                self._right_swing_time <= self.cfg.swing_max_time
        ).float()

        left_step_complete = (
                left_touchdown.float()
                * left_step_time_quality
        )

        right_step_complete = (
                right_touchdown.float()
                * right_step_time_quality
        )

        self._episode_sums["left_step_debug"] += left_step_complete
        self._episode_sums["right_step_debug"] += right_step_complete

        # Krok ma uczyć samej mechaniki chodu.
        # Kierunek, yaw i prędkość mają własne rewardy.
        step_complete = (
                                left_step_complete
                                + right_step_complete
                        ) * move_gate * step_direction_gate * knee_step_gate *backward_motion_gate
        # self._episode_sums["left_step_debug"] += left_step_complete
        # self._episode_sums["right_step_debug"] += right_step_complete
        # step_motion_gate = (
        #         0.25
        #         + 0.75 * track_lin_vel_xy
        # )
        #
        # step_complete = (
        #                         left_step_complete
        #                         + right_step_complete
        #                 ) * move_gate * motion_quality
        # self._episode_sums["left_step_debug"] += left_step_complete
        # self._episode_sums["right_step_debug"] += right_step_complete

        # ============================================================
        # NAPRZEMIENNE NAGRADZANIE KROKÓW
        # ============================================================

        # left_step_event = left_step_complete > 0.0
        # right_step_event = right_step_complete > 0.0
        #
        # # -1 = ostatnio nagrodzona lewa
        # # +1 = ostatnio nagrodzona prawa
        # left_allowed = self._last_rewarded_step != -1
        # right_allowed = self._last_rewarded_step != 1
        #
        # left_step_rewarded = (
        #         left_step_complete
        #         * left_allowed.float()
        # )
        #
        # right_step_rewarded = (
        #         right_step_complete
        #         * right_allowed.float()
        # )
        #
        # left_accepted = left_step_event & left_allowed
        # right_accepted = right_step_event & right_allowed
        #
        # self._last_rewarded_step = torch.where(
        #     left_accepted,
        #     torch.full_like(self._last_rewarded_step, -1),
        #     torch.where(
        #         right_accepted,
        #         torch.full_like(self._last_rewarded_step, 1),
        #         self._last_rewarded_step,
        #     ),
        # )
        # left_step_event = left_step_complete > 0.0
        # right_step_event = right_step_complete > 0.0
        #
        # # ============================================================
        # # PIERWSZY KROK TYLKO USTALA STRONĘ
        # # NAGRODA DOPIERO ZA KROK PRZECIWNĄ NOGĄ
        # # ============================================================
        #
        # no_previous_step = self._last_rewarded_step == 0
        #
        # first_left = (
        #         no_previous_step
        #         & left_step_event
        #         & (~right_step_event)
        # )
        #
        # first_right = (
        #         no_previous_step
        #         & right_step_event
        #         & (~left_step_event)
        # )
        #
        # # Jeżeli ostatnio była prawa, teraz nagradzamy lewą.
        # left_rewarded_mask = (
        #         (self._last_rewarded_step == 1)
        #         & left_step_event
        # )
        #
        # # Jeżeli ostatnio była lewa, teraz nagradzamy prawą.
        # right_rewarded_mask = (
        #         (self._last_rewarded_step == -1)
        #         & right_step_event
        # )
        #
        # left_step_rewarded = (
        #         left_step_complete
        #         * left_rewarded_mask.float()
        # )
        #
        # right_step_rewarded = (
        #         right_step_complete
        #         * right_rewarded_mask.float()
        # )
        #
        # # Pierwszy krok tylko zapisuje stronę.
        # self._last_rewarded_step = torch.where(
        #     first_left,
        #     torch.full_like(self._last_rewarded_step, -1),
        #     self._last_rewarded_step,
        # )
        #
        # self._last_rewarded_step = torch.where(
        #     first_right,
        #     torch.full_like(self._last_rewarded_step, 1),
        #     self._last_rewarded_step,
        # )
        #
        # # Kolejne nagrodzone kroki muszą być naprzemienne.
        # self._last_rewarded_step = torch.where(
        #     left_rewarded_mask,
        #     torch.full_like(self._last_rewarded_step, -1),
        #     self._last_rewarded_step,
        # )
        #
        # self._last_rewarded_step = torch.where(
        #     right_rewarded_mask,
        #     torch.full_like(self._last_rewarded_step, 1),
        #     self._last_rewarded_step,
        # )
        #
        # # Krok jest wartościowy tylko gdy robot rzeczywiście realizuje
        # # komendę prędkości i NIE kręci się bez komendy yaw.
        # step_velocity_gate = (
        #         0.25
        #         + 0.75 * track_lin_vel_xy
        # )
        #
        # # step_complete = (
        # #                         left_step_rewarded
        # #                         + right_step_rewarded
        # #                 ) * move_gate * motion_quality * step_velocity_gate * track_yaw_vel
        #
        # step_discovery_gate = (
        #         0.25
        #         + 0.75 * motion_quality
        # )
        #
        # step_complete = (
        #                         left_step_rewarded
        #                         + right_step_rewarded
        #                 ) * move_gate * step_discovery_gate * step_velocity_gate * track_yaw_vel

        both_feet_air = (~left_contact & ~right_contact).float()

        vel_l_xy = self._robot.data.body_lin_vel_w[:, self.stopa_l_idx, :2]
        vel_r_xy = self._robot.data.body_lin_vel_w[:, self.stopa_r_idx, :2]

        foot_slip = (
            left_contact.float() * torch.sum(torch.square(vel_l_xy), dim=1)
            + right_contact.float() * torch.sum(torch.square(vel_r_xy), dim=1)
        )
        # ============================================================
        # RESET HISTORII PO TOUCHDOWN
        # ============================================================

        self._left_swing_time = torch.where(
            left_touchdown,
            torch.zeros_like(self._left_swing_time),
            self._left_swing_time,
        )

        self._right_swing_time = torch.where(
            right_touchdown,
            torch.zeros_like(self._right_swing_time),
            self._right_swing_time,
        )

        self._left_max_clearance = torch.where(
            left_touchdown,
            torch.zeros_like(self._left_max_clearance),
            self._left_max_clearance,
        )

        self._right_max_clearance = torch.where(
            right_touchdown,
            torch.zeros_like(self._right_max_clearance),
            self._right_max_clearance,
        )

        self._prev_left_contact = left_contact.clone()
        self._prev_right_contact = right_contact.clone()

        # root_pos = self._robot.data.root_pos_w
        #
        # delta_pos = root_pos[:, :2] - self._prev_root_pos[:, :2]
        #
        # self._prev_root_pos = root_pos.clone()
        # forward_progress = delta_pos[:, 0] / self.step_dt
        # forward_progress = forward_vel
        # progress_quality = torch.clamp(
        #     forward_progress / self.cfg.forward_command_x,
        #     min=-1.0,
        #     max=1.0,
        # )

        # forward_progress = forward_vel
        #
        # progress_quality = torch.clamp(
        #     forward_progress / safe_forward_command,
        #     min=-1.0,
        #     max=1.0,
        # )

        forward_progress = vel_along_cmd

        progress_quality = torch.clamp(
            forward_progress / safe_command_speed,
            min=-1.0,
            max=1.0,
        )

        upright_gate = torch.clamp(
            1.0 - orientation_error / 0.25,
            min=0.0,
            max=1.0,
        )
        # progress_direction_gate = (
        #         0.25
        #         + 0.75 * track_lin_vel_xy
        # )

        progress_quality *= upright_gate

        progress_yaw_gate = (
                0.10
                + 0.90 * track_yaw_vel
        )

        progress_quality *= progress_yaw_gate

        # progress_quality *= progress_direction_gate

        # progress_quality *= upright_gate

        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions),
            dim=1,
        )

        dt = self.step_dt

        # lateral_excess = torch.clamp(
        #     torch.abs(lateral_vel) - self.cfg.lateral_vel_deadzone,
        #     min=0.0,
        # )
        yaw_rate = self._robot.data.root_ang_vel_b[:, 2]

        # obrot10_idx = self.debug_joint_indices["obrot10"]
        #
        # stand_obrot10_delta = joint_delta_all[:, obrot10_idx]
        #
        # stand_obrot10_excess = torch.clamp(
        #     torch.abs(stand_obrot10_delta) - 0.10,
        #     min=0.0,
        # )
        #
        # stand_obrot10_error = torch.square(
        #     stand_obrot10_excess
        # )

        obrot9_idx = self.debug_joint_indices["obrot9"]
        obrot10_idx = self.debug_joint_indices["obrot10"]

        obrot9_delta = joint_delta_all[:, obrot9_idx]
        obrot10_delta = joint_delta_all[:, obrot10_idx]

        obrot9_abs = torch.abs(obrot9_delta)
        obrot10_abs = torch.abs(obrot10_delta)

        # Kara za brak symetrii obu "bliźniaczych" stawów podczas stania.
        stand_ankle_sym_error = torch.square(
            torch.clamp(
                torch.abs(obrot9_abs - obrot10_abs) - self.cfg.stand_ankle_sym_deadband,
                min=0.0,
            )
        )

        # Dodatkowa kara tylko wtedy, gdy obrot9 jest bardziej zagięty niż obrot10.
        # stand_obrot9_extra_error = torch.square(
        #     torch.clamp(
        #         obrot9_abs - obrot10_abs - 0.01,
        #         min=0.0,
        #     )
        # )

        left_foot_tilt_deg = (
                torch.acos(
                    torch.clamp(
                        torch.abs(left_foot_normal_z),
                        min=0.0,
                        max=1.0,
                    )
                )
                * (180.0 / math.pi)
        )

        right_foot_tilt_deg = (
                torch.acos(
                    torch.clamp(
                        torch.abs(right_foot_normal_z),
                        min=0.0,
                        max=1.0,
                    )
                )
                * (180.0 / math.pi)
        )

        backward_speed = torch.clamp(
            -forward_vel,
            min=0.0,
        )

        backward_target_speed = torch.clamp(
            -self._commands[:, 0],
            min=0.01,
        )

        backward_speed_ratio = torch.clamp(
            backward_speed / backward_target_speed,
            min=0.0,
            max=1.0,
        )

        backward_discovery_quality = backward_cmd_gate * (
                0.35 * torch.tanh(backward_speed / 0.012)
                + 0.65 * backward_speed_ratio
        )

        backward_velocity_error = (
                                          forward_vel - self._commands[:, 0]
                                  ) ** 2

        backward_track_quality = (
                torch.exp(
                    -backward_velocity_error
                    / self.cfg.backward_tracking_sigma
                )
                * backward_cmd_gate
        )

        self._episode_sums["stand_left_foot_tilt_deg_debug"] += (
                left_foot_tilt_deg
                * stand_gate
                * self.step_dt
        )

        self._episode_sums["stand_right_foot_tilt_deg_debug"] += (
                right_foot_tilt_deg
                * stand_gate
                * self.step_dt
        )

        rewards = {
            "alive": (
                self.cfg.rew_scale_alive
                * (1.0 - self.reset_terminated.float())
                * dt
            ),
            # Jednorazowa kara terminalna.
            "terminated": (
                self.cfg.rew_scale_terminated
                * self.reset_terminated.float()
            ),
            "track_lin_vel_xy": (
                self.cfg.rew_scale_track_lin_vel_xy
                * track_lin_vel_xy
                * support_reward_gate
                * dt
            ),
            "track_yaw_vel": (
                self.cfg.rew_scale_track_yaw_vel
                * track_yaw_vel
                * support_reward_gate
                * dt
            ),
            "orientation": (
                self.cfg.rew_scale_orientation
                * orientation_error
                * dt
            ),
            "lin_vel_z": (
                self.cfg.rew_scale_lin_vel_z
                * lin_vel_z_error
                * dt
            ),
            "ang_vel_xy": (
                self.cfg.rew_scale_ang_vel_xy
                * ang_vel_xy_error
                * dt
            ),
            "both_feet_air": (
                self.cfg.rew_scale_both_feet_air
                * both_feet_air
                * dt
            ),
            "foot_slip": (
                self.cfg.rew_scale_foot_slip
                * foot_slip
                * dt
            ),
            "action_rate": (
                self.cfg.rew_scale_action_rate
                * action_rate
                * dt
            ),
            "swing_clearance":
                self.cfg.rew_scale_swing_clearance
                * swing_clearance_quality
                * self.step_dt,

            "step_complete":
                self.cfg.rew_scale_step_complete
                * step_complete,
            "forward_progress":
                self.cfg.rew_scale_forward_progress
                * progress_quality
                * move_gate
                * support_reward_gate
                * progress_gate
                * bias_progress_gate
                * roll_progress_gate
                * self.step_dt,
            "swing_knee_flex": (
                    self.cfg.rew_scale_swing_knee_flex
                    * swing_knee_flex
                    * move_gate
                    * gait_gate
                    * self.step_dt
            ),
            "double_support_stuck": (
                    self.cfg.rew_scale_double_support_stuck
                    * double_support_stuck
                    * self.step_dt
            ),
            "overlong_swing": (
                    self.cfg.rew_scale_overlong_swing
                    * overlong_swing
                    * self.step_dt
            ),
            # "lateral_vel": (
            #         self.cfg.rew_scale_lateral_vel
            #         * torch.square(lateral_vel)
            #         * dt
            # ),
            "lateral_vel": (
                    self.cfg.rew_scale_lateral_vel
                    * torch.square(vel_cross_cmd)
                    * move_gate
                    * dt
            ),
            # "yaw_rate": (
            #         self.cfg.rew_scale_yaw_rate
            #         * torch.square(yaw_rate)
            #         * straightness_gate
            #         * dt
            # ),
            "yaw_rate": (
                    self.cfg.rew_scale_yaw_rate
                    * yaw_error
                    * dt
            ),
            "swing_balance": (
                    self.cfg.rew_scale_swing_balance
                    * swing_balance_error
                    * move_gate
                    * dt
            ),
            "lateral_bias": (
                    self.cfg.rew_scale_lateral_bias
                    * lateral_bias_error
                    * move_gate
                    * dt
            ),
            # "lateral_bias": (
            #         self.cfg.rew_scale_lateral_bias
            #         * torch.square(self._lateral_vel_ema)
            #         * dt
            # ),
            "knee_flex_balance": (
                    self.cfg.rew_scale_knee_flex_balance
                    * knee_flex_balance_error
                    * move_gate
                    * dt
            ),
            "knee_pair_use": (
                    self.cfg.rew_scale_knee_pair_use
                    * knee_pair_quality
                    * move_gate
                    * backward_motion_gate
                    * dt
            ),
            "roll_bias": (
                    self.cfg.rew_scale_roll_bias
                    * roll_bias_error
                    * dt
            ),
            "stand_pose": (
                    self.cfg.rew_scale_stand_pose
                    * stand_pose_error
                    * stand_gate
                    * dt
            ),

            "stand_height": (
                    self.cfg.rew_scale_stand_height
                    * stand_height_quality
                    * stand_gate
                    * dt
            ),

            "stand_joint_vel": (
                    self.cfg.rew_scale_stand_joint_vel
                    * stand_joint_vel_error
                    * stand_gate
                    * dt
            ),

            "stand_double_contact": (
                    self.cfg.rew_scale_stand_double_contact
                    * stand_double_contact
                    * stand_gate
                    * dt
            ),
            "stand_lin_vel": (
                    self.cfg.rew_scale_stand_lin_vel
                    * stand_lin_vel_error
                    * stand_gate
                    * self.step_dt
            ),

            "stand_yaw_vel": (
                    self.cfg.rew_scale_stand_yaw_vel
                    * stand_yaw_vel_error
                    * stand_gate
                    * self.step_dt
            ),
            "stand_upright": (
                    self.cfg.rew_scale_stand_upright
                    * stand_tilt_error
                    * stand_gate
                    * self.step_dt
            ),
            "stand_action": (
                    self.cfg.rew_scale_stand_action
                    * stand_action_error
                    * stand_gate
                    * self.step_dt
            ),
            "stand_single_support": (
                    self.cfg.rew_scale_stand_single_support
                    * stand_single_support
                    * stand_gate
                    * self.step_dt
            ),
            "stand_knee_sym": (
                    self.cfg.rew_scale_stand_knee_sym
                    * stand_knee_sym_error
                    * stand_gate
                    * self.step_dt
            ),
            # "stand_obrot10": (
            #         self.cfg.rew_scale_stand_obrot10
            #         * stand_obrot10_error
            #         * stand_gate
            #         * self.step_dt
            # ),
            "stand_ankle_sym": (
                    self.cfg.rew_scale_stand_ankle_sym
                    * stand_ankle_sym_error
                    * stand_gate
                    * self.step_dt
            ),

            # "stand_obrot9_extra": (
            #         self.cfg.rew_scale_stand_obrot9_extra
            #         * stand_obrot9_extra_error
            #         * stand_gate
            #         * self.step_dt
            # ),
            "stand_foot_flat": (
                    self.cfg.rew_scale_stand_foot_flat
                    * stand_foot_flat_error
                    * stand_gate
                    * self.step_dt
            ),
            # "backward_discovery": (
            #         self.cfg.rew_scale_backward_discovery
            #         * backward_progress_quality
            #         * upright_gate
            #         * self.step_dt
            # ),
            "backward_discovery": (
                    self.cfg.rew_scale_backward_discovery
                    * backward_discovery_quality
                    * upright_gate
                    * self.step_dt
            ),

            "backward_track": (
                    self.cfg.rew_scale_backward_track
                    * backward_track_quality
                    * upright_gate
                    * self.step_dt
            ),

        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)


        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["single_support_debug"] += (
                single_support_debug * self.step_dt
        )

        self._episode_sums["double_support_debug"] += (
                double_support_debug * self.step_dt
        )

        self._episode_sums["foot_height_diff_debug"] += (
                foot_height_diff_debug * self.step_dt
        )
        self._episode_sums["load_transfer_debug"] += (
                load_transfer_quality * self.step_dt
        )
        self._episode_sums["swing_knee_flex_debug"] += (
                swing_knee_flex * self.step_dt
        )
        self._episode_sums["knee_l_pos_debug"] += (
                left_knee_delta * self.step_dt
        )

        self._episode_sums["knee_r_pos_debug"] += (
                right_knee_delta * self.step_dt
        )

        self._episode_sums["knee_l_abs_debug"] += (
                torch.abs(left_knee_delta) * self.step_dt
        )

        self._episode_sums["knee_r_abs_debug"] += (
                torch.abs(right_knee_delta) * self.step_dt
        )
        self._episode_sums["left_contact_debug"] += (
                left_contact.float() * self.step_dt
        )

        self._episode_sums["right_contact_debug"] += (
                right_contact.float() * self.step_dt
        )
        self._episode_sums["lateral_speed_debug"] += (
                torch.abs(lateral_vel) * self.step_dt
        )
        self._episode_sums["yaw_rate_debug"] += (
                torch.abs(yaw_rate) * self.step_dt
        )
        self._episode_sums["lateral_signed_debug"] += (
                lateral_vel * self.step_dt
        )

        self._episode_sums["yaw_rate_signed_debug"] += (
                yaw_rate * self.step_dt
        )
        self._episode_sums["left_swing_debug"] += (
                left_swing.float() * self.step_dt
        )

        self._episode_sums["right_swing_debug"] += (
                right_swing.float() * self.step_dt
        )
        self._episode_sums["straightness_gate_debug"] += (
                straightness_gate * self.step_dt
        )

        self._episode_sums["step_gate_debug"] += (
                step_gate * self.step_dt
        )

        self._episode_sums["progress_gate_debug"] += (
                progress_gate * self.step_dt
        )
        self._episode_sums["lateral_bias_debug"] += (
                torch.abs(self._lateral_vel_ema) * self.step_dt
        )
        self._episode_sums["bias_progress_gate_debug"] += (
                bias_progress_gate * self.step_dt
        )
        self._episode_sums["left_knee_flex_ema_debug"] += (
                self._left_knee_flex_ema * self.step_dt
        )

        self._episode_sums["right_knee_flex_ema_debug"] += (
                self._right_knee_flex_ema * self.step_dt
        )
        self._episode_sums["knee_pair_use_debug"] += (
                knee_pair_quality * self.step_dt
        )
        self._episode_sums["roll_signed_debug"] += (
                roll_signal * self.step_dt
        )

        self._episode_sums["roll_abs_debug"] += (
                torch.abs(roll_signal) * self.step_dt
        )

        self._episode_sums["roll_ema_debug"] += (
                self._roll_ema * self.step_dt
        )

        self._episode_sums["roll_progress_gate_debug"] += (
                roll_progress_gate * self.step_dt
        )

        # ============================================================
        # DEBUG WSZYSTKICH 10 STAWÓW
        #
        # delta = średnie podpisane odchylenie od pozycji neutralnej
        # abs   = średnia wielkość pracy danego stawu
        # ============================================================

        for joint_name, joint_idx in self.debug_joint_indices.items():
            joint_delta = joint_delta_all[:, joint_idx]

            self._episode_sums[f"{joint_name}_delta_debug"] += (
                    joint_delta * self.step_dt
            )

            self._episode_sums[f"{joint_name}_abs_debug"] += (
                    torch.abs(joint_delta) * self.step_dt
            )
        self._episode_sums["forward_vel_debug"] += (
                forward_vel * self.step_dt
        )

        self._episode_sums["forward_command_debug"] += (
                self._commands[:, 0] * self.step_dt
        )

        self._episode_sums["forward_error_debug"] += (
                torch.abs(
                    self._commands[:, 0] - forward_vel
                )
                * self.step_dt
        )
        self._episode_sums["stand_fraction_debug"] += (
                stand_gate * self.step_dt
        )

        self._episode_sums["stand_forward_speed_debug"] += (
                torch.abs(forward_vel)
                * stand_gate
                * self.step_dt
        )

        self._episode_sums["stand_lateral_speed_debug"] += (
                torch.abs(lateral_vel)
                * stand_gate
                * self.step_dt
        )

        self._episode_sums["stand_yaw_rate_debug"] += (
                torch.abs(yaw_rate)
                * stand_gate
                * self.step_dt
        )

        self._episode_sums["stand_pose_error_debug"] += (
                stand_pose_error
                * stand_gate
                * self.step_dt
        )

        self._episode_sums["stand_height_error_debug"] += (
                torch.abs(
                    base_height - self.cfg.base_height_target
                )
                * stand_gate
                * self.step_dt
        )
        self._episode_sums["stand_tilt_debug"] += (
                stand_tilt_error
                * stand_gate
                * self.step_dt
        )
        self._episode_sums["stand_double_contact_debug"] += (
                stand_double_contact
                * stand_gate
                * self.step_dt
        )

        self._episode_sums["stand_single_support_debug"] += (
                stand_single_support
                * stand_gate
                * self.step_dt
        )
        self._episode_sums["stand_knee_sym_debug"] += (
                stand_knee_sym_diff
                * stand_gate
                * self.step_dt
        )
        # self._episode_sums["stand_obrot9_debug"] += (
        #         joint_delta_all[:, self.debug_joint_indices["obrot9"]]
        #         * stand_gate
        #         * self.step_dt
        # )
        #
        # self._episode_sums["stand_obrot10_debug"] += (
        #         joint_delta_all[:, self.debug_joint_indices["obrot10"]]
        #         * stand_gate
        #         * self.step_dt
        # )
        # self._episode_sums["stand_obrot9_abs_debug"] += (
        #         torch.abs(joint_delta_all[:, self.debug_joint_indices["obrot9"]])
        #         * stand_gate
        #         * self.step_dt
        # )
        #
        # self._episode_sums["stand_obrot10_abs_debug"] += (
        #         torch.abs(joint_delta_all[:, self.debug_joint_indices["obrot10"]])
        #         * stand_gate
        #         * self.step_dt
        # )
        self._episode_sums["stand_obrot9_abs_debug"] += (
                obrot9_abs * stand_gate * self.step_dt
        )

        self._episode_sums["stand_obrot10_abs_debug"] += (
                obrot10_abs * stand_gate * self.step_dt
        )

        self._episode_sums["stand_ankle_sym_debug"] += (
                torch.abs(obrot9_abs - obrot10_abs) * stand_gate * self.step_dt
        )
        self._episode_sums["backward_fraction_debug"] += (
                backward_cmd_gate * self.step_dt
        )

        self._episode_sums["backward_command_debug"] += (
                torch.abs(self._commands[:, 0])
                * backward_cmd_gate
                * self.step_dt
        )

        self._episode_sums["backward_vel_debug"] += (
                (-forward_vel)
                * backward_cmd_gate
                * self.step_dt
        )

        self._episode_sums["backward_error_debug"] += (
                torch.abs(
                    self._commands[:, 0] - forward_vel
                )
                * backward_cmd_gate
                * self.step_dt
        )

        self._episode_sums["backward_cross_speed_debug"] += (
                torch.abs(vel_cross_cmd)
                * backward_cmd_gate
                * self.step_dt
        )
        self._episode_sums["forward_fraction_debug"] += (
                forward_cmd_gate * self.step_dt
        )

        self._episode_sums["forward_only_command_debug"] += (
                self._commands[:, 0]
                * forward_cmd_gate
                * self.step_dt
        )

        self._episode_sums["forward_only_vel_debug"] += (
                forward_vel
                * forward_cmd_gate
                * self.step_dt
        )

        self._episode_sums["forward_only_error_debug"] += (
                torch.abs(
                    self._commands[:, 0] - forward_vel
                )
                * forward_cmd_gate
                * self.step_dt
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_height = self._robot.data.root_pos_w[:, 2]

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        too_low = base_height < self.cfg.base_height_fall
        physics_glitch = base_height > 1.0

        tilt_error = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]),
            dim=1,
        )
        fell_over = tilt_error > self.cfg.max_tilt_error

        died = too_low | physics_glitch | fell_over
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self._sample_training_commands(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        # Małe losowe odchylenie rozbija idealną symetrię L/R.
        if self.cfg.reset_joint_noise > 0.0:
            joint_pos += (
                2.0 * torch.rand_like(joint_pos) - 1.0
            ) * self.cfg.reset_joint_noise

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = self.cfg.base_height_target

        self._current_targets[env_ids] = joint_pos.clone()

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._prev_root_pos[env_ids] = default_root_state[:, :3]

        self._left_foot_contact.reset(env_ids)
        self._right_foot_contact.reset(env_ids)
        self._left_swing_time[env_ids] = 0.0
        self._right_swing_time[env_ids] = 0.0
        self._left_swing_ema[env_ids] = 0.0
        self._right_swing_ema[env_ids] = 0.0
        self._double_support_time[env_ids] = 0.0
        self._lateral_vel_ema[env_ids] = 0.0
        self._left_max_clearance[env_ids] = 0.0
        self._right_max_clearance[env_ids] = 0.0
        self._left_knee_flex_ema[env_ids] = 0.0
        self._right_knee_flex_ema[env_ids] = 0.0
        self._prev_left_contact[env_ids] = True
        self._prev_right_contact[env_ids] = True
        self._roll_ema[env_ids] = 0.0
        # self._last_rewarded_step[env_ids] = 0

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/time_out"] = (
            torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        )

