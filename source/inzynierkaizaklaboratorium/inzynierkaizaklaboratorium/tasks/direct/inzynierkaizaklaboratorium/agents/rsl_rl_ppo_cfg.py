# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # ZWIĘKSZONE: Kaczka musi zebrać dane z pełnego kroku przed aktualizacją
    max_iterations = 150000  # Żebyś nie musiał wpisywać w terminalu z palca
    save_interval = 50
    experiment_name = "inzynierka_kaczka_clean_walk_v5_knee_fixed_straight"  # Zmienione z cartpole!
    empirical_normalization = True  # BARDZO WAŻNE: Automatycznie skaluje obserwacje do równych wartości

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.80,
        # MÓZG ROBOTA: Powiększony do standardów lokomocji (było 32x32)
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,  # ZWIĘKSZONE: Wymusza większą kreatywność w ruchach na początku # zwiększone z 0,006
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )