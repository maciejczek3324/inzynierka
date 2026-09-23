# from isaaclab.app import AppLauncher
#
# app_launcher = AppLauncher(headless=False)
# simulation_app = app_launcher.app
#
# import torch
#
# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
#
# from isaaclab_assets.robots.inzynierka import PORZADKI_ROBOT_CFG
#
#
# # ============================================================
# # CZYSTY VIEWER - ZERO RL, ZERO POLICY
# # ============================================================
#
# sim_cfg = sim_utils.SimulationCfg(
#     dt=1.0 / 60.0,
#     gravity=(0.0, 0.0, 0.0),   # ZERO GRAWITACJI
# )
#
# sim = sim_utils.SimulationContext(sim_cfg)
#
# sim.set_camera_view(
#     eye=(1.0, -1.0, 0.55),
#     target=(0.0, 0.0, 0.18),
# )
#
#
# # ============================================================
# # ROBOT
# # ============================================================
#
# robot_cfg = PORZADKI_ROBOT_CFG.replace(
#     prim_path="/World/Robot"
# )
#
# robot = Articulation(robot_cfg)
#
# light_cfg = sim_utils.DomeLightCfg(
#     intensity=2000.0,
# )
# light_cfg.func("/World/Light", light_cfg)
#
# # ============================================================
# # CZARNA ZIEMIA
# # ============================================================
#
# ground_cfg = sim_utils.GroundPlaneCfg(
#     color=(7.0, 3.0, 3.0),
# )
#
# ground_cfg.func(
#     "/World/Ground",
#     ground_cfg,
# )
#
# sim.reset()
#
#
# # ============================================================
# # JOINTY
# # ============================================================
#
# print("\nJOINT NAMES:")
# for i, name in enumerate(robot.data.joint_names):
#     print(i, name)
#
# joint_pos = torch.zeros_like(
#     robot.data.default_joint_pos
# )
#
# joint_vel = torch.zeros_like(joint_pos)
#
#
# # ============================================================
# # TU WPISUJEMY POZYCJĘ
# # ============================================================
#
# def set_joint(name, value):
#     idx = robot.data.joint_names.index(name)
#     joint_pos[:, idx] = value
#
#
# # na razie znane kolana
# #
# # UWAGA:
# # to są tylko wartości do TESTU.
# # Znaki dopasujemy do Twojej geometrii.
# # set_joint("obrot1", +0.1)
# set_joint("obrot5",  +0.65)
# set_joint("obrot6",  -0.65)
#
# set_joint("obrot7",  -0.85)
# set_joint("obrot8",  +0.85)
#
# set_joint("obrot9",  +0.25)
# set_joint("obrot10", +0.25)
#
#
#
# # PÓŹNIEJ DODAMY:
# #
# # lewe hip pitch   ~ 0.630 rad
# # prawe hip pitch  ~ 0.635 rad
# #
# # lewe ankle       ~ 0.784 rad
# # prawe ankle      ~ 0.796 rad
# #
# # ale najpierw ustalimy które obrotX są którymi stawami.
#
#
# # ============================================================
# # ROOT - SZTYWNO W POWIETRZU
# # ============================================================
#
# root_state = robot.data.default_root_state.clone()
#
# root_state[:, 0] = 0.0
# root_state[:, 1] = 0.0
# root_state[:, 2] = 0.183
#
# root_state[:, 3] = 1.0
# root_state[:, 4:7] = 0.0
# root_state[:, 7:] = 0.0
#
#
# # ============================================================
# # SAM RENDER
# # ============================================================
#
# while simulation_app.is_running():
#
#     robot.write_root_pose_to_sim(
#         root_state[:, :7]
#     )
#
#     robot.write_root_velocity_to_sim(
#         root_state[:, 7:]
#     )
#
#     robot.write_joint_state_to_sim(
#         joint_pos,
#         joint_vel,
#     )
#
#     sim.step()
#
#
# simulation_app.close()