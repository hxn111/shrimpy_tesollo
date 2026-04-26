from robot_motion_interface.isaacsim.config.bimanual_arm_env_config import BimanualArmSceneCfg, ActionsCfg, ObservationsCfg, EventCfg

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch

from pathlib import Path


USD_DIR = Path(__file__).resolve().parent.parent / "usds"
    


@configclass
class BimanualArmObjectSceneCfg(BimanualArmSceneCfg):
    """Configuration for the Bimanual Arm with a bunch of objects"""
    
    # bowl = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/bowl",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "bowl.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # bowl_1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/bowl_1",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "bowl.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # bowl_2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/bowl_2",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "bowl.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # cup = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cup",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "cup.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # cup_1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cup_1",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "cup.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # cup_2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cup_2",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "cup.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # cup_3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cup_3",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "cup.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # fork = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/fork",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "fork.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # fork_1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/fork_1",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "fork.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # fork_2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/fork_2",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "fork.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # fork_3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/fork_3",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "fork.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    # spoon = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/spoon",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "spoon.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False,
    #                                                        solver_position_iteration_count=16
    #                                                        ),
    #         visible=False,
    #     ),
    # )

    # spoon_1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/spoon_1",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "spoon.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False,
    #                                                        solver_position_iteration_count=16
    #                                                        ),
    #         visible=False,
    #     ),
    # )

    # spoon_2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/spoon_2",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "spoon.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False,
    #                                                        solver_position_iteration_count=16
    #                                                        ),
    #         visible=False,
    #     ),
    # )

    # spoon_3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/spoon_3",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "spoon.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False,
    #                                                        solver_position_iteration_count=16
    #                                                        ),
    #         visible=False,
    #     ),
    # )

    # funnel = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/funnel",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "funnel.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )


    # bin = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/bin",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "bin.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         visible=False,
    #     ),
    # )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            # size=(0.04, 0.06, 0.06),
            size=(0.06, 0.06, 0.06),
            mass_props = sim_utils.MassPropertiesCfg(mass=0.08),
            rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.2, 0.962)),
    )

    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_1",
        spawn=sim_utils.CuboidCfg(
            # size=(0.12, 0.12, 0.02),
            size=(0.06, 0.06, 0.06),
            mass_props = sim_utils.MassPropertiesCfg(mass=0.08),
            rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.0, 0.952)),
    )
    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.05 ),
            mass_props = sim_utils.MassPropertiesCfg(mass=0.1),
            rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.4,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            visible=False,
        ),
    )
    # cylinder = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cylinder",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.05, height=0.1,
    #         mass_props = sim_utils.MassPropertiesCfg(mass=0.1),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.5,
    #             dynamic_friction=0.4,
    #             friction_combine_mode="max",
    #         ),
    #         visible=False,
    #     ),
    # )

    # sphere = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/sphere",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.05,
    #         mass_props = sim_utils.MassPropertiesCfg(mass=0.1),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.5,
    #             dynamic_friction=0.4,
    #             friction_combine_mode="max",
    #         ),
    #         visible=False,
    #     ),
    # )


    # marker = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/marker",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=str(USD_DIR / "marker.usd"),
    #         scale=(1.0, 1.0, 1.0),
    #         visible=False,
    #     ),
    # )


# ####################### Many Object Generation ####################### 
# cube_spawn_cfg = sim_utils.CuboidCfg(
#             size=(0.01, 0.01, 0.01),
#             mass_props = sim_utils.MassPropertiesCfg(mass=0.01),
#             rigid_props = sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False),
#             collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
#             visual_material=sim_utils.PreviewSurfaceCfg(
#                 diffuse_color=(0.0, 0.0, 1.0)
#             ),
#             visible=False,
#         )

# NUM_CUBES = 50

# for i in range(NUM_CUBES):
#     setattr(
#         BimanualArmObjectSceneCfg,
#         f"cube_{i}",
#         RigidObjectCfg(
#             prim_path=f"{{ENV_REGEX_NS}}/cube_{i}",
#             spawn=cube_spawn_cfg,
#             init_state=RigidObjectCfg.InitialStateCfg(
#                 pos=(0.01 * i, 0.0, 0.05),
#             ),
#         ),
#     )

# ######################################################################


def cube_out_of_table(env, min_height: float = 0.85) -> torch.Tensor:
    """Terminate when the cube falls below the table surface (z < min_height)."""
    cube = env.scene["cube"]
    return cube.data.root_state_w[:, 2] < min_height


def timeout_unless_grasped(env, min_grasp_height: float = 0.99) -> torch.Tensor:
    """Timeout reset only if the cube has NOT been lifted off the table.
    If the cube is up in the air (being grasped), hold off the reset."""
    cube = env.scene["cube"]
    cube_not_lifted = cube.data.root_state_w[:, 2] < min_grasp_height
    return mdp.time_out(env) & cube_not_lifted


@configclass
class ObjectTerminationsCfg:
    """Reset on timeout (only if cube not grasped) OR if the cube falls off the table."""
    timeout = DoneTerm(func=timeout_unless_grasped, time_out=True, params={"min_grasp_height": 0.99})
    cube_fell = DoneTerm(func=cube_out_of_table, params={"min_height": 0.85})


@configclass
class BimanualArmObjectEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the Bimanual Arm environment."""

    scene = BimanualArmObjectSceneCfg(num_envs=1, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    terminations = ObjectTerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        self.viewer.eye = [0.0, -1.3, 2.5]
        self.viewer.lookat = [0.0, 0.0, 1.2]
        self.decimation = 1
        self.sim.dt = 0.005
        self.sim.render_interval = 0.02 / self.sim.dt  # 50 FPS
        self.episode_length_s = 20.0  # 20s per episode — resets unless cube is actively being held