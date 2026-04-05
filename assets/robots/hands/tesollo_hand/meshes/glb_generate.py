import yourdfpy
import trimesh
import os

# 1. 指定你的 URDF 文件路径
urdf_file = "C:/Users/lenovo/Documents/GitHub/dex-retargeting/assets/robots/hands/tesollo_hand/tesollo_hand_left.urdf"

# 2. 加载机器人模型
# yourdfpy 加载后会生成一个包含所有 link 的场景
robot = yourdfpy.URDF.load(urdf_file)

# 3. 使用 robot.scene (这是一个 trimesh.Scene 对象) 来导出
if hasattr(robot, 'scene'):
    # 导出为 GLB
    output_path = "tesollo_hand_left.glb"
    robot.scene.export(output_path)
    print(f"成功！文件已保存至: {os.path.abspath(output_path)}")
else:
    print("未能成功加载场景，请检查 URDF 中的 STL 路径是否正确。")