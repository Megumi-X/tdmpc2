import mujoco
import sys
import os

def convert_global_to_local(input_xml_path, output_xml_path):
    # 1. 加载原始模型
    # MuJoCo 加载时会计算所有的运动学链
    try:
        model = mujoco.MjModel.from_xml_path(input_xml_path)
    except ValueError as e:
        print(f"Error loading XML: {e}")
        return

    # 2. 保存模型
    # mujoco.mj_saveLastXML 或者是 model 对象的保存功能
    # 这里的关键是：当你把加载后的 model 存回 XML 时，
    # MuJoCo 引擎会自动把 body 的 pos/quat 写成相对于 parent 的形式。
    mujoco.mj_saveLastXML(output_xml_path, model)
    
    print(f"✅ Conversion complete!")
    print(f"   Input:  {input_xml_path}")
    print(f"   Output: {output_xml_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_local.py <input_file.xml> [output_file.xml]")
        sys.exit(1)

    input_file = sys.argv[1]
    
    # 如果没指定输出文件名，就加一个 _local 后缀
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_local{ext}"

    convert_global_to_local(input_file, output_file)