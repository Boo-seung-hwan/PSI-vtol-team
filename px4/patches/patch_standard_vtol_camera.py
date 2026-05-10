from pathlib import Path

MODEL_FILE = Path("/workspace/px4/PX4-Autopilot/Tools/simulation/gz/models/standard_vtol/model.sdf")

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"standard_vtol model not found: {MODEL_FILE}")

text = MODEL_FILE.read_text()

if "vtol_camera_link" in text:
    print("[INFO] vtol camera already exists.")
    raise SystemExit(0)

camera_block = """
    <link name="vtol_camera_link">
      <pose>0.35 0 -0.08 0 0 0</pose>
      <inertial>
        <mass>0.05</mass>
        <inertia>
          <ixx>0.00001</ixx>
          <iyy>0.00001</iyy>
          <izz>0.00001</izz>
        </inertia>
      </inertial>
      <sensor name="camera" type="camera">
        <pose>0 0 0 0 0 0</pose>
        <topic>/vtol/camera</topic>
        <update_rate>10</update_rate>
        <camera>
          <horizontal_fov>1.3962634</horizontal_fov>
          <image>
            <width>1280</width>
            <height>960</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.02</near>
            <far>300</far>
          </clip>
        </camera>
        <always_on>1</always_on>
        <visualize>true</visualize>
        <gz_frame_id>vtol_camera_link</gz_frame_id>
      </sensor>
    </link>

    <joint name="vtol_camera_joint" type="fixed">
      <parent>base_link</parent>
      <child>vtol_camera_link</child>
      <pose>0 0 0 0 0 0</pose>
    </joint>
"""

if "</model>" not in text:
    raise RuntimeError("Could not find </model> in standard_vtol/model.sdf")

text = text.replace("</model>", camera_block + "\n  </model>")
MODEL_FILE.write_text(text)

print("[INFO] Added vtol camera to standard_vtol.")
