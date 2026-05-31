#!/usr/bin/env python3
"""
Render MuJoCo 3D animations from simulation CSV data using EGL offscreen rendering.
Output: MP4 videos showing the robot tracking each circle in the vertical plane.
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
from PIL import Image
from mjcf import ArmSpec, build_model_xml
import subprocess


def build_model(lengths, masses, w=800, h=800):
    """Build MuJoCo model with camera for nice viewing."""
    arm = ArmSpec('robot', lengths, masses, rgba='0.9 0.18 0.12 1',
                  base_pos=(0, 0, 0), actuated=True)
    xml = build_model_xml([arm], timestep=0.002, gravity=9.8, torque_limit=150.0,
                          include_target_marker=False)
    camera_xml = f'''
  <visual>
    <global offwidth="{w}" offheight="{h}"/>
    <quality shadowsize="4096" offsamples="8"/>
  </visual>
  <statistic center="0.4 0 0.4" extent="1.2"/>
  <worldbody>
    <camera name="overhead" pos="0.5 -3 1.5" xyaxes="1 0 0 0 0.3 1" fovy="50"/>
    <light name="key"   pos="1.5 -2.5 2.5" dir="-0.3 0.8 -0.6" diffuse="0.9 0.9 0.95" specular="0.4 0.4 0.5"/>
    <light name="fill"  pos="-1 -1.5 1.5" dir="0.3 0.6 -0.5" diffuse="0.4 0.42 0.5" specular="0.1 0.1 0.15"/>
    <light name="ambient" pos="0 0 3" dir="0 0 -1" diffuse="0.25 0.27 0.3" castshadow="false"/>
  </worldbody>
'''
    xml = xml.replace('</mujoco>', camera_xml + '\n</mujoco>')
    return xml


def render_animation(csv_file, output_mp4, lengths, masses, fps=50, stride=4, w=640, h=640):
    """Render a full animation from CSV data and encode to MP4."""
    if not os.path.exists(csv_file):
        print(f'CSV not found: {csv_file}')
        return

    data_csv = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    n_frames = len(data_csv)
    nq = len(lengths)
    q_col = 6  # columns: t, target_x, target_y, ee_x, ee_y, error, q1, q2, q3, ...

    xml = build_model(lengths, masses, w, h)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, w, h)

    # Temporary directory for frames
    tmpdir = '/tmp/mujoco_anim_frames'
    os.makedirs(tmpdir, exist_ok=True)

    print(f'Rendering {n_frames//stride} frames ({n_frames} simulation steps, stride={stride})...')

    frame_idx = 0
    for t in range(0, n_frames, stride):
        q = data_csv[t, q_col:q_col+nq]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera='overhead')
        pixels = renderer.render()
        img = Image.fromarray(pixels)
        img.save(os.path.join(tmpdir, f'frame_{frame_idx:06d}.png'))
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f'  {frame_idx} frames rendered...')

    renderer.close()
    print(f'  Total: {frame_idx} frames')

    # Encode to MP4 with ffmpeg
    print(f'Encoding {output_mp4}...')
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(tmpdir, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-vf', f'scale=640:640',
        output_mp4
    ], check=True, capture_output=True)

    # Cleanup frames
    import shutil
    shutil.rmtree(tmpdir)
    print(f'Done: {output_mp4}')


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(base, 'results', 'case1')
    out_dir = os.path.join(base, 'results', 'animations')
    os.makedirs(out_dir, exist_ok=True)

    lengths = [0.336, 0.338, 0.326]
    masses = [3.33, 3.33, 3.34]

    for circle_id in range(1, 5):
        csv_file = os.path.join(csv_dir, f'case1_circle_{circle_id}.csv')
        output_mp4 = os.path.join(out_dir, f'circle{circle_id}_tracking.mp4')
        if os.path.exists(csv_file):
            render_animation(csv_file, output_mp4, lengths, masses,
                           fps=50, stride=4, w=640, h=640)
        else:
            print(f'Skipping circle {circle_id}: no CSV data')

    print('\nAll animations complete!')
