"""保存轨迹图片与可拖动进度条的独立HTML，不需要启动网页服务。"""
import json
import math
from pathlib import Path
import numpy as np
from config import vehicle_parameters
from coarse_route import world_to_grid


def body_polygons(state, parameters=None):
    # 第1步：从前桥状态算出铲斗、前车体和后车体矩形，绘图应能看到铰接姿态。
    x, y, heading, theta = state
    parameters = vehicle_parameters if parameters is None else parameters
    front_length = parameters["front_axle_to_hitch_m"]
    front_width = parameters["front_body_width_m"]
    bucket_length = parameters["bucket_front_m"]
    bucket_width = parameters["bucket_width_m"]
    rear_length = parameters["hitch_to_rear_axle_m"] + parameters["rear_axle_to_tail_m"]
    rear_width = parameters["rear_body_width_m"]
    hitch = np.array([x - front_length * math.cos(heading), y - front_length * math.sin(heading)])
    parts = [(np.array([x, y]), heading, -front_length, 0.0, front_width),
             (np.array([x, y]), heading, 0.0, bucket_length, bucket_width),
             (hitch, heading - theta, -rear_length, 0.0, rear_width)]
    polygons = []
    for origin, angle, low, high, width in parts:
        points = np.array([[low, -width / 2], [high, -width / 2], [high, width / 2], [low, width / 2], [low, -width / 2]])
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        polygons.append(points @ rotation.T + origin)
    return polygons


def save_visualization(output_dir, map_data, start, goal, route, states, directions,
                       optimized_states=None, optimized_directions=None, reference=None,
                       prediction_label="Network", parameters=None, optimized_label="Optimized"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(map_data["map_features"][0], origin="lower", cmap="gray_r", vmin=0, vmax=1)
    route_grid = world_to_grid(route, map_data)
    ax.plot(route_grid[:, 0], route_grid[:, 1], ":", color="gray", label="Map-only coarse route")
    groups = [(states, directions, prediction_label, 0.65, "--")]
    if optimized_states is not None:
        groups.append((optimized_states, optimized_directions, optimized_label, 1.0, "-"))
    for path, path_directions, label, alpha, style in groups:
        xy = world_to_grid(path[:, :2], map_data)
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        colors = ["#16864b" if direction > 0 else "#dc6b22" for direction in path_directions]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=2, alpha=alpha, linestyles=style, zorder=3))
        ax.plot([], [], color="black", linestyle=style, label=label)
        boundaries = np.flatnonzero(path_directions[1:] != path_directions[:-1]) + 1
        ax.scatter(xy[boundaries, 0], xy[boundaries, 1], marker="*", s=130, color="#b83faa")
    shown = optimized_states if optimized_states is not None else states
    for index in np.linspace(0, len(shown) - 1, min(9, len(shown)), dtype=int):
        for polygon in body_polygons(shown[index], parameters):
            points = world_to_grid(polygon, map_data)
            ax.plot(points[:, 0], points[:, 1], color="#2762ad", alpha=0.5, linewidth=0.8)
    if reference is not None:
        xy = world_to_grid(reference[:, :2], map_data)
        ax.plot(xy[:, 0], xy[:, 1], color="#a641b7", alpha=0.4, linewidth=1.2, label="Reference (display only)")
    endpoints = world_to_grid(np.array([start[:2], goal[:2]]), map_data)
    ax.scatter(endpoints[:, 0], endpoints[:, 1], c=["blue", "red"], s=60)
    ax.set_title("Loader trajectory: green=forward, orange=reverse, star=switch")
    ax.set_xlabel("Map column (cell)")
    ax.set_ylabel("Map row (cell)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory.png", dpi=140)
    plt.close(fig)

    # 第2步：把地图和车辆轮廓写入HTML；成功后浏览器打开文件即可拖动进度条。
    paths = []
    for path, path_directions, label, alpha, style in groups:
        frames = []
        for state in path:
            frames.append([world_to_grid(polygon, map_data).tolist() for polygon in body_polygons(state, parameters)])
        paths.append({"name": label, "xy": world_to_grid(path[:, :2], map_data).tolist(),
                      "frames": frames, "directions": np.asarray(path_directions).tolist(),
                      "states": np.asarray(path).tolist()})
    payload = json.dumps({"occupancy": map_data["map_features"][0].tolist(), "paths": paths,
                          "route": route_grid.tolist()}, allow_nan=False)
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>装载机轨迹回放</title>
<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;background:#f5f7fa;color:#253247}canvas{background:white;width:100%;border:1px solid #ddd}input{width:70%}button,select{padding:8px;margin:8px}p{line-height:1.6}</style>
<h1>装载机轨迹回放</h1><p>绿色为前进，橙色为倒车，紫点为换挡。选择轨迹并拖动进度条，可查看前车体、后车体和铲斗的位置。请以同目录 result.json 的完整车体、终点和控制重放检查为准；画出轨迹不表示已通过验收。</p>
<select id="path"></select><button id="play">播放／暂停</button><input id="step" type="range" min="0" value="0"><p id="state"></p><canvas id="canvas" width="880" height="880"></canvas>
<script>const data=PAYLOAD;const canvas=document.getElementById('canvas');const ctx=canvas.getContext('2d');
const select=document.getElementById('path');const slider=document.getElementById('step');
data.paths.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=p.name==='Network'?'网络预测':p.name==='Optimized'?'优化结果':p.name;select.appendChild(o)});select.value=data.paths.length-1;
const h=data.occupancy.length,w=data.occupancy[0].length;const scale=Math.min(840/w,840/h),ox=(880-w*scale)/2,oy=(880-h*scale)/2;
function px(p){return [ox+p[0]*scale,880-oy-p[1]*scale]}
function line(points,color,width){ctx.beginPath();points.forEach((p,i)=>{const q=px(p);if(i===0)ctx.moveTo(q[0],q[1]);else ctx.lineTo(q[0],q[1])});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function draw(){const p=data.paths[Number(select.value)];slider.max=p.xy.length-1;let k=Math.min(Number(slider.value),p.xy.length-1);slider.value=k;ctx.clearRect(0,0,880,880);
for(let y=0;y<h;y++)for(let x=0;x<w;x++){if(data.occupancy[y][x]>=0.5){ctx.fillStyle='#354052';ctx.fillRect(ox+x*scale,880-oy-(y+1)*scale,scale+0.5,scale+0.5)}}
line(data.route,'#b0b8c1',1);for(let i=0;i<p.xy.length-1;i++){line([p.xy[i],p.xy[i+1]],p.directions[i]>0?'#16864b':'#dc6b22',2);if(i>0&&p.directions[i]!==p.directions[i-1]){const q=px(p.xy[i]);ctx.fillStyle='#b83faa';ctx.beginPath();ctx.arc(q[0],q[1],6,0,Math.PI*2);ctx.fill()}}
p.frames[k].forEach(poly=>line(poly,'#165ca8',2));const s=p.states[k];document.getElementById('state').textContent='点 '+k+' / '+(p.xy.length-1)+'　x='+s[0].toFixed(2)+' m　y='+s[1].toFixed(2)+' m　航向='+(s[2]*180/Math.PI).toFixed(1)+'°　铰接角='+(s[3]*180/Math.PI).toFixed(1)+'°';}
select.onchange=()=>{slider.value=0;draw()};slider.oninput=draw;let playing=false;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);draw()}},100);draw();</script></html>'''
    (output_dir / "trajectory.html").write_text(html.replace("PAYLOAD", payload), encoding="utf-8")
