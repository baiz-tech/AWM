#!/usr/bin/env python3
"""Generate paper-ready EK100 HASP evidence traces from V23 checkpoints."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont

from src.experiments.awm_ek100_multiscale_adapter_seed239.readout import EK100MultiScaleReadout
from src.experiments.awm_ek100_multiscale_adapter_seed239.model import SharedMultiScaleWorldExtractor


def load_cache(path: Path):
    blobs = [torch.load(x, map_location="cpu", weights_only=False) for x in sorted(path.glob("features-rank*.pt"))]
    return torch.cat([x["grid"].float() for x in blobs]), sum((x["records"] for x in blobs), [])


def annotation_map(path: Path):
    with path.open(newline="") as handle: return {row["narration_id"]: row for row in csv.DictReader(handle)}


def video_frames(root: Path, row: dict, count=8):
    path = root / row["participant_id"] / "videos" / f"{row['video_id']}.MP4"
    if not path.exists(): path = path.with_suffix(".mp4")
    reader = VideoReader(str(path), num_threads=1, ctx=cpu(0)); start=max(0,int(row["start_frame"])-1); stop=min(len(reader)-1,max(start,int(row["stop_frame"])-1)); index=np.linspace(start,stop,count).round().astype(np.int64)
    return reader.get_batch(index).asnumpy()


def label(words, indices): return [words[int(i)] for i in indices]


try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    SMALL_FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    TITLE_FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
except OSError:
    FONT = ImageFont.load_default(); SMALL_FONT = FONT; TITLE_FONT = FONT
from src.core.run_context import apply_cli_defaults, task_context


def heatmap_overlay(frame: np.ndarray, attention: np.ndarray, size: int = 224) -> Image.Image:
    """Overlay normalized local-attention evidence without claiming attribution."""
    image = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    evidence = cv2.resize(attention, (size, size), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(np.clip(evidence, 0, 1) * 255), cv2.COLORMAP_MAGMA)
    return Image.fromarray(cv2.addWeighted(image, 0.58, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.42, 0))


def line_panel(values: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.new('RGB', (width, height), 'white'); draw = ImageDraw.Draw(image)
    left, right, top, bottom = 56, 18, 22, 32
    draw.line((left, height-bottom, width-right, height-bottom), fill='#6b746f', width=1)
    draw.line((left, top, left, height-bottom), fill='#6b746f', width=1)
    vmax = max(float(values.max()), 1e-8)
    points = [(left + i * (width-left-right) / max(len(values)-1, 1), height-bottom - float(v) / vmax * (height-top-bottom)) for i, v in enumerate(values)]
    draw.line(points, fill='#236147', width=3)
    for x, y in points: draw.ellipse((x-4, y-4, x+4, y+4), fill='#236147')
    draw.text((left, 3), 'Dynamic / temporal change (higher = more slot change)', fill='#17221d', font=SMALL_FONT)
    for i, (x, _) in enumerate(points): draw.text((x-3, height-bottom+7), str(i+1), fill='#47514c', font=SMALL_FONT)
    return image


def relation_panel(relation: np.ndarray, size: int = 320) -> Image.Image:
    vmax = max(float(relation.max()), 1e-8)
    color = cv2.applyColorMap(np.uint8(relation / vmax * 255), cv2.COLORMAP_VIRIDIS)
    color = cv2.cvtColor(cv2.resize(color, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(color); draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size, 28), fill='white'); draw.text((4, 6), 'Relation / slot-pair strength', fill='#17221d', font=SMALL_FONT)
    return image


def render_figure(frames, attention, motion, relation, title: str, text: str) -> Image.Image:
    margin, panel_width, frame_size = 26, 1600, 184
    canvas = Image.new('RGB', (panel_width, 850), '#f4f7f5'); draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), title, fill='#17221d', font=TITLE_FONT)
    for t, frame in enumerate(frames):
        tile = heatmap_overlay(frame, attention[t], frame_size)
        x = margin + t * (frame_size + 12); canvas.paste(tile, (x, 42)); draw.text((x, 230), f'Frame {t+1}', fill='#47514c', font=SMALL_FONT)
    canvas.paste(line_panel(motion, panel_width - 2*margin, 180), (margin, 256))
    canvas.paste(relation_panel(relation, 330), (margin, 470))
    text_x, text_y = 390, 470
    for line in text.splitlines():
        draw.text((text_x, text_y), line, fill='#17221d', font=SMALL_FONT); text_y += 24
    return canvas


def main():
    ctx=task_context();p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--readout',required=True);p.add_argument('--adapter',required=True);p.add_argument('--annotations',required=True);p.add_argument('--video-root',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=8);p.add_argument('--offset',type=int,default=0);p.add_argument('--device',default='cuda:0');apply_cli_defaults(p,ctx);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);grid,records=load_cache(Path(a.cache)); annotations=annotation_map(Path(a.annotations)); payload=torch.load(a.readout,map_location='cpu',weights_only=False);action_map=payload['action_map'];pairs=[None]*len(action_map)
    for pair,index in action_map.items():pairs[int(index)]=pair
    readout=EK100MultiScaleReadout(payload['verb_classes'],payload['noun_classes'],pairs);readout.load_state_dict(payload['model'],strict=True);shared=SharedMultiScaleWorldExtractor();shared.load_state_dict(torch.load(a.adapter,map_location='cpu',weights_only=False)['shared'],strict=True)
    device=torch.device(a.device);readout.to(device).eval();shared.to(device).eval();verb_words={};noun_words={}
    for row in annotations.values():verb_words[int(row['verb_class'])]=row['verb'];noun_words[int(row['noun_class'])]=row['noun']
    verb_vocab=payload['verb_vocabulary'];noun_vocab=payload['noun_vocabulary'];verb_names=[verb_words.get(x,str(x)) for x in verb_vocab];noun_names=[noun_words.get(x,str(x)) for x in noun_vocab]
    cards=[]
    with torch.inference_mode():
      for index in range(a.offset,min(len(records),a.offset+a.count)):
        record=records[index]; row=annotations.get(record['narration_id']);
        if row is None:continue
        x=grid[index:index+1].to(device); refined=shared(x);verb,noun,action,_,info=readout(refined,return_analysis=True)
        frames=video_frames(Path(a.video_root),row,8); attention=info['local_attention'][0].mean((0,1))[:8*64].reshape(8,8,8).cpu().numpy();attention=(attention-attention.min())/(attention.max()-attention.min()+1e-8)
        dynamics=info['objects'][0].float();motion=(dynamics[1:]-dynamics[:-1]).norm(dim=-1).mean(-1).cpu().numpy();relation=info['relations'][0].norm(dim=-1).cpu().numpy();np.fill_diagonal(relation,0)
        vt=verb[0].topk(5).indices.cpu();nt=noun[0].topk(5).indices.cpu();at=action[0].topk(5).indices.cpu();action_names=[f'{verb_names[pairs[int(i)][0]]} {noun_names[pairs[int(i)][1]]}' for i in at]
        text=f"GT: {row['verb']} {row['noun']}\nNarration: {row['narration']}\n\nVerb Top-5: {', '.join(label(verb_names,vt))}\nNoun Top-5: {', '.join(label(noun_names,nt))}\nAction Top-5: {', '.join(action_names)}"
        name=f'{index:05d}_{record["narration_id"]}.png';render_figure(frames,attention,motion,relation,f'HASP action evidence trace | {record["narration_id"]}',text).save(output/name);cards.append({'file':name,'id':record['narration_id'],'ground_truth':f"{row['verb']} {row['noun']}"})
    (output/'manifest.json').write_text(json.dumps({'count':len(cards),'samples':cards},indent=2)+'\n')
    links='\n'.join(f'<article><h2>{html.escape(x["id"])} | GT: {html.escape(x["ground_truth"])}</h2><img src="{html.escape(x["file"])}"><p class="caption">从左到右：模型在视频中看到的 8 个时刻。紫/黄热图表示局部对象证据；曲线表示槽位状态变化；矩阵表示槽位之间的关系强度；右侧文字是分类结果。</p></article>' for x in cards)
    guide='''<section class="guide"><h1>EK100 世界模型可视化</h1><p>这张图回答一个简单问题：模型是否从视频中提取了“物体、变化、物体之间的关系”，并用它们预测动作？</p><div class="steps"><div><b>1. 物体证据</b><br>上方 8 帧是动作片段。颜色越亮，表示 readout 更关注该局部区域。它是对象证据，不是严格因果归因。</div><div><b>2. 动态证据</b><br>绿色曲线越高，表示相邻时刻的槽位状态变化越大。</div><div><b>3. 关系证据</b><br>矩阵中第 i 行第 j 列表示第 i、j 个对象槽位的关系状态强度。对角线已清零。</div><div><b>4. 分类输出</b><br>右侧列出真实标签和模型认为最可能的 Verb、Noun、Action 前五名。Top-5 中出现真实词，说明候选语义已被捕捉。</div></div></section>'''
    page=f'''<!doctype html><meta charset="utf-8"><title>EK100 HASP Evidence Traces</title><style>body{{font-family:Arial,sans-serif;margin:32px;background:#eef3f0;color:#17221d}}.guide{{max-width:1200px;margin:0 auto 28px;padding:24px 28px;background:#fff;border-left:5px solid #236147}}.guide h1{{margin:0 0 10px;font-size:28px}}.guide p{{font-size:16px;line-height:1.6}}.steps{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.steps div{{padding:14px;background:#f4f7f5;line-height:1.5;font-size:14px}}article{{max-width:1650px;margin:30px auto;padding:20px;background:white;box-shadow:0 1px 5px #ccd5d0}}article h2{{font-size:18px;margin:0 0 12px}}img{{max-width:100%;height:auto;display:block}}.caption{{font-size:14px;line-height:1.5;color:#47514c}}@media(max-width:900px){{.steps{{grid-template-columns:1fr 1fr}}}}@media(max-width:600px){{.steps{{grid-template-columns:1fr}}}}</style>{guide}{links}'''
    (output/'index.html').write_text(page)

if __name__=='__main__':main()
