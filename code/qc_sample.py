"""Human QC: stratified-sample captions, emit an HTML review sheet with thumbnails + mark columns."""
import json,glob,os,html
from PIL import Image
R="/data/haiyuez/visteon_cabin_vlm"; OUT=R+"/data/qc_pack"; os.makedirs(OUT+"/thumbs",exist_ok=True)
PER=12  # per source
def src(p):
    for k in ['nuscenes','stanford_cars','gtsrb','textvqa','sun397','road_traffic','jaad_frames']:
        if k in p: return k
    return 'other'
# 优先抽 caption 丰富的文件
prefer=['exterior_cot_v2_sharegpt.json','trafficlight_sharegpt.json','jaad_vru_sharegpt.json','cars_sharegpt.json','signs_sharegpt.json','landscape_sharegpt.json','textvqa_sharegpt.json']
seen=set(); rows=[]; cnt={}
for f in prefer:
    p=R+"/data/"+f
    if not os.path.exists(p): continue
    for x in json.load(open(p)):
        im=x['images'][0]; s=src(im)
        if cnt.get(s,0)>=PER or im in seen: continue
        seen.add(im); cnt[s]=cnt.get(s,0)+1
        conv=x['conversations']; cap=conv[1]['value'] if len(conv)>1 else ''
        qa=[(conv[i]['value'],conv[i+1]['value']) for i in range(2,len(conv)-1,2)]
        rows.append((im,s,cap,qa))
# 缩略图
for i,(im,s,cap,qa) in enumerate(rows):
    try:
        t=Image.open(im).convert('RGB'); t.thumbnail((360,360)); t.save(f"{OUT}/thumbs/{i:03d}.jpg",quality=85)
    except: pass
# HTML
h=['<html><head><meta charset=utf-8><style>body{font-family:sans-serif;font-size:13px}table{border-collapse:collapse}td{border:1px solid #ccc;padding:6px;vertical-align:top}img{width:340px}.cap{max-width:520px}.qa{color:#444;font-size:12px}</style></head><body>',
   f'<h2>Caption 人工抽检表 ({len(rows)} 条,每源{PER})</h2>',
   '<p>逐条核对 caption 是否与图相符、有无幻觉。在"判定"列填:OK / 幻觉 / 需修;"备注"写问题。</p>',
   '<table><tr><th>#</th><th>图</th><th>来源</th><th>Caption + QA</th><th>判定</th><th>备注</th></tr>']
for i,(im,s,cap,qa) in enumerate(rows):
    qstr='<br>'.join(f'Q: {html.escape(q[:120])}<br>A: {html.escape(a[:120])}' for q,a in qa[:3])
    h.append(f'<tr><td>{i}</td><td><img src="thumbs/{i:03d}.jpg"></td><td>{s}</td>'
             f'<td class=cap>{html.escape(cap[:600])}<div class=qa>{qstr}</div></td><td></td><td></td></tr>')
h.append('</table></body></html>')
open(OUT+"/qc_review.html","w").write('\n'.join(h))
json.dump([{"id":i,"image":im,"source":s,"caption":cap} for i,(im,s,cap,qa) in enumerate(rows)],open(OUT+"/qc_sample.json","w"),ensure_ascii=False,indent=1)
print("QC_PACK",len(rows),"sources",cnt)
