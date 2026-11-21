import json, pandas as pd, numpy as np
rows=[]; att={}
for l in open('ledger_copy.jsonl'):
    try: d=json.loads(l)
    except: continue
    if d.get('type')=='attempt': att[d['attempt_id']]=d['ts']
    if d.get('type')!='response' or d.get('http_status')!=200: continue
    rows.append(dict(ep=d['endpoint'],pid=d.get('prompt_id'),prompt=d.get('prompt'),rep=d.get('rep'),mt=d.get('max_tokens'),run=d.get('run_id'),
      fr=d.get('finish_reason'),ts=d['ts'],ats=att.get(d.get('attempt_id')),el=d.get('elapsed_s'),text=d.get('text') or ''))
df=pd.DataFrame(rows)
df['ts']=pd.to_datetime(df.ts); df['ats']=pd.to_datetime(df.ats)
df['nch']=df.text.str.len(); df['nw']=df.text.str.split().str.len()
df['plen']=df.prompt.fillna('').str.len()
df.to_pickle('resp.pkl')
