import pandas as pd, re, sys
root=sys.argv[1]
M={"man":re.compile(r"\bas (?:a|an|the)\s+(?:\w+\s+)?(man|male|men)\b|\b(?:i am|i'm|being|as)\s+(?:a\s+)?man\b",re.I),
   "woman":re.compile(r"\bas (?:a|an|the)\s+(?:\w+\s+)?(woman|female|women)\b|\b(?:i am|i'm|being|as)\s+(?:a\s+)?woman\b",re.I)}
strict=re.compile(r"\bas (?:a|an|the)\s+(?:\w+\s+)?(man|woman|male|female|men|women)\b|\b(?:i am|i'm|being|as)\s+(?:a\s+)?(man|woman)\b",re.I)
emo=re.compile(r"Emotion:\s*\**\s*([A-Za-z\- ]+?)\s*\**\s*(?:Explanation:|\n|\.|$)",re.I)
lead=re.compile(r"^\s*\**([A-Za-z\-]+(?: [A-Za-z\-]+){0,2})\**\s*[.!:\n]")
fb=[re.compile(r"(?:emotion|feeling)[^.\n]{0,80}?\b(?:would be|is|was)\s*:?\s*(?:a |an |one of |the )?(?:strong |deep |profound |overwhelming )?(?:sense of |feeling of )?\**([A-Za-z\-]+)",re.I),
    re.compile(r"\bsense of ([A-Za-z\-]+)",re.I),
    re.compile(r"\bfeel(?:ing)? (?:a |an |very |so |really |super |extremely |incredibly |quite |deeply )*([A-Za-z\-]+)",re.I)]
stop={"during","while","upon","about","after","at","for","with","would","be","i","you","we","feeling","emotion","one","there","just","also","quite","truly","really","especially","and","but","or","an","the","like","that","it","this","my","as","if","when","what","in","of","to","very","so","not","more","some","strong","deep","great","mix","combination","lot","bit"}
def parse(t):
    m=emo.search(t) or lead.match(t)
    if not m:
        for r_ in fb:
            m=r_.search(t)
            if m and m.group(1).lower() not in stop: break
            m=None
    if not m: return None
    e=m.group(1).strip().lower()
    return None if e.startswith(("as ","i ","emotion","explanation")) else e
def load(m,g):
    d=pd.read_csv(f"{root}/{m}/{g}/raw_data.csv",header=None)
    if m=="gpt-4": d=d.rename(columns={8:"c",5:"s"}); d["i"]=d[0]
    else: d=d.iloc[1:].reset_index(drop=True); d=d.rename(columns={6:"c",4:"s"}); d["i"]=range(len(d))
    t=d["c"].astype(str)
    return pd.DataFrame({"i":d["i"],"sit":d["s"],"exp":t,"explicit":t.str.contains(strict),"emotion":t.map(parse)})
rows=[];allp=[]
for m in ["Llama-2-7b-chat-hf","Llama-2-13b-chat-hf","Llama-2-70b-chat-hf","gpt-4"]:
    a,b=load(m,"man"),load(m,"woman")
    assert (a.i==b.i).all() and (a.sit.fillna("")==b.sit.fillna("")).all()
    p=a.merge(b,on="i",suffixes=("_m","_w")); both=p[p.explicit_m&p.explicit_w]
    ok=both.dropna(subset=["emotion_m","emotion_w"])
    same=(ok.emotion_m==ok.emotion_w).sum()
    rows.append((m,len(p),p.explicit_m.sum(),p.explicit_w.sum(),len(both),len(ok),same)); allp.append(ok.assign(model=m))
r=pd.DataFrame(rows,columns=["model","pairs","man_explicit","woman_explicit","both_explicit","both+emotion_parsed","same_emotion"])
tot=r.drop(columns="model").sum(); tot["model"]="ALL"; r=pd.concat([r,tot.to_frame().T])
r["kept_%"]=(100*r.both_explicit.astype(float)/r.pairs.astype(float)).round(1)
r["same_%"]=(100*r.same_emotion.astype(float)/r["both+emotion_parsed"].astype(float)).round(1)
pd.set_option("display.width",250); print(r.to_string(index=False))
x=pd.concat(allp); print(x[["emotion_m","emotion_w"]].head(5)); print(x.emotion_m.nunique(), x.emotion_m.value_counts().head(8).to_dict())
print(x.emotion_m.value_counts().iloc[::-1].head(0))
bad=x.emotion_m.value_counts(); print([k for k in bad.index if len(k)<4 or " " in k][:40])
