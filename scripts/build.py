#!/usr/bin/env python3
"""Generate the README tables and the GitHub Pages site from data/*.yaml.

The YAML files under data/ are the single source of truth. Everything between the
<!-- BEGIN:x --> / <!-- END:x --> markers in README.md is machine-written; edit the
YAML, then run `python scripts/build.py`.

Usage:
    python scripts/build.py            # rewrite README.md and docs/index.html
    python scripts/build.py --check    # fail if the generated output is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
README = ROOT / "README.md"
SITE = ROOT / "docs" / "index.html"

CATEGORIES = [
    ("molecule-text", "Molecule ↔ Text", "Contrastive alignment, translation and text-guided generation between molecular structure and natural language."),
    ("multi-view", "Multi-View: 2D + 3D", "Alternative encodings of the *same* molecule — graph topology fused with geometry."),
    ("mllm", "Multimodal LLMs & Assistants", "Instruction-tuned models that inject structural embeddings into a language backbone."),
    ("image", "Images, OCR & Document Intelligence", "Molecular depictions and reaction schemes as they actually appear in papers and patents."),
    ("reaction", "Reactions, Conditions & Procedures", "Reaction context, condition recommendation and experimental procedure prediction."),
    ("spectra", "Spectra & Structure Elucidation", "IR, NMR and MS as model inputs — retrieval, interpretation and de novo generation."),
    ("materials", "Materials & Crystals", "Property-conditioned generation and simulation of inorganic solids."),
]

MODALITY_LABEL = {
    "text": "text", "smiles": "SMILES", "graph": "graph", "3d": "3D",
    "image": "image", "ir": "IR", "nmr": "NMR", "ms": "MS",
    "reaction": "reaction", "kg": "KG", "crystal": "crystal",
}

KIND_LABEL = {
    "corpus": "Corpus",
    "pretraining": "Pretraining set",
    "instruction": "Instruction set",
    "benchmark": "Benchmark",
}

# Datasets are grouped by the modality pair they align, because that is what
# decides which models can be trained or evaluated on them.
PAIRINGS = [
    ("structure-text", "Structure ↔ Text",
     "Molecule–description, QA and instruction pairs. The scarcest resource in the field, and the one most constrained by text licensing."),
    ("spectra", "Structure ↔ Spectra",
     "IR, NMR and MS paired with the structure that produced them. Mostly simulated, mostly patent-derived."),
    ("image", "Structure ↔ Image",
     "Depictions and reaction schemes paired with machine-readable structure. Largely synthetic renderings rather than scraped figures."),
    ("reaction", "Reaction, Conditions & Procedures",
     "Reaction records with roles, conditions, yields and free-text procedures."),
    ("geometry", "Structure ↔ 3D Geometry",
     "Conformers and quantum labels. These carry no text, but they are what every 3D branch of a multi-modal model is pretrained on."),
    ("materials", "Crystals & Materials",
     "Periodic structures with computed properties."),
]


# --------------------------------------------------------------------------- io

def load(name: str):
    with (DATA / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def mods(entry) -> list[str]:
    return [MODALITY_LABEL.get(m, m) for m in entry.get("modalities", [])]


def link(url: str | None, label: str) -> str:
    return f"[{label}]({url})" if url else "—"


# ---------------------------------------------------------------------- readme

def md_models(models) -> str:
    out: list[str] = []
    for key, title, blurb in CATEGORIES:
        rows = sorted((m for m in models if m["category"] == key),
                      key=lambda m: (m["year"], m["name"].lower()))
        if not rows:
            continue
        out.append(f"### {title}\n")
        out.append(f"{blurb}\n")
        out.append("| Model | Year | Modalities | Paper | Code | Licence | Why it matters |")
        out.append("|---|:--:|---|---|---|---|---|")
        for m in rows:
            tags = " ".join(f"`{t}`" for t in mods(m))
            code = link(m.get("code"), "repo")
            out.append(
                f"| **{m['name']}** | {m['year']} | {tags} | "
                f"[{m['venue']}]({m['paper']}) | {code} | {m.get('license', 'unspecified')} | {m['note']} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def md_datasets(datasets) -> str:
    out: list[str] = []
    for key, title, blurb in PAIRINGS:
        rows = sorted((d for d in datasets if d.get("pairing") == key),
                      key=lambda d: (d["year"], d["name"].lower()))
        if not rows:
            continue
        out.append(f"### {title}\n")
        out.append(f"{blurb}\n")
        out.append("| Dataset | Year | Type | Modalities | Scale | Licence | Paper | Data |")
        out.append("|---|:--:|---|---|---|---|---|---|")
        for d in rows:
            tags = " ".join(f"`{t}`" for t in mods(d))
            out.append(
                f"| **{d['name']}** | {d['year']} | {KIND_LABEL.get(d.get('kind'), '—')} | {tags} | "
                f"{d['scale']} | {d.get('license', 'unspecified')} | "
                f"{link(d.get('paper'), 'paper')} | {link(d.get('data'), 'download')} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def md_reading(reading) -> str:
    out: list[str] = []
    sections = [
        ("surveys", "Surveys & reviews"),
        ("critiques", "Critical evaluations & benchmark hygiene"),
        ("governance", "Safety, dual use & governance"),
    ]
    for key, title in sections:
        items = reading.get(key, [])
        if not items:
            continue
        out.append(f"### {title}\n")
        for it in sorted(items, key=lambda i: (-i["year"], i["title"])):
            out.append(f"- **[{it['title']}]({it['url']})** — {it['authors']}, *{it['venue']}* ({it['year']}).  ")
            out.append(f"  {it['note']}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def md_timeline(models) -> str:
    by_year: dict[int, list[str]] = {}
    for m in models:
        by_year.setdefault(m["year"], []).append(m["name"])
    lines = ["```mermaid", "timeline",
             "    title Multi-modal chemistry, by first public release"]
    for year in sorted(by_year):
        names = sorted(by_year[year], key=str.lower)
        lines.append(f"    {year} : {names[0]}")
        for n in names[1:]:
            lines.append(f"         : {n}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def md_stats(models, datasets, reading) -> str:
    n_ref = sum(len(v) for v in reading.values())
    n_code = sum(1 for m in models if m.get("code"))
    return (
        f"**{len(models)}** models and systems · **{len(datasets)}** datasets and benchmarks · "
        f"**{n_ref}** surveys, critiques and governance references · "
        f"**{n_code}/{len(models)}** models with public code\n"
    )


def inject(text: str, key: str, body: str) -> str:
    begin, end = f"<!-- BEGIN:{key} -->", f"<!-- END:{key} -->"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"marker {begin} … {end} not found in README.md")
    return pattern.sub(f"{begin}\n{body}\n{end}", text)


# ------------------------------------------------------------------------ site

SITE_TEMPLATE = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Awesome Multi-Modal Chemistry</title>
<meta name="description" content="A curated, searchable index of multi-modal models, datasets and benchmarks in chemistry — graphs, 3D structures, spectra, images, reactions and text.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='42' fill='none' stroke='%235eead4' stroke-width='8'/><circle cx='50' cy='50' r='12' fill='%23a78bfa'/></svg>">
<style>
:root{
  --bg:#0a0e1a; --bg2:#0f1526; --card:#131a2e;
  --line:#243050; --fg:#e8edf9; --dim:#93a3c4; --dim2:#65759a;
  --a1:#5eead4; --a2:#7dd3fc; --a3:#a78bfa; --a4:#f0abfc;
  --radius:14px;
}
html[data-theme="light"]{
  --bg:#f7f8fc; --bg2:#eef1f8; --card:#ffffff; --line:#dfe4f0;
  --fg:#141a2b; --dim:#4f5c78; --dim2:#7c88a5;
  --a1:#0d9488; --a2:#0284c7; --a3:#7c3aed; --a4:#c026d3;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Inter,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}
a{color:var(--a2); text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1240px; margin:0 auto; padding:0 20px}

header{
  position:relative; overflow:hidden;
  background:
    radial-gradient(1000px 380px at 12% -10%, rgba(94,234,212,.16), transparent 60%),
    radial-gradient(900px 420px at 88% 0%, rgba(167,139,250,.20), transparent 62%),
    linear-gradient(180deg,var(--bg2),var(--bg));
  border-bottom:1px solid var(--line);
}
header .wrap{padding-top:56px; padding-bottom:40px}
h1{
  margin:0; font-size:clamp(30px,5vw,52px); letter-spacing:-1.4px; font-weight:800;
  background:linear-gradient(90deg,var(--a1),var(--a2) 38%,var(--a3) 72%,var(--a4));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.tagline{margin:14px 0 0; color:var(--dim); max-width:74ch; font-size:16px}
.kicker{
  font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:3px;
  color:var(--dim2); text-transform:uppercase; margin-bottom:16px;
}
.stats{display:flex; flex-wrap:wrap; gap:10px; margin-top:24px}
.stat{
  background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
  padding:10px 16px; min-width:104px;
}
.stat b{display:block; font-size:22px; letter-spacing:-.5px}
.stat span{font-size:11px; color:var(--dim2); text-transform:uppercase; letter-spacing:1.2px}
.hlinks{display:flex; gap:8px; margin-top:24px; flex-wrap:wrap}
.hlinks a, .ghost{
  border:1px solid var(--line); background:var(--card); color:var(--fg);
  border-radius:999px; padding:7px 15px; font-size:13px; font-weight:500; cursor:pointer;
}
.hlinks a:hover,.ghost:hover{border-color:var(--a2); text-decoration:none}

.controls{position:sticky; top:0; z-index:20; background:var(--bg); border-bottom:1px solid var(--line)}
.controls .wrap{padding-top:14px; padding-bottom:14px}
.searchrow{display:flex; gap:10px; align-items:center}
#q{
  flex:1; background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:11px 14px; color:var(--fg); font-size:15px; outline:none;
}
#q:focus{border-color:var(--a2)}
.tabs{display:flex; gap:6px; flex-wrap:wrap; margin-top:12px}
.tab{
  border:1px solid var(--line); background:transparent; color:var(--dim);
  border-radius:999px; padding:6px 13px; font-size:13px; cursor:pointer;
}
.tab.on{background:var(--a3); border-color:var(--a3); color:#0a0e1a; font-weight:600}
html[data-theme="light"] .tab.on{color:#fff}
.filters{display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; align-items:center}
.chip{
  border:1px solid var(--line); background:transparent; color:var(--dim);
  border-radius:8px; padding:4px 10px; font-size:12px; cursor:pointer;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
}
.chip.on{border-color:var(--a1); color:var(--a1); background:rgba(94,234,212,.10)}
.flabel{font-size:11px; color:var(--dim2); text-transform:uppercase; letter-spacing:1.4px; margin-right:4px}

main .wrap{padding-top:28px; padding-bottom:80px}
#count{color:var(--dim2); font-size:13px; margin-bottom:18px}
.group{margin-bottom:34px}
.group > h2{
  font-size:13px; letter-spacing:2.2px; text-transform:uppercase; color:var(--dim2);
  margin:0 0 4px; font-weight:700;
}
.group > p{margin:0 0 14px; color:var(--dim); font-size:13.5px; max-width:80ch}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px}
.card{
  background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
  padding:16px 17px; display:flex; flex-direction:column; gap:9px;
  transition:border-color .15s, transform .15s;
}
.card:hover{border-color:var(--a3); transform:translateY(-2px)}
.ctop{display:flex; justify-content:space-between; align-items:baseline; gap:10px}
.cname{font-weight:700; font-size:16.5px; letter-spacing:-.3px}
.cyear{
  font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--dim2);
  border:1px solid var(--line); border-radius:6px; padding:4px 7px; flex:none;
}
.ctitle{font-size:12.5px; color:var(--dim2); line-height:1.45}
.cnote{font-size:13.5px; color:var(--dim); flex:1}
.tags{display:flex; gap:5px; flex-wrap:wrap}
.tag{
  font:500 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace; padding:4px 7px;
  border-radius:6px; background:rgba(125,211,252,.11); color:var(--a2);
  border:1px solid rgba(125,211,252,.22);
}
.cfoot{display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding-top:3px; border-top:1px solid var(--line)}
.cfoot a{font-size:12.5px; font-weight:600}
.lic{font-size:11px; color:var(--dim2); margin-left:auto; font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.scale{font-size:12px; color:var(--a1); font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.readitem{
  background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
  padding:14px 16px; margin-bottom:10px;
}
.readitem .rt{font-weight:650; font-size:15px}
.readitem .rm{font-size:12px; color:var(--dim2); margin:3px 0 6px}
.readitem .rn{font-size:13.5px; color:var(--dim)}
.hide{display:none !important}
footer{border-top:1px solid var(--line); color:var(--dim2); font-size:13px}
footer .wrap{padding:24px 20px}
mark{background:rgba(240,171,252,.28); color:inherit; border-radius:3px; padding:0 2px}
@media (max-width:640px){ .stats{gap:8px} .stat{min-width:88px; padding:8px 12px} }
</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="kicker">graphs · 3D · spectra · images · reactions · text</div>
    <h1>Awesome Multi-Modal Chemistry</h1>
    <p class="tagline">A curated, searchable index of the models, datasets and benchmarks that learn across
    chemically meaningful modalities — every entry checked against its primary source.</p>
    <div class="stats" id="stats"></div>
    <div class="hlinks">
      <a href="__REPO__">GitHub repository</a>
      <a href="__REPO__/blob/main/REVIEW.md">Full review</a>
      <a href="__REPO__/blob/main/CONTRIBUTING.md">Contribute</a>
      <button class="ghost" id="theme">Light mode</button>
    </div>
  </div>
</header>

<div class="controls">
  <div class="wrap">
    <div class="searchrow">
      <input id="q" type="search" placeholder="Search models, datasets, tasks, modalities…  (press /)" autocomplete="off">
    </div>
    <div class="tabs" id="tabs"></div>
    <div class="filters" id="modfilters"></div>
  </div>
</div>

<main>
  <div class="wrap">
    <div id="count"></div>
    <div id="results"></div>
  </div>
</main>

<footer><div class="wrap">
  Content licensed <a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0 1.0</a>.
  Generated from <code>data/*.yaml</code> by <code>scripts/build.py</code>. Last built __DATE__.
</div></footer>

<script>
const DATA = __PAYLOAD__;
const CATS = __CATS__;
const PAIRS = __PAIRS__;
const MODS = __MODS__;

let tab = 'models';
let activeMods = new Set();
const $ = s => document.querySelector(s);

function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function hl(s, q){
  if(!q) return esc(s);
  const re = new RegExp('(' + q.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&') + ')','ig');
  return esc(s).replace(re, '<mark>$1</mark>');
}

function renderStats(){
  const s = DATA.stats;
  $('#stats').innerHTML = [
    [s.models,'models'],[s.datasets,'datasets'],[s.reading,'papers'],[s.withCode,'with code']
  ].map(([n,l]) => `<div class="stat"><b>${n}</b><span>${l}</span></div>`).join('');
}

function renderTabs(){
  $('#tabs').innerHTML = [['models','Models & systems'],['datasets','Datasets & benchmarks'],['reading','Reading list']]
    .map(([k,l]) => `<button class="tab ${k===tab?'on':''}" data-tab="${k}">${l}</button>`).join('');
  $('#tabs').querySelectorAll('.tab').forEach(b => b.onclick = () => { tab = b.dataset.tab; renderTabs(); renderFilters(); render(); });
}

function renderFilters(){
  const box = $('#modfilters');
  if(tab === 'reading'){ box.innerHTML = ''; return; }
  box.innerHTML = '<span class="flabel">modality</span>' + MODS
    .map(m => `<button class="chip ${activeMods.has(m)?'on':''}" data-mod="${m}">${m}</button>`).join('')
    + ' <button class="chip" data-mod="__clear">clear</button>';
  box.querySelectorAll('.chip').forEach(c => c.onclick = () => {
    const m = c.dataset.mod;
    if(m === '__clear') activeMods.clear();
    else activeMods.has(m) ? activeMods.delete(m) : activeMods.add(m);
    renderFilters(); render();
  });
}

function matches(e, q){
  if(activeMods.size && ![...activeMods].every(m => (e.modalities||[]).includes(m))) return false;
  if(!q) return true;
  return (e._blob || '').includes(q.toLowerCase());
}

function cardModel(e, q){
  const links = [];
  if(e.paper) links.push(`<a href="${e.paper}">paper</a>`);
  if(e.code) links.push(`<a href="${e.code}">code</a>`);
  return `<div class="card">
    <div class="ctop"><span class="cname">${hl(e.name,q)}</span><span class="cyear">${e.year}</span></div>
    <div class="ctitle">${hl(e.title,q)} · ${esc(e.venue)}</div>
    <div class="tags">${(e.modalities||[]).map(m=>`<span class="tag">${esc(m)}</span>`).join('')}</div>
    <div class="cnote">${hl(e.note,q)}</div>
    <div class="cfoot">${links.join('')}<span class="lic">${esc(e.license||'')}</span></div>
  </div>`;
}

function cardDataset(e, q){
  const links = [];
  if(e.paper) links.push(`<a href="${e.paper}">paper</a>`);
  if(e.data) links.push(`<a href="${e.data}">data</a>`);
  return `<div class="card">
    <div class="ctop"><span class="cname">${hl(e.name,q)}</span><span class="cyear">${e.year}</span></div>
    <div class="ctitle">${esc(e.kindLabel)} · ${esc(e.venue)}</div>
    <div class="tags">${(e.modalities||[]).map(m=>`<span class="tag">${esc(m)}</span>`).join('')}</div>
    <div class="scale">${hl(e.scale,q)}</div>
    <div class="cnote">${hl(e.note,q)}</div>
    <div class="cfoot">${links.join('')}<span class="lic">${esc(e.license||'')}</span></div>
  </div>`;
}

function render(){
  const q = $('#q').value.trim();
  const out = [];
  let n = 0;

  if(tab === 'models'){
    for(const [key,label,blurb] of CATS){
      const rows = DATA.models.filter(m => m.category===key && matches(m,q));
      if(!rows.length) continue;
      n += rows.length;
      out.push(`<section class="group"><h2>${label}</h2><p>${blurb}</p>
        <div class="grid">${rows.map(r=>cardModel(r,q)).join('')}</div></section>`);
    }
  } else if(tab === 'datasets'){
    const rows = DATA.datasets.filter(d => matches(d,q));
    for(const [key,label,blurb] of PAIRS){
      const g = rows.filter(d => d.pairing === key);
      if(!g.length) continue;
      n += g.length;
      out.push(`<section class="group"><h2>${label}</h2><p>${blurb}</p>
        <div class="grid">${g.map(r=>cardDataset(r,q)).join('')}</div></section>`);
    }
  } else {
    for(const [key,label] of [['surveys','Surveys & reviews'],['critiques','Critical evaluations'],['governance','Safety & governance']]){
      const rows = (DATA.reading[key]||[]).filter(r => matches(r,q));
      if(!rows.length) continue;
      n += rows.length;
      out.push(`<section class="group"><h2>${label}</h2>` + rows.map(r =>
        `<div class="readitem"><div class="rt"><a href="${r.url}">${hl(r.title,q)}</a></div>
         <div class="rm">${esc(r.authors)} · ${esc(r.venue)} · ${r.year}</div>
         <div class="rn">${hl(r.note,q)}</div></div>`).join('') + `</section>`);
    }
  }

  $('#results').innerHTML = out.join('') || `<p style="color:var(--dim)">Nothing matches that. Try a broader term.</p>`;
  $('#count').textContent = `${n} ${n===1?'entry':'entries'}`;
}

$('#q').addEventListener('input', render);
document.addEventListener('keydown', e => {
  if(e.key === '/' && document.activeElement !== $('#q')){ e.preventDefault(); $('#q').focus(); }
  if(e.key === 'Escape'){ $('#q').value=''; render(); $('#q').blur(); }
});
$('#theme').onclick = () => {
  const el = document.documentElement;
  const next = el.dataset.theme === 'dark' ? 'light' : 'dark';
  el.dataset.theme = next;
  $('#theme').textContent = next === 'dark' ? 'Light mode' : 'Dark mode';
};

renderStats(); renderTabs(); renderFilters(); render();
</script>
</body>
</html>
"""


def build_site(models, datasets, reading, stats, repo_url: str, date: str) -> str:
    def blob(*parts) -> str:
        return " ".join(str(p) for p in parts if p).lower()

    m_out = []
    for m in sorted(models, key=lambda m: (m["year"], m["name"].lower())):
        e = {
            "name": m["name"], "title": m["title"], "year": m["year"], "venue": m["venue"],
            "paper": m["paper"], "code": m.get("code"), "license": m.get("license", ""),
            "category": m["category"], "modalities": mods(m), "note": m["note"],
        }
        e["_blob"] = blob(m["name"], m["title"], m["venue"], m["note"],
                          " ".join(mods(m)), " ".join(m.get("tasks", [])), m["category"])
        m_out.append(e)

    d_out = []
    for d in sorted(datasets, key=lambda d: (d["year"], d["name"].lower())):
        e = {
            "name": d["name"], "title": d["title"], "year": d["year"], "venue": d["venue"],
            "paper": d.get("paper"), "data": d.get("data"), "license": d.get("license", ""),
            "kindLabel": KIND_LABEL.get(d.get("kind"), "Other"),
            "pairing": d.get("pairing", ""),
            "modalities": mods(d), "scale": d["scale"], "note": d["note"],
        }
        e["_blob"] = blob(d["name"], d["title"], d["venue"], d["note"], d["scale"],
                          " ".join(mods(d)), d.get("pairing", ""))
        d_out.append(e)

    r_out = {}
    for key, items in reading.items():
        lst = []
        for it in sorted(items, key=lambda i: (-i["year"], i["title"])):
            e = dict(it)
            e["_blob"] = blob(it["title"], it["authors"], it["venue"], it["note"])
            lst.append(e)
        r_out[key] = lst

    used = []
    for m in m_out + d_out:
        for t in m["modalities"]:
            if t not in used:
                used.append(t)

    payload = {"models": m_out, "datasets": d_out, "reading": r_out, "stats": stats}
    return (SITE_TEMPLATE
            .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
            .replace("__CATS__", json.dumps([[k, t, b] for k, t, b in CATEGORIES], ensure_ascii=False))
            .replace("__PAIRS__", json.dumps([[k, t, b] for k, t, b in PAIRINGS], ensure_ascii=False))
            .replace("__MODS__", json.dumps(sorted(used, key=str.lower), ensure_ascii=False))
            .replace("__REPO__", repo_url)
            .replace("__DATE__", date))


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if output is stale")
    ap.add_argument("--repo", default="https://github.com/AxelRolov/awesome-multimodal-chemistry")
    ap.add_argument("--date", default="", help="build date stamp for the site footer")
    args = ap.parse_args()

    models = load("models.yaml")
    datasets = load("datasets.yaml")
    reading = load("reading.yaml")

    stats = {
        "models": len(models),
        "datasets": len(datasets),
        "reading": sum(len(v) for v in reading.values()),
        "withCode": sum(1 for m in models if m.get("code")),
    }

    readme = README.read_text(encoding="utf-8")
    new_readme = readme
    new_readme = inject(new_readme, "stats", md_stats(models, datasets, reading))
    new_readme = inject(new_readme, "models", md_models(models))
    new_readme = inject(new_readme, "datasets", md_datasets(datasets))
    new_readme = inject(new_readme, "reading", md_reading(reading))
    new_readme = inject(new_readme, "timeline", md_timeline(models))

    site = build_site(models, datasets, reading, stats, args.repo, args.date or "from data/")

    if args.check:
        stale = []
        if new_readme != readme:
            stale.append("README.md")
        if not SITE.exists() or SITE.read_text(encoding="utf-8") != site:
            stale.append("docs/index.html")
        if stale:
            print("stale, run `python scripts/build.py`: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("up to date")
        return 0

    README.write_text(new_readme, encoding="utf-8")
    SITE.parent.mkdir(parents=True, exist_ok=True)
    SITE.write_text(site, encoding="utf-8")
    print(f"wrote README.md and docs/index.html "
          f"({stats['models']} models, {stats['datasets']} datasets, {stats['reading']} papers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
