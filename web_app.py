"""
web_app.py — Flask replacement for app.py
Serves the ScholarRAG UI matching scholarrag_pure_white_final.html exactly.
Run with:  python web_app.py
"""

import os, sys, json, threading
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context

app = Flask(__name__)
app.secret_key = "scholarrag-secret"

# ── In-memory session store (single-user dev server) ─────────────────────────
_state = {
    "vector_store":  None,
    "papers_df":     None,   # all fetched papers (raw)
    "chat_messages": [],
    "current_topic": "",
    "pipeline_ready": False,
    "ragas_enabled": False,
    "history":       [],     # [{topic, messages}]
}

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ScholarRAG</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&family=Crimson+Pro:ital,wght@0,400;1,400&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;width:100%;overflow:hidden;font-family:'DM Sans',sans-serif;}
.app{display:flex;width:100vw;height:100vh;background:#fff;overflow:hidden;}

/* ── Sidebar ── */
.sb{width:270px;background:#f5f5f5;border-right:1px solid #e8e8e8;padding:16px 14px 12px;display:flex;flex-direction:column;flex-shrink:0;transition:width 0.25s ease,padding 0.25s ease;overflow:hidden;}
.sb.collapsed{width:0;padding:0;}
.sb-inner{width:242px;display:flex;flex-direction:column;height:100%;}
.sb-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;}
.sb-name{display:flex;align-items:center;gap:8px;font-size:18px;font-weight:700;color:#141414;font-family:'Playfair Display',serif;white-space:nowrap;}
.sb-collapse{cursor:pointer;opacity:0.35;line-height:0;flex-shrink:0;user-select:none;}
.sb-collapse:hover{opacity:0.8;}
.nav-item{display:flex;align-items:center;gap:10px;padding:7px 6px;border-radius:6px;font-size:15px;color:#141414;cursor:pointer;white-space:nowrap;}
.nav-item:hover{background:#ebebeb;}
.nav-icon{width:16px;height:16px;flex-shrink:0;display:flex;align-items:center;justify-content:center;}
.model-row{display:flex;align-items:center;gap:10px;padding:7px 6px;border-radius:6px;font-size:15px;color:#141414;cursor:pointer;position:relative;}
.model-row:hover{background:#ebebeb;}
.model-val{font-size:13px;color:#707070;margin-left:auto;}
.model-dropdown{position:absolute;left:0;top:34px;background:#fff;border:1px solid #e8e8e8;border-radius:8px;padding:4px;z-index:200;min-width:190px;box-shadow:0 4px 16px rgba(0,0,0,0.10);display:none;}
.model-dropdown.show{display:block;}
.model-opt{padding:6px 10px;border-radius:5px;font-size:13px;cursor:pointer;color:#141414;}
.model-opt:hover{background:#f5f5f5;}
.model-opt.active-opt{font-weight:600;}
.ragas-row{display:flex;align-items:center;gap:10px;padding:7px 6px;border-radius:6px;font-size:15px;color:#141414;cursor:pointer;}
.ragas-row:hover{background:#ebebeb;}
.toggle{width:28px;height:16px;background:#d0d0d0;border-radius:8px;margin-left:auto;position:relative;flex-shrink:0;transition:background 0.2s;}
.toggle.on{background:#404040;}
.toggle-dot{width:12px;height:12px;background:#fff;border-radius:50%;position:absolute;top:2px;left:2px;transition:left 0.2s;}
.toggle.on .toggle-dot{left:14px;}
.sb-divider{height:1px;background:#e8e8e8;margin:8px 0;}
.sb-recents-label{font-size:13px;font-weight:600;color:#a0a0a0;margin:4px 6px 4px;}
.hist-scroll{flex:1;overflow-y:auto;}
.hist-item{display:flex;align-items:center;padding:6px 6px;border-radius:6px;font-size:14px;color:#141414;cursor:pointer;position:relative;margin-bottom:1px;}
.hist-item:hover{background:#ebebeb;}
.hist-item.active{background:#e8e8e8;}
.hist-text{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.hist-dots{background:none;border:none;color:#707070;cursor:pointer;font-size:14px;padding:0 3px;display:none;line-height:1;font-weight:700;flex-shrink:0;}
.hist-item:hover .hist-dots,.hist-item.active .hist-dots{display:flex;align-items:center;}
.dropdown{position:absolute;right:0;top:28px;background:#fff;border:1px solid #e8e8e8;border-radius:8px;padding:4px;z-index:100;min-width:110px;box-shadow:0 4px 16px rgba(0,0,0,0.10);display:none;}
.dropdown.show{display:block;}
.dd-item{display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:5px;font-size:13px;cursor:pointer;}
.dd-item.danger{color:#c0392b;}
.dd-item.danger:hover{background:#fdf0ee;}
.sb-user{display:flex;align-items:center;gap:8px;padding:8px 6px 4px;border-top:1px solid #e8e8e8;margin-top:4px;cursor:pointer;border-radius:6px;}
.sb-user:hover{background:#ebebeb;}
.sb-avatar{width:28px;height:28px;border-radius:50%;background:#141414;color:#fff;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.sb-userinfo{display:flex;flex-direction:column;}
.sb-username{font-size:14px;font-weight:600;color:#141414;}
.sb-plan{font-size:12px;color:#a0a0a0;}

/* ── Main ── */
.main-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.topbar{display:flex;align-items:center;padding:5px 12px;border-bottom:1px solid #e8e8e8;background:#fff;min-height:30px;}
.sb-open{display:none;cursor:pointer;opacity:0.35;line-height:0;margin-right:auto;user-select:none;}
.sb-open:hover{opacity:0.8;}
.ref-toggle{display:flex;align-items:center;gap:5px;cursor:pointer;opacity:0.45;margin-left:auto;user-select:none;}
.ref-toggle:hover{opacity:0.85;}
.ref-toggle-lbl{font-size:9px;font-weight:600;color:#404040;}
.body-wrap{flex:1;display:flex;overflow:hidden;}
.chat-body{flex:1;padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;}

/* Messages */
.welcome{margin:auto;display:flex;flex-direction:column;align-items:center;gap:8px;text-align:center;}
.welcome-greet{font-size:26px;font-weight:700;color:#141414;font-family:'Playfair Display',serif;}
.welcome-sub{font-size:14px;color:#707070;max-width:300px;line-height:1.6;}
.user-msg{background:#efefef;border-radius:8px 8px 2px 8px;padding:10px 14px;font-size:14px;color:#141414;align-self:flex-end;max-width:80%;}
.system-msg{background:#f5f5f5;border:1px solid #e8e8e8;border-radius:6px;padding:10px 14px;font-size:13px;color:#707070;}
.loading-bar{background:#e8e8e8;border-radius:4px;height:3px;overflow:hidden;margin-top:6px;}
.loading-fill{height:100%;background:#404040;border-radius:4px;animation:fill 2s ease-in-out infinite;}
@keyframes fill{0%{width:10%}50%{width:80%}100%{width:10%}}
.step{font-size:13px;color:#a0a0a0;display:flex;align-items:center;gap:6px;margin-top:4px;}
.dot{width:5px;height:5px;border-radius:50%;background:#d0d0d0;flex-shrink:0;}
.dot.done{background:#404040;}
.dot.active{background:#404040;animation:pulse 1s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.refer-tag{font-size:11px;color:#a0a0a0;font-style:italic;}
.answer-box{background:#f5f5f5;border:1px solid #e8e8e8;border-left:2px solid #606060;border-radius:6px;padding:14px 16px;font-size:15px;line-height:1.7;color:#141414;font-family:'Crimson Pro',serif;}
.ah{font-size:13px;font-weight:700;color:#404040;font-family:'DM Sans',sans-serif;margin:8px 0 2px;}
.ah:first-child{margin-top:0;}
.cite{display:inline-flex;width:14px;height:14px;border-radius:50%;background:#e8e8e8;color:#404040;font-size:7px;font-weight:700;align-items:center;justify-content:center;margin:0 1px;vertical-align:middle;cursor:pointer;}
.cite:hover{background:#d0d0d0;}
.followups{display:flex;flex-direction:column;gap:3px;}
.fq{background:#f5f5f5;border:1px solid #e8e8e8;border-radius:4px;padding:7px 12px;font-size:13px;color:#141414;cursor:pointer;display:flex;justify-content:space-between;}
.fq:hover{background:#efefef;}
.hint{font-size:13px;color:#a0a0a0;text-align:center;padding:3px 0 4px;background:#fff;}
.chat-input{padding:8px 16px;border-top:1px solid #e8e8e8;display:flex;gap:8px;background:#fff;}
.input-box{flex:1;background:#f5f5f5;border:1px solid #e0e0e0;border-radius:12px;padding:12px 16px;font-size:15px;color:#141414;outline:none;}
.input-box:focus{border-color:#b0b0b0;}
.send-btn{background:#303030;color:#fff;border:none;border-radius:12px;padding:12px 20px;font-size:14px;font-weight:600;cursor:pointer;}
.send-btn:hover{background:#141414;}
.send-btn:disabled{background:#c0c0c0;cursor:not-allowed;}

/* ── Right panel ── */
.rp{width:0;background:#fff;border-left:1px solid #e8e8e8;transition:width 0.3s ease;overflow:hidden;flex-shrink:0;}
.rp.open{width:265px;}
.rp-inner{width:265px;padding:12px;display:flex;flex-direction:column;height:100%;}
.rp-tabs{display:flex;align-items:center;border-bottom:1px solid #e8e8e8;padding-bottom:7px;margin-bottom:8px;}
.rp-tab{font-size:13px;font-weight:600;color:#a0a0a0;cursor:pointer;padding-bottom:4px;margin-right:14px;}
.rp-tab.active{color:#141414;border-bottom:2px solid #303030;}
.rp-close{background:none;border:none;font-size:13px;color:#a0a0a0;cursor:pointer;padding:0;margin-left:auto;line-height:1;opacity:0.6;}
.rp-close:hover{opacity:1;}
.ref-list{flex:1;overflow-y:auto;}
.ref-item{border-bottom:1px solid #ececec;padding:9px 2px;}
.ref-item:last-child{border-bottom:none;}
.ref-num{font-size:11px;color:#a0a0a0;font-weight:600;margin-bottom:2px;}
.ref-title{font-size:13px;font-weight:600;color:#141414;line-height:1.4;margin-bottom:3px;}
.ref-meta{font-size:11px;color:#a0a0a0;}
.ref-link{font-size:11px;color:#404040;}
.ragas-box{background:#f5f5f5;border-radius:6px;padding:9px;margin-top:10px;border:1px solid #e8e8e8;display:none;}
.ragas-box.visible{display:block;}
.ragas-lbl{font-size:11px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:#a0a0a0;margin-bottom:7px;}
.r-row{display:flex;justify-content:space-between;font-size:12px;color:#404040;margin-bottom:2px;}
.r-bg{height:4px;background:#e0e0e0;border-radius:2px;margin-bottom:6px;}
.r-fill{height:4px;border-radius:2px;background:#404040;}
</style>
</head>
<body>

<div class="app">
  <!-- Sidebar -->
  <div class="sb" id="sb">
    <div class="sb-inner">
      <div class="sb-top">
        <div class="sb-name">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><rect x="2" y="3" width="16" height="13" rx="2" stroke="#505050" stroke-width="1.4" fill="none"/><line x1="6" y1="7" x2="14" y2="7" stroke="#505050" stroke-width="1.2" stroke-linecap="round"/><line x1="6" y1="10" x2="14" y2="10" stroke="#505050" stroke-width="1.2" stroke-linecap="round"/><line x1="6" y1="13" x2="10" y2="13" stroke="#505050" stroke-width="1.2" stroke-linecap="round"/></svg>
          ScholarRAG
        </div>
        <div class="sb-collapse" onclick="toggleSidebar()">
          <svg width="17" height="17" viewBox="0 0 18 18" fill="none"><rect x="1.5" y="1.5" width="15" height="15" rx="3" stroke="#404040" stroke-width="1.3" fill="none"/><line x1="6" y1="2" x2="6" y2="16" stroke="#404040" stroke-width="1.3"/></svg>
        </div>
      </div>

      <div class="nav-item" onclick="newChat()">
        <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 15 15" fill="none"><line x1="7.5" y1="2" x2="7.5" y2="13" stroke="#404040" stroke-width="1.5" stroke-linecap="round"/><line x1="2" y1="7.5" x2="13" y2="7.5" stroke="#404040" stroke-width="1.5" stroke-linecap="round"/></svg></div>
        New chat
      </div>

      <!-- Model selector -->
      <div class="model-row" onclick="toggleModelMenu(event)">
        <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 15 15" fill="none"><circle cx="7.5" cy="7.5" r="5.5" stroke="#404040" stroke-width="1.3" fill="none"/><circle cx="7.5" cy="7.5" r="2" stroke="#404040" stroke-width="1.3" fill="none"/></svg></div>
        Model<span class="model-val" id="model-val">Gemini Flash ▾</span>
        <div class="model-dropdown" id="model-menu"></div>
      </div>

      <div class="ragas-row" onclick="toggleRagas()">
        <div class="nav-icon"><svg width="15" height="15" viewBox="0 0 15 15" fill="none"><rect x="1" y="4" width="13" height="2" rx="1" fill="#404040"/><rect x="1" y="8" width="9" height="2" rx="1" fill="#404040"/><rect x="1" y="12" width="11" height="2" rx="1" fill="#404040"/></svg></div>
        RAGAS scores
        <div class="toggle" id="ragas-toggle"><div class="toggle-dot"></div></div>
      </div>

      <div class="sb-divider"></div>

      <div style="flex:1;"></div>
      <div class="sb-user">
        <div class="sb-avatar">S</div>
        <div class="sb-userinfo">
          <span class="sb-username">ScholarRAG</span>
          <span class="sb-plan">Research Assistant</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Main -->
  <div class="main-wrap">
    <div class="topbar">
      <div class="sb-open" id="sb-open" onclick="toggleSidebar()">
        <svg width="17" height="17" viewBox="0 0 18 18" fill="none"><rect x="1.5" y="1.5" width="15" height="15" rx="3" stroke="#404040" stroke-width="1.3" fill="none"/><line x1="6" y1="2" x2="6" y2="16" stroke="#404040" stroke-width="1.3"/></svg>
      </div>
      <div class="ref-toggle" onclick="togglePanel()">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><line x1="2" y1="4" x2="14" y2="4" stroke="#404040" stroke-width="1.4" stroke-linecap="round"/><line x1="2" y1="8" x2="14" y2="8" stroke="#404040" stroke-width="1.4" stroke-linecap="round"/><line x1="2" y1="12" x2="10" y2="12" stroke="#404040" stroke-width="1.4" stroke-linecap="round"/></svg>
        <span class="ref-toggle-lbl" id="panel-label">References</span>
      </div>
    </div>

    <div class="body-wrap">
      <div style="flex:1;display:flex;flex-direction:column;overflow:hidden;">
        <div class="chat-body" id="chat-body">
          <div id="welcome" class="welcome">
            <div class="welcome-greet" id="greet"></div>
            <div class="welcome-sub">Type a research topic below to get started. I'll find and index the most relevant papers for you.</div>
          </div>
        </div>
        <div class="hint" id="hint">Type your research topic to get started</div>
        <div class="chat-input">
          <input class="input-box" id="inp" placeholder="e.g. transformer attention mechanisms…">
          <button class="send-btn" id="send-btn" onclick="handleSend()">Send →</button>
        </div>
      </div>

      <!-- Right panel -->
      <div class="rp" id="rp">
        <div class="rp-inner">
          <div class="rp-tabs">
            <div class="rp-tab active" onclick="switchTab(this,'r')">References</div>
            <div class="rp-tab" onclick="switchTab(this,'p')">All Papers</div>
            <button class="rp-close" onclick="togglePanel()">✕</button>
          </div>
          <div class="ref-list" id="r-content"></div>
          <div class="ref-list" id="p-content" style="display:none;"></div>
          <div class="ragas-box" id="ragas-box">
            <div class="ragas-lbl">RAGAS Scores</div>
            <div id="ragas-scores"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let pipelineReady = false;
let panelOpen = false;
let sbOpen = true;
let ragasEnabled = false;
let openMenu = null;
let modelMenuOpen = false;
let activeModel = 'gemini';
let models = {};
let loadingPipeline = false;
let lastSources = [];
let lastPapers = [];
let history = []; // [{topic, id}]

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  const h = new Date().getHours();
  document.getElementById('greet').textContent = h < 12 ? 'Good Morning!' : h < 17 ? 'Good Afternoon!' : 'Good Evening!';
  const res = await fetch('/api/models');
  const data = await res.json();
  models = data.models;
  activeModel = data.active;
  renderModelMenu();
  updateModelLabel();
  await loadHistory();
})();

document.addEventListener('click', e => {
  if (openMenu && !e.target.closest('.hist-dots') && !e.target.closest('.dropdown')) {
    document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('show')); openMenu = null;
  }
  if (modelMenuOpen && !e.target.closest('.model-row')) {
    document.getElementById('model-menu').classList.remove('show'); modelMenuOpen = false;
  }
});

document.getElementById('inp').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
});

// ── Models ─────────────────────────────────────────────────────────────────
function renderModelMenu() {
  const menu = document.getElementById('model-menu');
  menu.innerHTML = Object.entries(models).map(([k, v]) =>
    `<div class="model-opt ${k === activeModel ? 'active-opt' : ''}" onclick="selectModel('${k}')">${v.label}</div>`
  ).join('');
}

function updateModelLabel() {
  const lbl = models[activeModel]?.label || activeModel;
  const short = lbl.split('—')[0].trim();
  document.getElementById('model-val').textContent = short + ' ▾';
}

function toggleModelMenu(e) {
  e.stopPropagation();
  modelMenuOpen = !modelMenuOpen;
  document.getElementById('model-menu').classList.toggle('show', modelMenuOpen);
}

async function selectModel(key) {
  activeModel = key;
  updateModelLabel();
  renderModelMenu();
  document.getElementById('model-menu').classList.remove('show');
  modelMenuOpen = false;
  await fetch('/api/set_model', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({model: key}) });
}

// ── History sidebar ────────────────────────────────────────────────────────
async function loadHistory() {
  const res = await fetch('/api/history');
  history = await res.json();
  renderHistory();
}

function renderHistory() {
  const el = document.getElementById('hist-scroll');
  el.innerHTML = history.map((h, i) => `
    <div class="hist-item ${i === 0 && pipelineReady ? 'active' : ''}" onclick="loadSession(${i})">
      <span class="hist-text">${escHtml(h.topic)}</span>
      <button class="hist-dots" onclick="toggleMenu(event,'hm${i}')">···</button>
      <div class="dropdown" id="hm${i}">
        <div class="dd-item danger" onclick="deleteSession(${i})">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 3h8M5 3V2h2v1M4.5 3l.5 7M7.5 3l-.5 7" stroke="#c0392b" stroke-width="1.2" stroke-linecap="round"/></svg>
          Delete
        </div>
      </div>
    </div>`).join('');
}

async function deleteSession(idx) {
  await fetch('/api/history/' + idx, { method: 'DELETE' });
  await loadHistory();
}

async function loadSession(idx) {
  const res = await fetch('/api/history/' + idx + '/load', { method: 'POST' });
  const data = await res.json();
  if (data.ok) {
    pipelineReady = true;
    document.getElementById('welcome').style.display = 'none';
    document.getElementById('hint').textContent = 'Ask a follow-up question';
    document.getElementById('inp').placeholder = 'Ask about the papers…';
    renderMessages(data.messages);
    renderPapersPanel(data.all_papers || []);
    if (!panelOpen) togglePanel();
    await loadHistory();
  }
}

// ── Chat rendering ─────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function circledNum(n) {
  const nums = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩'];
  return nums[n - 1] || String(n);
}

function renderMessages(msgs) {
  const body = document.getElementById('chat-body');
  // Clear except welcome
  Array.from(body.children).forEach(c => { if (c.id !== 'welcome') c.remove(); });
  msgs.forEach(m => appendMessage(m, false));
}

function appendMessage(msg, scroll = true) {
  const body = document.getElementById('chat-body');
  const div = document.createElement('div');
  if (msg.role === 'user') {
    div.className = 'user-msg';
    div.textContent = msg.content;
  } else {
    div.style.display = 'flex';
    div.style.flexDirection = 'column';
    div.style.gap = '5px';
    let html = '';
    if (msg.sources && msg.sources.length) {
      html += `<div class="refer-tag">Referring to ${msg.sources.length} article${msg.sources.length > 1 ? 's' : ''} for the answer</div>`;
    }
    html += `<div class="answer-box">${formatAnswer(msg.content, msg.sources)}</div>`;
    if (msg.followups && msg.followups.length) {
      html += '<div class="followups">' + msg.followups.map(q =>
        `<div class="fq" onclick="askFollowup(${JSON.stringify(escHtml(q))})">${escHtml(q)} →</div>`
      ).join('') + '</div>';
    }
    div.innerHTML = html;
    if (msg.sources) {
      lastSources = msg.sources;
      renderRefPanel(msg.sources);
    }
    if (msg.scores) renderRagas(msg.scores);
  }
  body.appendChild(div);
  if (scroll) div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function formatAnswer(text, sources) {
  // Replace [N] citation markers with styled circles
  return escHtml(text).replace(/\[(\d+)\]/g, (_, n) => {
    const i = parseInt(n) - 1;
    const src = sources && sources[i];
    const title = src ? escHtml(src.title) : '';
    return `<span class="cite" title="${title}">${circledNum(parseInt(n))}</span>`;
  });
}

// ── Panels ─────────────────────────────────────────────────────────────────
function renderRefPanel(sources) {
  const el = document.getElementById('r-content');
  el.innerHTML = sources.map((s, i) => `
    <div class="ref-item">
      <div class="ref-num">${circledNum(i+1)} ${s.year || ''}</div>
      <div class="ref-title">${escHtml(s.title)}</div>
      <div class="ref-meta">${escHtml((s.authors||'').substring(0,50))} · arXiv
        ${s.url ? `&nbsp;<a class="ref-link" href="${escHtml(s.url)}" target="_blank">↗</a>` : ''}
      </div>
    </div>`).join('');
}

function renderPapersPanel(papers) {
  lastPapers = papers || [];
  const el = document.getElementById('p-content');
  el.innerHTML = lastPapers.map((p, i) => `
    <div class="ref-item">
      <div class="ref-num">#${i+1} · ${p.year||''}</div>
      <div class="ref-title">${escHtml(p.title)}</div>
      <div class="ref-meta">${escHtml((p.authors||'').substring(0,50))}
        ${p.paper_url ? `&nbsp;<a class="ref-link" href="${escHtml(p.paper_url)}" target="_blank">↗</a>` : ''}
      </div>
    </div>`).join('');
}

function renderRagas(scores) {
  if (!ragasEnabled) return;
  const box = document.getElementById('ragas-box');
  box.classList.add('visible');
  document.getElementById('ragas-scores').innerHTML = Object.entries(scores).map(([k,v]) => `
    <div class="r-row"><span>${k.replace(/_/g,' ')}</span><span>${v.toFixed(2)}</span></div>
    <div class="r-bg"><div class="r-fill" style="width:${Math.round(v*100)}%"></div></div>
  `).join('');
}

// ── Send logic ─────────────────────────────────────────────────────────────
async function handleSend() {
  const inp = document.getElementById('inp');
  const text = inp.value.trim();
  if (!text || loadingPipeline) return;
  inp.value = '';

  if (!pipelineReady) {
    await startPipeline(text);
  } else {
    await askQuestion(text);
  }
}

async function askFollowup(q) {
  document.getElementById('inp').value = q;
  await askQuestion(q);
}

async function startPipeline(topic) {
  loadingPipeline = true;
  setBusy(true);
  document.getElementById('welcome').style.display = 'none';

  // User msg
  const body = document.getElementById('chat-body');
  const userDiv = document.createElement('div');
  userDiv.className = 'user-msg';
  userDiv.textContent = topic;
  body.appendChild(userDiv);

  // Loading msg
  const loadDiv = document.createElement('div');
  loadDiv.className = 'system-msg';
  loadDiv.id = 'loading-msg';
  loadDiv.innerHTML = `
    <div style="font-weight:600;color:#141414;margin-bottom:4px;font-size:11px;">🔍 Building pipeline…</div>
    <div class="loading-bar"><div class="loading-fill"></div></div>
    <div class="step" id="step0"><div class="dot active"></div><span>Expanding topic queries…</span></div>
    <div class="step" id="step1"><div class="dot"></div><span>Fetching papers from arXiv…</span></div>
    <div class="step" id="step2"><div class="dot"></div><span>Ranking papers…</span></div>
    <div class="step" id="step3"><div class="dot"></div><span>Semantic chunking…</span></div>
    <div class="step" id="step4"><div class="dot"></div><span>Building vector index…</span></div>`;
  body.appendChild(loadDiv);
  loadDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });

  document.getElementById('hint').textContent = 'Indexing papers, please wait…';
  document.getElementById('inp').placeholder = 'Please wait…';

  try {
    // Stream pipeline progress
    const evtSource = new EventSource('/api/build_pipeline?topic=' + encodeURIComponent(topic));
    evtSource.onmessage = e => {
      const msg = JSON.parse(e.data);
      if (msg.step !== undefined) {
        // mark previous steps done
        for (let i = 0; i < msg.step; i++) {
          const s = document.getElementById('step' + i);
          if (s) {
            s.querySelector('.dot').className = 'dot done';
            s.querySelector('span').style.color = '#404040';
            s.querySelector('span').textContent = '✓ ' + s.querySelector('span').textContent.replace(/^✓ /, '');
          }
        }
        const cur = document.getElementById('step' + msg.step);
        if (cur) {
          cur.querySelector('.dot').className = 'dot active';
          if (msg.text) cur.querySelector('span').textContent = msg.text;
        }
      }
      if (msg.done) {
        evtSource.close();
        finishPipeline(msg);
      }
      if (msg.error) {
        evtSource.close();
        loadDiv.innerHTML = `<span style="color:#c0392b">⚠️ ${escHtml(msg.error)}</span>`;
        loadingPipeline = false;
        setBusy(false);
        document.getElementById('hint').textContent = 'Type your research topic to get started';
        document.getElementById('inp').placeholder = 'e.g. transformer attention mechanisms…';
      }
    };
    evtSource.onerror = () => {
      evtSource.close();
      loadDiv.innerHTML = `<span style="color:#c0392b">⚠️ Connection error. Try again.</span>`;
      loadingPipeline = false;
      setBusy(false);
    };
  } catch (err) {
    loadDiv.innerHTML = `<span style="color:#c0392b">⚠️ ${escHtml(err.message)}</span>`;
    loadingPipeline = false;
    setBusy(false);
  }
}

function finishPipeline(msg) {
  loadingPipeline = false;
  pipelineReady = true;
  setBusy(false);

  const loadDiv = document.getElementById('loading-msg');
  if (loadDiv) {
    loadDiv.innerHTML = `✅ ${msg.paper_count} papers indexed · Ask your first question below`;
  }

  document.getElementById('hint').textContent = 'Ask a follow-up question';
  document.getElementById('inp').placeholder = 'Ask about the papers…';

  renderPapersPanel(msg.all_papers || []);
  if (!panelOpen) togglePanel();
  // Switch to All Papers tab since there are no references yet
  const allPapersTab = document.querySelectorAll('.rp-tab')[1];
  if (allPapersTab) switchTab(allPapersTab, 'p');

  // Auto-generate overview
  if (msg.overview) {
    const ovMsg = { role: 'assistant', content: msg.overview, sources: [], followups: [] };
    appendMessage(ovMsg);
  }

  loadHistory();
}

async function askQuestion(question) {
  setBusy(true);
  appendMessage({ role: 'user', content: question });

  // Thinking indicator
  const body = document.getElementById('chat-body');
  const thinkDiv = document.createElement('div');
  thinkDiv.className = 'system-msg';
  thinkDiv.id = 'thinking-msg';
  thinkDiv.innerHTML = '<span style="color:#a0a0a0;font-style:italic;font-size:10px;">Reading the papers…</span>';
  body.appendChild(thinkDiv);
  thinkDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, ragas: ragasEnabled })
    });
    const data = await res.json();
    thinkDiv.remove();
    appendMessage({
      role: 'assistant',
      content: data.answer,
      sources: data.sources || [],
      followups: data.followups || [],
      scores: data.scores
    });
  } catch (err) {
    thinkDiv.innerHTML = `<span style="color:#c0392b">⚠️ ${escHtml(err.message)}</span>`;
  }
  setBusy(false);
}

function setBusy(busy) {
  const btn = document.getElementById('send-btn');
  btn.disabled = busy;
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function newChat() {
  fetch('/api/new_chat', { method: 'POST' });
  pipelineReady = false;
  lastSources = [];
  lastPapers = [];
  const body = document.getElementById('chat-body');
  Array.from(body.children).forEach(c => { if (c.id !== 'welcome') c.remove(); });
  document.getElementById('welcome').style.display = '';
  document.getElementById('hint').textContent = 'Type your research topic to get started';
  document.getElementById('inp').placeholder = 'e.g. transformer attention mechanisms…';
  document.getElementById('r-content').innerHTML = '';
  document.getElementById('p-content').innerHTML = '';
  if (panelOpen) togglePanel();
}

function toggleRagas() {
  ragasEnabled = !ragasEnabled;
  document.getElementById('ragas-toggle').classList.toggle('on', ragasEnabled);
  document.getElementById('ragas-box').classList.toggle('visible', ragasEnabled && document.getElementById('ragas-scores').innerHTML);
}

function toggleMenu(e, id) {
  e.stopPropagation();
  const m = document.getElementById(id);
  const was = m.classList.contains('show');
  document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('show'));
  if (!was) { m.classList.add('show'); openMenu = id; } else openMenu = null;
}

function togglePanel() {
  panelOpen = !panelOpen;
  document.getElementById('rp').classList.toggle('open', panelOpen);
  document.getElementById('panel-label').textContent = panelOpen ? 'Close' : 'References';
}

function toggleSidebar() {
  sbOpen = !sbOpen;
  document.getElementById('sb').classList.toggle('collapsed', !sbOpen);
  document.getElementById('sb-open').style.display = sbOpen ? 'none' : 'block';
}

function switchTab(el, t) {
  document.querySelectorAll('.rp-tab').forEach(x => x.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('r-content').style.display = t === 'r' ? 'block' : 'none';
  document.getElementById('p-content').style.display = t === 'p' ? 'block' : 'none';
}
</script>
</body>
</html>
"""

# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/models")
def api_models():
    import config
    labels = {k: {"label": v["label"]} for k, v in config.MODEL_OPTIONS.items()}
    return jsonify({"models": labels, "active": config.ACTIVE_PROVIDER})


@app.route("/api/set_model", methods=["POST"])
def api_set_model():
    import config
    key = request.json.get("model", "gemini")
    if key in config.MODEL_OPTIONS:
        config.ACTIVE_PROVIDER = key
    return jsonify({"ok": True})


@app.route("/api/history")
def api_history():
    return jsonify(_state["history"])


@app.route("/api/history/<int:idx>", methods=["DELETE"])
def api_delete_history(idx):
    if 0 <= idx < len(_state["history"]):
        _state["history"].pop(idx)
    return jsonify({"ok": True})


@app.route("/api/history/<int:idx>/load", methods=["POST"])
def api_load_history(idx):
    if 0 <= idx < len(_state["history"]):
        sess = _state["history"][idx]
        _state["vector_store"]   = sess.get("vector_store")
        _state["papers_df"]      = sess.get("papers_df")
        _state["chat_messages"]  = sess.get("messages", [])
        _state["current_topic"]  = sess.get("topic", "")
        _state["pipeline_ready"] = True

        papers = []
        if _state["papers_df"] is not None:
            papers = _state["papers_df"].to_dict("records")

        return jsonify({
            "ok": True,
            "messages": _state["chat_messages"],
            "all_papers": papers,
        })
    return jsonify({"ok": False})


@app.route("/api/new_chat", methods=["POST"])
def api_new_chat():
    _state["vector_store"]     = None
    _state["papers_df"]      = None
    _state["chat_messages"]  = []
    _state["current_topic"]  = ""
    _state["pipeline_ready"] = False
    return jsonify({"ok": True})


@app.route("/api/build_pipeline")
def api_build_pipeline():
    topic = request.args.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    def generate():
        import config
        def _send(step=None, text=None, **kw):
            payload = {}
            if step is not None: payload["step"] = step
            if text:             payload["text"]  = text
            payload.update(kw)
            yield f"data: {json.dumps(payload)}\n\n"

        try:
            yield from _send(step=0, text="Expanding topic queries…")
            from pipeline import build_pipeline
            from src.answer_generator import generate_topic_overview

            yield from _send(step=1, text="Fetching papers from arXiv…")

            # Run the pipeline (blocking)
            rebuild = request.args.get("rebuild", "false") == "true"
            vs, df = build_pipeline(topic, force_rebuild=rebuild)

            yield from _send(step=2, text="Chunking all papers…")
            yield from _send(step=3, text="Building vector index…")
            yield from _send(step=4, text="Generating overview…")

            _state["vector_store"]   = vs
            _state["papers_df"]      = df
            _state["current_topic"]  = topic
            _state["pipeline_ready"] = True

            overview = ""
            try:
                overview = generate_topic_overview(topic, vs)
            except Exception as e:
                overview = f"⚠️ Could not generate overview: {e}"

            all_papers = df.to_dict("records")

            # Save to history
            _state["history"].insert(0, {
                "topic":      topic,
                "vector_store": vs,
                "papers_df":  df,
                "messages":   [],
            })

            yield from _send(
                done=True,
                paper_count=len(df),
                all_papers=all_papers,
                overview=overview,
            )

        except Exception as e:
            yield from _send(error=str(e))

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    if not _state["pipeline_ready"] or _state["vector_store"] is None:
        return jsonify({"error": "Pipeline not ready"}), 400

    data     = request.json
    question = data.get("question", "").strip()
    run_eval = data.get("ragas", False)

    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        from pipeline import ask_question
        result = ask_question(question, _state["vector_store"], run_evaluation=run_eval)

        # Generate follow-up suggestions
        followups = _generate_followups(question, result["answer"])

        msg = {
            "role":      "assistant",
            "content":   result["answer"],
            "sources":   result.get("sources", []),
            "followups": followups,
            "scores":    result.get("scores"),
        }
        _state["chat_messages"].append({"role": "user",      "content": question})
        _state["chat_messages"].append(msg)

        # Update history entry
        if _state["history"]:
            _state["history"][0]["messages"] = _state["chat_messages"]

        return jsonify({
            "answer":    result["answer"],
            "sources":   result.get("sources", []),
            "followups": followups,
            "scores":    result.get("scores"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _generate_followups(question: str, answer: str) -> list:
    """Generate 2-3 follow-up questions using the active LLM."""
    try:
        import config
        provider = config.ACTIVE_PROVIDER
        opts     = config.MODEL_OPTIONS[provider]

        prompt = (
            f"Given the research question: '{question}'\n"
            f"And the answer summary: '{answer[:400]}'\n\n"
            "Suggest 3 short follow-up research questions the user might ask next. "
            "Return only the questions, one per line, no numbering, no quotes."
        )

        if provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=opts["api_key"])
            m = genai.GenerativeModel(opts["chat_model"])
            r = m.generate_content(prompt)
            lines = r.text.strip().split("\n")
        elif provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=opts["api_key"])
            r = client.messages.create(model=opts["chat_model"], max_tokens=200,
                                        messages=[{"role":"user","content":prompt}])
            lines = r.content[0].text.strip().split("\n")
        elif provider == "llama":
            from groq import Groq
            client = Groq(api_key=opts["api_key"])
            r = client.chat.completions.create(model=opts["chat_model"], max_tokens=200,
                messages=[{"role":"user","content":prompt}])
            lines = r.choices[0].message.content.strip().split("\n")
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=opts["api_key"])
            r = client.chat.completions.create(model=opts["chat_model"], max_tokens=200,
                messages=[{"role":"user","content":prompt}])
            lines = r.choices[0].message.content.strip().split("\n")
        else:
            return []

        return [l.strip().lstrip("•-–").strip() for l in lines if l.strip()][:3]
    except Exception:
        return []


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 ScholarRAG — http://localhost:5000")
    app.run(debug=True, port=5000, threaded=True)