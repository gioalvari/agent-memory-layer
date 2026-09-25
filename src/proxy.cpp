#include "memorylayer/proxy.h"
#include "memorylayer/injector.h"
#include "memorylayer/logger.h"
#include "httplib.h"
#include "json.hpp"
#include <iostream>
#include <sstream>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <vector>
#include <utility>

using json = nlohmann::json;

namespace memorylayer {

MemoryProxy::MemoryProxy(const Config& cfg, MemoryStore& store, EmbeddingWorker& embedder)
    : cfg_(cfg), store_(store), embedder_(embedder),
      sticky_cache_(static_cast<std::size_t>(cfg.sticky_cache_entries)) {
    saver_thread_ = std::thread(&MemoryProxy::saver_loop, this);
}

MemoryProxy::~MemoryProxy() {
    shutdown();
}

void MemoryProxy::stop() {
    auto* svr = svr_ptr_.load();
    if (svr) svr->stop();
}

void MemoryProxy::shutdown() {
    stop_ = true;
    save_cv_.notify_all();
    if (saver_thread_.joinable()) saver_thread_.join();
}

bool MemoryProxy::check_admin_auth(const httplib::Request& req, httplib::Response& res) const {
    if (cfg_.admin_token.empty()) return true;  // Auth disabled
    std::string auth = req.get_header_value("Authorization");
    if (auth == "Bearer " + cfg_.admin_token) return true;
    res.status = 401;
    res.set_content(R"({"error":"Unauthorized — set Authorization: Bearer <admin-token>"})",
                    "application/json");
    return false;
}

std::string MemoryProxy::extract_last_user_message(const json& messages) const {    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if ((*it).value("role", "") == "user") {
            return (*it).value("content", "");
        }
    }
    return "";
}

std::string MemoryProxy::extract_conversation_context(const json& messages, int max_turns) const {
    std::vector<std::pair<std::string, std::string>> turns;

    for (const auto& msg : messages) {
        std::string role = msg.value("role", "");
        std::string content = msg.value("content", "");
        if (role == "user" || role == "assistant") {
            turns.push_back({role, content});
        }
    }

    int start = std::max(0, static_cast<int>(turns.size()) - max_turns * 2);

    std::string context;
    for (int i = start; i < static_cast<int>(turns.size()); i++) {
        if (!context.empty()) context += " | ";
        context += turns[i].first + ": " + turns[i].second;
    }

    return context;
}

std::string MemoryProxy::retrieve_and_inject(const std::string& agent_id,
                                               const std::string& user_text,
                                               json& messages) {
    const bool sticky_mode = cfg_.inject_mode == "sticky";
    std::optional<std::size_t> last_user;
    uint64_t last_key = 0;
    StickyHistoryResult sticky_history;
    if (sticky_mode) {
        last_user = last_user_message_index(messages);
        if (!last_user) return "";
        last_key = sticky_message_key(agent_id, messages, *last_user);
        sticky_history = apply_sticky_history(messages, agent_id, sticky_cache_);
    }

    if (user_text.empty() || !embedder_.is_ready()) {
        if (sticky_mode) sticky_cache_.put(last_key, {});
        return "";
    }

    auto query_emb = embedder_.embed(user_text, EmbedPriority::HIGH);
    if (query_emb.empty()) {
        LOG_WARN("proxy", "Skipping memory retrieval because query embedding is empty for agent='" +
                 (agent_id.empty() ? "global" : agent_id) + "'");
        if (sticky_mode) sticky_cache_.put(last_key, {});
        return "";
    }

    // Context overflow guard: estimate total tokens and reduce top_k if needed
    int effective_top_k = cfg_.top_k;
    if (cfg_.max_context_tokens > 0) {
        // Rough estimate: 4 chars per token for user prompt
        int user_token_est = static_cast<int>(user_text.size() / 4);
        int budget = cfg_.max_context_tokens - user_token_est;
        // Each memory is ~max_inject_tokens/top_k tokens; scale down if over budget
        if (budget < cfg_.max_inject_tokens) {
            int tokens_per_mem = std::max(1, cfg_.max_inject_tokens / std::max(1, cfg_.top_k));
            effective_top_k = std::max(1, budget / tokens_per_mem);
        }
    }

    const int search_top_k = effective_top_k +
        static_cast<int>(sticky_history.shown_memory_ids.size());
    auto memories = store_.search(query_emb, agent_id, search_top_k,
                                    cfg_.min_score_threshold, cfg_.decay_days,
                                    cfg_.agent_boost);

    if (sticky_mode) {
        memories = filter_excluded_memories(memories, sticky_history.shown_memory_ids,
                                            effective_top_k);
    }

    if (memories.empty()) {
        if (sticky_mode) {
            sticky_cache_.put(last_key, {});
            LOG_INFO("proxy", "Sticky injection: reused blocks=" +
                     std::to_string(sticky_history.reused_blocks) + ", new memories=0");
        }
        return "";
    }

    std::string context;
    std::vector<int64_t> injected_ids;
    if (sticky_mode) {
        const StickyFormattedContext formatted =
            format_sticky_memory_context_budgeted(memories, cfg_.max_inject_tokens);
        context = formatted.block;
        injected_ids = formatted.memory_ids;
        const auto injected_last_user = last_user_message_index(messages);
        if (injected_last_user) inject_sticky_block(messages, *injected_last_user, context);
        sticky_cache_.put(last_key, {context, injected_ids});
        LOG_INFO("proxy", "Sticky injection: reused blocks=" +
                 std::to_string(sticky_history.reused_blocks) + ", new memories=" +
                 std::to_string(injected_ids.size()));
    } else {
        context = format_memory_context_budgeted(memories,
                                                 static_cast<double>(std::time(nullptr)),
                                                 cfg_.max_inject_tokens);
        inject_memories(messages, context, parse_inject_mode(cfg_.inject_mode));
    }

    {
        std::lock_guard<std::mutex> lock(debug_mutex_);
        injection_ring_.push_front({agent_id, user_text, context,
                                    static_cast<double>(std::time(nullptr))});
        if (static_cast<int>(injection_ring_.size()) > kInjectionRingSize) {
            injection_ring_.pop_back();
        }
    }

    LOG_INFO("proxy", "Injected " + std::to_string(sticky_mode ? injected_ids.size() : memories.size()) + " memories for agent='" + (agent_id.empty() ? "global" : agent_id) + "'");

    return context;
}

void MemoryProxy::enqueue_save(const std::string& agent_id,
                                const std::string& user_text,
                                const std::string& assist_text,
                                const std::string& embed_text) {
    {
        std::lock_guard<std::mutex> lock(save_mutex_);
        if (save_queue_.size() >= 100) {
            ++save_drop_count_;
            LOG_WARN("proxy", "Save queue full (drops=" + std::to_string(save_drop_count_.load()) + "), dropping oldest");
            save_queue_.pop();
        }
        save_queue_.push({agent_id, user_text, assist_text, embed_text});
    }
    save_cv_.notify_one();
}

void MemoryProxy::saver_loop() {
    auto last_purge = std::chrono::steady_clock::now();
    while (!stop_) {
        SaveJob job;
        {
            std::unique_lock<std::mutex> lock(save_mutex_);
            save_cv_.wait_for(lock, std::chrono::seconds(30),
                              [this] { return !save_queue_.empty() || stop_; });
            if (stop_ && save_queue_.empty()) return;
            if (save_queue_.empty()) {
                // Woke on timeout — only run housekeeping
                goto housekeeping;
            }
            job = std::move(save_queue_.front());
            save_queue_.pop();
        }

        if (!embedder_.is_ready()) continue;

        {
            const std::string& to_embed = job.embed_text.empty() ? job.user_text : job.embed_text;
            auto user_emb = embedder_.embed(to_embed);
            auto assist_emb = embedder_.embed(job.assist_text);
            if (user_emb.empty() || assist_emb.empty()) continue;

            int64_t dup_id = store_.find_duplicate(user_emb, cfg_.dedup_threshold, job.agent_id);
            if (dup_id > 0) {
                store_.touch(dup_id);
                LOG_INFO("memory", "Deduplicated: merged with memory #" + std::to_string(dup_id));
            } else {
                store_.insert(job.agent_id, job.user_text, job.assist_text, user_emb, assist_emb);
                store_.evict(job.agent_id);
                LOG_INFO("memory", "Saved new memory for agent='" +
                         (job.agent_id.empty() ? "global" : job.agent_id) + "'");
            }
        }

        housekeeping:
        {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::hours>(now - last_purge).count() >= 1) {
                if (cfg_.memory_ttl_days > 0) {
                    store_.purge_expired(cfg_.memory_ttl_days);
                    LOG_INFO("memory", "Purged memories older than " +
                             std::to_string(cfg_.memory_ttl_days) + " days");
                }
                last_purge = now;
            }
        }
    }
}

bool MemoryProxy::run() {
    httplib::Server svr;
    svr_ptr_.store(&svr);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // Admin dashboard UI
    svr.Get("/admin/ui", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        static const std::string html = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Memory Layer — Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}
  header{background:#1a1d27;border-bottom:1px solid #2d3147;padding:16px 24px;display:flex;align-items:center;gap:12px}
  header h1{font-size:1.2rem;font-weight:600;color:#a78bfa}
  header span{font-size:.8rem;color:#64748b;margin-left:auto}
  .main{padding:24px;max-width:1400px;margin:0 auto}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}
  .card{background:#1a1d27;border:1px solid #2d3147;border-radius:10px;padding:20px}
  .card .label{font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
  .card .value{font-size:2rem;font-weight:700;color:#a78bfa}
  .card .sub{font-size:.75rem;color:#64748b;margin-top:4px}
  .row{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}
  .panel{background:#1a1d27;border:1px solid #2d3147;border-radius:10px;padding:20px;flex:1;min-width:280px}
  .panel h2{font-size:.9rem;font-weight:600;color:#94a3b8;margin-bottom:14px;display:flex;align-items:center;gap:8px}
  .agent-list{display:flex;flex-direction:column;gap:6px}
  .agent-row{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:#13151f;border-radius:6px;cursor:pointer;border:2px solid transparent;transition:border .15s}
  .agent-row:hover,.agent-row.active{border-color:#a78bfa}
  .agent-name{font-size:.85rem;font-weight:500}
  .agent-count{font-size:.75rem;background:#2d3147;padding:2px 8px;border-radius:10px;color:#94a3b8}
  .toolbar{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center}
  .toolbar input{background:#13151f;border:1px solid #2d3147;border-radius:6px;padding:6px 12px;color:#e2e8f0;font-size:.85rem;flex:1;min-width:200px;outline:none}
  .toolbar input:focus{border-color:#a78bfa}
  .btn{background:#2d3147;border:none;color:#e2e8f0;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.8rem;white-space:nowrap;transition:background .15s}
  .btn:hover{background:#3d4267}
  .btn.danger{background:#450a0a;color:#fca5a5}
  .btn.danger:hover{background:#7f1d1d}
  .btn.primary{background:#5b21b6;color:#fff}
  .btn.primary:hover{background:#7c3aed}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th{text-align:left;padding:8px 12px;color:#64748b;font-weight:500;border-bottom:1px solid #2d3147;position:sticky;top:0;background:#1a1d27}
  td{padding:10px 12px;border-bottom:1px solid #1e2132;vertical-align:top;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  tr:hover td{background:#13151f}
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.72rem;font-weight:600}
  .badge.hits{background:#14532d;color:#86efac}
  .tbl-wrap{overflow-x:auto;max-height:520px;overflow-y:auto;border-radius:8px;border:1px solid #2d3147}
  .detail-box{position:fixed;inset:0;background:#000a;display:flex;align-items:center;justify-content:center;z-index:100;padding:20px}
  .detail-inner{background:#1a1d27;border:1px solid #2d3147;border-radius:12px;max-width:680px;width:100%;max-height:80vh;overflow-y:auto;padding:24px}
  .detail-inner h3{font-size:1rem;margin-bottom:16px;color:#a78bfa}
  .detail-field{margin-bottom:12px}
  .detail-field .lbl{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
  .detail-field .val{background:#13151f;border-radius:6px;padding:10px 12px;font-size:.82rem;white-space:pre-wrap;word-break:break-word;line-height:1.5}
  .inject-preview{background:#0c1a0c;border:1px solid #14532d;border-radius:8px;padding:12px;font-size:.8rem;color:#86efac;margin-top:8px;white-space:pre-wrap;max-height:200px;overflow-y:auto}
  .empty{color:#64748b;text-align:center;padding:40px;font-size:.9rem}
  .spinner{display:inline-block;width:14px;height:14px;border:2px solid #2d3147;border-top-color:#a78bfa;border-radius:50%;animation:spin .7s linear infinite;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .toast{position:fixed;bottom:24px;right:24px;background:#14532d;color:#86efac;padding:10px 18px;border-radius:8px;font-size:.85rem;z-index:200;opacity:0;transition:opacity .3s;pointer-events:none}
  .toast.show{opacity:1}
</style>
</head>
<body>
<header>
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>
  <h1>Agent Memory Layer</h1>
  <span id="ts">—</span>
</header>
<div class="main">
  <div class="cards">
    <div class="card"><div class="label">Total Memories</div><div class="value" id="stat-total">—</div><div class="sub">across all agents</div></div>
    <div class="card"><div class="label">Active Agents</div><div class="value" id="stat-agents">—</div></div>
    <div class="card"><div class="label">Queue Drops</div><div class="value" id="stat-drops">—</div><div class="sub">save queue overflows</div></div>
    <div class="card"><div class="label">Last Injection</div><div class="value" id="stat-inj" style="font-size:1.1rem;padding-top:6px">—</div></div>
  </div>
  <div class="row">
    <div class="panel" style="max-width:260px;flex:0 0 260px">
      <h2>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg>
        Agents
      </h2>
      <div class="agent-list" id="agent-list"><div class="empty">Loading…</div></div>
    </div>
    <div class="panel" style="flex:1">
      <h2>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        Memories
        <span id="mem-agent-label" style="color:#a78bfa;font-size:.8rem"></span>
        <button class="btn" style="margin-left:auto" onclick="loadMemories()">↻ Refresh</button>
      </h2>
      <div class="toolbar">
        <input type="text" id="search-box" placeholder="Filter by text…" oninput="filterTable()">
        <select id="sort-sel" onchange="filterTable()" style="background:#13151f;border:1px solid #2d3147;border-radius:6px;padding:6px 10px;color:#e2e8f0;font-size:.85rem;outline:none">
          <option value="created_desc">Newest first</option>
          <option value="created_asc">Oldest first</option>
          <option value="access_desc">Most accessed</option>
        </select>
        <button class="btn danger" onclick="purgeSelected()">Delete selected</button>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr>
            <th><input type="checkbox" id="sel-all" onchange="toggleAll(this)"></th>
            <th>ID</th><th>Agent</th><th>User message</th><th>Assistant reply</th><th>Accesses</th><th>Created</th><th></th>
          </tr></thead>
          <tbody id="mem-tbody"><tr><td colspan="8" class="empty">Select an agent or load all</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>
  <div class="row">
    <div class="panel" style="flex:1">
      <h2>Last Injection Context</h2>
      <div id="inj-detail" class="empty">No injection recorded yet</div>
    </div>
  </div>
</div>

<div class="detail-box" id="detail-modal" style="display:none" onclick="if(event.target===this)closeDetail()">
  <div class="detail-inner">
    <h3 id="detail-title">Memory detail</h3>
    <div id="detail-body"></div>
    <div style="margin-top:16px;display:flex;gap:8px">
      <button class="btn danger" id="detail-del-btn">Delete</button>
      <button class="btn" onclick="closeDetail()">Close</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let allMemories = [];
let activeAgent = '';

function toast(msg, ok=true){
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.style.background=ok?'#14532d':'#450a0a';
  t.style.color=ok?'#86efac':'#fca5a5';
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2500);
}

function fmt(ts){
  if(!ts)return '—';
  const d=new Date(ts*1000);
  return d.toLocaleString();
}

async function loadStats(){
  try{
    const r=await fetch('/admin/stats');
    const d=await r.json();
    document.getElementById('stat-total').textContent=d.total_memories??'—';
    document.getElementById('stat-agents').textContent=d.total_agents??'—';
    document.getElementById('stat-drops').textContent=d.save_queue_drops??0;
    document.getElementById('ts').textContent='Updated '+new Date().toLocaleTimeString();
  }catch(e){console.error(e)}
}

async function loadAgents(){
  try{
    const r=await fetch('/admin/agents');
    const agents=await r.json();
    const el=document.getElementById('agent-list');
    if(!agents.length){el.innerHTML='<div class="empty">No agents yet</div>';return;}
    el.innerHTML='<div class="agent-row'+(activeAgent===''?' active':'')+'" onclick="selectAgent(\'\')"><span class="agent-name">All agents</span><span class="agent-count">all</span></div>'
      +agents.map(a=>`<div class="agent-row${activeAgent===a.agent_id?' active':''}" onclick="selectAgent('${a.agent_id.replace(/'/g,"\\'")}')">
        <span class="agent-name">${a.agent_id||'<global>'}</span>
        <span class="agent-count">${a.memory_count}</span>
      </div>`).join('');
  }catch(e){console.error(e)}
}

function selectAgent(id){
  activeAgent=id;
  document.getElementById('mem-agent-label').textContent=id?`— ${id}`:'';
  loadAgents();
  loadMemories();
}

async function loadMemories(){
  const url='/admin/memories?limit=200'+(activeAgent?'&agent_id='+encodeURIComponent(activeAgent):'');
  try{
    const r=await fetch(url);
    allMemories=await r.json();
    filterTable();
  }catch(e){console.error(e)}
}

function filterTable(){
  const q=document.getElementById('search-box').value.toLowerCase();
  const sort=document.getElementById('sort-sel').value;
  let mems=[...allMemories];
  if(q)mems=mems.filter(m=>(m.user_text+m.assist_text+m.agent_id).toLowerCase().includes(q));
  if(sort==='created_asc')mems.sort((a,b)=>a.created_at-b.created_at);
  else if(sort==='access_desc')mems.sort((a,b)=>b.access_count-a.access_count);
  else mems.sort((a,b)=>b.created_at-a.created_at);
  renderTable(mems);
}

function renderTable(mems){
  const tb=document.getElementById('mem-tbody');
  if(!mems.length){tb.innerHTML='<tr><td colspan="8" class="empty">No memories found</td></tr>';return;}
  tb.innerHTML=mems.map(m=>`<tr>
    <td><input type="checkbox" class="sel-cb" data-id="${m.id}"></td>
    <td>${m.id}</td>
    <td><span style="color:#a78bfa">${m.agent_id||'<global>'}</span></td>
    <td style="max-width:240px" title="${esc(m.user_text)}">${esc(m.user_text).substring(0,80)}${m.user_text.length>80?'…':''}</td>
    <td style="max-width:240px" title="${esc(m.assist_text)}">${esc(m.assist_text).substring(0,80)}${m.assist_text.length>80?'…':''}</td>
    <td><span class="badge hits">${m.access_count}</span></td>
    <td>${fmt(m.created_at)}</td>
    <td><button class="btn" onclick='openDetail(${JSON.stringify(m)})'>View</button></td>
  </tr>`).join('');
}

function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}

function toggleAll(cb){document.querySelectorAll('.sel-cb').forEach(c=>c.checked=cb.checked);}

async function purgeSelected(){
  const ids=[...document.querySelectorAll('.sel-cb:checked')].map(c=>+c.dataset.id);
  if(!ids.length){toast('Select memories first',false);return;}
  if(!confirm(`Delete ${ids.length} memory/memories?`))return;
  let ok=0;
  for(const id of ids){
    const r=await fetch('/admin/memories/'+id,{method:'DELETE'});
    if(r.ok)ok++;
  }
  toast(`Deleted ${ok}/${ids.length}`);
  loadMemories();loadStats();loadAgents();
}

function openDetail(m){
  document.getElementById('detail-title').textContent=`Memory #${m.id}`;
  document.getElementById('detail-body').innerHTML=`
    <div class="detail-field"><div class="lbl">Agent</div><div class="val">${esc(m.agent_id||'<global>')}</div></div>
    <div class="detail-field"><div class="lbl">Created</div><div class="val">${fmt(m.created_at)}</div></div>
    <div class="detail-field"><div class="lbl">Access count</div><div class="val">${m.access_count}</div></div>
    <div class="detail-field"><div class="lbl">User message</div><div class="val">${esc(m.user_text)}</div></div>
    <div class="detail-field"><div class="lbl">Assistant reply</div><div class="val">${esc(m.assist_text)}</div></div>`;
  document.getElementById('detail-del-btn').onclick=async()=>{
    if(!confirm('Delete this memory?'))return;
    const r=await fetch('/admin/memories/'+m.id,{method:'DELETE'});
    if(r.ok){toast('Deleted');closeDetail();loadMemories();loadStats();loadAgents();}
    else toast('Failed',false);
  };
  document.getElementById('detail-modal').style.display='flex';
}
function closeDetail(){document.getElementById('detail-modal').style.display='none';}

async function loadLastInjection(){
  try{
    const r=await fetch('/admin/debug/last-injection');
    const items=await r.json();
    const el=document.getElementById('inj-detail');
    if(!items.length){el.innerHTML='<div class="empty">No injection recorded yet</div>';return;}
    const i=items[0];
    el.innerHTML=`<div class="detail-field"><div class="lbl">Agent</div><div class="val">${esc(i.agent_id||'<global>')}</div></div>
      <div class="detail-field"><div class="lbl">Query</div><div class="val">${esc(i.query)}</div></div>
      <div class="detail-field"><div class="lbl">Injected context</div><div class="inject-preview">${esc(i.injected_context)}</div></div>
      <div class="detail-field"><div class="lbl">Time</div><div class="val">${fmt(i.timestamp)}</div></div>`;
    document.getElementById('stat-inj').textContent=i.agent_id||'<global>';
  }catch(e){console.error(e)}
}

async function refresh(){
  await Promise.all([loadStats(),loadAgents(),loadMemories(),loadLastInjection()]);
}

refresh();
setInterval(refresh,10000);
</script>
</body>
</html>)HTML";
        res.set_content(html, "text/html; charset=utf-8");
    });

    svr.Get("/admin/memories", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        std::string agent_id = req.has_param("agent_id") ? req.get_param_value("agent_id") : "";
        int limit = 50;
        int offset = 0;
        if (req.has_param("limit")) {
            try { limit = std::stoi(req.get_param_value("limit")); } catch (...) {}
        }
        if (req.has_param("offset")) {
            try { offset = std::stoi(req.get_param_value("offset")); } catch (...) {}
        }

        auto memories = store_.list_memories(agent_id, limit, offset);

        json result = json::array();
        for (const auto& m : memories) {
            result.push_back({
                {"id", m.id},
                {"agent_id", m.agent_id},
                {"created_at", m.created_at},
                {"user_text", m.user_text},
                {"assist_text", m.assist_text},
                {"access_count", m.access_count}
            });
        }
        res.set_content(result.dump(2), "application/json");
    });

    // Admin API: delete a memory
    svr.Delete(R"(/admin/memories/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        int64_t id;
        try {
            id = std::stoll(req.matches[1]);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid memory ID"})", "application/json");
            return;
        }
        bool removed = store_.remove(id);
        if (removed) {
            res.set_content(R"({"status":"deleted"})", "application/json");
        } else {
            res.status = 404;
            res.set_content(R"({"error":"Memory not found"})", "application/json");
        }
    });

    // Admin API: stats
    svr.Get("/admin/stats", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        auto stats = store_.get_stats();
        json result = {
            {"total_memories", stats.total_memories},
            {"total_agents", stats.total_agents},
            {"save_queue_drops", save_drop_count_.load()},
            {"per_agent", json::object()}
        };
        for (const auto& [agent, count] : stats.per_agent_counts) {
            result["per_agent"][agent] = count;
        }
        res.set_content(result.dump(2), "application/json");
    });

    // Admin API: per-agent summary (id, count, last_active)
    svr.Get("/admin/agents", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        auto stats = store_.get_stats();
        json result = json::array();
        for (const auto& [agent, count] : stats.per_agent_counts) {
            result.push_back({{"agent_id", agent}, {"memory_count", count}});
        }
        res.set_content(result.dump(2), "application/json");
    });

    // Admin API: injection history ring buffer (newest first, up to kInjectionRingSize entries)
    svr.Get("/admin/debug/last-injection", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        std::lock_guard<std::mutex> lock(debug_mutex_);
        json result = json::array();
        for (const auto& inj : injection_ring_) {
            result.push_back({
                {"agent_id", inj.agent_id},
                {"query", inj.query},
                {"injected_context", inj.injected_context},
                {"timestamp", inj.timestamp}
            });
        }
        res.set_content(result.dump(2), "application/json");
    });

    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        auto t_start = std::chrono::steady_clock::now();

        // Body size limit: 1MB
        if (req.body.size() > 1024 * 1024) {
            res.status = 413;
            res.set_content(R"({"error":"Request body too large"})", "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid JSON"})", "application/json");
            return;
        }

        std::string agent_id;
        if (req.has_header("X-Agent-Id")) {
            agent_id = req.get_header_value("X-Agent-Id");
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":"Missing messages array"})", "application/json");
            return;
        }

        auto& messages = body["messages"];
        std::string user_text = extract_last_user_message(messages);
        // Captured before injection: in suffix mode the memory block is written
        // into the last user turn and must not be saved back as memory.
        const std::string conv_context = extract_conversation_context(messages);

        retrieve_and_inject(agent_id, user_text, messages);

        bool streaming = body.value("stream", false);

        // Parse backend host and port
        httplib::Client cli(cfg_.backend_url);
        cli.set_read_timeout(300);
        cli.set_connection_timeout(10);

        std::string modified_body = body.dump();

        if (!streaming) {
            auto backend_res = cli.Post("/v1/chat/completions",
                                         modified_body, "application/json");
            if (!backend_res) {
                res.status = 502;
                res.set_content(R"({"error":"Backend unreachable"})", "application/json");
                return;
            }

            res.status = backend_res->status;
            res.set_content(backend_res->body, backend_res->get_header_value("Content-Type"));
            // Latency header: total proxy wall-time in milliseconds
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            res.set_header("X-Memory-Layer-Latency-Ms", std::to_string(ms));
            res.set_header("Access-Control-Expose-Headers", "X-Memory-Layer-Latency-Ms");

            if (backend_res->status == 200 && !user_text.empty()) {
                try {
                    auto resp_json = json::parse(backend_res->body);
                    std::string assist_text = resp_json["choices"][0]["message"]["content"].get<std::string>();
                    std::string save_user = conv_context.empty() ? user_text : conv_context;
                    enqueue_save(agent_id, save_user, assist_text, user_text);
                } catch (const std::exception& e) {
                    LOG_WARN("proxy", std::string("Failed to parse backend response for memory save: ") + e.what());
                }
            }
        } else {
            // TRUE streaming: forward SSE chunks in real-time via chunked response
            std::string agent_id_copy = agent_id;
            std::string user_text_copy = user_text;
            std::string backend_url = cfg_.backend_url;
            std::string mod_body = modified_body;
            std::string save_user = conv_context.empty() ? user_text_copy : conv_context;

            res.set_chunked_content_provider(
                "text/event-stream",
                [this, mod_body, backend_url, agent_id_copy, save_user, user_text_copy]
                (size_t /*offset*/, httplib::DataSink& sink) -> bool {
                    httplib::Client backend(backend_url);
                    backend.set_read_timeout(300);
                    backend.set_connection_timeout(10);

                    std::string accumulated_content;

                    auto result = backend.Post(
                        "/v1/chat/completions",
                        mod_body.size(),
                        [&mod_body](size_t offset, size_t length, httplib::DataSink& body_sink) -> bool {
                            size_t remaining = mod_body.size() - offset;
                            size_t to_write = std::min(length, remaining);
                            body_sink.write(mod_body.data() + offset, to_write);
                            return true;
                        },
                        "application/json",
                        [&sink, &accumulated_content](const char* data, size_t len) -> bool {
                            // Forward chunk to client in real-time
                            sink.write(data, len);

                            // Parse SSE lines to accumulate assistant content
                            std::string chunk(data, len);
                            std::istringstream stream(chunk);
                            std::string line;
                            while (std::getline(stream, line)) {
                                if (line.size() > 6 && line.substr(0, 6) == "data: ") {
                                    std::string payload = line.substr(6);
                                    if (payload == "[DONE]") continue;
                                    try {
                                        auto j = nlohmann::json::parse(payload);
                                        if (j.contains("choices") && !j["choices"].empty()) {
                                            auto& delta = j["choices"][0]["delta"];
                                            if (delta.contains("content")) {
                                                accumulated_content += delta["content"].get<std::string>();
                                            }
                                        }
                                    } catch (...) {
                                        // Partial SSE chunks commonly fail JSON parse — skip silently
                                    }                                }
                            }
                            return true;
                        }
                    );

                    // Save accumulated content to memory after streaming completes
                    if (!accumulated_content.empty() && !save_user.empty()) {
                        enqueue_save(agent_id_copy, save_user, accumulated_content, user_text_copy);
                    }

                    sink.done();
                    return false;
                },
                [](bool) {}
            );
        }
    });

    // Catch-all for other /v1/* endpoints
    svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
        // Only catch unhandled paths under /v1/
        if (req.path.compare(0, 4, "/v1/") == 0 && req.path != "/v1/chat/completions") {
            httplib::Client cli(cfg_.backend_url);
            cli.set_read_timeout(60);
            httplib::Result result = (req.method == "POST")
                ? cli.Post(req.path, req.body, req.get_header_value("Content-Type"))
                : cli.Get(req.path);

            if (result) {
                res.status = result->status;
                res.set_content(result->body, result->get_header_value("Content-Type"));
            } else {
                res.status = 502;
                res.set_content(R"({"error":"Backend unreachable"})", "application/json");
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    LOG_INFO("proxy", "Listening on http://0.0.0.0:" + std::to_string(cfg_.port));
    LOG_INFO("proxy", "Backend: " + cfg_.backend_url);
    LOG_INFO("proxy", "Memory injection: mode=" + cfg_.inject_mode + ", top_k=" + std::to_string(cfg_.top_k) + ", decay_days=" + std::to_string(cfg_.decay_days));

    const bool served = svr.listen("0.0.0.0", cfg_.port);
    const bool stopping = svr_ptr_.load() == nullptr || stop_;
    svr_ptr_.store(nullptr);  // Clear before svr goes out of scope (stop() safety)
    if (!served && !stopping) {
        LOG_ERROR("proxy", "Failed to listen on port " + std::to_string(cfg_.port) +
                  " (address in use?)");
        return false;
    }
    return true;
}

} // namespace memorylayer
