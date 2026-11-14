const state = { data: null, split: 'dev', selectedTrace: null };
const methodMeta = {
  dense: { label: 'Dense retrieval', short: 'Dense', color: '#8067e7' },
  bm25: { label: 'Lexical · BM25', short: 'BM25 lexical', color: '#f0a06a' },
  query_transformation: { label: 'Query transformation', short: 'Query transform', color: '#37b9c5' }
};
const $ = (id) => document.getElementById(id);
const pct = (n, d) => d ? `${Math.round((n / d) * 100)}%` : '—';
const safe = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));

async function loadData() {
  $('noticeText').textContent = 'Đang đọc dữ liệu benchmark trong project…';
  try {
    const response = await fetch('/api/dashboard');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    render();
  } catch (error) {
    $('noticeText').textContent = 'Không tải được dữ liệu. Hãy chạy dashboard bằng lệnh python scripts/serve_dashboard.py.';
    $('noticeText').style.color = '#bb6969';
    console.error(error);
  }
}

function splitData() {
  const split = state.split;
  const cases = state.data.cases.filter(item => item.split === split);
  const questions = state.data.questions.filter(item => item.split === split);
  const retrieval = state.data.retrieval[split] || {};
  const agent = state.data.agent[split] || null;
  const traces = state.data.agentTraces[split] || [];
  const reviews = state.data.agentReviews[split] || [];
  const qMap = Object.fromEntries(questions.map(item => [item.id, item]));
  const agentTraceMap = Object.fromEntries(traces.map(item => [item.query_id, item]));
  const reviewMap = Object.fromEntries(reviews.map(item => [item.query_id, item]));
  const retrievalTrace = retrieval.traces || {};
  const retrievalById = {};
  Object.entries(retrievalTrace).forEach(([method, rows]) => rows.forEach(row => {
    (retrievalById[row.query_id] ||= {})[method] = row;
  }));
  return { cases, questions, retrieval, agent, qMap, agentTraceMap, reviewMap, retrievalById };
}

function render() {
  const d = splitData();
  const nAnswerable = d.cases.filter(item => item.answerable).length;
  const allMethods = Object.values(d.retrieval.summary || {});
  const best = [...allMethods].sort((a, b) => b.hit3_rate - a.hit3_rate)[0];
  const agent = d.agent;
  const grade = agent?.grounded_answer;
  const abstention = agent?.abstention;
  $('noticeText').textContent = `${d.cases.length} câu hỏi · ${nAnswerable} có đáp án · ${d.cases.length - nAnswerable} thiếu evidence · ${state.data.chunks.length} chunks`;
  $('metricHit3').textContent = best ? pct(best.hit3, best.N) : '—';
  $('metricHit3Detail').textContent = best ? `${methodMeta[best.method]?.label || best.method} · ${best.hit3}/${best.N}` : 'Chưa có kết quả retrieval';
  const contextBest = [...allMethods].sort((a, b) => b.context_evidence_rate - a.context_evidence_rate)[0];
  $('metricContext').textContent = contextBest ? pct(contextBest.context_hit, contextBest.N) : '—';
  $('metricContextDetail').textContent = contextBest ? `${methodMeta[contextBest.method]?.label || contextBest.method} · ${contextBest.context_hit}/${contextBest.N}` : 'Chưa có context metrics';
  $('metricGrounded').textContent = grade?.mean != null ? Number(grade.mean).toFixed(2) : '—';
  $('metricGroundedDetail').textContent = grade ? `${grade['2'] || 0} tốt · ${grade['1'] || 0} một phần · ${grade['0'] || 0} chưa đạt` : 'Chưa có agent summary';
  $('metricAbstain').textContent = abstention ? pct(abstention.n, abstention.N) : '—';
  $('metricAbstainDetail').textContent = abstention ? `${abstention.n}/${abstention.N} câu thiếu evidence` : 'Chưa có agent summary';
  $('groundedLarge').textContent = grade?.mean != null ? Number(grade.mean).toFixed(2) : '—';
  $('toolUse').textContent = agent ? `${agent.tool_use.n}/${agent.tool_use.N}` : '—';
  $('citationValid').textContent = agent ? `${agent.citation_validity.n}/${agent.citation_validity.N}` : '—';
  $('falseAbstain').textContent = agent ? `${agent.false_abstention.n}/${agent.false_abstention.N}` : '—';
  $('retrievalBudget').textContent = allMethods.length ? `Budget ${allMethods[0].budget} chars` : 'Chưa có budget';
  renderChart(d.retrieval.summary || {});
  renderMethodRows(d.retrieval.summary || {});
  renderGrades(grade);
  renderQuestions(d);
}

function renderChart(summary) {
  const entries = Object.values(summary);
  if (!entries.length) { $('methodChart').innerHTML = '<div class="empty-state">Chưa có kết quả retrieval cho tập này.</div>'; return; }
  $('methodChart').innerHTML = entries.map(row => {
    const method = methodMeta[row.method] || { short: row.method };
    const h = (row.hit3_rate || 0) * 100;
    const c = (row.context_evidence_rate || 0) * 100;
    return `<div class="chart-method"><div class="chart-method-name" title="${safe(method.label || method.short)}">${safe(method.short)}</div><div class="bar-stack"><div class="bar hit3" style="width:${h}%"></div><div class="bar context" style="width:${c}%"></div></div><div class="bar-value">${Math.round(h)}%</div></div>`;
  }).join('');
}

function renderMethodRows(summary) {
  const entries = Object.values(summary);
  $('methodRows').innerHTML = entries.length ? entries.map(row => {
    const method = methodMeta[row.method] || { label: row.method, color: '#aaa' };
    return `<tr><td class="method-name-cell"><i class="method-dot" style="background:${method.color}"></i>${safe(method.label)}</td><td>${row.hit1}/${row.N} <span class="muted">(${pct(row.hit1,row.N)})</span></td><td>${row.hit3}/${row.N} <span class="muted">(${pct(row.hit3,row.N)})</span></td><td>${row.context_hit}/${row.N} <span class="muted">(${pct(row.context_hit,row.N)})</span></td><td>${Number(row.median_ms).toFixed(1)} ms</td></tr>`;
  }).join('') : '<tr><td colspan="5" class="muted">Chưa có dữ liệu</td></tr>';
}

function renderGrades(grade) {
  if (!grade?.N) {
    $('gradeDonut').style.background = 'conic-gradient(#eeeef2 0deg 360deg)';
    $('gradeLegend').innerHTML = '<span class="muted">Chưa có nhãn agent</span>';
    return;
  }
  const n0 = grade['0'] || 0, n1 = grade['1'] || 0, n2 = grade['2'] || 0;
  const a = n2 / grade.N * 360, b = a + n1 / grade.N * 360;
  $('gradeDonut').style.background = `conic-gradient(#8067e7 0deg ${a}deg,#f0a06a ${a}deg ${b}deg,#dedfe6 ${b}deg 360deg)`;
  $('gradeLegend').innerHTML = [
    ['#8067e7','Đúng & đủ',n2,grade.N], ['#f0a06a','Đúng một phần',n1,grade.N], ['#dedfe6','Chưa đạt',n0,grade.N]
  ].map(([color,label,count,total]) => `<div class="score-row"><i style="background:${color}"></i><span>${label}</span><strong>${count}/${total}</strong><small>${pct(count,total)} câu</small></div>`).join('');
}

function renderQuestions(d) {
  const search = $('querySearch').value.trim().toLocaleLowerCase('vi');
  const type = $('typeFilter').value;
  const rows = d.cases.filter(item => {
    if (type === 'answerable' && !item.answerable) return false;
    if (type === 'unanswerable' && item.answerable) return false;
    const question = item.question || d.qMap[item.id]?.question || '';
    return !search || `${item.id} ${question}`.toLocaleLowerCase('vi').includes(search);
  });
  $('questionRows').innerHTML = rows.map(item => {
    const id = item.id;
    const trace = d.agentTraceMap[id];
    const review = d.reviewMap[id];
    const question = item.question || d.qMap[id]?.question || 'Không có nội dung câu hỏi';
    const grade = item.answerable ? review?.answer_grade : null;
    const status = trace?.status || 'missing';
    const retrieval = d.retrievalById[id]?.query_transformation || d.retrievalById[id]?.dense || d.retrievalById[id]?.bm25;
    const retrievalPercent = retrieval ? (retrieval.hit3 ? 100 : 0) : null;
    const score = grade == null ? '—' : `${grade}/2`;
    const scoreClass = grade === 2 ? 'good' : grade === 0 ? 'bad' : '';
    return `<tr class="data-row" data-query-id="${safe(id)}"><td class="query-id">${safe(id)}</td><td class="question-cell" title="${safe(question)}">${safe(question)}</td><td><span class="type-badge ${item.answerable ? 'answerable' : 'unanswerable'}">${item.answerable ? '● Có đáp án' : '○ Thiếu evidence'}</span></td><td><span class="agent-badge ${status === 'answered' ? 'answered' : 'abstain'}">${status === 'answered' ? '↗ Answered' : status === 'abstain' ? '↘ Abstain' : '— Chưa có trace'}</span></td><td class="score-cell ${scoreClass}">${score}</td><td>${retrievalPercent == null ? '<span class="mini-value">—</span>' : `<div class="retrieval-mini"><div class="mini-track"><div class="mini-fill" style="width:${retrievalPercent}%"></div></div><span class="mini-value">${retrieval.hit3 ? 'Hit@3' : 'Miss'}</span></div>`}</td><td class="row-arrow">›</td></tr>`;
  }).join('');
  $('emptyState').hidden = rows.length !== 0;
  $('rowCount').textContent = `Hiển thị ${rows.length} / ${d.cases.length} câu hỏi`;
  $('questionRows').querySelectorAll('.data-row').forEach(row => row.addEventListener('click', () => openDetail(row.dataset.queryId, d)));
}

function openDetail(id, d) {
  const item = d.cases.find(entry => entry.id === id);
  const question = item.question || d.qMap[id]?.question || '';
  const q = d.qMap[id];
  const trace = d.agentTraceMap[id];
  const review = d.reviewMap[id];
  const evalRows = d.retrievalById[id] || {};
  const retrieval = evalRows.query_transformation || evalRows.dense || evalRows.bm25;
  $('drawerTitle').textContent = id;
  let html = `<div class="detail-question">${safe(question)}</div><div class="detail-tags"><span class="type-badge ${item.answerable ? 'answerable' : 'unanswerable'}">${item.answerable ? 'Có đáp án' : 'Thiếu evidence'}</span>${trace ? `<span class="agent-badge ${trace.status === 'answered' ? 'answered' : 'abstain'}">${trace.status === 'answered' ? 'Answered' : 'Abstain'}</span>` : ''}${review?.answer_grade != null ? `<span class="agent-badge ${review.answer_grade === 2 ? 'answered' : 'abstain'}">Điểm ${review.answer_grade}/2</span>` : ''}</div>`;
  if (q?.expected_answer) html += `<section class="detail-section"><h3>ĐÁP ÁN THAM CHIẾU</h3><div class="answer-box">${safe(q.expected_answer)}</div></section>`;
  if (q?.gold?.length) html += `<section class="detail-section"><h3>GOLD EVIDENCE</h3>${q.gold.map(g => `<div class="evidence-box"><strong>${safe(g.source_id)} · trang ${safe(g.page)}</strong><br>${safe(g.evidence)}</div>`).join('<div style="height:6px"></div>')}</section>`;
  if (retrieval) {
    const name = methodMeta[retrieval.method]?.label || retrieval.method;
    html += `<section class="detail-section"><h3>RETRIEVAL · ${safe(name)}</h3>${retrieval.transformation ? `<div class="answer-box" style="margin-bottom:7px"><strong>Query viết lại:</strong> ${safe(retrieval.transformation.transformed_query)}</div>` : ''}<div class="result-list">${(retrieval.results || []).map((result,index) => `<div class="result-item"><span class="rank">#${index+1}</span><div><strong>${safe(result.id)} · ${safe(result.source || result.source_id)} · p.${safe(result.page)}</strong><p>${safe(result.text)}</p></div><span class="result-score">${Number(result.score || 0).toFixed(3)}</span></div>`).join('')}</div><div class="answer-box" style="margin-top:7px"><strong>Context:</strong> ${safe(retrieval.context || '(trống)')}</div></section>`;
  }
  if (trace) {
    html += `<section class="detail-section"><h3>AGENT ANSWER</h3><div class="answer-box">${safe(trace.answer || '—')}<br><br><strong>Citations:</strong> ${safe((trace.citations || []).join(', ') || 'không có')}<br><strong>Số search:</strong> ${(trace.events || []).filter(e => e.event === 'search').length}</div></section>`;
    html += `<section class="detail-section"><h3>TOOL TRACE</h3><div class="trace-list">${(trace.events || []).map((event,index) => `<div class="trace-item"><div class="trace-item-top"><span>${safe(event.event || 'event')} ${index+1}</span><small>${safe(event.query || '')}</small></div>${event.context ? `<p>${safe(event.context)}</p>` : ''}${event.event === 'search' ? `<p>${(event.results || []).length} kết quả · ${safe((event.context_ids || []).join(', '))}</p>` : ''}</div>`).join('')}</div></section>`;
  }
  $('drawerContent').innerHTML = html;
  document.body.classList.add('drawer-open');
  $('detailDrawer').setAttribute('aria-hidden','false');
}

function closeDrawer() { document.body.classList.remove('drawer-open'); $('detailDrawer').setAttribute('aria-hidden','true'); }
$('splitSelect').addEventListener('change', e => { state.split = e.target.value; render(); });
$('querySearch').addEventListener('input', () => renderQuestions(splitData()));
$('typeFilter').addEventListener('change', () => renderQuestions(splitData()));
$('refreshButton').addEventListener('click', async () => { await loadData(); $('toast').classList.add('show'); setTimeout(() => $('toast').classList.remove('show'), 1600); });
$('showAllResults').addEventListener('click', () => $('questions').scrollIntoView({ behavior:'smooth' }));
$('closeDrawer').addEventListener('click', closeDrawer);
$('drawerBackdrop').addEventListener('click', closeDrawer);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });
document.querySelectorAll('.nav-item').forEach(link => link.addEventListener('click', () => { document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active')); link.classList.add('active'); }));
loadData();
