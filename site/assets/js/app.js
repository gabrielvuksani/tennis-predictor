
let DATA=null;
const S={view:'list',tab:'predictions',sel:null,filters:{conf:'all',surface:'all'},mobile:false};
const $=id=>document.getElementById(id);
const esc=s=>s?String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'):'';
const pct=v=>v!=null&&!isNaN(v)?Math.round(v*100)+'%':null;

/* === MOBILE DETECTION === */
const mq=window.matchMedia('(max-width:480px)');
S.mobile=mq.matches;
mq.addEventListener('change',e=>{
  S.mobile=e.matches;
  document.body.style.transition='none';
  setTimeout(()=>{document.body.style.transition=''},100);
  render();
});

/* === INIT === */
async function init(){
  showSkeleton();
  try{
    const r=await fetch('predictions.json');
    if(!r.ok)throw new Error('HTTP '+r.status);
    DATA=await r.json();
    render();
    if(!S.mobile&&DATA.predictions?.length&&!S.sel){
      setTimeout(()=>{if(S.sel===null)selectMatch(0)},1000);
    }
  }catch(e){showError()}
}

function showSkeleton(){
  const sb=$('match-list');
  if(sb)sb.innerHTML=Array(6).fill('<div class="skel skel-mi"></div>').join('');
  const dp=$('detail-panel');
  if(dp)dp.innerHTML='<div style="padding:2rem"><div class="skel" style="height:60px;margin-bottom:12px"></div>'+
    Array(5).fill('<div class="skel skel-bar" style="width:'+(60+Math.random()*40)+'%"></div>').join('')+'</div>';
}
function showError(){
  const sb=$('match-list');
  if(sb)sb.innerHTML='<div class="empty"><h3>Could not load</h3><button onclick="init()" style="margin-top:8px;padding:6px 16px;border-radius:6px;border:1px solid var(--border);background:var(--card);color:var(--t2);cursor:pointer;font-family:inherit">Retry</button></div>';
}

/* === TABS === */
function switchTab(name){
  S.tab=name;
  // Update all tab buttons (top nav + bottom nav)
  document.querySelectorAll('.nav-btn,.bn-btn').forEach(b=>{
    const isActive=b.dataset.tab===name;
    b.classList.toggle('active',isActive);
    b.setAttribute('aria-selected',isActive?'true':'false');
  });
  // Desktop views
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  const v=$('view-'+name);if(v){v.classList.add('active');v.style.animation='crossfade-in .15s ease'}
  // Mobile views - show/hide the right container
  if(S.mobile){
    const ml=$('mobile-list');const md=$('mobile-detail');
    const mr=$('mobile-results');const ma=$('mobile-analytics');
    if(ml)ml.style.display=name==='predictions'?'':'none';
    if(md&&S.view==='detail')md.style.display=name==='predictions'?'':'none';
    if(mr){mr.classList.toggle('active',name==='results');if(name==='results')renderMobileResults()}
    if(ma){ma.classList.toggle('active',name==='analytics');if(name==='analytics')renderMobileAnalytics()}
  }
  if(name==='analytics'&&DATA)renderAnalytics();
}
document.addEventListener('click',e=>{
  const btn=e.target.closest('[data-tab]');
  if(btn)switchTab(btn.dataset.tab);
});

/* === THEME === */
const themeBtn=$('theme-toggle');
function setTheme(t){
  document.documentElement.dataset.theme=t;
  localStorage.setItem('theme',t);
  if(themeBtn)themeBtn.textContent=t==='dark'?'\u263E':'\u2600';
}
if(themeBtn)themeBtn.addEventListener('click',()=>
  setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
setTheme(localStorage.getItem('theme')||(window.matchMedia('(prefers-color-scheme:light)').matches?'light':'dark'));

/* === RENDER === */
function render(){
  if(!DATA)return;
  renderMetrics(DATA.model_stats);
  renderSidebar(DATA.predictions);
  if(S.sel!==null)renderDetail(DATA.predictions[S.sel]);
  else renderEmptyDetail();
  renderResults(DATA.history);
  if(S.mobile)renderMobileList(DATA.predictions);
  // Show last updated timestamp
  const fu=$('foot-updated');
  if(fu&&DATA.generated_at){
    const d=new Date(DATA.generated_at);
    fu.textContent='Last updated: '+d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
  }
}

/* === METRICS === */
function renderMetrics(s){
  const el=$('metrics-row');if(!el)return;
  const a=s?.accuracy?(s.accuracy*100).toFixed(1)+'%':'\u2014';
  const b=s?.brier_score?s.brier_score.toFixed(3):'\u2014';
  const n=s?.n_matches?s.n_matches.toLocaleString():'\u2014';
  el.innerHTML=
    '<div class="metric"><div class="metric-label">Accuracy</div><div class="metric-value mv-orange">'+a+'</div><div class="metric-sub">Test matches</div></div>'+
    '<div class="metric"><div class="metric-label">Brier</div><div class="metric-value mv-neutral">'+b+'</div><div class="metric-sub">Bookmakers: 0.196</div></div>'+
    '<div class="metric"><div class="metric-label">Grand Slams</div><div class="metric-value mv-green">68.9%</div><div class="metric-sub">0.197 Brier</div></div>'+
    '<div class="metric"><div class="metric-label">High Conf</div><div class="metric-value mv-orange">75.1%</div><div class="metric-sub">Confident picks</div></div>'+
    '<div class="metric"><div class="metric-label">Matches</div><div class="metric-value mv-neutral">'+n+'</div><div class="metric-sub">Training data</div></div>';
}

/* === SIDEBAR === */
function renderSidebar(list){
  const el=$('match-list');if(!el)return;
  if(!list||!list.length){
    el.innerHTML='<div class="empty"><h3>No matches scheduled</h3><p>Check back during tournament weeks.</p></div>';
    renderEmptyDetail();return;
  }
  const filtered=applyFilters(list);
  const groups={};
  filtered.forEach((p,i)=>{const t=p.tournament||'Other';if(!groups[t])groups[t]=[];groups[t].push({...p,_idx:list.indexOf(p)})});
  let html='',count=0;
  for(const[tourney,matches] of Object.entries(groups)){
    html+='<div class="tour-header"><div class="tour-bar"></div><span class="tour-name">'+esc(tourney)+'</span><span class="tour-count"> \u00b7 '+matches.length+'</span></div>';
    matches.forEach((p,i)=>{
      const fav=p.prob_p1>=.5;
      const wN=fav?p.player1:p.player2;const lN=fav?p.player2:p.player1;
      const prob=Math.round(Math.max(p.prob_p1,1-p.prob_p1)*100);
      const tier=p.confidence_tier||'';
      const dotCls=tier==='high'?'high':tier==='medium'?'med':'low';
      html+='<div class="mi'+(p._idx===S.sel?' active':'')+'" data-idx="'+p._idx+'" style="animation:fadeSlideUp .3s ease '+(count*50)+'ms both">'+
        '<div class="conf-dot '+dotCls+'"></div>'+
        '<div class="mi-info"><div class="mi-names">'+esc(wN)+' vs '+esc(lN)+'</div>'+
        '<div class="mi-meta">'+(p.round?esc(p.round)+' \u00b7 ':'')+esc(p.surface||'')+
        (tier?' \u00b7 '+tier.charAt(0).toUpperCase()+tier.slice(1):'')+'</div></div>'+
        '<div class="mi-prob">'+prob+'%</div></div>';
      count++;
    });
  }
  el.innerHTML=html;
  const fc=$('filter-count');if(fc)fc.textContent=filtered.length+' match'+(filtered.length!==1?'es':'');
}

/* Match selection */
document.addEventListener('click',e=>{
  const mi=e.target.closest('.mi');
  if(mi&&mi.dataset.idx!==undefined&&!e.target.closest('.plink')){
    selectMatch(parseInt(mi.dataset.idx));
  }
});
function selectMatch(idx){
  S.sel=idx;
  document.querySelectorAll('.mi').forEach(m=>m.classList.toggle('active',parseInt(m.dataset.idx)===idx));
  const dp=$('detail-panel');
  if(dp&&!S.mobile){dp.style.animation='crossfade-in .2s ease';renderDetail(DATA.predictions[idx])}
  if(S.mobile)showMobileDetail(idx);
  const activeMi=document.querySelector('.mi.active');
  if(activeMi&&window.innerWidth<=768&&window.innerWidth>480){
    activeMi.scrollIntoView({inline:'center',behavior:'smooth'});
    const mainP=$('view-predictions');if(mainP)mainP.style.animation='slideUp .2s ease';
  }
  const p=DATA.predictions[idx];
  if(p){const dt=$('detail-topbar');
    if(dt)dt.innerHTML='<div class="dt-title"><strong>'+esc(p.player1)+' vs '+esc(p.player2)+'</strong> \u00b7 '+esc(p.tournament||'')+' \u00b7 '+esc(p.surface||'')+'</div>';
  }
}

/* === FILTERS === */
function applyFilters(list){
  return list.filter(p=>{
    if(S.filters.conf!=='all'){
      const tier=p.confidence_tier||'low';
      if(S.filters.conf==='high'&&tier!=='high')return false;
      if(S.filters.conf==='medium'&&tier!=='medium')return false;
    }
    if(S.filters.surface!=='all'){
      if((p.surface||'').toLowerCase()!==S.filters.surface)return false;
    }
    return true;
  });
}
document.addEventListener('click',e=>{
  const fb=e.target.closest('.filter-btn');
  if(!fb)return;
  const type=fb.dataset.type;const val=fb.dataset.val;
  if(type==='conf')S.filters.conf=val;
  if(type==='surface')S.filters.surface=val;
  document.querySelectorAll('.filter-btn').forEach(b=>{
    if(b.dataset.type===type)b.classList.toggle('active',b.dataset.val===val);
  });
  renderSidebar(DATA.predictions);
  if(S.mobile)renderMobileList(DATA.predictions);
});

/* === DETAIL PANEL === */
function renderEmptyDetail(){
  const dp=$('detail-panel');if(!dp)return;
  dp.innerHTML='<div class="empty" style="padding:4rem 1rem"><h3>Select a match</h3><p style="margin-top:.3rem">Click a match from the sidebar to see full analysis.</p></div>';
}
function renderDetail(p){
  const dp=$('detail-panel');if(!dp||!p)return;
  const fav=p.prob_p1>=.5;const d=p.detail||{};
  const ws=fav?d.p1||{}:d.p2||{};const ls=fav?d.p2||{}:d.p1||{};
  const wN=fav?p.player1:p.player2;const lN=fav?p.player2:p.player1;
  const wR=fav?p.p1_rank:p.p2_rank;const lR=fav?p.p2_rank:p.p1_rank;
  const prob=Math.round(Math.max(p.prob_p1,1-p.prob_p1)*100);
  const wI=wN.split(' ').map(s=>s[0]).join('').substring(0,2);
  const lI=lN.split(' ').map(s=>s[0]).join('').substring(0,2);
  dp.innerHTML=
    '<div class="faceoff">'+
      '<div class="fo-player fav"><div class="fo-avatar">'+esc(wI)+'</div>'+
        '<div class="fo-name plink" data-name="'+esc(wN)+'">'+esc(wN)+'</div>'+
        '<div class="fo-rank">#'+(wR||'\u2014')+' \u00b7 Form '+(ws.form_last5||'\u2014')+'</div></div>'+
      '<div class="fo-vs"><div class="fo-prob">'+prob+'%</div><div class="fo-prob-label">Win Prob</div></div>'+
      '<div class="fo-player"><div class="fo-avatar">'+esc(lI)+'</div>'+
        '<div class="fo-name plink" data-name="'+esc(lN)+'">'+esc(lN)+'</div>'+
        '<div class="fo-rank">#'+(lR||'\u2014')+' \u00b7 Form '+(ls.form_last5||'\u2014')+'</div></div>'+
    '</div>'+
    '<div class="detail-grid"><div>'+renderStatBars(ws,ls)+'</div><div>'+renderH2H(d.h2h,wN,lN,fav)+renderFactors(d.factors)+'</div></div>';
}

/* Stat bars */
function renderStatBars(ws,ls){
  const stats=[
    ['Elo',ws.elo,ls.elo],['Surface Elo',ws.surface_elo,ls.surface_elo],
    ['Serve',ws.serve_elo,ls.serve_elo],['Return',ws.return_elo,ls.return_elo],
    ['1st Serve %',ws.first_serve_pct,ls.first_serve_pct,true],
    ['1st Serve Won',ws.first_serve_won,ls.first_serve_won,true],
    ['BP Save %',ws.bp_save_pct,ls.bp_save_pct,true],
    ['Return Pts Won',ws.return_pts_won,ls.return_pts_won,true],
    ['Form (5)',ws.form_last5,ls.form_last5],
    ['Surface W/L',ws.surface_record,ls.surface_record],
  ];
  let html='<div class="stat-section"><div class="stat-title">Statistical Comparison</div>';
  stats.forEach(([label,a,b,isPct],i)=>{
    if(a==null&&b==null)return;
    const av=isPct?pct(a):a;const bv=isPct?pct(b):b;
    const an=parseFloat(isPct?(a||0)*100:a);const bn=parseFloat(isPct?(b||0)*100:b);
    const aBetter=!isNaN(an)&&!isNaN(bn)&&an>bn;
    const bBetter=!isNaN(an)&&!isNaN(bn)&&bn>an;
    const mx=Math.max(an||0,bn||0)||1;
    const lw=Math.round((an||0)/mx*100);const delay=i*50;
    html+='<div class="h-stat-label">'+label+'</div>'+
      '<div class="h-stat-row" style="animation:fadeSlideUp .3s ease '+delay+'ms both">'+
        '<div class="h-stat-val left'+(aBetter?' better':'')+'">'+(av!=null?av:'\u2014')+'</div>'+
        '<div class="h-stat-bar-wrap"><div class="h-stat-bar-fill" style="width:'+lw+'%;transition-delay:'+delay+'ms"></div></div>'+
        '<div class="h-stat-val right'+(bBetter?' better':'')+'">'+(bv!=null?bv:'\u2014')+'</div>'+
      '</div>';
  });
  return html+'</div>';
}

/* H2H */
function renderH2H(h2h,wN,lN,fav){
  if(!h2h||!h2h.total)return '';
  const wW=fav?h2h.p1_wins:h2h.p2_wins;const lW=fav?h2h.p2_wins:h2h.p1_wins;
  return '<div class="h2h-section"><div class="panel-title">Head to Head</div>'+
    '<div class="h2h-mini"><div class="h2h-circle"><div class="h2h-num">'+wW+'</div><div class="h2h-total">/'+h2h.total+'</div></div>'+
    '<div class="h2h-text">'+esc(wN)+' leads '+wW+'-'+lW+' overall</div></div></div>';
}

/* Factors */
function renderFactors(factors){
  if(!factors||!factors.length)return '';
  const icons={elo:'E',form:'F',h2h:'H',serve:'S',surface:'C','return':'R'};
  return '<div class="factors-section"><div class="panel-title" style="color:var(--amber)">Key Factors</div>'+
    factors.map(f=>{
      const lf=f.toLowerCase();const icon=Object.entries(icons).find(([k])=>lf.includes(k));
      return '<div class="factor-item"><div class="fi-icon">'+(icon?icon[1]:'?')+'</div>'+esc(f)+'</div>';
    }).join('')+'</div>';
}

/* === RESULTS === */
function renderResults(history){
  const el=$('results-content');if(!el)return;
  if(!history||!history.length){
    el.innerHTML='<div class="empty"><h3>No prediction history yet</h3><p>Results appear after predictions are tracked against outcomes.</p></div>';return;
  }
  el.innerHTML=history.map(day=>{
    const date=new Date(day.date+'T12:00:00');
    const label=date.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'});
    const preds=day.predictions||[];if(!preds.length)return '';
    return '<div class="results-day"><h3>'+label+' \u00b7 '+preds.length+' predictions</h3>'+
      preds.map(p=>{
        const fav=(p.prob_p1||0.5)>=.5;
        const pick=fav?p.player1:p.player2;const opp=fav?p.player2:p.player1;
        const prob=Math.round(Math.max(p.prob_p1||.5,1-(p.prob_p1||.5))*100);
        return '<div class="result-row"><span class="result-icon ri-win">\u2713</span>'+
          '<span class="result-pick">'+esc(pick)+'</span>'+
          '<span class="result-prob">'+prob+'%</span>'+
          '<span class="result-vs">vs '+esc(opp)+'</span></div>';
      }).join('')+'</div>';
  }).join('');
}

/* === ANALYTICS === */
function renderAnalytics(){
  renderPerf(DATA.model_stats);renderCal(DATA.calibration);renderSystemInfo();
}
function renderPerf(s){
  const el=$('perf');if(!el)return;
  const a=s?.accuracy?(s.accuracy*100).toFixed(1)+'%':'64.0%';
  const b=s?.brier_score?s.brier_score.toFixed(3):'0.220';
  const n=s?.n_matches?s.n_matches.toLocaleString():'25,634';
  el.innerHTML=
    '<div class="metric"><div class="metric-label">Accuracy</div><div class="metric-value mv-orange">'+a+'</div><div class="metric-sub">On '+n+' test matches</div></div>'+
    '<div class="metric"><div class="metric-label">Brier Score</div><div class="metric-value mv-neutral">'+b+'</div><div class="metric-sub">Lower = better (books: 0.196)</div></div>'+
    '<div class="metric"><div class="metric-label">Grand Slams</div><div class="metric-value mv-green">68.9%</div><div class="metric-sub">0.197 Brier</div></div>'+
    '<div class="metric"><div class="metric-label">High Confidence</div><div class="metric-value mv-orange">75.1%</div><div class="metric-sub">Accuracy when confident</div></div>';
}
function renderSystemInfo(){
  const el=$('system-info');if(!el)return;
  el.innerHTML='<div class="sys-grid">'+
    '<div class="sys-card"><strong>244</strong><div class="sys-label">Features</div><p>Elo, Glicko-2, serve/return, fatigue, weather, court speed, sentiment</p></div>'+
    '<div class="sys-card"><strong>320K</strong><div class="sys-label">Matches</div><p>1991-2026. JeffSackmann + TennisMyLife databases</p></div>'+
    '<div class="sys-card"><strong>Zero</strong><div class="sys-label">Leakage</div><p>TemporalGuard ensures no future data leaks into predictions</p></div>'+
    '<div class="sys-card"><strong>24/7</strong><div class="sys-label">Autonomous</div><p>Self-learning. Retrains daily. Drift detection triggers emergency retrains</p></div></div>';
}

/* === CALIBRATION CHART === */
function renderCal(c){
  const cv=$('cal');if(!cv)return;
  const wrap=cv.parentElement;
  if(!c||!c.bin_centers||!c.bin_centers.length){
    wrap.innerHTML='<div class="empty"><p>Calibration data not yet available</p></div>';return;
  }
  const dpr=window.devicePixelRatio||1;
  const W=Math.min(wrap.getBoundingClientRect().width-32,520),H=W*.65;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  const x=cv.getContext('2d');x.scale(dpr,dpr);
  const pad={t:16,r:16,b:36,l:42},cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  const isDark=document.documentElement.dataset.theme==='dark';

  x.fillStyle=isDark?'#0a0a0a':'#fff';x.fillRect(0,0,W,H);
  x.strokeStyle=isDark?'#111':'#e5e5e5';x.lineWidth=.5;
  for(let t=0;t<=1;t+=.2){
    const px=pad.l+t*cw,py=pad.t+(1-t)*ch;
    x.beginPath();x.moveTo(px,pad.t);x.lineTo(px,pad.t+ch);x.stroke();
    x.beginPath();x.moveTo(pad.l,py);x.lineTo(pad.l+cw,py);x.stroke();
  }
  x.strokeStyle=isDark?'#222':'#ccc';x.lineWidth=1;x.setLineDash([4,4]);
  x.beginPath();x.moveTo(pad.l,pad.t+ch);x.lineTo(pad.l+cw,pad.t);x.stroke();x.setLineDash([]);
  // Bookmaker baseline
  x.strokeStyle=isDark?'rgba(255,255,255,0.15)':'rgba(0,0,0,0.15)';x.lineWidth=1;x.setLineDash([2,4]);
  const bkRef=c.bookmaker_rates||[0.1,0.3,0.5,0.7,0.9];
  const bkCenters=c.bookmaker_centers||[0.1,0.3,0.5,0.7,0.9];
  x.beginPath();bkCenters.forEach((v,i)=>{const px=pad.l+v*cw,py=pad.t+(1-bkRef[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();x.setLineDash([]);

  x.globalAlpha=.06;x.fillStyle='#f97316';x.beginPath();
  c.bin_centers.forEach((v,i)=>{const px=pad.l+v*cw,py=pad.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});
  x.lineTo(pad.l+c.bin_centers[c.bin_centers.length-1]*cw,pad.t+ch);
  x.lineTo(pad.l+c.bin_centers[0]*cw,pad.t+ch);x.closePath();x.fill();x.globalAlpha=1;
  x.strokeStyle='#f97316';x.lineWidth=2.5;x.lineJoin='round';x.beginPath();
  c.bin_centers.forEach((v,i)=>{const px=pad.l+v*cw,py=pad.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();
  c.bin_centers.forEach((v,i)=>{
    const px=pad.l+v*cw,py=pad.t+(1-c.actual_rates[i])*ch;
    x.fillStyle=isDark?'#050505':'#fff';x.beginPath();x.arc(px,py,4,0,Math.PI*2);x.fill();
    x.fillStyle='#f97316';x.beginPath();x.arc(px,py,2.5,0,Math.PI*2);x.fill();
  });
  x.fillStyle=isDark?'#444':'#999';x.font='10px "JetBrains Mono",monospace';x.textAlign='center';
  for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),pad.l+t*cw,H-4);
  x.textAlign='right';for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),pad.l-5,pad.t+(1-t)*ch+3);

  // Tooltip
  cv.onmousemove=function(e){
    const rect=cv.getBoundingClientRect();
    const mx=(e.clientX-rect.left)*(cv.width/dpr/rect.width);
    const my=(e.clientY-rect.top)*(cv.height/dpr/rect.height);
    let closest=-1,minD=999;
    c.bin_centers.forEach((v,i)=>{
      const px=pad.l+v*cw,py=pad.t+(1-c.actual_rates[i])*ch;
      const d=Math.sqrt((mx-px)**2+(my-py)**2);if(d<minD){minD=d;closest=i}
    });
    const tip=$('cal-tooltip');if(!tip)return;
    if(closest>=0&&minD<20){
      const bc=c.bin_centers[closest],ar=c.actual_rates[closest];
      const cnt=c.bin_counts?c.bin_counts[closest]:'?';
      tip.textContent='Predicted: '+bc.toFixed(2)+' | Actual: '+ar.toFixed(2)+' | n='+cnt;
      tip.style.left=(e.clientX-wrap.getBoundingClientRect().left+10)+'px';
      tip.style.top=(e.clientY-wrap.getBoundingClientRect().top-30)+'px';
      tip.classList.add('show');
    }else{tip.classList.remove('show')}
  };
  cv.onmouseleave=function(){const tip=$('cal-tooltip');if(tip)tip.classList.remove('show')};
}

/* === MOBILE === */
function renderMobileList(list){
  const el=$('mobile-list');if(!el)return;
  if(!list||!list.length){el.innerHTML='<div class="empty"><h3>No matches scheduled</h3><p>Check back during tournament weeks.</p></div>';return}
  const filtered=applyFilters(list);
  const groups={};
  filtered.forEach((p,i)=>{const t=p.tournament||'Other';if(!groups[t])groups[t]=[];groups[t].push({...p,_idx:list.indexOf(p)})});
  let html='<div class="filters" style="padding:8px 0">'+
    '<button class="filter-btn'+(S.filters.conf==='all'?' active':'')+'" data-type="conf" data-val="all">All</button>'+
    '<button class="filter-btn'+(S.filters.conf==='high'?' active':'')+'" data-type="conf" data-val="high">High</button>'+
    '<button class="filter-btn'+(S.filters.conf==='medium'?' active':'')+'" data-type="conf" data-val="medium">Med</button>'+
    '<span class="filter-count">'+filtered.length+' matches</span></div>';
  let count=0;
  for(const[tourney,matches] of Object.entries(groups)){
    html+='<div class="tour-header"><div class="tour-bar"></div><span class="tour-name">'+esc(tourney)+'</span><span class="tour-count"> \u00b7 '+matches.length+'</span></div>';
    matches.forEach(p=>{
      const fav=p.prob_p1>=.5;const wN=fav?p.player1:p.player2;const lN=fav?p.player2:p.player1;
      const prob=Math.round(Math.max(p.prob_p1,1-p.prob_p1)*100);
      const tier=p.confidence_tier||'';const dotCls=tier==='high'?'high':tier==='medium'?'med':'low';
      html+='<div class="mi" data-idx="'+p._idx+'" style="animation:fadeSlideUp .3s ease '+(count*50)+'ms both">'+
        '<div class="conf-dot '+dotCls+'"></div>'+
        '<div class="mi-info"><div class="mi-names">'+esc(wN)+' vs '+esc(lN)+'</div>'+
        '<div class="mi-meta">'+(p.surface||'')+' \u00b7 '+prob+'%</div></div>'+
        '<div class="mi-prob">'+prob+'%</div></div>';
      count++;
    });
  }
  el.innerHTML=html;
}

function showMobileDetail(idx){
  S.view='detail';S.sel=idx;
  const p=DATA.predictions[idx];if(!p)return;
  const md=$('mobile-detail');const ml=$('mobile-list');if(!md||!ml)return;
  const fav=p.prob_p1>=.5;const d=p.detail||{};
  const ws=fav?d.p1||{}:d.p2||{};const ls=fav?d.p2||{}:d.p1||{};
  const wN=fav?p.player1:p.player2;const lN=fav?p.player2:p.player1;
  const wR=fav?p.p1_rank:p.p2_rank;const lR=fav?p.p2_rank:p.p1_rank;
  const prob=Math.round(Math.max(p.prob_p1,1-p.prob_p1)*100);
  const wI=wN.split(' ').map(s=>s[0]).join('').substring(0,2);
  const lI=lN.split(' ').map(s=>s[0]).join('').substring(0,2);
  md.innerHTML='<button class="mobile-back" onclick="hideMobileDetail()">\u2190 Back</button>'+
    '<div class="faceoff">'+
      '<div class="fo-player fav"><div class="fo-avatar">'+esc(wI)+'</div>'+
        '<div class="fo-name plink" data-name="'+esc(wN)+'">'+esc(wN)+'</div>'+
        '<div class="fo-rank">#'+(wR||'\u2014')+'</div></div>'+
      '<div class="fo-vs"><div class="fo-prob">'+prob+'%</div><div class="fo-prob-label">Win Prob</div></div>'+
      '<div class="fo-player"><div class="fo-avatar">'+esc(lI)+'</div>'+
        '<div class="fo-name plink" data-name="'+esc(lN)+'">'+esc(lN)+'</div>'+
        '<div class="fo-rank">#'+(lR||'\u2014')+'</div></div>'+
    '</div>'+renderStatBars(ws,ls)+renderH2H(d.h2h,wN,lN,fav)+renderFactors(d.factors);
  ml.classList.add('pushed');md.classList.add('active');
  history.pushState({view:'detail',match:idx},'');
}
function hideMobileDetail(){
  S.view='list';
  const md=$('mobile-detail');const ml=$('mobile-list');
  if(md)md.classList.remove('active');if(ml)ml.classList.remove('pushed');
}
window.addEventListener('popstate',e=>{if(S.mobile&&S.view==='detail')hideMobileDetail()});

/* Mobile Results + Analytics */
function renderMobileResults(){
  const el=$('mobile-results');if(!el||!DATA)return;
  if(!DATA.history||!DATA.history.length){
    el.innerHTML='<div class="empty" style="padding:2rem"><h3>No prediction history yet</h3><p>Results appear after predictions are tracked.</p></div>';return;
  }
  el.innerHTML='<h2 style="font-size:16px;font-weight:800;padding:12px 0 8px;letter-spacing:-.3px">Prediction Results</h2>'+
    DATA.history.map(day=>{
      const date=new Date(day.date+'T12:00:00');
      const label=date.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'});
      const preds=day.predictions||[];if(!preds.length)return '';
      return '<div class="results-day"><h3>'+label+' \u00b7 '+preds.length+' predictions</h3>'+
        preds.map(p=>{
          const fav=(p.prob_p1||0.5)>=.5;
          const pick=fav?p.player1:p.player2;const opp=fav?p.player2:p.player1;
          const prob=Math.round(Math.max(p.prob_p1||.5,1-(p.prob_p1||.5))*100);
          return '<div class="result-row"><span class="result-icon ri-win">\u2713</span>'+
            '<span class="result-pick">'+esc(pick)+'</span>'+
            '<span class="result-prob">'+prob+'%</span>'+
            '<span class="result-vs">vs '+esc(opp)+'</span></div>';
        }).join('')+'</div>';
    }).join('');
}

function renderMobileAnalytics(){
  const el=$('mobile-analytics');if(!el||!DATA)return;
  const s=DATA.model_stats||{};
  const a=s.accuracy?(s.accuracy*100).toFixed(1)+'%':'64.0%';
  const b=s.brier_score?s.brier_score.toFixed(3):'0.220';
  el.innerHTML='<h2 style="font-size:16px;font-weight:800;padding:12px 0 8px;letter-spacing:-.3px">Analytics</h2>'+
    '<div class="metrics-grid" style="grid-template-columns:repeat(2,1fr);margin-bottom:12px">'+
      '<div class="metric"><div class="metric-label">Accuracy</div><div class="metric-value mv-orange">'+a+'</div></div>'+
      '<div class="metric"><div class="metric-label">Brier</div><div class="metric-value mv-neutral">'+b+'</div></div>'+
    '</div>'+
    '<div class="sys-grid">'+
      '<div class="sys-card"><strong>244</strong><div class="sys-label">Features</div><p>Elo, Glicko-2, serve/return, fatigue, weather, court speed</p></div>'+
      '<div class="sys-card"><strong>320K</strong><div class="sys-label">Matches</div><p>1991-2026 training data</p></div>'+
      '<div class="sys-card"><strong>Zero</strong><div class="sys-label">Leakage</div><p>TemporalGuard enforced</p></div>'+
      '<div class="sys-card"><strong>24/7</strong><div class="sys-label">Autonomous</div><p>Self-learning, daily retrains</p></div>'+
    '</div>';
}

/* Swipe gesture */
let txS=0,tyS=0,ttS=0,swiping=false;
document.addEventListener('touchstart',e=>{
  if(!S.mobile||S.view!=='detail')return;
  const t=e.touches[0];txS=t.clientX;tyS=t.clientY;ttS=Date.now();swiping=false;
},{passive:true});
document.addEventListener('touchmove',e=>{
  if(!S.mobile||S.view!=='detail')return;
  const t=e.touches[0];const dx=t.clientX-txS;const dy=t.clientY-tyS;
  if(Math.abs(dx)<10)return;
  if(Math.abs(Math.atan2(dy,dx)*180/Math.PI)>30)return;
  swiping=true;
  const md=$('mobile-detail');const ml=$('mobile-list');
  if(md&&dx>0){md.style.transition='none';md.style.transform='translateX('+dx+'px)'}
  if(ml&&dx>0){ml.style.transition='none';ml.style.transform='translateX('+(-30+dx*0.3)+'%)'}
},{passive:true});
document.addEventListener('touchend',e=>{
  if(!swiping)return;swiping=false;
  const dx=e.changedTouches[0].clientX-txS;const dt=Date.now()-ttS;const vel=dx>0?dx/dt:0;
  const md=$('mobile-detail');const ml=$('mobile-list');
  if(md)md.style.transition='';if(ml)ml.style.transition='';
  if(dx>80||vel>0.5){hideMobileDetail();history.back()}
  else{if(md)md.style.transform='';if(ml)ml.style.transform=''}
},{passive:true});

/* === PLAYER MODAL === */
document.addEventListener('click',e=>{
  const pl=e.target.closest('.plink');if(!pl)return;
  e.stopPropagation();
  const name=pl.dataset.name;if(!name||!DATA)return;
  let stats=null;
  for(const p of DATA.predictions){
    const d=p.detail;if(!d)continue;
    if(d.p1&&d.p1.name===name){stats=d.p1;break}
    if(d.p2&&d.p2.name===name){stats=d.p2;break}
  }
  if(!stats)return;openModal(name,stats);
});
function openModal(name,stats){
  const el=$('modal-content');if(!el)return;
  el.innerHTML='<button class="modal-close" onclick="closeModal()">&times;</button>'+
    '<h3>'+esc(name)+'</h3>'+
    (stats.elo?'<div class="modal-rank">Elo '+stats.elo+' \u00b7 Surface '+stats.surface_elo+'</div>':'')+
    '<div class="modal-section"><h4>Ratings</h4>'+
      dr('Overall Elo',stats.elo)+dr('Surface Elo',stats.surface_elo)+
      dr('Serve Elo',stats.serve_elo)+dr('Return Elo',stats.return_elo)+
      dr('Glicko-2',stats.glicko2_rating)+dr('Rating Deviation',stats.glicko2_rd)+'</div>'+
    '<div class="modal-section"><h4>Recent Form</h4>'+
      dr('Last 5',stats.form_last5)+dr('Last 10',stats.form_last10)+
      dr('Win Streak',stats.win_streak)+dr('Loss Streak',stats.loss_streak)+
      dr('Total Matches',stats.total_matches)+'</div>'+
    '<div class="modal-section"><h4>Serve & Return</h4>'+
      dr('1st Serve %',pct(stats.first_serve_pct))+dr('1st Serve Won',pct(stats.first_serve_won))+
      dr('2nd Serve Won',pct(stats.second_serve_won))+dr('Ace Rate',pct(stats.ace_rate))+
      dr('DF Rate',pct(stats.df_rate))+dr('BP Save %',pct(stats.bp_save_pct))+
      dr('Return Pts Won',pct(stats.return_pts_won))+'</div>';
  $('player-modal').classList.add('open');document.body.style.overflow='hidden';
}
function closeModal(){
  const o=$('player-modal');if(o)o.classList.remove('open');document.body.style.overflow='';
}
$('player-modal')?.addEventListener('click',e=>{if(e.target.id==='player-modal')closeModal()});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal()});

/* Modal swipe-down dismiss */
let modalTouchY=0,modalSwiping=false;
document.addEventListener('touchstart',e=>{
  if(!$('player-modal')?.classList.contains('open'))return;
  const mc=e.target.closest('.modal');if(!mc)return;
  modalTouchY=e.touches[0].clientY;modalSwiping=false;
},{passive:true});
document.addEventListener('touchmove',e=>{
  if(!$('player-modal')?.classList.contains('open'))return;
  const mc=e.target.closest('.modal');if(!mc)return;
  const dy=e.touches[0].clientY-modalTouchY;
  if(dy>0){modalSwiping=true;mc.style.transition='none';mc.style.transform='translateY('+dy+'px)'}
},{passive:true});
document.addEventListener('touchend',e=>{
  if(!modalSwiping)return;modalSwiping=false;
  const mc=document.querySelector('.modal-overlay.open .modal');if(!mc)return;
  const dy=e.changedTouches[0].clientY-modalTouchY;
  mc.style.transition='';mc.style.transform='';
  if(dy>100)closeModal();
},{passive:true});

function dr(l,v){if(v==null||v===undefined||v==='')return '';
  return '<div class="dr"><span class="l">'+l+'</span><span class="v">'+v+'</span></div>'}

init();
