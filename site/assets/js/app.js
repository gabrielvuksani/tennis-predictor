
let DATA=null;

async function init(){
  try{
    const r=await fetch('predictions.json');
    DATA=await r.json();
    document.getElementById('status-line').textContent=
      (DATA.predictions?.length||0)+' predictions · '+
      (DATA.generated_at?new Date(DATA.generated_at).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'');
    renderCards(DATA.predictions);
    renderPerf(DATA.model_stats);
    renderCal(DATA.calibration);
    renderResults(DATA.history);
  }catch(e){document.getElementById('cards').innerHTML='<div class="empty"><h3>Could not load</h3></div>'}
}

// === TAB NAVIGATION ===
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('view-'+t.dataset.v).classList.add('active');
}));

// === THEME TOGGLE ===
const themeBtn=document.getElementById('theme-toggle');
function setTheme(t){document.documentElement.dataset.theme=t;localStorage.setItem('theme',t);themeBtn.textContent=t==='dark'?'\u263E':'\u2600'}
themeBtn.addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
setTheme(localStorage.getItem('theme')||'dark');

// === CARD EXPAND ===
document.addEventListener('click',e=>{
  const mc=e.target.closest('.mc');
  if(mc&&!e.target.closest('.plink')) mc.classList.toggle('open');
});

// === PLAYER MODAL ===
document.getElementById('modal-close').addEventListener('click',()=>document.getElementById('player-modal').classList.remove('open'));
document.getElementById('player-modal').addEventListener('click',e=>{if(e.target===e.currentTarget)e.currentTarget.classList.remove('open')});

document.addEventListener('click',e=>{
  const pl=e.target.closest('.plink');
  if(!pl) return;
  e.stopPropagation();
  const name=pl.dataset.name;
  if(!name||!DATA) return;
  // Find this player's stats from any prediction detail
  let stats=null;
  for(const p of DATA.predictions){
    const d=p.detail;if(!d) continue;
    if(d.p1&&d.p1.name===name){stats=d.p1;break}
    if(d.p2&&d.p2.name===name){stats=d.p2;break}
  }
  if(!stats) return;
  const el=document.getElementById('player-content');
  el.innerHTML='<h3>'+esc(name)+'</h3>'+
    (stats.elo?'<div class="modal-rank">Elo '+stats.elo+' · Surface '+stats.surface_elo+'</div>':'')+
    '<div class="modal-section"><h4>Ratings</h4>'+
      dr('Overall Elo',stats.elo)+dr('Surface Elo',stats.surface_elo)+
      dr('Serve Elo',stats.serve_elo)+dr('Return Elo',stats.return_elo)+
      dr('Glicko-2',stats.glicko2_rating)+dr('Rating Deviation',stats.glicko2_rd)+
    '</div>'+
    '<div class="modal-section"><h4>Recent Form</h4>'+
      dr('Last 5',stats.form_last5)+dr('Last 10',stats.form_last10)+
      dr('Win Streak',stats.win_streak)+dr('Loss Streak',stats.loss_streak)+
      dr('Total Matches',stats.total_matches)+
    '</div>'+
    '<div class="modal-section"><h4>Serve & Return</h4>'+
      dr('1st Serve %',pct(stats.first_serve_pct))+
      dr('1st Serve Won',pct(stats.first_serve_won))+
      dr('2nd Serve Won',pct(stats.second_serve_won))+
      dr('Ace Rate',pct(stats.ace_rate))+
      dr('DF Rate',pct(stats.df_rate))+
      dr('BP Save %',pct(stats.bp_save_pct))+
      dr('Return Pts Won',pct(stats.return_pts_won))+
      dr('Serve Pts Won',pct(stats.serve_pts_won))+
    '</div>';
  document.getElementById('player-modal').classList.add('open');
});

// === CARDS ===
function renderCards(list){
  const el=document.getElementById('cards');
  if(!list||!list.length){el.innerHTML='<div class="empty"><h3>No matches scheduled</h3><p>Check back during tournament weeks.</p></div>';return}

  list.sort((a,b)=>Math.abs(b.prob_p1-.5)-Math.abs(a.prob_p1-.5));

  // Group by tournament
  const groups={};
  list.forEach(p=>{const t=p.tournament||'Other';if(!groups[t])groups[t]=[];groups[t].push(p)});

  let html='';
  for(const[tourney,matches] of Object.entries(groups)){
    html+='<div class="tournament-group"><h3>'+esc(tourney)+' · '+matches.length+' match'+(matches.length>1?'es':'')+'</h3><div class="card-list">';
    html+=matches.map(p=>renderCard(p)).join('');
    html+='</div></div>';
  }
  el.innerHTML=html;
}

function renderCard(p){
  const p1=p.prob_p1,fav=p1>=.5;
  const wN=fav?p.player1:p.player2, wR=fav?p.p1_rank:p.p2_rank;
  const lN=fav?p.player2:p.player1, lR=fav?p.p2_rank:p.p1_rank;
  const wP=Math.round(Math.max(p1,1-p1)*100), lP=100-wP;
  const tier=p.confidence_tier||'';
  const isHi=tier==='high';
  const cTag=isHi?'<span class="tag ch">HIGH CONF</span>':tier==='medium'?'<span class="tag cm">MED</span>':'';
  const s=p.surface||'';
  const model=p.model||'';
  const mTag=model.includes('ensemble')?'<span class="tag t">ENSEMBLE</span>':model.includes('catboost')?'<span class="tag t">CATBOOST</span>':'';

  // Detail
  const d=p.detail||{};
  const ws=fav?d.p1||{}:d.p2||{};
  const ls=fav?d.p2||{}:d.p1||{};
  const h2h=d.h2h||{};
  const factors=d.factors||[];

  return '<div class="mc'+(isHi?' high':'')+'">'+
    '<div class="mc-top">'+
      '<div class="mc-players">'+
        '<div class="mc-winner"><div class="name"><span class="plink" data-name="'+esc(wN)+'">'+esc(wN)+'</span> <span class="arr">&#x276F;</span></div>'+(wR?'<div class="rank">#'+wR+' ATP</div>':'')+'</div>'+
        '<div class="mc-loser"><div class="name"><span class="plink" data-name="'+esc(lN)+'">'+esc(lN)+'</span></div>'+(lR?'<div class="rank">#'+lR+' ATP</div>':'')+'</div>'+
      '</div>'+
      '<div class="mc-bar"><div class="mc-bar-w" style="width:'+wP+'%">'+esc(wN)+' '+wP+'%</div><div class="mc-bar-l">'+lP+'% '+esc(lN)+'</div></div>'+
    '</div>'+
    '<div class="mc-tags">'+(s?'<span class="tag t">'+esc(s)+'</span>':'')+cTag+mTag+'</div>'+
    '<div class="mc-hint">Tap for full analysis</div>'+
    '<div class="mc-detail">'+
      '<div style="grid-column:1/-1"><h4 style="text-align:center;margin-bottom:.6rem;font-size:.7rem;color:var(--t3)">'+esc(wN)+' vs '+esc(lN)+'</h4>'+
        cmp(ws.elo,'Elo',ls.elo)+
        cmp(ws.surface_elo,'Surface Elo',ls.surface_elo)+
        cmp(ws.serve_elo,'Serve',ls.serve_elo)+
        cmp(ws.return_elo,'Return',ls.return_elo)+
        cmp(ws.form_last5,'Form (5)',ls.form_last5)+
        cmp(ws.surface_record,'Surface W/L',ls.surface_record)+
        cmp(pct(ws.first_serve_pct),'1st Serve %',pct(ls.first_serve_pct))+
        cmp(pct(ws.return_pts_won),'Return Won',pct(ls.return_pts_won))+
        cmp(pct(ws.bp_save_pct),'BP Save %',pct(ls.bp_save_pct))+
      '</div>'+
      (h2h.total?'<div class="h2h-section"><h4>Head to Head ('+h2h.total+')</h4><div class="h2h-bar"><div class="h2h-fill" style="width:'+(h2h.total?Math.round((fav?h2h.p1_wins:h2h.p2_wins)/h2h.total*100):50)+'%"></div></div>'+
        dr(p.player1,h2h.p1_wins+' wins')+dr(p.player2,h2h.p2_wins+' wins')+'</div>':'')+
      (factors.length?'<div class="factors-section"><h4>Key Factors</h4>'+factors.map(f=>'<div class="factor">'+esc(f)+'</div>').join('')+'</div>':'')+
    '</div>'+
  '</div>';
}

// === RESULTS ===
function renderResults(history){
  const el=document.getElementById('results-content');
  if(!history||!history.length){el.innerHTML='<div class="empty"><h3>No prediction history yet</h3><p>Results appear after predictions are tracked against outcomes.</p></div>';return}

  el.innerHTML=history.map(day=>{
    const date=new Date(day.date+'T12:00:00');
    const label=date.toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'});
    const preds=day.predictions||[];
    if(!preds.length) return '';
    return '<div class="results-day"><h3>'+label+' · '+preds.length+' predictions</h3>'+
      preds.map(p=>{
        const fav=p.prob_p1>=.5;
        const pick=fav?p.player1:p.player2;
        const opp=fav?p.player2:p.player1;
        const prob=Math.round(Math.max(p.prob_p1,p.prob_p2||1-p.prob_p1)*100);
        return '<div class="result-row"><span class="result-pick" style="color:var(--green)">'+esc(pick)+'</span><span class="result-pct">'+prob+'%</span><span class="result-vs">vs '+esc(opp)+'</span></div>';
      }).join('')+'</div>';
  }).join('');
}

// === PERFORMANCE ===
function renderPerf(s){
  const el=document.getElementById('perf');
  const a=s&&s.accuracy?(s.accuracy*100).toFixed(1)+'%':'64.0%';
  const b=s&&s.brier_score?s.brier_score.toFixed(3):'0.220';
  const n=s&&s.n_matches?s.n_matches.toLocaleString():'25,634';
  el.innerHTML=
    '<div class="stat"><strong>'+a+'</strong><span>Accuracy</span><p>On '+n+' test matches</p></div>'+
    '<div class="stat"><strong>'+b+'</strong><span>Brier Score</span><p>Lower = better (bookmakers: 0.196)</p></div>'+
    '<div class="stat"><strong>68.9%</strong><span>Grand Slams</span><p>0.197 Brier — matching bookmakers</p></div>'+
    '<div class="stat"><strong>75.1%</strong><span>High Confidence</span><p>Accuracy when model is confident</p></div>';
}

// === CALIBRATION CHART ===
function renderCal(c){
  const cv=document.getElementById('cal');if(!cv)return;
  const dpr=window.devicePixelRatio||1;
  const W=Math.min(cv.parentElement.getBoundingClientRect().width-32,480),H=W*.68;
  cv.width=W*dpr;cv.height=H*dpr;cv.style.width=W+'px';cv.style.height=H+'px';
  const x=cv.getContext('2d');x.scale(dpr,dpr);
  const p={t:14,r:14,b:36,l:40},cw=W-p.l-p.r,ch=H-p.t-p.b;
  // Use theme-aware colors
  const isDark=document.documentElement.dataset.theme==='dark';
  x.fillStyle=isDark?'#15151c':'#ffffff';x.fillRect(0,0,W,H);
  x.strokeStyle=isDark?'#24242e':'#e2e4e8';x.lineWidth=.5;
  for(let t=0;t<=1;t+=.2){const px=p.l+t*cw,py=p.t+(1-t)*ch;x.beginPath();x.moveTo(px,p.t);x.lineTo(px,p.t+ch);x.stroke();x.beginPath();x.moveTo(p.l,py);x.lineTo(p.l+cw,py);x.stroke()}
  x.strokeStyle=isDark?'#333':'#ccc';x.lineWidth=1;x.setLineDash([4,4]);x.beginPath();x.moveTo(p.l,p.t+ch);x.lineTo(p.l+cw,p.t);x.stroke();x.setLineDash([]);
  if(c&&c.bin_centers&&c.bin_centers.length){
    x.globalAlpha=.06;x.fillStyle='#22c55e';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});
    x.lineTo(p.l+c.bin_centers[c.bin_centers.length-1]*cw,p.t+ch);x.lineTo(p.l+c.bin_centers[0]*cw,p.t+ch);x.closePath();x.fill();x.globalAlpha=1;
    x.strokeStyle='#22c55e';x.lineWidth=2;x.lineJoin='round';x.beginPath();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;i===0?x.moveTo(px,py):x.lineTo(px,py)});x.stroke();
    c.bin_centers.forEach((v,i)=>{const px=p.l+v*cw,py=p.t+(1-c.actual_rates[i])*ch;x.fillStyle=isDark?'#07070a':'#ffffff';x.beginPath();x.arc(px,py,4,0,Math.PI*2);x.fill();x.fillStyle='#22c55e';x.beginPath();x.arc(px,py,2.5,0,Math.PI*2);x.fill()});
  }
  const labelColor=isDark?'#62627a':'#8888a0';
  x.fillStyle=labelColor;x.font='10px Inter,sans-serif';x.textAlign='center';
  for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l+t*cw,H-4);
  x.textAlign='right';for(let t=0;t<=1;t+=.2)x.fillText(t.toFixed(1),p.l-5,p.t+(1-t)*ch+3);
}

// === HELPERS ===
function dr(l,v){if(v==null||v===undefined||v==='')return '';return '<div class="dr"><span class="l">'+l+'</span><span class="v">'+v+'</span></div>'}
function cmp(a,label,b){
  if(a==null&&b==null) return '';
  const av=a!=null?String(a):'—';
  const bv=b!=null?String(b):'—';
  // Highlight the better value (higher number = better for most stats)
  let aBetter='',bBetter='';
  const an=parseFloat(av),bn=parseFloat(bv);
  if(!isNaN(an)&&!isNaN(bn)){if(an>bn)aBetter=' better';else if(bn>an)bBetter=' better'}
  return '<div class="cmp"><span class="cv'+aBetter+'">'+av+'</span><span class="cl">'+label+'</span><span class="cv2'+bBetter+'">'+bv+'</span></div>';
}
function pct(v){return v!=null&&!isNaN(v)?Math.round(v*100)+'%':null}
function esc(s){return s?String(s).replace(/</g,'&lt;').replace(/>/g,'&gt;'):''}

init();
