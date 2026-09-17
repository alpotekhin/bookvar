# Systems — журнал четырёх согласованных исправлений

Дата: 2026-09-15. Сначала полностью прочитаны и зафиксированы 60/60 исходных страниц в `systems.json` и `systems.md`; лишь затем изменены пять перечисленных ниже русских глав. Исходные `read_sha256`, excerpts и вердикты аудита оставлены снимком до исправлений. Другие замечания не исправлялись. Английские страницы, исходники, общая структура и чужие правки не затронуты; коммитов и push не было.

## 1. Собственные K/V текущей позиции в decode

[00 Учебник/07 Анатомия современной LLM/04 RoPE.md](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/07 Анатомия современной LLM/04 RoPE.md:198>) теперь включает текущие `R₄k₄,v₄` в доступный набор до вычисления внимания. У запроса позиции 4 пять ключей (0…4), а его выход предсказывает позицию 5; это не утечка будущего. Строка Decode в таблице исправлена согласованно.

[00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md:41>) задаёт `S−1` старых позиций и текущую позицию `S`, поэтому `Attn(q_S,K₁:S,V₁:S)` включает собственные K/V. Последующие оценки с `S` ключами сохранили своё значение, а не получили скрытое смещение на один.

Это согласуется с [reference implementation Llama 3](https://raw.githubusercontent.com/meta-llama/llama3/main/llama/model.py): текущие K/V помещаются в cache перед чтением до `start_pos + seqlen`. Малый пример с четырьмя нулевыми историческими значениями и текущим значением 1 даёт внимание 0,5; ошибочное исключение текущей позиции даёт 0. Второй query той же KV-головы даёт 0,2, что также проверяет общий K/V в GQA. Для текущей позиции явно применяется двумерный RoPE-поворот.

## 2. Нелинейность и её производная в TP-MLP

[00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md:31>) теперь задаёт поэлементную `φ`, локальные `Z_r=XA_r`, `U_r=φ(Z_r)` и `Y_r=U_rB_r`. В backward восстановлено `dZ_r=dU_r⊙φ′(Z_r)`, после чего суммируются `dX_r=dZ_rA_rᵀ`. Размещения и коллективные операции не изменены.

Проверка двух TP-rank при `T=1,D=2,H=4,φ=ReLU`: плотный и разрезанный forward одинаково дают `[8,4]`, тогда как пропуск `φ` даёт `[5.5,4.5]`. Градиент входа равен `[6,−3]` и совпадает с центральной конечной разностью при шаге `10⁻⁶` с ошибкой меньше `10⁻⁸`; пропуск `φ′` даёт `[10,0]`. Числа выбраны вне излома ReLU.

## 3. Новые позиции против полной истории в scheduling

[00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md:99>) различает `S_i` (длину KV-контекста) и `T_i` (новые позиции текущей итерации). Вход token-wise операций имеет форму `[ΣT_i,d]`: обычный decode даёт одну строку на запрос, prefill — длину назначенного фрагмента. Добавлен малый пример: контексты 100 и 200 плюс prefill-фрагмент 4 дают `1+1+4=6` строк, а не 304. Семантика attention, маски и адреса KV-блоков не менялись.

## 4. Первичный источник GLU

[00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md](</Users/aleksan.potekhin/Documents/knowledge base/02 Areas/ML & DL/.worktrees/ml-systems-ingestion/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md:9>) и библиография той же главы, строка 308: неверный `2002.12327` заменён на [Dauphin et al., Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083). Проверены название и авторы первичного источника. Соседняя ссылка [Shazeer, GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) сохранена: это другая работа, которая сама ссылается на исходный GLU. Старый URL ведёт на [A Primer in BERTology](https://arxiv.org/abs/2002.12327), не на GLU.

Первый запуск проверки нашёл ещё одно вхождение неправильного URL в `primary_sources`, хотя библиография уже была исправлена. После согласованного исправления обоих вхождений повторный запуск прошёл. Это изменение осталось в том же разрешённом файле и в рамках той же ошибки ссылки.

## Проверки и границы результата

Локальный Node.js v25.2.1 выполнил все проверки ниже с кодом выхода 0. `git diff --check` по пяти главам также завершился с кодом 0; diff просмотрен. После правок повторное сравнение с исходным аудитом обнаружило ровно пять ожидаемых изменённых страниц и 55 неизменённых. Эти проверки доказывают малые математические примеры, согласованность формул и целевых строк Markdown; они не являются запуском distributed training, serving benchmark или визуальным рендерингом. Полная сборка сайта здесь не запускалась.

Для повторения из корня рабочей копии выполнить следующий код в Node.js. Он читает только аудит и пять глав, не меняет файлы.

```javascript
const fs = require("fs"), assert = require("assert");
const audit = JSON.parse(fs.readFileSync("docs/audit/2026-09-15/systems.json"));
const page = i => fs.readFileSync(audit.pages[i - 1].path, "utf8");
const near = (a, b, eps = 1e-9) => assert(Math.abs(a - b) < eps, a + " != " + b);
const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
const rot = (v,t) => [Math.cos(t)*v[0]-Math.sin(t)*v[1],Math.sin(t)*v[0]+Math.cos(t)*v[1]];
const attn = (q,K,V) => {
  const w=K.map(k=>Math.exp(dot(q,k)/Math.sqrt(2))), z=w.reduce((a,b)=>a+b);
  return w.reduce((s,x,i)=>s+x*V[i]/z,0);
};
// Four old positions and one current position. Two query heads share one KV head.
const angle=4*Math.PI/8;
const currentKey=rot(rot([Math.sqrt(2)*Math.log(4),0],-angle),angle);
const Kold=Array.from({length:4},()=>[0,0]), Vold=[0,0,0,0];
const K=Kold.concat([currentKey]), V=Vold.concat([1]);
const q=rot(rot([1,0],-angle),angle);
const fixedAttention=attn(q,K,V), missingCurrent=attn(q,Kold,Vold);
near(fixedAttention,0.5); near(missingCurrent,0);
near(attn([0,1],K,V),0.2); assert.equal(K.length,5);
assert(page(4).includes("сравнивается с пятью ключами"));
assert(!page(4).includes("всеми четырьмя cached keys"));
assert(page(6).includes("\\operatorname{Attn}(q_S,K_{1:S},V_{1:S})"));
assert(page(6).includes("$S-1$"));
assert(!page(6).includes("q_{S+1}"));
// Two TP ranks must compute the same nonlinear MLP as its dense counterpart.
const x=[1,-2], A=[[1,2,-1,0.5],[1,-1,2,1]];
const B=[[1,0],[2,1],[0,-1],[1,3]], dy=[1,1];
const mul=(v,M)=>M[0].map((_,j)=>v.reduce((s,a,i)=>s+a*M[i][j],0));
const relu=v=>v.map(a=>Math.max(0,a));
const dense=v=>mul(relu(mul(v,A)),B);
const denseY=dense(x), tpY=[0,0], tpDX=[0,0];
for(let r=0;r<2;r++){
  const Ar=A.map(row=>row.slice(2*r,2*r+2)), Br=B.slice(2*r,2*r+2);
  const zr=mul(x,Ar), ur=relu(zr), yr=mul(ur,Br);
  const dur=Br.map(row=>dot(dy,row)), dzr=dur.map((g,j)=>zr[j]>0?g:0);
  for(let j=0;j<2;j++){tpY[j]+=yr[j];tpDX[j]+=dot(dzr,Ar[j]);}
}
denseY.forEach((v,j)=>near(v,tpY[j]));
const eps=1e-6, numericDX=x.map((_,j)=>{
  const xp=x.slice(),xm=x.slice(); xp[j]+=eps; xm[j]-=eps;
  return (dot(dense(xp),dy)-dot(dense(xm),dy))/(2*eps);
});
numericDX.forEach((v,j)=>near(v,tpDX[j],1e-8));
const oldY=mul(mul(x,A),B), oldDX=A.map(row=>dot(B.map(b=>dot(dy,b)),row));
assert(oldY.some((v,j)=>Math.abs(v-denseY[j])>0.1));
assert(oldDX.some((v,j)=>Math.abs(v-tpDX[j])>0.1));
assert(page(32).includes("U_r=\\phi(Z_r)"));
assert(page(32).includes("dZ_r=dU_r\\odot\\phi'(Z_r)"));
assert(page(32).includes("dX_r=dZ_rA_r^\\top"));
// Histories are cached; only scheduled positions enter token-wise operations.
const history=[100,200,4], scheduled=[1,1,4];
const rows=scheduled.reduce((a,b)=>a+b), oldRows=history.reduce((a,b)=>a+b);
assert.equal(rows,6); assert.equal(oldRows,304);
assert(page(44).includes("X_{packed}\\in\\mathbb{R}^{(\\sum_i T_i)\\times d}"));
assert(!page(44).includes("X_{packed}\\in\\mathbb{R}^{(\\sum_i S_i)\\times d}"));
assert(page(10).includes("https://arxiv.org/abs/1612.08083"));
assert(!page(10).includes("https://arxiv.org/abs/2002.12327"));
console.log(JSON.stringify({attention:{fixed:fixedAttention,old:missingCurrent,shared_kv_second_head:attn([0,1],K,V)},
  tp:{denseY,tpY,oldY,tpDX,numericDX,oldDX},
  scheduling:{rows,oldRows},source_link:"1612.08083"},null,2));
```

Фактический результат последнего успешного запуска:

```json
{
  "attention": {
    "fixed": 0.5,
    "old": 0,
    "shared_kv_second_head": 0.2
  },
  "tp": {
    "denseY": [
      8,
      4
    ],
    "tpY": [
      8,
      4
    ],
    "oldY": [
      5.5,
      4.5
    ],
    "tpDX": [
      6,
      -3
    ],
    "numericDX": [
      6.000000000838668,
      -2.9999999995311555
    ],
    "oldDX": [
      10,
      0
    ]
  },
  "scheduling": {
    "rows": 6,
    "oldRows": 304
  },
  "source_link": "1612.08083"
}
```

## Хеши изменённых глав

### 00 Учебник/07 Анатомия современной LLM/04 RoPE.md

До: `21387b3d5e33d3ffb1820acb1b4b57eec626756fb3ee540e58bdfbb5f6d03e74`  
После: `86fb1f58f6ddaa72f6424c55ee810359d05749cd4f284d67ba4bf0c92e635bbc`

### 00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA.md

До: `f8c00ab66e2eac6daee2a9bc9781d04e69ed99b109a7f4cad5a6cf573d2820bc`  
После: `60402496763f3a3c4f1464d6ae354770805dbadb0566830001a1d9a0a6196f53`

### 00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md

До: `8007224380b5ae387b12aac83dbf9564e7a3ad4be8dd7f602911117dc48ba3e6`  
После: `943a8beb5b22a2c17df3173e63691af6db3bc1430416d9a58fec471413124795`

### 00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md

До: `48f6acd65360d7c9955aeab76a1e1c1a42fb2d96aa871722912c5ac7411bb9b5`  
После: `eec3a2ed6c7b1f09d740f3f76a440b9b79bfce873efa8a506790fbb80868660e`

### 00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching.md

До: `170ad643c0731ea0121b2018cd13f3a2375224d55a3fc45da01b6fb04c1210b2`  
После: `a2e0c582adfb539b97aaa701c544c7d0a083791ee580f46df474f388644b27ab`

## Осталось в очереди

Четыре согласованные группы закрывают пять записей аудита: RoPE, GQA, TP-MLP, scheduling и ссылку GLU. Остальные 83 из 88 замечаний остаются вне этого пакета. Отдельно не исправлены GB/GiB в 57a, 1F1B в арифметике Transformer, источник speculative sampling, автор SpinQuant, autocast/Triton/Poisson, языковые проблемы и повторение глав. Эти изменения входят в дальнейшую редакционную работу; ограничение пакета не означает, что пользователь их запретил.
