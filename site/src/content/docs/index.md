---
title: Bookvar
description: Открытый русскоязычный учебник по машинному и глубокому обучению — от математических основ до современных LLM.
template: splash
hero:
  title: Машинное обучение и языковые модели
  tagline: "Последовательный учебник: от линейной алгебры и нейронных сетей до Transformer, обучения по предпочтениям, RAG и агентных систем. Объяснения опираются на университетские курсы и первичные статьи."
  actions:
    - text: Начать с учебника
      link: ./textbook/
      icon: right-arrow
      variant: primary
    - text: Открыть практику
      link: ./practice/causal-self-attention/
      icon: open-book
      variant: minimal
---

<div class="landing-intro">
  <p class="eyebrow">Карта знаний</p>
  <h2 id="choose-a-route">Выберите маршрут</h2>
  <p>Материал организован по задаче: последовательно изучать, быстро уточнять механизм, сравнивать модели или воспроизводить результат.</p>
</div>

<nav class="entry-grid" aria-labelledby="choose-a-route">
  <a class="entry-card" href="./textbook/">
    <span class="entry-index" aria-hidden="true">01</span>
    <strong>Учебник</strong>
    <span>Последовательный путь от основ нейросетей к архитектуре и обучению LLM.</span>
  </a>
  <a class="entry-card" href="./reference/">
    <span class="entry-index" aria-hidden="true">02</span>
    <strong>Справочник</strong>
    <span>Короткий путь к механизму, модели, обзору направления или хронологии развития технологий.</span>
  </a>
  <a class="entry-card" href="./models/">
    <span class="entry-index" aria-hidden="true">03</span>
    <strong>Атлас моделей</strong>
    <span>Атлас моделей: что наследуется, что меняется и чем приходится за это платить.</span>
  </a>
  <a class="entry-card" href="./sources/">
    <span class="entry-index" aria-hidden="true">04</span>
    <strong>Источники</strong>
    <span>Курсы, книги, статьи и реализации, на которых построены объяснения и иллюстрации.</span>
  </a>
  <a class="entry-card" href="./practice/causal-self-attention/">
    <span class="entry-index" aria-hidden="true">05</span>
    <strong>Практика</strong>
    <span>Небольшие реализации и эксперименты, которые проверяют понимание кодом.</span>
  </a>
  <a class="entry-card" href="./questions/llm/">
    <span class="entry-index" aria-hidden="true">06</span>
    <strong>Вопросы</strong>
    <span>Короткий вход через конкретный вопрос со ссылкой на развёрнутый ответ.</span>
  </a>
</nav>

<section class="curriculum" aria-labelledby="curriculum-title">
  <div class="curriculum-copy">
    <p class="eyebrow">Учебный маршрут</p>
    <h2 id="curriculum-title">От механизма к системе</h2>
    <p>Основной маршрут проходит от математического аппарата к работающей системе. Ответвления для рекомендательных систем, ML-инфраструктуры и углублённых тем явно помечены в оглавлении.</p>
  </div>
  <ol class="curriculum-steps">
    <li><span>01 · Основания</span><strong>Линейная алгебра, вероятность и обучение моделей</strong></li>
    <li><span>02 · Язык</span><strong>Токены, языковое моделирование, RNN и Seq2Seq</strong></li>
    <li><span>03 · Transformer</span><strong>Attention, BERT, GPT и три архитектурных паттерна</strong></li>
    <li><span>04 · Современная LLM</span><strong>LLaMA-подобный блок, длинный контекст и MoE</strong></li>
    <li><span>05 · Обучение</span><strong>Данные, масштабирование, SFT, предпочтения и RLVR</strong></li>
    <li><span>06 · Система</span><strong>Инференс, поиск, мультимодальность и агенты</strong></li>
  </ol>
</section>

<section class="contribution" aria-labelledby="contribution-title">
  <div>
    <p class="eyebrow">Открытый конспект</p>
    <h2 id="contribution-title">Помогите сделать объяснение точнее</h2>
  </div>
  <p>Нашли неточность, слабый источник или пропущенный эксперимент? <a href="https://github.com/alpotekhin/bookvar/issues/new">Создайте issue</a> и укажите страницу, спорное утверждение и первичный источник. Текст хранится в Markdown, поэтому содержательную правку можно прислать обычным pull request.</p>
</section>
