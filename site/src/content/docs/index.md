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
  <a class="entry-card" href="./textbook/transformer/self-attention/">
    <span class="entry-index" aria-hidden="true">02</span>
    <strong>Технологии</strong>
    <span>Механизмы, из которых собраны модели: внимание, MoE, RAG и обучение после предобучения.</span>
  </a>
  <a class="entry-card" href="./models/families/deepseek/">
    <span class="entry-index" aria-hidden="true">03</span>
    <strong>Семейства</strong>
    <span>Атлас моделей: что наследуется, что меняется и чем приходится за это платить.</span>
  </a>
  <a class="entry-card" href="./sources/papers/deepseek-r1/">
    <span class="entry-index" aria-hidden="true">04</span>
    <strong>Статьи</strong>
    <span>Разборы первичных источников: постановка задачи, метод, эксперимент и границы выводов.</span>
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
    <p>Главы можно читать по порядку. Каждый следующий уровень опирается на предыдущий и отделяет архитектуру, обучение и продуктовое поведение.</p>
  </div>
  <ol class="curriculum-steps">
    <li><span>Основания</span><strong>Градиент, оптимизация, представления</strong></li>
    <li><span>Архитектура</span><strong>Механизм внимания, Transformer, MoE</strong></li>
    <li><span>Системы</span><strong>Дообучение, инференс, RAG</strong></li>
  </ol>
</section>

<section class="contribution" aria-labelledby="contribution-title">
  <div>
    <p class="eyebrow">Открытый конспект</p>
    <h2 id="contribution-title">Помогите сделать объяснение точнее</h2>
  </div>
  <p>Нашли неточность, слабый источник или пропущенный эксперимент? Предложите правку через канал обратной связи проекта, указав страницу, утверждение и первичный источник.</p>
</section>
