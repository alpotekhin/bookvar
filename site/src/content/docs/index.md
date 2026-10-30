---
title: Bookvar
description: Открытый русскоязычный учебник по машинному и глубокому обучению — от математических основ до современных LLM.
template: splash
hero:
  title: Машинное обучение и языковые модели
  tagline: "Основной курс от линейной алгебры до обучения и оценивания языковых моделей. Отдельные модули по ML-системам, инференсу, RAG, мультимодальности, агентам и рекомендациям."
  actions:
    - text: Начать с учебника
      link: ./textbook/
      icon: right-arrow
      variant: primary
    - text: Найти тему
      link: ./reference/
      icon: open-book
      variant: minimal
---

<div class="landing-intro">
  <h2 id="choose-a-route">Чтение, поиск и первоисточники</h2>
  <p>Учебник задаёт порядок изучения. Справочник ведёт к тем же объяснениям по названию темы. В библиотеке находятся оригинальные курсы, книги и статьи.</p>
</div>

<nav class="entry-grid" aria-labelledby="choose-a-route">
  <a class="entry-card" href="./textbook/">
    <span class="entry-index" aria-hidden="true">01</span>
    <strong>Учебник</strong>
    <span>Основной курс ML и LLM, затем углублённые модули с указанными предпосылками.</span>
  </a>
  <a class="entry-card" href="./reference/">
    <span class="entry-index" aria-hidden="true">02</span>
    <strong>Справочник</strong>
    <span>Темы ML и DL, механизмы LLM, модели, сравнения и хронология. Ссылки ведут к полным объяснениям.</span>
  </a>
  <a class="entry-card" href="./sources/">
    <span class="entry-index" aria-hidden="true">03</span>
    <strong>Библиотека источников</strong>
    <span>Курсы, книги, статьи и реализации, на которых построены объяснения и иллюстрации.</span>
  </a>
</nav>

<p><a href="./practice/">Практика и интерактивные примеры</a> · <a href="./questions/">Вопросы со ссылками на ответы</a></p>

<section class="curriculum" aria-labelledby="curriculum-title">
  <div class="curriculum-copy">
    <p class="eyebrow">Учебный маршрут</p>
    <h2 id="curriculum-title">От механизма к системе</h2>
    <p>Основной курс заканчивается генерацией и оцениванием языковой модели. Для дальнейшего чтения выберите модуль в оглавлении: каждый имеет свой порядок глав и необходимые предпосылки.</p>
  </div>
  <ol class="curriculum-steps">
    <li><span>01 · Основания</span><strong>Линейная алгебра, вероятность и обучение моделей</strong></li>
    <li><span>02 · Язык</span><strong>Токены, языковое моделирование, RNN и Seq2Seq</strong></li>
    <li><span>03 · Transformer</span><strong>Attention, BERT, GPT и три архитектурных паттерна</strong></li>
    <li><span>04 · Современная LLM</span><strong>Декодер, RoPE, FFN, формы тензоров и параметры</strong></li>
    <li><span>05 · Обучение</span><strong>Данные, масштабирование, SFT, предпочтения и RLVR</strong></li>
    <li><span>06 · Проверка модели</span><strong>Генерация ответов, оценивание и контаминация</strong></li>
  </ol>
</section>

<section class="contribution" aria-labelledby="contribution-title">
  <div>
    <p class="eyebrow">Открытый конспект</p>
    <h2 id="contribution-title">Помогите сделать объяснение точнее</h2>
  </div>
  <p>Нашли неточность, слабый источник или пропущенный эксперимент? <a href="https://github.com/alpotekhin/bookvar/issues/new">Создайте issue</a> и укажите страницу, спорное утверждение и первичный источник. Текст хранится в Markdown, поэтому содержательную правку можно прислать обычным pull request.</p>
</section>
