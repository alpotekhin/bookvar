---
title: "Week06 Dl Arithmetic — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week06_dl_arithmetic/lecture.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week06_dl_arithmetic/lecture.pdf?raw=1" title="Week06 Dl Arithmetic — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
Арифметика глубокого обучения




Михаил Хрущев
Руководитель группы претрейна Alice AI
```

### Page 2

```text
Познакомимся


Я – Михаил Хрущёв, руковожу группой
претрейна Alice AI.

Более 4 лет я и моя команда обучаем
большие языковые модели и разрабатываем
инфраструктуру для их обучения.

Основная часть LLM моделей Яндекса были
обучены нами или на нашей
инфраструктуре.

Наши open source:
• YaFSDP
• YaLM 100B
• Yandex GPT5 Lite
```

### Page 3

```text
Содержание

 01 История 1
 02 История 2
```

### Page 4

```text
История 1


Перед итерацией обучения нам нужно вычитать и подготовить батч.
Обычно подготовка происходит на CPU:



      Поход в БД       Подготовка батча                   Итерация обучения
```

### Page 5

```text
История 1


Перед итерацией обучения нам нужно вычитать и подготовить батч.
Обычно подготовка происходит на CPU:



      Поход в БД        Подготовка батча                   Итерация обучения



Проблема: Из-за подготовки батча простаивает GPU – это очень дорогой ресурс.
```

### Page 6

```text
История 1


Перед итерацией обучения нам нужно вычитать и подготовить батч.
Обычно подготовка происходит на CPU:



      Поход в БД              Подготовка батча



Что можно сделать: Выносим чтение и подготовку батча в отдельный процесс.


    Подготовка батча          Подготовка батча     Подготовка батча


          Итерация обучения                      Итерация обучения          Итерация обучения
```

### Page 7

```text
История 1. Выводы


• Подготовка данных может привести к замедлению
  обучения и простою GPU.
• GPU – самый дорогой ресурс, его простой нежелателен.
• Чтобы избежать простоя, можно сделать загрузку данных
  асинхронной.



  Подготовка батча          Подготовка батча     Подготовка батча


        Итерация обучения                      Итерация обучения    Итерация обучения
```

### Page 8

```text
История 2. Внутри итерации




                 Итерация обучения
```

### Page 9

```text
История 2. Небольшой экскурс в DL
      Batch (B, H)




      Linear H -> I


          ReLU


      Linear I -> 10



        SoftMax



  Loss(outputs, targets)
```

### Page 10

```text
История 2. Небольшой экскурс в DL
      Batch (B, H)                        A


                                                          W
                           активации             Linear
                                                 Linear
      Linear H -> I


          ReLU             активации   A @ W.T



      Linear I -> 10       активации



        SoftMax



  Loss(outputs, targets)
```

### Page 11

```text
История 2. Небольшой экскурс в DL
      Batch (B, H)                        A           G@W


                                                              W
                                                 Linear
      Linear H -> I        активации                          A @ G.T



          ReLU             активации   A @ W.T            G



      Linear I -> 10       активации



        SoftMax



  Loss(outputs, targets)
```

### Page 12

```text
История 2. Небольшой экскурс в DL
      Batch (B, H)                        A           G@W


                                                              W
                                                 Linear
      Linear H -> I        активации                          A @ G.T



          ReLU             активации   A @ W.T            G



      Linear I -> 10       активации



        SoftMax



  Loss(outputs, targets)
```

### Page 13

```text
История 2. Небольшой экскурс в DL
      Batch (B, H)                        A             G@W


                                                                      W
                                                   Linear
      Linear H -> I        активации                                  A @ G.T



          ReLU             активации   A @ W.T              G



      Linear I -> 10       активации      A            G * I(A > 0)



        SoftMax                                    ReLU


  Loss(outputs, targets)
                                       max(A, 0)            G
```

### Page 14

```text
История 2. Небольшой экскурс в DL
      Batch (B, H)                        A             G@W


                                                                      W
                                                   Linear
      Linear H -> I        активации                                  A @ G.T



          ReLU             активации   A @ W.T              G



      Linear I -> 10       активации      A            G * I(A > 0)



        SoftMax                                    ReLU


  Loss(outputs, targets)
                                       max(A, 0)            G
```

### Page 15

```text
История 2. Небольшой экскурс в DL
      Batch (B, H)




      Linear H -> I        активации
                                       градиенты для весов


          ReLU             активации

                                                             Оптимизатор
      Linear I -> 10       активации
                                       градиенты для весов


        SoftMax



  Loss(outputs, targets)
```

### Page 16

```text
История 2. Небольшой экскурс в DL
      Batch (B, H)




      Linear H -> I        активации
                                            градиенты для весов
                                       обновленные веса

          ReLU             активации

                                                                   Оптимизатор
      Linear I -> 10       активации
                                             градиенты для весов
                                       обновленные веса

        SoftMax



  Loss(outputs, targets)
```

### Page 17

```text
История 2

    Embedding




                     Attention

         +
                                 xN


                       MLP

         +


  Debedding + Loss
```

### Page 18

```text
История 2
       Batch


    Embedding



                                  активации
                     Attention

         +


                                  активации
                       MLP

         +

                      активации
  Debedding + Loss


       Loss
```

### Page 19

```text
История 2
       Batch


    Embedding



                                  активации
                     Attention

         +


                                  активации
                       MLP                       A

         +
                                                                 W
                      активации                         Linear
  Debedding + Loss


       Loss                                   A @ W.T
```

### Page 20

```text
История 2
       Batch


    Embedding



                                  активации
                     Attention

         +


                                  активации                 G@W
                       MLP                       A

         +
                                                                     W
                      активации                         Linear
  Debedding + Loss                                                   A @ G.T


       Loss                                   A @ W.T            G
```

### Page 21

```text
История 2
       Batch


    Embedding         градиенты для весов



                                            активации
                     Attention              градиенты для весов

         +


                                            активации
                       MLP                  градиенты для весов
         +

                      активации
  Debedding + Loss    градиенты для весов

       Loss
```

### Page 22

```text
История 2
       Batch


    Embedding         градиенты для весов



                                            активации
                     Attention              градиенты для весов

         +
                                                                  Оптимизатор
                                            активации
                       MLP                  градиенты для весов
         +

                      активации
  Debedding + Loss    градиенты для весов

       Loss
```

### Page 23

```text
История 2
       Batch


    Embedding         градиенты для весов

                                            обновленные веса

                                            активации
                     Attention              градиенты для весов

         +                                  обновленные веса
                                                                  Оптимизатор
                                            активации
                       MLP                  градиенты для весов
         +                                  обновленные веса

                      активации
  Debedding + Loss    градиенты для весов

                                    обновленные веса
       Loss
```

### Page 24

```text
100M модель
            Batch


         Embedding             градиенты для весов



                                                     активации (42 MB)
                          Attention (2 MB)           градиенты для весов
                                                                           Оптимизатор
              +                                                            (x18 от числа
24 x                                                                        параметров
                                                                            для AdamW)
                                                     активации (84 MB)
                           MLP (6.3 MB)              градиенты для весов      18 GB
              +

                               активации
       Debedding + Loss        градиенты для весов

            Loss
```

### Page 25

```text
100M модель
            Batch


         Embedding             градиенты для весов



                                                     активации (42 MB)
                          Attention (2 MB)           градиенты для весов
                                                                           Оптимизатор
              +                                                            (x18 от числа
24 x                                                                        параметров
                                                                            для AdamW)
                                                     активации (84 MB)
                           MLP (6.3 MB)              градиенты для весов      1.8 GB
              +

                               активации
       Debedding + Loss        градиенты для весов

            Loss
                                              Всего: 5 GB
```

### Page 26

```text
1B модель
            Batch


         Embedding              градиенты для весов



                                                      активации (126 MB)
                          Attention (18 MB)           градиенты для весов
                                                                            Оптимизатор
              +                                                             (x18 от числа
36 x                                                                         параметров
                                                                             для AdamW)
                                                      активации (185 MB)
                            MLP (38 MB)               градиенты для весов      1.8 GB
              +

                                активации
       Debedding + Loss         градиенты для весов

            Loss
                                              Всего: 31.5 GB
```

### Page 27

```text
7B модель
            Batch


         Embedding              градиенты для весов



                                                      активации (235 MB)
                          Attention (84 MB)           градиенты для весов
                                                                            Оптимизатор
              +                                                             (x18 от числа
36 x                                                                         параметров
                                                                             для AdamW)
                                                      активации (492 MB)
                           MLP (268 MB)               градиенты для весов     126 GB
              +

                                активации
       Debedding + Loss         градиенты для весов

            Loss
                                               Всего: 170 GB
```

### Page 28

```text
Llama 70B
            Batch


         Embedding              градиенты для весов



                                                      активации (436 MB)
                          Attention (302 MB)          градиенты для весов
                                                                            Оптимизатор
              +                                                             (x18 от числа
80 x                                                                         параметров
                                                                             для AdamW)
                                                      активации (1.07 GB)
                           MLP (1409 MB)              градиенты для весов     1232 GB
              +

                                активации
       Debedding + Loss         градиенты для весов

            Loss
                                               Всего: 1352 GB
```

### Page 29

```text
Qwen 30B-A3B
            Batch


         Embedding              градиенты для весов



                                                      активации (201 MB)
                          Attention (42 MB)           градиенты для весов
                                                                            Оптимизатор
              +                                                             (x18 от числа
48 x                                                                         параметров
                                                                             для AdamW)
                                                      активации (503 MB)
                          MLP (1207 MB)               градиенты для весов     540 GB
              +

                                активации
       Debedding + Loss         градиенты для весов

            Loss
                                               Всего: 634 GB
```

### Page 30

```text
Qwen 235B-A32B
            Batch


         Embedding              градиенты для весов



                                                      активации (352 MB)
                          Attention (142 MB)          градиенты для весов
                                                                            Оптимизатор
              +                                                             (x18 от числа
94 x                                                                         параметров
                                                                             для AdamW)
                                                      активации (1000 MB)
                           MLP (4831 MB)              градиенты для весов     4.23 TB
              +

                                активации
       Debedding + Loss         градиенты для весов

            Loss
                                               Всего: 4.8 TB
```

### Page 31

```text
История 2. Проблема

• Для обучения модели нужны терабайты памяти.
• Но на GPU не так много: 40GB-190GB для серверных GPU,
  единицы GB для игровых.


 Что делать?
```

### Page 32

```text
История 2. Проблема

• Для обучения модели нужны терабайты памяти.
• Но на GPU не так много: 40GB-190GB для серверных GPU,
  единицы GB для игровых.


 Что делать?
 Хранить где-то еще!

 Как использовать веса/активации/состояния оптимизатора, если они лежат где-то?
```

### Page 33

```text
История 2. Проблема

• Для обучения модели нужны терабайты памяти.
• Но на GPU не так много: 40GB-190GB для серверных GPU,
  единицы GB для игровых.


 Что делать?
 Хранить где-то еще!

 Как использовать веса/активации/состояния оптимизатора, если они лежат где-то?
 Организовать пересылку.
```

### Page 34

```text
История 2. Проблема

• Для обучения модели нужны терабайты памяти.
• Но на GPU не так много: 40GB-190GB для серверных GPU,
  единицы GB для игровых.


 Что делать?
 Хранить где-то еще!

 Как использовать веса/активации/состояния оптимизатора, если они лежат где-то?
 Организовать пересылку.

 Но GPU будут простаивать, если пересылать данные?
 Нет, если пересылка асинхронная.
```

### Page 35

```text
История 1


Перед итерацией обучения нам нужно вычитать и подготовить батч.
Обычно подготовка происходит на CPU:



      Поход в БД              Подготовка батча



Что можно сделать: Выносим чтение и подготовку батча в отдельный процесс.


    Подготовка батча          Подготовка батча     Подготовка батча


          Итерация обучения                      Итерация обучения          Итерация обучения
```

### Page 36

```text
Эффективное обучение – логистическая
проблема

• Нужно разнести состояния оптимизатора, веса и активации на разные ресурсы:
  другие GPU, RAM, SSD.
• Нужно обеспечить сборку этих данных в момент использования на каждой GPU.
• Нужно организовать логистику так, чтобы GPU простаивала минимально.
```

### Page 37

```text
Эффективное обучение – логистическая
проблема

• Нужно разнести состояния оптимизатора, веса и активации на разные ресурсы:
  другие GPU, RAM, SSD.
• Нужно обеспечить сборку этих данных в момент использования на каждой GPU.
• Нужно организовать логистику так, чтобы GPU простаивала минимально.




 Этой задачей занимаются многие исследователи DL уже больше 7 лет.
```

### Page 38

```text
Содержание

 01   Определяемся
      с логистикой          06   Заключение


      Локальная логистика
 02   на GPU



 03   Оптимизация
      elementwise



 04   Коммуникации между
      GPU. FSDP


 05   Кейс с MoE
```

### Page 39

```text
Определяемся
с логистикой
```

### Page 40

```text
Логистика
```

### Page 41

```text
Что нужно для правильного построения
логистики

• Нужна карта маршрутов.
• Нужно понимание того, сколько стоит пройти тот или иной маршрут.
```

### Page 42

```text
Что нужно для правильного построения
логистики

• Нужна карта маршрутов.
• Нужно понимание того, сколько стоит пройти тот или иной маршрут.


Как получить такие данные?
• Спросить у экспертов
• Почитать спецификацию
• Замерить самим
```

### Page 43

```text
Что нужно для правильного построения
логистики

• Нужна карта маршрутов.
• Нужно понимание того, сколько стоит пройти тот или иной маршрут.


Как получить такие данные?
• Спросить у экспертов
• Почитать спецификацию
• Замерить самим
```

### Page 44

```text
Смотрим в спецификацию
```

### Page 45

```text
Смотрим в спецификацию
```

### Page 46

```text
Смотрим в спецификацию

                           L  OPS
                    400 TF


                               PS
                    800 TFLO

                        TF    LOPS
                   1600




                                c
                            B/se
                      2.4 T

                                      c
                                GB/se
                              0
                          + 40
                    400
```

### Page 47

```text
Что нужно для правильного построения
логистики

• Нужна карта маршрутов.
• Нужно понимание того, сколько стоит пройти тот или иной маршрут.


Как получить такие данные?
• Спросить у экспертов
• Почитать спецификацию
• Замерить самим
```

### Page 48

```text
Как замерить время и память?


• Использовать torch.profiler или nsys.
```

### Page 49

```text
Как замерить время и память?


• Использовать torch.profiler или nsys.
  Важно: время CPU искажается.
```

### Page 50

```text
Как замерить время и память?


• Использовать torch.profiler или nsys.
  Важно: время CPU искажается.
• Снять профиль памяти




https://pytorch.org/memory_viz
```

### Page 51

```text
Карта GPU (H100)

                                   HBM (80 GB)



                                 L2 Cache (50 Mb)

                                   GPU (132 SM)


   Shared Memory (256 Kb)     Shared Memory (256 Kb)     Shared Memory (256 Kb)

             SM                         SM                         SM


       Registers (64 Kb)          Registers (64 Kb)          Registers (64 Kb)

   Warp(Tensor + Cuda Core)   Warp(Tensor + Cuda Core)   Warp(Tensor + Cuda Core)
```

### Page 52

```text
Карта GPU (H100)

                                   HBM (80 GB)

                                             2.4 TB/sec in+out

                                 L2 Cache (50 Mb)

                                   GPU (132 SM)
               12 TB/sec
   Shared Memory (256 Kb)     Shared Memory (256 Kb)             Shared Memory (256 Kb)

             SM                         SM                                 SM
               31 TB/sec
       Registers (64 Kb)          Registers (64 Kb)                  Registers (64 Kb)

   Warp(Tensor + Cuda Core)   Warp(Tensor + Cuda Core)           Warp(Tensor + Cuda Core)
```

### Page 53

```text
Карта Хоста (H100 SXM)
                                       Switch




                                RAM             RAM                              50 GB/sec



    NIC    NIC   NIC      NIC   CPU             CPU   NIC    NIC     NIC    NIC
                                      256 GB/sec                                 64 GB/sec

                   PCIe                                     PCIe

                                                                                 64 GB/sec

     GPU     GPU          GPU    GPU         GPU      GPU          GPU     GPU

                                                                                 400 GB/sec

                                       NVLink
```

### Page 54

```text
Доступность памяти (H100)
                             Моментально
  Warp(Tensor + Cuda Core)                    Registers (64 Kb)
                              31 TB/sec
                                           Shared Memory (256 Kb)
                              12 TB/sec
                                              L2 Cache (50 Mb)
                              2.4 TB/sec
                                                HBM (80 GB)
                              400 GB/sec
                                            Host GPUs (640 GB)

                              64 GB/sec
                                              RAM (0.8 - 3 TB)

                               50 GB/sec
                                               NIC - InfiniBand

                             ← 12 GB/sec
                                                    SSD
```

### Page 55

```text
Достаточно ли вводных?


С логистикой разобрались, осталось понять, сколько времени занимают
вычисления на GPU:
• Вычисления на Tensor Cores (H100): 800 TFLOPS в bf16.
• Вычисления на CUDA Cores (H100): < 25 TFLOPS в bf16.
```

### Page 56

```text
Достаточно ли вводных?


С логистикой разобрались, осталось понять, сколько времени занимают
вычисления на GPU:
• Вычисления на Tensor Cores (H100): 800 TFLOPS в bf16.
• Вычисления на CUDA Cores (H100): < 25 TFLOPS в bf16.
• Запуск кернела 6-12 мкрсек.
```

### Page 57

```text
Достаточно ли вводных?


С логистикой разобрались, осталось понять, сколько времени занимают
вычисления на GPU:
• Вычисления на Tensor Cores (H100): 800 TFLOPS в bf16.
• Вычисления на CUDA Cores (H100): < 25 TFLOPS в bf16.
• Запуск кернела 6-12 мкрсек.

Мы построили карту логистики данных.
Пора делать что-то конкретное!
```

### Page 58

```text
Локальная
логистика на GPU
```

### Page 59

```text
Упражнение 1


Пусть скорость HBM-памяти - 2.4 GB/sec, а скорость GPU в BF16 –
800TFLOPS/sec. Сколько по времени займет такая операция в BF16
(размерности тензоров – 8192x8192):

   C=A+B
```

### Page 60

```text
Карта GPU (H100)

                                       HBM (80 GB)

                                                 2.4 TB/sec in+out

                                     L2 Cache (50 Mb)

                                       GPU (132 SM)
               12 TB/sec in+out
   Shared Memory (256 Kb)         Shared Memory (256 Kb)             Shared Memory (256 Kb)

             SM                             SM                                 SM
               31 TB/sec in+out
       Registers (64 Kb)              Registers (64 Kb)                  Registers (64 Kb)

   Warp(Tensor + Cuda Core)       Warp(Tensor + Cuda Core)           Warp(Tensor + Cuda Core)
```

### Page 61

```text
Упражнение 1


Пусть скорость HBM-памяти - 2.4 GB/sec, а скорость GPU в BF16 –
800TFLOPS/sec. Сколько по времени займет такая операция в BF16
(размерности тензоров – 8192x8192):
   C=A+B
• Время на загрузку и выгрузку:
```

### Page 62

```text
Скорость вычислений на GPU


• Вычисления на Tensor Cores (H100): 800 TFLOPS в bf16.
• Вычисления на CUDA Cores (H100): < 25 TFLOPS в bf16.
• Запуск кернела 6-12 мкрсек.
```

### Page 63

```text
Упражнение 1


Cкорость HBM-памяти - 2.4 TB/sec, а скорость GPU в BF16 – 800 TFLOPS.
Сколько по времени займет такая операция в BF16 (размерности
тензоров – 8192x8192):
   C=A+B
• Время на загрузку и выгрузку:



• Время на вычисление:
```

### Page 64

```text
Упражнение 1


Cкорость HBM-памяти - 2.4 TB/sec, а скорость GPU в BF16 – 800 TFLOPS.
Сколько по времени займет такая операция в BF16 (размерности
тензоров – 8192x8192):
   C=A+B
• Время на загрузку и выгрузку:



• Время на вычисление:



• Итоговое время выполнения: 168+12 мкрсек – операция memory bound.
```

### Page 65

```text
Упражнение 2


Cкорость HBM-памяти - 2.4 TB/sec, а скорость GPU в BF16 – 800 TFLOPS.
Сколько по времени займет такая операция в BF16 (размерности
тензоров – 1024x1024):
   C=A@B
• Время на загрузку и выгрузку:



• Время на вычисление:



• Ожидаемое время выполнения: 2.6+12 мкрсек.
```

### Page 66

```text
Упражнение 3


Cкорость HBM-памяти - 2.4 TB/sec, а скорость GPU в BF16 – 800 TFLOPS.
Сколько по времени займет такая операция в BF16 (размерности
тензоров – A - 4x8192, B – 8192x8192):
   C=A@B
• Время на загрузку и выгрузку:



• Время на вычисление:



• Ожидаемое время выполнения: 56+12 мкрсек.
```

### Page 67

```text
Упражнение 3


Cкорость HBM-памяти - 2.4 TB/sec, а скорость GPU в BF16 – 800 TFLOPS.
Сколько по времени займет такая операция в BF16 (размерности
тензоров – A - 4x8192, B – 8192x8192):
   C=A@B
• Время на загрузку и выгрузку:



• Время на вычисление:



• Ожидаемое время выполнения: 56+12 мкрсек.
                      При большом дисбалансе матричные умножения могут быть Memory Bound.
```

### Page 68

```text
Пример Attention
```

### Page 69

```text
Упражнение 4. Attention




 S = 8192, H = 4096, nh = 64.

 • Линейные слои умножения:

 • Матричные умножения в attention:

 • Scaling, softmax:
```

### Page 70

```text
Flash Attention
```

### Page 71

```text
Оптимизации
elementwise
```

### Page 72

```text
Последовательность elementwise-операций


    ResidualAdd                              RMSNorm



                  RMSNorm                     Rotary




                            Sigmoid(x)*x*y
```

### Page 73

```text
Что делать с elementwise?


●   torch.compile - собирает последовательности elementwise в triton-
    кернелы. Хорошо ужимает неэффективный код, но руками пока можно
    лучше.
●   triton - позволяет писать код, близкий к torch, который выполняет
    матричные операции на SM’ках.
●   CuTe - мощный (но непростой) инструмент для написания сложной
    логики.
●   CUDA - C на уровне SM-thread.
```

### Page 74

```text
Где можно достать хорошие кернелы?


●   Liger-Kernel
     ○ Все базовые сценарии
     ○ Эффективная реализация Cross Entropy Loss
●   Apex
     ○ Старый, но memory-efficient RMSNorm
●   TransformerEngine
     ○ Большой набор инструментов для быстрого обучения и
        квантизации
●   FlashAttention, SonicMoE
```

### Page 75

```text
Выводы


• Даже на одной GPU важно просчитывать логистику.
• Memory bound операции могут занимать существенное время,
  не загружая GPU. Особенно, если они следуют друг за другом
• Матричные умножения могут стать неэффективными на маленьких
  размерностях или при дизбалансе размеров матриц.
• Для многих операций уже реализованы хорошие кернелы — их стоит
  изучать и использовать.
```

### Page 76

```text
Коммуникации
между GPU
```

### Page 77

```text
Карта Хоста (H100 SXM)
                                       Switch




                                RAM             RAM                              50 GB/sec



    NIC    NIC   NIC      NIC   CPU             CPU   NIC    NIC     NIC    NIC
                                      256 GB/sec                                 64 GB/sec

                   PCIe                                     PCIe

                                                                                 64 GB/sec

     GPU     GPU          GPU    GPU         GPU      GPU          GPU     GPU

                                                                                 400 GB/sec

                                       NVLink
```

### Page 78

```text
Типы коммуникаций


•   Симметричные
•   Ассиметричные
```

### Page 79

```text
Типы коммуникаций


• Симметричные
• Ассиметричные



                  GPU   GPU




                  GPU   GPU
```

### Page 80

```text
Типы коммуникаций


• Симметричные
• Ассиметричные



                  GPU   GPU




                  GPU   GPU
```

### Page 81

```text
Типы коммуникаций


• Симметричные
• Ассиметричные



                  GPU   GPU




                  GPU   GPU
```

### Page 82

```text
NCCL
```

### Page 83

```text
7B модель
            Batch


         Embedding              градиенты для весов



                                                      активации (235 MB)
                          Attention (84 MB)           градиенты для весов
                                                                            Оптимизатор
              +                                                             (x18 от числа
36 x                                                                         параметров
                                                                             для AdamW)
                                                      активации (492 MB)
                           MLP (268 MB)               градиенты для весов     126 GB
              +

                                активации
       Debedding + Loss         градиенты для весов

            Loss
                                               Всего: 170 GB
```

### Page 84

```text
Бьем состояния оптимизатора на разные GPU


• … а заодно и веса.
• Собираем веса перед каждым forward.
• Усредняем градиенты после каждого backward.
```

### Page 85

```text
FSDP


• Разделим веса и состояния оптимизатора между процессами.
• Сделаем коммуникации асинхронными.
```

### Page 86

```text
Математика (Llama 7B)
```

### Page 87

```text
Время forward одного слоя 7B


• S = 8192 токенов, H = 4096, I = 11008, nh⋅dh = 4096.
• Суммарное время матричных умножений:




• Время memory bound операций (2 RMSNorm, 2 Add, 1 SwiGLU):




• Время на AllGather FSDP (в одном слое 400MB):
```

### Page 88

```text
Время forward одного слоя 7B


• S = 8192 токенов, H = 4096, I = 11008, nh⋅dh = 4096.
• Суммарное время матричных умножений:




• Время memory bound операций (2 RMSNorm, 2 Add, 1 SwiGLU):




• Время на AllGather FSDP (в одном слое 400MB):
```

### Page 89

```text
Особенности коммуникаций в NCCL
```

### Page 90

```text
Особенности коммуникаций в NCCL
```

### Page 91

```text
7B модель на 1024 GPU
            Batch


         Embedding              градиенты для весов



                                                      активации (235 MB)
                          Attention (84 MB)           градиенты для весов
                                                                            Оптимизатор
              +                                                             (x18 от числа
36 x                                                                         параметров
                                                                             для AdamW)
                                                      активации (492 MB)
                           MLP (268 MB)               градиенты для весов      0.1 GB
              +

                                активации
       Debedding + Loss         градиенты для весов

            Loss
                                               Всего: 26 GB
```

### Page 92

```text
Llama 70B на 1024 GPU
            Batch


         Embedding              градиенты для весов



                                                      активации (436 MB)
                          Attention (151 MB)          градиенты для весов
                                                                            Оптимизатор
              +                                                             (x18 от числа
80 x                                                                         параметров
                                                                             для AdamW)
                                                      активации (1.07 GB)
                           MLP (1409 MB)              градиенты для весов     1.23 GB
              +

                                активации
       Debedding + Loss         градиенты для весов

            Loss
                                                  Всего: 128 GB
                                               Активации: 120.8 GB
```

### Page 93

```text
2 варианта логистики активаций


•   Через перевычисление (чекпоинт активаций).
•   Через пересылку.
```

### Page 94

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 95

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 96

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 97

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 98

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 99

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 100

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 101

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 102

```text
Чекпоинт активаций


 134 MB

              Attention (42 MB)   активации (201 MB)

          +
 134 MB

              MLP (1207 MB)       активации (1.07 GB)

          +
```

### Page 103

```text
Частичный рекомпьют


• Не все активации одинаково дорого
 вычислять
```

### Page 104

```text
Частичный рекомпьют


• Не все активации одинаково дорого
 вычислять
```

### Page 105

```text
2 варианта логистики активаций


•   Через перевычисление (чекпоинт активаций).
•   Через пересылку (Тензорный параллелизм).
```

### Page 106

```text
Тензорный параллелизм


Как было раньше:
• Шардируем только веса, собираем их по необходимости.

Как хотим:
• Шардируем веса и активации, собираем их.
```

### Page 107

```text
Тензорный параллелизм


       batch



    Linear H -> I


       ReLU


    Linear I -> H


    MLP output
```

### Page 108

```text
Тензорный параллелизм
                      Batch part 1                       Batch part 2


       batch                              All gather



    Linear H -> I     Linear H -> I/2                     Linear H -> I/2


       ReLU               ReLU                                ReLU


    Linear I -> H     Linear I/2 -> H                     Linear I/2 -> H


    MLP output
                                        Reduce scatter


                    MLP output part 1                  MLP output part 2
```

### Page 109

```text
Время forward одного слоя 70B (TP=2)
• S = 4096 токенов, H = 8192, I = 28672, nh⋅dh = 8192, kvh = 4.

• Суммарное время матричных умножений:



• Время memory bound операций (2 RMSNorm, 2 Add, 1 SwiGLU):



• Время на AllGather FSDP (в одном слое 1560 MB)



• Время TP коммуникаций (NVLink)
```

### Page 110

```text
Время forward одного слоя 70B (TP=2)
• S = 4096 токенов, H = 8192, I = 28672, nh⋅dh = 8192, kvh = 4.

• Суммарное время матричных умножений:



• Время memory bound операций (2 RMSNorm, 2 Add, 1 SwiGLU):



• Время на AllGather FSDP (в одном слое 1560 MB)


                                                   Почему слой модели не стал меньше в 2 раза?
• Время TP коммуникаций (NVLink)
```

### Page 111

```text
Карта Хоста (H100 SXM)
                                       Switch




                                RAM             RAM                              50 GB/sec



    NIC    NIC   NIC      NIC   CPU             CPU   NIC    NIC     NIC    NIC
                                      256 GB/sec                                 64 GB/sec

                   PCIe                                     PCIe

                                                                                 64 GB/sec

     GPU     GPU          GPU    GPU         GPU      GPU          GPU     GPU

                                                                                 400 GB/sec

                                       NVLink
```

### Page 112

```text
Карта Хоста (H100 SXM)
                                       Switch




                                RAM             RAM                              50 GB/sec



    NIC    NIC   NIC      NIC   CPU             CPU   NIC    NIC     NIC    NIC
                                      256 GB/sec                                 64 GB/sec

                   PCIe                                     PCIe

                                                                                 64 GB/sec

     GPU     GPU          GPU    GPU         GPU      GPU          GPU     GPU

                                                                                 400 GB/sec

                                       NVLink
```

### Page 113

```text
Время forward одного слоя 70B (TP=2)
• S = 4096 токенов, H = 8192, I = 28672, nh⋅dh = 8192, kvh = 4.

• Суммарное время матричных умножений:



• Время memory bound операций (2 RMSNorm, 2 Add, 1 SwiGLU):



• Время на AllGather FSDP (в одном слое 1560 MB)



• Время TP коммуникаций (NVLink)


                                                     Эти коммуникации тормозят обучение?
```

### Page 114

```text
Асинхронный TP


           All gather               Linear              Activation            Linear       Reduce Scatter




                All                All
                        Linear               Linear   Activation     Linear      RS       Linear   RS
              gather             gather

    All                   All
              Linear             Linear                                          Linear   RS       Linear   RS
  gather                gather
```

### Page 115

```text
Асинхронный TP. Риски


• Матричные умножения могут оказаться memory bound – это приведет к
 замедлению
```

### Page 116

```text
Выводы
• Симметричные коммуникации: FSDP и TP позволяют производить
 обучение LLM без потери скорости.

• Рекомпьют активаций – альтернатива доставке.

• Увеличение TP внутри хоста не приводит к уменьшению FSDP
 коммуникаций.

• Это основная причина, по которой подход FSDP+TP не является
 универсальным рецептом масштабирования обучения.

• Альтернативы: Context parallelism, Expert parallelism для MoE, Pipeline
 parallelism.
```

### Page 117

```text
Кейс с MoE
```

### Page 118

```text
MoE
Современные модели учатся с MoE

                                        token emb


                                         Router



          probas      E1          E2                 E3   E4



                                       WeightedSum
```

### Page 119

```text
MoE. Как это работает
                        W1       W2        W3




                             GroupedGEMM
```

### Page 120

```text
А что поменялось?
В первую очередь, sparsity:

●   Соотношение вычислений к параметрам существенно меньше:
     ○ DeepSeek v3: 1/18
     ○ Qwen 3, 235B: 1/11
     ○ Qwen 3.5: 1/23
```

### Page 121

```text
А что поменялось?
В первую очередь, sparsity:

●   Соотношение вычислений к параметрам существенно меньше:
     ○ DeepSeek v3: 1/18
     ○ Qwen 3, 235B: 1/11
     ○ Qwen 3.5: 1/23

Как итог: большой шанс того, что операция будет memory bound, FSDP
коммуникации становятся очень дорогими.
```

### Page 122

```text
Пример: Qwen 3, 235B
Оценки по времени на H100:

●   Размер одного слоя в BF16: 5 GB
●   Оптимальное время Forward (S=8k): 5 ms
●   Оптимальное время FSDP-коммуникации: 12.5 ms
```

### Page 123

```text
Пример: Qwen 3, 235B
Оценки по времени на H100:

●   Размер одного слоя в BF16: 5 GB
●   Оптимальное время Forward (S=8k): 5 ms
●   Оптимальное время FSDP-коммуникации: 12.5 ms

Что можно делать?
 ● Растить локальный батч: 8k -> 16k
 ● Уменьшать объем FSDP-пересылок через квантизацию:
     ○ bf16 -> fp8
```

### Page 124

```text
Пример 2: DeepSeek v3, 671B
Оценки по времени на H100:

●   Размер одного слоя в BF16: 21 GB
●   Оптимальное время Forward (S=8k): 10 ms
●   Оптимальное время FSDP-коммуникации: 57 ms
```

### Page 125

```text
Пример 2: DeepSeek v3, 671B
Оценки по времени на H100:

●   Размер одного слоя в BF16: 21 GB
●   Оптимальное время Forward (S=8k): 10 ms
●   Оптимальное время FSDP-коммуникации: 57 ms

Проблемы:

●   2 слоя весов и градиентов будут занимать слишком много памяти.
```

### Page 126

```text
Пример 2: DeepSeek v3, 671B
Оценки по времени на H100:

●   Размер одного слоя в BF16: 21 GB
●   Оптимальное время Forward (S=8k): 10 ms
●   Оптимальное время FSDP-коммуникации: 57 ms

Проблемы:

●   2 слоя весов и градиентов будут занимать слишком много памяти.
●   Коммуникации FSDP огромные и не перекрываются вычислениями.
```

### Page 127

```text
Нужно шардировать модель между хостами
●   Тензорный параллелизм
●   Экспертный параллелизм
●   Пайплайн параллелизм
```

### Page 128

```text
TP vs EP

     Тензорный параллелизм                         Экспертный параллелизм

 GPU 1                                        GPU 1
         B1


 GPU 2   B2                 B1             E1 In


                AllGather        Permute                     SwiGLU
                            B2             E2 In
 GPU 3   B3                 B3             E3 In
                            B4
                                           E4 In
 GPU 4   B4
```

### Page 129

```text
TP vs EP

     Тензорный параллелизм                         Экспертный параллелизм

 GPU 1                                        GPU 1
         B1


 GPU 2   B2                 B1             E1 In


                AllGather        Permute                     SwiGLU
                            B2             E2 In
 GPU 3   B3                 B3             E3 In
                            B4
                                           E4 In
 GPU 4   B4
```

### Page 130

```text
TP vs EP

     Тензорный параллелизм                        Экспертный параллелизм

 GPU 1                                       GPU 1
         B1


 GPU 2   B2                B1             E1 In


                Dispatch        Permute                     SwiGLU
                           B2
 GPU 3   B3                B3
                           B4
 GPU 4   B4
```

### Page 131

```text
Для TP=8




  S=8192, E=256, topK=8, H=7168, I=2048

   ●   GroupedGEMM compute time:
           (S*TP)*topK*H*(I/TP)*2 / 800e12 = 2.5 ms
   ●   GroupedGEMM memory time:
           (S*TP*top_k*H + E*H*I/TP)*2 / 2.4e12 = 3.5 ms
```

### Page 132

```text
Для EP=8




  S=8192, E=256, topK=8, H=7168, I=2048

   ●   GroupedGEMM compute time:
           S*topK*H*I*2 / 800e12 = 2.5 ms
   ●   GroupedGEMM memory time:
           (S*top_k*H + E/TP*H*I)*2 / 2.4e12 = 0.78 ms
```

### Page 133

```text
Что по коммуникациям?

   ●   Стоимость полного оптимального AllGather/RS для TP=8:
           2*S*H*TP / 400e9 = 2.2 ms
   ●   Таких коммуникации нужно 2 на слой.
```

### Page 134

```text
Что по коммуникациям?

   ●   Стоимость полного оптимального AllGather/RS для TP=8:
           2*S*H*TP / 400e9 = 2.2 ms
   ●   Таких коммуникации нужно 2 на слой.
   ●   EP может быть реализован дешевле, но при небольших EP оценка
       по времени коммуникации близка к TP.
   ●   Время Forward одного слоя - 10 ms
```

### Page 135

```text
Что по коммуникациям?

   ●   Стоимость полного оптимального AllGather/RS для TP=8:
           2*S*H*TP / 400e9 = 2.2 ms
   ●   Таких коммуникации нужно 2 на слой.
   ●   EP может быть реализован дешевле, но при небольших EP оценка
       по времени коммуникации близка к TP.
   ●   Время Forward одного слоя - 10 ms - лишь в 2 раза больше
       вычислений.
```

### Page 136

```text
Что по коммуникациям?

   ●   Стоимость полного оптимального AllGather/RS для TP=8:
           2*S*H*TP / 400e9 = 2.2 ms
   ●   Таких коммуникации нужно 2 на слой.
   ●   EP может быть реализован дешевле, но при небольших EP оценка
       по времени коммуникации близка к TP.
   ●   Время Forward одного слоя - 10 ms - лишь в 2 раза больше
       вычислений. При больших TP/EP проигрыш будет больше.
   ●   Вариант. Можно пробовать запускать 2 батча в параллель:


                   Attn1   Attn2   MLP1   MLP2

                           AG1     AG2    RS1    RS22
```

### Page 137

```text
Что еще можно сделать с коммуникациями?

   ●   Pipeline Parallelism
```

### Page 138

```text
Что еще можно сделать с коммуникациями?

   ●   Pipeline Parallelism
   ●   Это ассиметричное шардирование - нужно строить сложные
       схемы.
```

### Page 139

```text
Что еще можно сделать с коммуникациями?

   ●   Pipeline Parallelism
   ●   Это ассиметричное шардирование - нужно строить сложные
       схемы.
   ●   Но коммуникации максимально дешевы и подходят для пересылки
       между хостами:
           S*H*2 / 50e9 = 2 ms
```

### Page 140

```text
Pipeline parallelism

    ●   PP = 4
    ●   FSDP коммуникаций в PP раз меньше
    ●   PP коммуникаций мало и они дешевые
    ●   Но: схема ассиметричная, есть пузыри
```

### Page 141

```text
1F1B
```

### Page 142

```text
ZeroBubble
```

### Page 143

```text
EP+PP: DualPipeV
```

### Page 144

```text
Выводы

   ●   C MoE работать сложнее: FSDP коммуникации становятся очень
       дорогими, а слои могут не помещаться на GPU после сборки.
   ●   Иногда эту проблему можно решить увеличением плотности
       вычислений и ускорением коммуникаций.
   ●   В общем случае, модель нужно шардировать по различным
       хостам для оптимизации FSDP.
   ●   PP - наиболее эффективный и сложный способ сделать это. Часто
       для этого используют EP.
```

### Page 145

```text
Заключение
```

### Page 146

```text
Заключение


• Эффективное обучение LLM связано в первую очередь с правильно
 выстроенной логистикой данных.

• Можно пересылать веса и их градиенты через FSDP, в том числе
 на CPU RAM.

• Можно пересылать активации и их градиенты через TP и другие
 параллелизмы.

• Можно делать их рекомпьют.
```

### Page 147

```text
Михаил Хрущев
```
