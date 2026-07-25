---
title: "Sem03 Gd"
type: external-resource
status: imported-source
source_kind: jupyter-notebook
source_commit: 4b21051531fb72dc9eef58632332ad971c92d006
language: ru
---

> [!note] Полный оригинальный материал HSE
> Источник: [`ml1-2026-spring/seminars/sem03-gd.ipynb`](https://github.com/esokolov/ml-course-hse/blob/4b21051531fb72dc9eef58632332ad971c92d006/ml1-2026-spring/seminars/sem03-gd.ipynb), commit `4b21051531fb72dc9eef58632332ad971c92d006`.
> В репозитории не найдено общей лицензии; материал перенесён без перевода
> по прямому разрешению владельца Bookvar для некоммерческого учебного архива.
> Ссылка на оригинал и provenance сохранены.


# Sem03 Gd

# Машинное обучение 1, ПМИ ФКН ВШЭ

## Семинар 3 (градиентный спуск)

На этом семинаре мы обсудим градиентный спуск, его модификации и подбор параметров в них.

В ноутбуке используется библиотека plotly, которая позволяет создавать интерактивные диаграммы. Эти диаграммы не отображаются в nbviewer или на github, поэтому для удобного просмотра лучше скачать ноутбук, открыть локально и перезапустить его.

## Градиентный спуск

Напомним, что в градиентном спуске значения параметров на следующем шаге получаются из значений параметров на предыдущем шаге смещением в сторону антиградиента функционала:

$$w^{(t)} = w^{(t-1)} - \eta_t \nabla Q(w^{(t-1)}),$$
где $\eta_t$ — длина шага (learning rate) градиентного спуска.

### Асимптотическая сложность

Оптимальный набор весов для линейной регрессии с точки зрения MSE выглядит как $w = (X^TX)^{-1}X^Ty$. В этой формуле присутствует обращение матрицы $X^TX$ — очень трудоёмкая операция при большом количестве признаков. Нетрудно подсчитать, что сложность вычислений $O(d^3 + d^2 \ell)$. При решении задач такая трудоёмкость часто оказывается непозволительной, поэтому параметры ищут итерационными методами, стоимость которых меньше. Один из них — градиентный спуск.

Градиент MSE записывается как

$$\nabla Q(w) = 2X^T(Xw - y).$$

Сложность вычислений в данном случае $O(d \ell)$. Стохастический градиентный спуск отличается от обычного заменой градиента на несмещённую оценку по одному или нескольким объектам. В этом случае сложность становится $O(kd)$, где $k$ — количество объектов, по которым оценивается градиент, $k \ll \ell$. Это отчасти объясняет популярность стохастических методов оптимизации.

### Визуализация траекторий GD и SGD
На простом примере разберём основные тонкости, связанные со стохастической оптимизацией.

Сгенерируем матрицу объекты—признаки $X$ и вектор весов $w_{true}$, вектор целевых переменных $y$ вычислим как $Xw_{true}$ и добавим нормальный шум:

```python
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```

```python
n_features = 2
n_objects = 300

np.random.seed(100)
w_true = np.random.normal(size=(n_features, ))

X = np.random.uniform(-5, 5, (n_objects, n_features))
X *= (np.arange(n_features) * 2 + 1)[np.newaxis, :]  # for different scales

Y = X.dot(w_true) + np.random.normal(0, 1, (n_objects))

w_0 = np.random.uniform(-1, 1, (n_features))
```

Зададим параметры градиентного спуска:

```python
batch_size = 10
num_steps = 50
```

Обучим на полученных данных линейную регрессию для MSE при помощи полного градиентного спуска — тем самым получим вектор параметров.

```python
step_size = 1e-2

w = w_0.copy()
w_list = [w.copy()]

for i in range(num_steps):
    w -= 2 * step_size * np.dot(X.T, np.dot(X, w) - Y) / Y.shape[0]
    w_list.append(w.copy())

w_list = np.array(w_list)
```

```python
import plotly.graph_objects as go


def compute_limits(w_list):
    dx = np.max(np.abs(w_list[:, 0] - w_true[0])) * 1.1
    dy = np.max(np.abs(w_list[:, 1] - w_true[1])) * 1.1

    return (w_true[0] - dx, w_true[0] + dx), (w_true[1] - dy, w_true[1] + dy)


def compute_levels(w_list, x_range, y_range, num: int = 100):
    x, y = np.linspace(x_range[0], x_range[1], num), np.linspace(y_range[0], y_range[1], num)
    A, B = np.meshgrid(x, y)

    levels = np.empty_like(A)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            w_tmp = np.array([A[i, j], B[i, j]])
            levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - Y, 2))

    return x, y, levels


def make_contour(x, y, levels, name: str=None):
    return go.Contour(
        x=x,
        y=y,
        z=levels,
        contours_coloring='lines',
        line_smoothing=1,
        line_width=2,
        ncontours=100,
        opacity=0.5,
        name=name
    )


def make_arrow(figure, x, y):
    x, dx = x
    y, dy = y

    figure.add_annotation(
        x=x,
        y=y,
        ax=x + dx,
        ay=y + dy,
        xref='x',
        yref='y',
        text='',
        showarrow=True,
        axref = 'x',
        ayref='y',
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
    )


def plot_trajectory(w_list, name):
    # compute limits
    x_range, y_range = compute_limits(w_list)

    # compute level set
    x, y, levels = compute_levels(w_list, x_range, y_range)

    # plot levels
    contour = make_contour(x, y, levels, 'Loss function levels')

    # plot weights
    w_path = go.Scatter(
        x=w_list[:, 0][:-1],
        y=w_list[:, 1][:-1],
        mode='lines+markers',
        name='W',
        marker=dict(size=7, color='red')
    )

    # plot final weight
    w_final = go.Scatter(
        x=[w_list[:, 0][-1]],
        y=[w_list[:, 1][-1]],
        mode='markers',
        name='W_final',
        marker=dict(size=10, color='black'),
    )

    # plot true optimum
    w_true_point = go.Scatter(
        x=[w_true[0]],
        y=[w_true[1]],
        mode='markers',
        name='W_true',
        marker=dict(size=10, color='black'),
        marker_symbol='star'
    )

    # make the figure
    fig = go.Figure(data=[contour, w_path, w_final, w_true_point])

    fig.update_xaxes(type='linear', range=x_range)
    fig.update_yaxes(type='linear', range=y_range)

    fig.update_layout(title=name)

    fig.update_layout(
        autosize=True,
        width=700,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=100,
            pad=4
        ),
        paper_bgcolor='LightSteelBlue',
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    fig.update_traces(showlegend=True)

    fig.show()
```

```python
plot_trajectory(w_list, 'Gradient descent')
```

На лекции обсуждалось, что градиент перпендикулярен линиям уровня. Это объясняет такие зигзагообразные траектории градиентного спуска. Для большей наглядности в каждой точке пространства посчитаем градиент функционала и покажем его направление.

```python
# make new figure with contour lines
x_range, y_range = compute_limits(w_list)
x, y, levels = compute_levels(w_list, x_range, y_range)
contour = make_contour(x, y, levels, 'Loss function levels')
fig = go.Figure(data=[contour])

# visualize the gradients

x_smol, y_smol, _ = compute_levels(w_list, x_range, y_range, num=10)
x_smol, y_smol = x_smol[1:-1], y_smol[1:-1]
A_smol, B_smol = np.meshgrid(x_smol, y_smol)

for i in range(A_smol.shape[0]):
    for j in range(A_smol.shape[1]):
        w_tmp = np.array([A_smol[i, j], B_smol[i, j]])

        antigrad = 0.003 * np.dot(X.T, np.dot(X, w_tmp) - Y) / Y.shape[0]

        make_arrow(fig, (A_smol[i, j], antigrad[0]), (B_smol[i, j], antigrad[1]))


fig.update_xaxes(type='linear', range=x_range)
fig.update_yaxes(type='linear', range=y_range)

fig.update_layout(title = 'Antigradient')

fig.update_layout(
    autosize=True,
    width=700,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=100,
        pad=4
    ),
    paper_bgcolor='LightSteelBlue',
)

fig.show()
```

Визуализируем теперь траектории стохастического градиентного спуска, повторив те же самые действия, оценивая при этом градиент по подвыборке.

```python
def calc_grad_on_batch(X, Y, w, batch_size):
    sample = np.random.randint(n_objects, size=batch_size)
    return 2 * np.dot(X[sample].T, np.dot(X[sample], w) - Y[sample]) / batch_size
```

```python
step_size = 1e-2

w = w_0.copy()
w_list = [w.copy()]

for i in range(num_steps):
    w -= step_size * calc_grad_on_batch(X, Y, w, batch_size)
    w_list.append(w.copy())

w_list = np.array(w_list)
```

```python
plot_trajectory(w_list, 'Stochastic gradient descent')
```

Как видно, метод стохастического градиента «бродит» вокруг оптимума. Это объясняется подбором шага градиентного спуска $\eta_k$. Дело в том, что для сходимости стохастического градиентного спуска для последовательности шагов $\eta_k$ должны выполняться [условия Роббинса-Монро](https://projecteuclid.org/download/pdf_1/euclid.aoms/1177729586):
$$
\sum_{k = 1}^\infty \eta_k = \infty, \qquad \sum_{k = 1}^\infty \eta_k^2 < \infty.
$$
Интуитивно это означает следующее: (1) последовательность должна расходиться, чтобы метод оптимизации мог добраться до любой точки пространства, (2) но при этом расходиться не слишком быстро.

Попробуем посмотреть на траектории SGD, последовательность шагов которой удовлетворяет условиям Роббинса-Монро:

```python
step_size_0 = 0.01

w = w_0.copy()
w_list = [w.copy()]


for i in range(num_steps):
    step_size = step_size_0 / (i+1)
    w -= step_size * calc_grad_on_batch(X, Y, w, batch_size)
    w_list.append(w.copy())

w_list = np.array(w_list)
```

```python
plot_trajectory(w_list, 'Stochastic gradient descent with dynamic learning rate')
```

Теперь градиентный спуск движется более направленно, но не доходит до оптимума. Попробуем более сложную схему изменения длины шага:
$$
    \eta_t
    =
    \lambda
    \left(
        \frac{s_0}{s_0 + t}
    \right)^p.
$$
Попробуем взять $s_0 = 1$и поэкспериментируем с разными $\lambda$ и $p$.

```python
def sgd_with_lr_schedule(lambda_param, p=0.5, s_init=1.0, batch_size=10):
    w = w_0.copy()
    w_list = [w.copy()]


    for i in range(num_steps):
        step_size = lambda_param * np.power(s_init / (s_init + i), p)
        w -= step_size * calc_grad_on_batch(X, Y, w, batch_size)
        w_list.append(w.copy())

    return np.array(w_list)
```

```python
w_list = sgd_with_lr_schedule(lambda_param=0.01, p=0.8)
plot_trajectory(w_list, 'SGD with learning rate schedule')
```

```python
w_list = sgd_with_lr_schedule(lambda_param=0.01, p=0.5)
plot_trajectory(w_list, 'SGD with learning rate schedule')
```

```python
w_list = sgd_with_lr_schedule(lambda_param=0.01, p=0.35)
plot_trajectory(w_list, 'SGD with learning rate schedule')
```

По сути, коэффициенты в формуле для длины шага являются гиперпараметрами, и их нужно подбирать. Желательно использовать для этого валидационную выборку.

Посмотрим, как размер подвыборки, по которой оценивается градиент, влияет на сходимость.

```python
w_list = sgd_with_lr_schedule(lambda_param=0.01, p=0.35, batch_size=1)
plot_trajectory(w_list, 'SGD with learning rate schedule')
```

```python
w_list = sgd_with_lr_schedule(lambda_param=0.01, p=0.35, batch_size=10)
plot_trajectory(w_list, 'SGD with learning rate schedule')
```

```python
w_list = sgd_with_lr_schedule(lambda_param=0.01, p=0.35, batch_size=100)
plot_trajectory(w_list, 'SGD with learning rate schedule')
```

Вывод, в общем-то, очевидный: чем больше размер подвыборки, тем более стабильная траектория градиентного спуска. Интереснее посмотреть, как это влияет на скорость сходимости.

### Сравнение скоростей сходимости

Изучим, насколько быстро достигают оптимума методы полного и стохастического градиентного спуска. Сгенерируем выборку и построим график зависимости функционала от итерации.

```python
num_steps = 100
batch_size = 10
```

```python
# data generation
n_features = 50
n_objects = 10000

w_true = np.random.uniform(-2, 2, n_features)

X = np.random.uniform(-10, 10, (n_objects, n_features))
Y = X.dot(w_true) + np.random.normal(0, 5, n_objects)
```

```python
from scipy.linalg import norm

step_size_sgd = 1e-2
step_size_gd = 1e-2

w_sgd = np.random.uniform(-1, 1, n_features)
w_gd = w_sgd.copy()

residuals_sgd = [np.mean(np.power(np.dot(X, w_sgd) - Y, 2))]
residuals_gd = [np.mean(np.power(np.dot(X, w_gd) - Y, 2))]

norm_sgd = []
norm_gd = []


for i in range(num_steps):
    step_size = step_size_sgd / ((i+1) ** 0.4)
    sample = np.random.randint(n_objects, size=batch_size)

    w_sgd -= step_size * calc_grad_on_batch(X, Y, w_sgd, batch_size)
    residuals_sgd.append(np.mean(np.power(np.dot(X, w_sgd) - Y, 2)))
    norm_sgd.append(norm(np.dot(X[sample].T, np.dot(X[sample], w_sgd) - Y[sample])))

    w_gd -= 2 * step_size_gd * np.dot(X.T, np.dot(X, w_gd) - Y) / Y.shape[0]
    residuals_gd.append(np.mean(np.power(np.dot(X, w_gd) - Y, 2)))
    norm_gd.append(norm(np.dot(X.T, np.dot(X, w_gd) - Y)))
```

```python
full_gd = go.Scatter(x=np.arange(num_steps+1), y=residuals_gd, name='Full GD')
sgd = go.Scatter(x=np.arange(num_steps+1), y=residuals_sgd, name='SGD')

fig = go.Figure(data=[full_gd, sgd])

fig.update_xaxes(type='linear', range=[-1, num_steps + 1])
fig.update_yaxes(type='linear')

fig.update_layout(title = 'Residuals comparison', xaxis=dict(title="Iteration"))

fig.update_layout(
    autosize=True,
    width=700,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=100,
        pad=4
    ),
    paper_bgcolor='LightSteelBlue',
)

fig.show()
```

```python
full_gd = go.Scatter(x=np.arange(num_steps+1), y=norm_gd, name='Full GD')
sgd = go.Scatter(x=np.arange(num_steps+1), y=norm_sgd, name='SGD')

fig = go.Figure(data=[full_gd, sgd])

fig.update_xaxes(type='linear', range=[-1, num_steps + 1])
fig.update_yaxes(type='linear')

fig.update_layout(title = 'Gradient norm comparison', xaxis=dict(title="Iteration"))

fig.update_layout(
    autosize=True,
    width=700,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=100,
        pad=4
    ),
    paper_bgcolor='LightSteelBlue',
)

fig.show()
```

Как видно, GD буквально за несколько итераций оказывается вблизи оптимума, в то время как поведение SGD может быть весьма нестабильным. Как правило, для более сложных моделей наблюдаются ещё большие флуктуации в зависимости качества функционала от итерации при использовании стохастических градиентных методов. Путём подбора величины шага можно добиться лучшей скорости сходимости, и существуют методы, адаптивно подбирающие величину шага (AdaGrad, Adam, RMSProp).

Ещё интересно посмотреть, как сильно использование mini-batch GD ускоряет сходимость. Посчитаем, за сколько шагов стохастический градиентный спуск достаточно сильно сближается с истинным решением в зависимости от размера батча.

```python
step_size_sgd = 1e-2
step_size_gd = 1e-2
num_steps = 500

w_init = np.random.uniform(-1, 1, n_features)
w_gd = w_init.copy()

for i in range(num_steps):
    w_gd -= 2 * step_size_gd * np.dot(X.T, np.dot(X, w_gd) - Y) / Y.shape[0]

best_error = np.mean(np.power(np.dot(X, w_gd) - Y, 2))
steps_before_conv = []
batch_sizes = np.arange(0, 500, 10)

for batch_size in batch_sizes:
    w_sgd = w_init.copy()
    for i in range(num_steps):
        step_size = step_size_sgd / ((i+1) ** 0.4)
        sample = np.random.randint(n_objects, size=batch_size)

        w_sgd -= step_size * calc_grad_on_batch(X, Y, w_sgd, batch_size)
        err = np.mean(np.power(np.dot(X, w_sgd) - Y, 2))
        if np.abs(err - best_error) < 1:
            break

    steps_before_conv.append(i)
```

```python
conv_speed = go.Scatter(x=batch_sizes, y=steps_before_conv, name='Number of steps to convergence')

fig = go.Figure(data=conv_speed)

fig.update_layout(title='Convergence speed',
                 xaxis=dict(title="batch size"),
                yaxis=dict(title="steps before convergence")
)

fig.update_layout(
    autosize=True,
    width=700,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=100,
        pad=4
    ),
    paper_bgcolor='LightSteelBlue',
)

fig.show()
```

Видно, что конкретно на этом наборе данных увеличение размера батча примерно до 100 позволяет добиться существенного ускорения сходимости. В то же время увеличение батча в 100 раз приводит и к пропорциональному замедлению каждого шага градиентного спуска. Поэтому, как правило, имеет смысл оценивать градиент по небольшой подвыборке.

### Качество оценки градиента

Интересно посмотреть, как соотносятся стохастический и полный градиенты. Для этого на каждой итерации градиентного спуска будет вычислять полный градиент и смотреть, какой косинус угла в среднем оказывается между ним и стохастической оценкой. Чтобы было с чем сравнить, посчитаем также средний косинус получается между полным градиентом и случайным вектором.

```python
from scipy.spatial import distance

step_size_sgd = 1e-2
step_size_gd = 1e-2
batch_size = 1
num_steps = 200

w_init = np.random.uniform(-1, 1, n_features)
w_sgd = w_init.copy()

mean_cosine_between_grads = []
mean_cosine_between_rand = []

for t in range(num_steps):
    full_grad_at_w = 2 * np.dot(X.T, np.dot(X, w_sgd) - Y) / Y.shape[0]

    cosine_between_grads = []
    cosine_between_rand = []
    for _ in range(1000):
        stoch_grad = calc_grad_on_batch(X, Y, w_sgd, batch_size)
        random_vector = np.random.normal(0, 1, full_grad.shape)
        cosine_between_grads.append(distance.cosine(stoch_grad, full_grad_at_w))
        cosine_between_rand.append(distance.cosine(random_vector, full_grad_at_w))
    mean_cosine_between_grads.append(np.mean(cosine_between_grads))
    mean_cosine_between_rand.append(np.mean(cosine_between_rand))

    step_size = step_size_sgd / ((t+1) ** 0.4)
    w_sgd -= step_size * calc_grad_on_batch(X, Y, w_sgd, batch_size)
```

```python
cos_grad = go.Scatter(x=np.arange(num_steps+1), y=mean_cosine_between_grads,
                      name='Cosine similarity: stochastic vs full gradient')
cos_rand = go.Scatter(x=np.arange(num_steps+1), y=mean_cosine_between_rand,
                      name='Cosine similarity: random vs full gradient')

fig = go.Figure(data=[cos_grad, cos_rand])

fig.update_layout(title='Stochastic gradient estimate',
                 xaxis=dict(title="Iteration")
)

fig.update_layout(
    autosize=True,
    width=700,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=100,
        pad=4
    ),
    paper_bgcolor='LightSteelBlue',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()
```

Видно, что в самом начале стохастическая оценка чуть ближе к случайной, но в целом оценка по одному объекту стабильно и значимо лучше, чем движение в случайном направлении.
