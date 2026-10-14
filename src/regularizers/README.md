# Регуляризаторы

`SIGReg` и `VISReg` принимают эмбеддинги разных изображений формы `[B, D]`, где `B >= 2`, и возвращают скаляр. Для полного выхода энкодера `features` формы `[B, P, D]` функция `apply_regularization` подаёт каждому регуляризатору `features[:, p, :]` и усредняет результаты по `P` позициям патчей. Эмбеддинги патчей перед этим не усредняются.

```python
from src.losses import prediction_loss
from src.regularizers.apply_regularization import apply_regularization

prediction = prediction_loss(predictions, targets)
regularization = apply_regularization(features, args.regularizer)
loss = prediction + args.regularizer_weight * regularization
```

Второй аргумент — `"none"`, `"sigreg"` или `"visreg"`. Для `"none"` функция возвращает `0` и допускает `features=None`. Для `"sigreg"` и `"visreg"` `features` вычисляется обучаемым энкодером по полным изображениям в режиме `joint`. Используется полный выход до дополнительной нормализации целей; маска контекста на этот проход не влияет. В режиме `ema` target encoder заморожен, поэтому регуляризация его эмбеддингов не обучит энкодер.

## SIGReg

Адаптирован из [минимального примера LeJEPA](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md) Randall Balestriero и Yann LeCun; исходник распространяется по [CC BY-NC 4.0](https://github.com/galilai-group/lejepa/blob/main/LICENSE), без гарантий. Сохраняются 256 случайных единичных направлений, 17 узлов на `[0, 3]`, симметричные трапециевидные веса с гауссовым окном и множитель `B`. Узлы, характеристическая функция стандартной нормали и веса зарегистрированы как буферы модуля.

Адаптация сохраняет интерфейс `[B, D]`, проверку входов, выбор устройства по входному тензору и независимые направления для каждой позиции патча. В отличие от примера авторов, весь расчёт SIGReg выполняется вне autocast: FP16/BF16-входы переводятся в FP32, FP64 сохраняется. Проверяются совпадение loss и градиентов с исходной формулой в FP32 и отсутствие влияния BF16 autocast. Это не доказывает устранение редкого сбоя backward в полном обучении.

## Формулы VISReg

В текущей реализации `VISReg(num_slices=64, scale_weight=1, shape_weight=1, center_weight=1, eps=1e-6)` получает `z` формы `[B, D]`. Ниже `mean` означает усреднение по всем элементам аргумента, `std` считается по батчу с `unbiased=False`, а `q_i` — квантиль стандартного нормального распределения при вероятности `i / (B + 1)`.

```python
center = z.mean(dim=0)
centered = z - center
std = centered.std(dim=0, unbiased=False)
scale_loss = (1 - std).square().mean()
center_loss = center.square().mean()
normalized = centered / (std.detach() + 1e-6)
shape_loss = (projected - quantiles[:, None]).square().mean()
loss = scale_weight * scale_loss + shape_weight * shape_loss + center_weight * center_loss
```

Здесь `projected` — нормализованные эмбеддинги после проекции на 64 случайных единичных направления и сортировки по батчу; `quantiles` содержит `q_i`. Направления выбираются заново на каждом вызове. `std.detach()` убирает градиент через знаменатель только в `shape_loss`.

В [статье VISReg, формула 6](https://arxiv.org/html/2606.02572) центр задан как квадрат L2-нормы среднего эмбеддинга: `L_center = ||mean(z, dim=0)||_2²`. На [странице авторов в псевдокоде](https://haiyuwu.github.io/visreg/) используется `.mean()`; код следует этому варианту.

При размерности `D` вариант с `.sum()` в `D` раз сильнее варианта с `.mean()` при одинаковом `center_weight`.

Для `shape_loss` код, как и псевдокод авторов, усредняет квадраты по изображениям и проекциям. Это в `B` раз меньше суммы по изображениям, усреднённой по проекциям. Определение одномерного расстояния Вассерштейна в лемме 3.2 статьи также включает деление на размер батча.

Статья добавляет `eps` только в знаменатель нормализованных эмбеддингов: `centered / (std.detach() + eps)`. Псевдокод опускает `eps`. В этом проекте выбран вариант формулы статьи; `scale_loss` использует стандартное отклонение без добавки `eps`.
