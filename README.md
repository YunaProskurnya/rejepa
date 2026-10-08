# rejepa

Воспроизведение I-JEPA.

## Мини-эксперимент на одном изображении

Нужны совместимые версии `torch` и `torchvision`, а также `Pillow`.
Запуск из корня репозитория в окружении с этими зависимостями:

```sh
python example/mini_experiment.py
python example/mini_experiment.py --image path/to/image.jpg --steps 30 --device cuda --seed 42
```

По умолчанию используются `example/lenna.png`, CPU и 10 шагов.
Скрипт один раз вызывает `process_image`, добавляет ось батча (`B=1`)
и повторно использует полученные изображение и маски на всех шагах.
Отдельный collator для этого не нужен; целевые блоки могут иметь разную длину.

Каждый шаг: target encoder → нормализованные цели; context encoder → predictor
для каждого целевого блока → средний по блокам Smooth L1 loss → backward →
AdamW для encoder и predictor → EMA для target encoder.
Bias и одномерные параметры исключены из weight decay.
Скрипт выводит формы входов, loss и норму градиентов; при нечисловых или
бесконечных loss/градиентах останавливается. Градиенты не обрезаются.

Это проверка совместной работы компонентов на компактной модели:
encoder — 2 блока, размерность 48; predictor — 2 блока, размерность 24;
по 3 головы, изображение 224×224, патч 16×16. Learning rate равен `1e-3`,
weight decay — `0.04`, EMA momentum — `0.996`; расписаний пока нет.
Снижение loss на одном изображении не подтверждает качество представлений
или воспроизведение результатов статьи. Из-за обновления целей по EMA loss
не обязан убывать на каждом шаге.

Маски берутся из текущего `src/masks.py`. В этой версии ещё ожидаются
исправления трактовки `scale` как доли площади и повторного создания генератора
с фиксированным seed. Аргумент `--seed` задаёт seed PyTorch для инициализации
и случайного crop; внутренний seed генераторов масок сейчас отдельный.

## Download dataset imagenet100
```sh
kaggle datasets download ambityga/imagenet100
unzip imagenet100.zip
```

## Регуляризация эмбеддингов

[Описание и использование модулей](src/regularizers/README.md).

## Линейная оценка

`evaluate_linear_probe` использует target encoder без обновления весов, усредняет его патч-токены и обучает один `nn.Linear`. Возвращает точность top-1 на валидации.

Когда общий загрузчик будет готов, передать два загрузчика батчей `(images, labels)` с одинаковой нумерацией классов. Для оценки нужны полные изображения без масок:

```python
from src.eval_linear import evaluate_linear_probe

train_loader = make_labeled_loader(split="train", deterministic=True)
val_loader = make_labeled_loader(split="val", deterministic=True)

top1 = evaluate_linear_probe(
    target_encoder, train_loader, val_loader,
    num_classes=100, device=device,
)
```

## Обратить внимание

1. **Размеры масок в `src/masks.py`.** Сейчас `scale` применяется к длине стороны, хотя должен задавать долю площади. В строках 58–59 для целевого блока написано:

   ```python
   h = min(grid, max(1, round(grid * scale_crop)))
   w = min(grid, max(1, round(h * aspect_crop)))
   ```

   В строках 135–136 для контекстного блока написано:

   ```python
   crop_h = min(grid, max(1, round(grid * scale_crop)))
   crop_w = min(grid, max(1, round(grid * scale_crop * aspect_ratio_crop)))
   ```

   Пусть `A = scale_crop * grid * grid` — площадь блока в патчах, а `r = h / w` — отношение высоты к ширине. Тогда `h ≈ √(A * r)` и `w ≈ √(A / r)`. Для целевого блока строки должны выглядеть так:

   ```python
   area = scale_crop * grid * grid
   h = max(1, round((area * aspect_crop) ** 0.5))
   w = max(1, round((area / aspect_crop) ** 0.5))
   if h > grid or w > grid:
       continue
   ```

   Для контекстного блока:

   ```python
   area = scale_crop * grid * grid
   crop_h = max(1, round((area * aspect_ratio_crop) ** 0.5))
   crop_w = max(1, round((area / aspect_ratio_crop) ** 0.5))
   ```

   Для контекстного блока с текущим `aspect_ratio=(1, 1)` размеры помещаются в сетку. Если диапазон отношения сторон изменится, перед выбором позиции также потребуется повторять выборку при `crop_h > grid` или `crop_w > grid`. Ограничение стороны через `min` меняет заданную площадь. На сетке 14×14 при `scale_crop=0.2`, `r=1` текущий код даёт блок 3×3 (4,6% площади), а исправленный — 6×6 (18,4%).

2. **VISReg.** Перед экспериментами сверить формулы и масштабы трёх слагаемых с [детальным описанием в README регуляризаторов](src/regularizers/README.md#формулы-visreg): особенно усреднение `center_loss` и `shape_loss`, а также `eps=1e-6` при нормализации.
