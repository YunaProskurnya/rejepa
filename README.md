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

Маски берутся из текущего `src/masks.py`. Аргумент `--seed` задаёт
случайный crop, маски и инициализацию модели. Для разных изображений при обучении
нужны разные seed: `process_image` по умолчанию использует `seed=0`.

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

1. **Размеры масок в `src/masks.py`.** Длина стороны теперь вычисляется из доли площади, но ограничивается `grid - 1`. При сетке 14×14 и `aspect_ratio=(1, 1)` контекстный crop всегда не больше 13×13 (86,2% площади), хотя диапазон `scale=(0.85, 1.0)` допускает всю сетку. Граница `torch.randint(0, grid - block_h)` также исключает последнее допустимое положение блока. Если `min_keep` не удаётся выполнить, генератор постепенно перестаёт исключать целевые блоки из контекста.

2. **VISReg.** Перед экспериментами сверить формулы и масштабы трёх слагаемых с [детальным описанием в README регуляризаторов](src/regularizers/README.md#формулы-visreg): особенно усреднение `center_loss` и `shape_loss`, а также `eps=1e-6` при нормализации.
