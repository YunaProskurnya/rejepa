# rejepa

Воспроизведение I-JEPA на PyTorch. Обучение на одном CPU или GPU, ImageNet100,
мониторинг представлений и отдельная итоговая оценка.

[Данные](#imagenet100) · [Обучение](#обучение) · [Comet](#логирование-в-comet) ·
[Оценка](#оценка-представлений) · [Продолжение запуска](#результаты-и-продолжение) · [Код](#структура-проекта)

## Установка

Все команды выполняются **из корня репозитория** в активированном Python-окружении.
Многострочные примеры используют синтаксис Bash.
Для GPU нужны совместимые `torch` и `torchvision` с CUDA — используйте окружение
кластера или [инструкцию PyTorch](https://pytorch.org/get-started/locally/).

```sh
python -m pip install -r requirements.txt
python scripts/train.py --config configs/smoke.json
```

Проверка `smoke` использует 8 синтетических изображений и маленькую модель на CPU;
данные не скачиваются, Comet выключен.

## ImageNet100

```sh
bash scripts/setup_imagenet100.sh
```

Скрипт скачивает датасет и раскладывает его в папки, уже указанные в конфигах:

```text
data/imagenet100/
├── train/<класс>/<изображение>
└── val/<класс>/<изображение>
```

В train и val должны совпадать названия классов. Если данные уже есть на сервере,
укажите оба пути в `data` / `val_data` конфига. Датасеты исключены из Git.
Подробнее — в [инструкции по данным](data/README.md).

## Обучение

GPU-конфиги используют **ViT-B/16, BF16, наш collator**, лёгкий probe после каждой
эпохи и Comet. Для Comet задайте `COMET_API_KEY` в окружении задачи кластера,
для запуска без него добавьте `--no-comet`.

| Запуск | Конфиг | Эпохи / warmup | Батч × накопление |
| --- | --- | --- | --- |
| Проверка обучения | [imagenet100_vit_base.json](configs/imagenet100_vit_base.json) | 10 / 1 | 128 × 1 |
| Длительный, батч 128 | [imagenet100_vit_base_600ep.json](configs/imagenet100_vit_base_600ep.json) | 600 / 1 | 128 × 1 |
| Полный запуск, ≈375 тыс. обновлений | [imagenet100_vit_base_370ep_batch128.json](configs/imagenet100_vit_base_370ep_batch128.json) | 370 / 9 | 128 × 1 |

**Сначала — 10 полных эпох:**

```sh
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name()); print('BF16:', torch.cuda.is_bf16_supported())"

python scripts/train.py --config configs/imagenet100_vit_base.json
```

**Для длительного запуска:**

```sh
python scripts/train.py --config configs/imagenet100_vit_base_370ep_batch128.json
```

Результаты сохраняются в `runs/imagenet100-vit-base-10ep/` и
`runs/imagenet100-vit-base-370ep-batch128/` соответственно. Ограничения числа шагов нет.
Итоговая оценка запускается отдельно.

При 130 000 train-изображений полный запуск даёт `370 × floor(130000 / 128) = 375550`
обновлений, warmup — `9 × 1015 = 9135` шагов. Это близко к 375 000 обновлений
и 9375 шагам warmup авторского ViT-B/16 (ImageNet-1K, batch 2048, 600 эпох).
LR во всех трёх конфигах — `6.25e-6 → 6.25e-5 → 6.25e-8`:
значения из [статьи, A.1](https://arxiv.org/html/2301.08243v3#A1.SS1), умноженные на `128 / 2048`.
Weight decay меняется по косинусу, как в авторском коде. Наш датасет — ImageNet100,
в статье — ImageNet-1K.

### Как изменить запуск

Скопируйте JSON из `configs/`, измените настройки и задайте новый `output`.
Параметры командной строки переопределяют JSON, относительные пути считаются от корня проекта при запуске команд отсюда. Например, при нехватке памяти можно уменьшить batch_size:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json \
  --batch-size 32 --accumulation-steps 4 \
  --output runs/imagenet100-small-batch
```

Доступны модели `tiny`, `vit_base`, `vit_large`, режимы `fp32` / `bf16` и collator
`custom` / `reference`. Полный список параметров: `python scripts/train.py --help`.
`max_steps` ограничивает шаги оптимизатора, длительность расписаний задаётся `epochs`.

## Логирование в Comet

[Открыть графики команды](https://www.comet.com/superustam/rejepa/view/new/panels)
— workspace `superustam`, project `rejepa`.

Ключ берётся только из `COMET_API_KEY`: получите его в
[настройках Comet](https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/) и задайте через секреты задачи кластера. В конфиги и checkpoint ключ не записывается.
Проект можно изменить через `--comet-workspace` / `--comet-project`.

| Что наблюдать | Метрики | Ось графика |
| --- | --- | --- |
| Обучение | `train/loss`, `train/grad_norm`, `train/lr`, скорость и память | Step |
| Качество представлений | `linear_probe/train_top1`, `linear_probe/val_top1`, соответствующие `train_loss` / `val_loss` | Epoch |
| Время | `linear_probe/probe_seconds`, `runtime/session_seconds` | Epoch |

Top-1 измеряется в процентах. Конфиг и CSV также отправляются как файлы результатов,
локальные результаты сохраняются и с `--no-comet`. При переносе запуска сохраните
`comet.json`: он связывает `--resume` с прежним облачным экспериментом.
Метрики ковариации будут подключены отдельно, после их реализации.

При штатном завершении с `comet: true` итоговые веса отправляются в **Comet → Models →
rejepa**, файл `final_weights.pt`: encoder, predictor, EMA target encoder, конфиг и шаг.
При остановке по лимиту шагов/времени отправляются достигнутые веса с отметкой о
незавершённом обучении.

## Оценка представлений

В обоих режимах encoder заморожен: используется EMA target, а при его отсутствии — context.
Линейная голова обучается на train. train и val оцениваются отдельно.

| | Мониторинг по эпохам | Итоговый linear probe |
| --- | --- | --- |
| Когда | Автоматически после полной эпохи I-JEPA | Отдельной командой после предобучения |
| Признаки | Средние patch-токены последнего слоя, один проход train/val, кэш в RAM | Последний слой или конкатенация последних четырёх, новые crop и flip каждую эпоху |
| Голова | Новая Linear, Adam, 30 эпох, batch 256, LR 0.001 | LARS, 50 эпох, эффективный batch 16384, с BatchNorm и без |
| Результат | `linear_probe.csv`, графики `linear_probe/*` | JSON со всеми вариантами и лучшим по val, графики `linear_final/*` |

Мониторинг использует Resize(256) → CenterCrop(224), не меняет encoder и состояние
генераторов обучения. Кэш train-признаков ViT-B занимает около 400 МБ для 130 тысяч
изображений. Отключение: `--no-linear-probe`. Оценка `eta_seconds` не включает будущие probe.

**Итоговая оценка checkpoint:**

```sh
python scripts/evaluate_linear.py --mode final \
  --checkpoint runs/imagenet100-vit-base-10ep/checkpoint.pt \
  --train data/imagenet100/train --val data/imagenet100/val \
  --output runs/imagenet100-vit-base-10ep/linear_final.json \
  --device cuda --precision bf16 --batch-size 128 --num-workers 4 --comet
```

Для другого запуска замените пути к checkpoint и результату. `--batch-size` задаёт
порцию изображений для encoder, `--probe-batch-size` — батч головы.
По [статье, A.2](https://arxiv.org/html/2301.08243v3#A1.SS2) перебираются reference LR
`{0.01, 0.05, 0.001}` и weight decay `{0.0005, 0}`, LR делится на 10 каждые 15 эпох.
С учётом признаков и BatchNorm получается 24 варианта (12 для `tiny`). Эта оценка
существенно дороже мониторинга, протокол и допущения записываются в JSON.
Если train меньше 16384 изображений, уменьшите `--probe-batch-size`.
Comet использует прежний эксперимент, если рядом с итоговым JSON есть `comet.json`.

**Один выбранный вариант вместо повторного перебора:**

Отредактируйте [configs/linear_probe_selected.json](configs/linear_probe_selected.json):
`checkpoint`, `train`, `val`, `output` задают пути для вашего сервера.
В конфиге уже выбран рецепт: `last4`, `batchnorm_linear`, reference LR `0.05`, weight decay `0`.

```sh
python scripts/evaluate_linear_selected.py --config configs/linear_probe_selected.json
```

`probe_epochs: 50` — эпохи головы; `probe_batch_size: 16384` — её батч;
`batch_size: 128` — порция изображений для encoder. Фактический LR равен
`reference_lr × probe_batch_size / 256` (здесь `3.2`). CLI переопределяет JSON,
относительные пути считаются от рабочей папки; команды выполняйте из корня проекта.
Предыдущий отчёт с метриками не нужен. На новом checkpoint заново обучается одна
голова и рассчитываются train/val loss и top-1. Для сравнения checkpoint сохраняйте
одинаковый рецепт. Полная фактическая конфигурация записывается в выходной JSON.
Альтернатива: без явного рецепта укажите `--best-from <предыдущий linear_final.json>`
и пути через CLI — тогда настройки головы, её эпохи и батч будут взяты из отчёта.
Это убирает перебор голов, но сохраняет проход encoder на каждой эпохе с новыми crops:
ускорение не будет 24-кратным, поскольку и раньше головы делили общие проходы encoder.
Для дешёвых промежуточных оценок используйте `--mode monitor` ниже.

<details>
<summary>Дополнительно: лёгкий probe для checkpoint и kNN</summary>

Лёгкий probe отдельно от обучения, с теми же настройками, что у мониторинга:

```sh
python scripts/evaluate_linear.py --mode monitor \
  --checkpoint runs/imagenet100-vit-base-10ep/checkpoint.pt \
  --train data/imagenet100/train --val data/imagenet100/val \
  --output runs/imagenet100-vit-base-10ep/linear_monitor.json \
  --device cuda --precision bf16 --batch-size 128 --num-workers 4
```

kNN классифицирует val по ближайшим train-признакам без обучения головы.
`--compare-random` сравнивает результат со случайно инициализированным encoder:

```sh
python scripts/evaluate_knn.py \
  --checkpoint runs/imagenet100-vit-base-10ep/checkpoint.pt \
  --train data/imagenet100/train --val data/imagenet100/val \
  --output runs/imagenet100-vit-base-10ep/knn.json \
  --device cuda --bf16 --compare-random
```

По умолчанию: 20 соседей, 20 train- и 5 val-изображений на класс. Меняются через
`--k`, `--reference-per-class`, `--queries-per-class`. Сравнивайте одинаковые данные
и настройки; kNN не заменяет итоговый linear probe.

</details>

## Результаты и продолжение

В папке `output` каждого запуска:

| Файл | Содержимое |
| --- | --- |
| `config.json` | Фактическая конфигурация, версии библиотек, классы и протокол оценки |
| `metrics.csv` | Loss, градиенты, расписания, скорость и память |
| `linear_probe.csv` | Train/val-метрики после эпох |
| `checkpoint.pt` | Модели, оптимизатор, расписания, RNG, прогресс и конфигурация |
| `final_weights.pt` | Веса и конфиг, отправляемые в Comet при штатном завершении с `comet: true` |
| `checkpoint_epoch_0010.pt`, `…0020.pt`, … | Отдельные полные checkpoint после каждых 10 завершённых эпох |
| `comet.json` | Идентификатор облачного эксперимента |

`runs/.../config.json` — отчёт о запуске, не входной файл для `--config`.
Новые эксперименты настраивайте через `configs/` и сохраняйте в отдельные папки.
Checkpoint обновляется каждые 100 шагов по умолчанию, в конце эпохи и при штатной остановке.
Дополнительные копии сохраняются каждые 10 эпох, до запуска probe.
Период задаётся `checkpoint_every_epochs` в JSON или `--checkpoint-every-epochs`;
`0010` означает 10 полностью завершённых эпох. При остановке посреди эпохи её копия не создаётся.

Для продолжения повторите исходную команду, добавив `--resume`:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json \
  --resume runs/imagenet100-vit-base-10ep/checkpoint.pt
```

Сохраните прежние `output`, данные и параметры обучения, включая число эпох и батч.
Для продолжения с сохранённой эпохи укажите её файл вместо `checkpoint.pt`.
Для отдельной траектории от старого checkpoint задайте новый `--output`.
Изменение длительности с 10 на 370 или 600 эпох требует нового запуска. При нескольких DataLoader workers
точное повторение crop и масок после перезапуска не гарантируется из-за предзагрузки.

## Структура проекта

| Где | Что находится |
| --- | --- |
| `configs/` | JSON-конфиги экспериментов |
| `scripts/` | Команды подготовки данных, обучения и оценки checkpoint |
| `src/models/` | Encoder, predictor, target encoder и EMA |
| `src/train_step.py`, `src/optimizer.py`, `src/losses.py` | Шаг обучения, AdamW и loss |
| `src/linear_probe.py`, `src/linear_probe_final.py` | Реализации оценки; CLI — `scripts/evaluate_linear.py`, один выбранный вариант — `scripts/evaluate_linear_selected.py` |
| `src/datasets/`, `src/reference/` | Collator команды и авторские компоненты |
| `src/config.py`, `src/tracking.py` | Чтение настроек и Comet |
| `example/`, `tests/` | Учебные примеры и проверки |
| `data/`, `runs/` | Локальные датасеты и результаты, исключённые из Git |

Оба collator имеют общий контракт:
`((images, labels), [context_indices], [target_indices_1, ...])`, индексы — `[B, K]`
на сетке после crop/resize. Авторский вариант выбирается через `--collator reference`.
Источники I-JEPA/MAE, закреплённые версии и лицензия — в [src/reference/](src/reference/).
Лёгкий probe основан на функции из ветки `reg`, commit `cd0da4e`.

```sh
python scripts/train.py --help
python scripts/evaluate_linear.py --help
python scripts/evaluate_knn.py --help
python -m unittest discover -s tests
```
