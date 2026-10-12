# rejepa

Воспроизведение I-JEPA на PyTorch: наши encoder/predictor, Smooth L1, AdamW и EMA.
Поддерживается обучение на одном устройстве (CPU или GPU).

## Установка и проверка запуска

Команды выполняются из корня репозитория в активированном Python-окружении.
Для GPU используйте окружение кластера с PyTorch/CUDA либо установите совместимые
`torch` и `torchvision` по [инструкции PyTorch](https://pytorch.org/get-started/locally/),
затем остальные зависимости:

```sh
python -m pip install -r requirements.txt
set -a
source .env
set +a
python scripts/train.py --config configs/smoke.json
```

Данная проверка использует 8 синтетических изображений и компактную модель на CPU,
ничего не скачивает и сохраняет конфигурацию, метрики и checkpoint.
Каждому новому запуску нужна отдельная папка `--output`.

## Comet

Comet включён для каждого запуска в `superustam/rejepa`. Загрузите локальный API key:

```sh
source .venv/bin/activate
set -a
source .env
set +a
python scripts/train.py --config configs/imagenet100_vit_base.json
```

`.env` исключён из Git. На другом компьютере задайте `COMET_API_KEY` в окружении.
В Comet отправляются параметры запуска, метрики каждого шага и финальные
веса encoder, predictor и target encoder через
[`log_model`](https://www.comet.com/docs/v2/integrations/ml-frameworks/pytorch/).
Без `COMET_API_KEY` обучение не запускается. Флагов включения и отключения Comet нет.
Каждый запуск, включая smoke test и `--resume`, создаёт отдельный Comet experiment.

## ImageNet100

Команды находятся в `scripts/setup_imagenet100.sh`; вместо блока выше
можно выполнить `bash -e ../../scripts/setup_imagenet100.sh` из папки `data/imagenet100`.
После подготовки получатся `data/imagenet100/train/<класс>/` и
`data/imagenet100/val/<класс>/`. Конфиг обучения уже использует этот путь.

По умолчанию используется ImageFolder: `root/class_name/image.jpg`.
Для обучения передавайте корень train; метки классов I-JEPA не использует.
Для оценки нужны отдельные train и val с одинаковыми названиями папок классов.
Список классов сохраняется в конфигурации запуска. Данные автоматически не скачиваются.

Содержимое `data/`, кроме [инструкции](data/README.md), исключено из Git.
Если датасет уже есть на сервере,
укажите его путь в `data` конфига или через `--data`; копировать его в проект не нужно.

## Первый запуск на GPU

Запускайте внутри выделенной GPU-задачи или контейнера кластера, где доступна
одна H200. Скрипт использует одно устройство, запускайте обычным `python`.
Проверка окружения и запуск на 10 эпох:

```sh
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name()); print('BF16:', torch.cuda.is_bf16_supported())"
python scripts/train.py --config configs/imagenet100_vit_base.json
```

Базовый конфиг: ViT-B/16, BF16, batch size 128 без накопления градиентов,
наш collator и **10 полных эпох**, без ограничения числа шагов.
Одна эпоха отведена под warmup. Learning rate растёт от `6.25e-6` до `6.25e-5`,
затем снижается по косинусу до `6.25e-8`. Эти три значения взяты из
[статьи I-JEPA, приложение A.1](https://arxiv.org/pdf/2301.08243)
и умножены на `128 / 2048` для меньшего батча. Такое масштабирование — наш
стартовый выбор для короткого запуска; его качество ещё предстоит проверить.
Цель запуска — проверить стабильность обучения на ImageNet100 и затем оценить
качество представлений через kNN.

При успешном завершении в `runs/imagenet100-vit-base-10ep/` появятся `config.json`,
`metrics.csv` и `checkpoint.pt`. В метриках смотрите `loss`, `grad_norm`,
`steady_images_per_second` и `cuda_peak_memory_mb`.
Для другого запуска задайте новый `output`, при нехватке памяти уменьшите batch size.

## Конфиги экспериментов

В `configs/` лежат планы запусков: плоские JSON с именами параметров в `snake_case`
(`batch_size`, `warmup_epochs`, `max_steps`). Параметры командной строки
переопределяют JSON; пропущенные настройки берутся из значений по умолчанию.
Относительные пути считаются от текущей рабочей папки.
Все команды здесь нужно выполнять из корня репозитория.

Основной конфиг — `configs/imagenet100_vit_base.json`. При необходимости измените
в нём `data` на путь к train и `output` на отдельную папку результатов.
`configs/smoke.json` оставлен только для быстрой проверки на CPU.
В основном конфиге `max_steps: null` позволяет пройти все заданные эпохи.
Ограничение `max_steps` относится к суммарному числу шагов оптимизатора;
длительность расписаний по-прежнему определяется `epochs`.

Разовое изменение удобно передать через CLI:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json --batch-size 32 --accumulation-steps 4 --output runs/imagenet100-vit-base-10ep-small-batch
```

Все изображения проходят random resized crop до 224×224 и нормализацию ImageNet.
Маски генерируются заново для каждого батча.

## Режимы обучения и регуляризация

`--mode ema` сохраняет базовое обучение: отдельный target encoder заморожен и
обновляется через EMA. `--mode joint` использует один обучаемый encoder для
контекста и полного изображения. В joint доступны `--regularizer none`,
`sigreg` и `visreg`; по умолчанию выбран `sigreg`. Регуляризатор применяется
к эмбеддингам каждой позиции патча по батчу, его вклад задаётся
`--regularizer-weight` (по умолчанию `0.02` из ветки `reg`; значение не подбиралось для этих запусков).
В [статье VISReg](https://arxiv.org/html/2606.02572) `0.02` относится к SIGReg,
а для VISReg используются другие веса. Числа из статьи нельзя напрямую переносить:
там loss смешивается как `(1−λ)·prediction + λ·regularization`, здесь —
`prediction + regularizer_weight·regularization`.
Режим и регуляризатор задаются этими CLI-параметрами или ключами `mode`,
`regularizer`, `regularizer_weight` в JSON-конфиге. В обоих конфигах на 600 эпох
явно указаны `"mode": "ema"` и `"regularizer": "none"`. Для joint измените оба
значения на `"joint"` и `"sigreg"` или `"visreg"` и задайте новый `output`.

```sh
python scripts/train.py --config configs/imagenet100_vit_base_600ep_sk_container.json \
  --mode joint --regularizer sigreg --representation-data val \
  --output /home/container/shared/rejepa/runs/imagenet100-joint-sigreg
```

Для VISReg замените `sigreg` на `visreg`, явно выберите
`--regularizer-weight` и другой `--output`.

Joint checkpoint содержит обучаемый `encoder` без отдельного `target_encoder`.
Режим и регуляризатор нельзя менять при продолжении запуска. Формулы
регуляризаторов описаны в [src/regularizers/README.md](src/regularizers/README.md).

По умолчанию используется collator команды: `--collator custom`, реализация
в `src/datasets/imagenet100.py`. Авторский вариант включается через
`--collator reference`. Оба возвращают одинаковый формат батча, поэтому encoder,
predictor и training step не меняются. Для проверки авторского варианта:

```sh
python scripts/train.py --config configs/smoke.json --collator reference --output runs/reference-smoke
```

Оба варианта повторяют правило reference: после 20 неудачных попыток получить
достаточно большой контекст часть ограничений на пересечение с targets
постепенно снимается. При текущих настройках маски обоих collator проверены на
совпадение.

Основные параметры:

| Параметр | Назначение |
| --- | --- |
| `--model tiny / vit_base / vit_large` | По умолчанию `tiny` для проверки. `vit_base` — ViT-B/16: encoder 768×12 блоков, predictor 384×6 блоков. |
| `--device cpu / cuda` | По умолчанию CPU. |
| `--batch-size`, `--accumulation-steps` | Эффективный batch size — их произведение; optimizer, EMA и расписания обновляются после накопления. |
| `--precision fp32 / bf16` | По умолчанию FP32; BF16 уменьшает затраты памяти на поддерживаемом оборудовании. |
| `--epochs`, `--warmup-epochs` | По умолчанию 1 и 0; warmup должен быть короче обучения. |
| `--lr`, `--weight-decay`, `--ema-start` | Пиковый LR, начальные weight decay и EMA; остальные параметры расписаний доступны в `--help`. |
| `--num-workers`, `--seed` | Загрузка данных и воспроизводимость; по умолчанию 0 и 0. |

При нехватке памяти уменьшайте batch size, при необходимости компенсируйте его
накоплением градиентов. Маски согласуются внутри каждого микробатча, поэтому
накопление не полностью эквивалентно одному большому батчу.

## Результаты и продолжение

В папке `--output` сохраняются:

- `config.json` — полная фактическая конфигурация запуска после объединения JSON и CLI;
- `metrics.csv` — loss, градиенты, расписания, скорость и память GPU;
- `checkpoint.pt` — модели, оптимизатор, расписания, генераторы случайных чисел и прогресс.
- `checkpoint_0.pt`, `checkpoint_1.pt`, … — отдельный checkpoint после каждой полностью завершённой эпохи, с нумерацией от нуля; сохраняется по умолчанию.

В `config.json` и `checkpoint["config"]` сохраняются одинаковые параметры:
итоговые аргументы, модели со значениями по умолчанию, маски, preprocessing,
objective, optimizer, schedules, версии библиотек и список классов.
Пути `data`, `output`, `resume` и `config` записываются абсолютными, если заданы;
поле `config` указывает исходный файл эксперимента.
Этот snapshot содержит вычисленные значения и служебные сведения, поэтому
`runs/.../config.json` не является входным файлом для `--config`.
Для новых запусков редактируйте файлы в `configs/`.

Checkpoint обновляется каждые `--checkpoint-every` шагов оптимизатора (по умолчанию
100), в конце эпохи и при штатной остановке. Для продолжения повторите исходную
команду с тем же `--output` и добавьте `--resume path/to/checkpoint.pt`.
Для продолжения с конкретной эпохи используйте `--resume path/to/checkpoint_N.pt` (N — индекс эпохи от нуля).

Отдельный флаг для сохранения каждой эпохи не нужен. Пример запуска:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json --output runs/my-run
```

Продолжение с checkpoint первой эпохи с теми же настройками и папкой вывода:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json \
  --output runs/my-run --resume runs/my-run/checkpoint_0.pt
```

`--checkpoint-every 100` задаёт дополнительные сохранения в `checkpoint.pt`
каждые 100 шагов оптимизатора. Отдельные checkpoint эпох сохраняются автоматически
и занимают дополнительное место на диске.

Параметры данных и обучения, включая collator, общее число эпох, batch size и накопление,
должны совпадать. Если предел шагов достигнут, увеличьте `--max-steps`
или установите `"max_steps": null` в исходном JSON.
Если после проверки меняете общее число эпох, начните новый запуск с отдельным
`output`: продолжать старый checkpoint с другим расписанием нельзя.
При нескольких DataLoader workers точное повторение будущих crop и масок после
перезапуска не гарантируется из-за предварительной загрузки.

## Оценка представлений

### Diagnostics during training

Pass `--representation-data` with an ImageFolder validation directory. During
training, the first, every 25th, and final completed epochs are evaluated on the
same images. EMA evaluates the target encoder; joint evaluates the shared encoder.
Each evaluation appends a row to `representation_analysis/representations.csv`.
`configs/imagenet100_vit_base_600ep.json` already sets `representation_data` to `val`.
The VM container config also sets `representation_data` to `val`. Run it from
the directory containing `train/` and `val/`, since those paths are relative:

```sh
cd /home/container/shared/rejepa
python /home/container/rejepa/scripts/train.py \
  --config /home/container/rejepa/configs/imagenet100_vit_base_600ep_sk_container.json
```

Use `--representation-samples` to change the sample size (default: 1024).
The run also saves `spectrum_N.pt` with singular values and metrics, and
`settings.json` with the selected image paths. Covariance is computed but not
stored. Resuming training does not duplicate completed epochs in the CSV.

### Анализ сохранённых эпох

После обучения можно сравнить представления из готовых `checkpoint_N.pt`:
**target encoder** для EMA и обучаемого **encoder** для joint. Обучение и Comet
для этого не нужны. Скрипт выбирает одни и те же
изображения из ImageFolder, применяет детерминированные resize и center crop,
усредняет patch tokens каждого изображения и считает ковариационную матрицу
признаков. Энтропия вычисляется по нормированным сингулярным числам центрированной
матрицы признаков; `effective_rank = exp(entropy)`. Низкие ранг и средняя дисперсия
могут указывать на схлопывание представлений, но сами по себе не измеряют качество
разделения классов.

```sh
python scripts/analyze_checkpoints.py \
  --checkpoints /home/container/shared/rejepa/runs/imagenet100-vit-base-600ep \
  --data val \
  --device cuda \
  --bf16 \
  --every 25
```

Команду запускайте из корня репозитория на VM, заменив пути к папке с
`checkpoint_N.pt` и к ImageFolder `val`, если они расположены иначе.
Без `--every` анализируются все checkpoint; `--every 25` выбирает первую,
последнюю и каждую 25-ю завершённую эпоху. Результаты появятся в
`runs/imagenet100-vit-base-600ep/representation_analysis/`: таблица
`representations.csv`, `spectrum_N.pt` и `settings.json`.
Повторный запуск с теми же настройками пропускает уже обработанные эпохи.
Для другого набора изображений или настроек укажите новую папку `--output`.
Эффективный ранг ограничен `min(число изображений − 1, размерность признака)`;
для ViT-B/16 с 768 признаками нужно хотя бы 769 изображений, чтобы увидеть
полный возможный ранг.
На сервере удобно запускать команду после завершения обучения в отдельной
задаче GPU; от Comet и процесса обучения она не зависит.

### kNN

kNN проверяет, насколько хорошо классы разделяются в признаках замороженного
**target encoder** для EMA или общего **encoder** для joint. Его patch tokens усредняются в один вектор изображения и
нормализуются. Для каждого изображения из val ищутся ближайшие по косинусной
близости изображения из train; их метки голосуют за класс, с большим весом у
более близких соседей. Отдельный классификатор не обучается.

```sh
python scripts/evaluate_knn.py --checkpoint runs/imagenet100-vit-base-10ep/checkpoint.pt --train data/imagenet100/train --val data/imagenet100/val --device cuda --bf16 --compare-random --output runs/imagenet100-vit-base-10ep/knn.json
```

По умолчанию оценка идёт на CPU, с 20 соседями, 20 reference-изображениями из train
и 5 запросами из val на класс. Числа меняются через `--k`, `--reference-per-class`
и `--queries-per-class`; в каждом классе должно хватать изображений. Для GPU
добавьте `--device cuda`, при поддержке BF16 — `--bf16`.

`--compare-random` повторяет оценку той же архитектуры со случайной инициализацией
на тех же изображениях. Это проверяет прирост от обучения. Случайный encoder не
обязан давать точность случайного угадывания: он тоже может сохранять сходство
изображений. Метки используются только при оценке, не при обучении I-JEPA.

JSON содержит top-1/top-5 accuracy, прирост top-1 в процентных пунктах, разброс
признаков, эффективный ранг и списки изображений. При сравнении экспериментов
сохраняйте одинаковые данные и параметры оценки. kNN не заменяет linear probe
из статьи.

## Где читать и менять код

`src/` содержит метод, `scripts/` — рабочие запуски обучения и оценки,
`configs/` — планы экспериментов, `example/` — учебные примеры,
`data/` — локальные датасеты, `runs/` — локальные результаты.

- `src/models/` — encoder, predictor и функции target encoder/EMA.
- `src/train_step.py` — один шаг: цели → предсказания → loss → backward → optimizer → EMA.
- `src/optimizer.py` и `src/losses.py` — группы параметров AdamW и функция потерь.
- `scripts/train.py` — параметры, DataLoader, цикл, расписания и сохранение.
- `scripts/evaluate_knn.py` — оценка сохранённой модели.
- `scripts/setup_imagenet100.sh` — команды команды для скачивания и раскладки ImageNet100.
- `src/config.py` — чтение JSON и проверка настроек запуска.
- `src/reference/` — авторские collator и расписания LR/weight decay.
- `src/datasets/imagenet100.py` — collator команды и общая сборка DataLoader;
  `make_imagenet100` читает `root/train` или `root/val` и возвращает dataset, loader, sampler.
  При его отдельном использовании в обучении вызывайте `sampler.set_epoch(epoch)` перед каждой эпохой.

Файлы `reference` взяты без изменений из
[facebookresearch/ijepa, commit 52c1ae95](https://github.com/facebookresearch/ijepa/tree/52c1ae95d05f743e000e8f10a1f3a79b10cff048).
Источник и лицензия CC BY-NC 4.0 находятся рядом. Реализация команды перенесена
из ветки `data_encoder` (commit `930120e`) с адаптацией интерфейсов.
Оба collator сохраняют контракт батча:
`((images, labels), [context_indices], [target_indices_1, ...])`.
Индексы относятся к сетке патчей после crop/resize; каждый набор имеет форму `[B, K]`.

Полный список параметров и проверки:

```sh
python scripts/train.py --help
python scripts/evaluate_knn.py --help
python -m unittest discover -s tests
```
