# rejepa

Воспроизведение I-JEPA.

## Download dataset imagenet100
```sh
kaggle datasets download ambityga/imagenet100
unzip imagenet100.zip
```

## Регуляризация эмбеддингов

[Описание и использование модулей](src/regularizers/README.md).

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
