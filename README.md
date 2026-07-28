# Art Visor — ML Worker

Python-микросервис для анализа изображений картин. Принимает задачи из RabbitMQ, прогоняет изображение через нейросеть CLIP и возвращает: 768-мерный эмбеддинг (для поиска похожих картин), топ-5 художественных стилей с вероятностями и палитру из 5 доминирующих цветов.

Является ML-частью платформы **Art Visor**: Java-бэкенд принимает загрузку пользователя, ставит задачу в очередь, воркер её обрабатывает и публикует результат обратно.

## Как это работает

```
Пользователь → Java-бэкенд → RabbitMQ (art.analysis.queue)
                                   │
                                   ▼
                             worker.py
                  1. скачивает изображение по URL
                  2. CLIP → нормализованный эмбеддинг (768)
                  3. MLP-классификатор → топ-5 стилей
                  4. K-Means → палитра из 5 цветов (HEX)
                                   │
                                   ▼
              RabbitMQ (art.results.queue) → Java-бэкенд
                                   │
                                   ▼
        PostgreSQL + pgvector (HNSW-индекс) → поиск похожих картин
```

### Конвейер обработки задачи

1. **Приём задачи.** Воркер слушает очередь `art.analysis.queue` (exchange `art.exchange`, routing key `art.analyze`). Задача — JSON вида `{"taskId": "...", "imageUrl": "..."}`.
2. **Загрузка изображения.** Скачивается по URL с браузерным `User-Agent` (иначе некоторые серверы, например Википедия, отдают 403).
3. **Эмбеддинг.** Изображение проходит через **CLIP** (`openai/clip-vit-large-patch14`), вектор L2-нормализуется. Именно этот вектор Java-бэкенд сохраняет в pgvector и использует для косинусного поиска похожих работ.
4. **Классификация стиля.** Обученная на WikiArt модель (`minimalism_classifier.pkl`, scikit-learn MLP поверх CLIP-эмбеддингов, ~75% top-1 по 27 стилям) выдаёт вероятности; в ответ идут топ-5 с русскими названиями стилей. Опционально подмешивается zero-shot-прогноз CLIP по текстовым промптам (переменная `CLF_WEIGHT`, по умолчанию выключено).
5. **Палитра.** K-Means по пикселям уменьшенной копии изображения → 5 доминирующих цветов в HEX.
6. **Публикация результата.** Итоговый JSON уходит в `art.results.queue`. При ошибке публикуется `{"taskId", "status": "FAILED", "error"}`, чтобы задача на стороне Java не зависла в статусе PROCESSING.

### Формат результата

```json
{
  "taskId": "abc-123",
  "embedding": [0.0123, -0.0456, "... 768 чисел"],
  "palette": ["#1a2b3c", "#ffffff", "#c0ffee", "#123456", "#654321"],
  "styleBreakdown": [
    {"style": "Импрессионизм", "prob": "62.4%", "val": 62.4},
    {"style": "Постимпрессионизм", "prob": "18.1%", "val": 18.1}
  ]
}
```

### Надёжность

- **Reconnect-цикл**: при обрыве соединения с RabbitMQ воркер переподключается каждые 5 секунд; exchange и очереди объявляются заново (идемпотентно).
- **Fail-fast**: без файла классификатора воркер не стартует — лучше упасть на старте, чем публиковать «успешные» результаты без стилей.
- **Прогрев на старте**: первый forward-pass CLIP на CPU занимает 5–10 секунд (ленивая инициализация PyTorch), поэтому при запуске через пайплайн прогоняется фиктивное изображение — первая реальная загрузка пользователя обрабатывается быстро.
- **`prefetch_count=1`** — воркер берёт по одной задаче, что позволяет масштабироваться простым запуском нескольких контейнеров.

## Структура репозитория

| Путь | Назначение |
|---|---|
| `worker.py` | Основной сервис: RabbitMQ-консьюмер + инференс |
| `minimalism_classifier.pkl` | Обученный классификатор стилей (MLP + StandardScaler) |
| `embeddings.npz` | Кэш CLIP-эмбеддингов датасета WikiArt (~56 тыс. картин, 27 стилей) |
| `scripts/extract_features.py` | Извлечение эмбеддингов из локального датасета `datasets/<стиль>/*.jpg` |
| `scripts/train_classifier.py` | Обучение классификатора на `embeddings.npz` + отчёт по метрикам |
| `scripts/report_metrics.py` | Метрики и графики (confusion matrix, ROC, кривые обучения) → `report/` |
| `scripts/load_artworks.py` | Генерация CSV из `embeddings.npz` для загрузки в таблицу `artworks` (pgvector) |
| `scripts/npz_from_db.py`, `scripts/rebuild.py` | Утилиты пересборки данных из БД |
| `scripts/colab_extract_336.py` | Извлечение эмбеддингов на GPU в Colab (вариант CLIP 336px) |
| `Dockerfile` | Образ воркера (CPU-сборка PyTorch) |
| `.github/workflows/ci-cd.yml` | CI/CD: сборка образа → GHCR → деплой на сервер по SSH |
| `docs/` | Проектная документация |

## Запуск

### Локально

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python worker.py
```

Нужен запущенный RabbitMQ. При первом старте CLIP-веса (~1.7 ГБ) скачиваются с Hugging Face автоматически.

### Docker

```bash
docker build -t art-python-ai .
docker run --env-file .env art-python-ai
```

В compose-окружении рекомендуется примонтировать named volume в `/app/hf_cache`, чтобы веса CLIP не скачивались при каждом пересоздании контейнера.

### Переменные окружения (`.env`)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `RABBITMQ_HOST` | `localhost` | Хост RabbitMQ |
| `RABBITMQ_DEFAULT_USER` | `guest` | Логин RabbitMQ |
| `RABBITMQ_DEFAULT_PASS` | `guest` | Пароль RabbitMQ |
| `CLIP_MODEL` | `openai/clip-vit-large-patch14` | Backbone CLIP (должен совпадать с моделью, которой построены вектора в БД) |
| `CLF_WEIGHT` | `1.0` | Вес обученной модели при смешивании с zero-shot CLIP (1.0 = zero-shot выключен) |

## Переобучение классификатора

1. Разложить датасет по папкам `datasets/<Имя_стиля>/*.jpg`.
2. `python scripts/extract_features.py` → `embeddings.npz` (на GPU быстрее — см. `scripts/colab_extract_336.py`).
3. `python scripts/train_classifier.py` → `minimalism_classifier.pkl` + отчёт по метрикам.
4. `python scripts/load_artworks.py --out artworks.csv` и загрузить CSV в таблицу `artworks` через `psql \copy` — база и классификатор должны жить в одном векторном пространстве.
5. `python scripts/report_metrics.py` — обновить графики в `report/`.

**Важно:** версии `torch` / `transformers` / `scikit-learn` в `requirements.txt` зафиксированы — они соответствуют тем, на которых обучалась модель. Смена версий может сломать загрузку pickle или сдвинуть эмбеддинги CLIP (и ухудшить поиск похожих).

## CI/CD

При пуше в `main` GitHub Actions собирает Docker-образ, публикует его в GHCR (`ghcr.io/<owner>/art-python-ai:latest` и `:sha`) и деплоит на сервер по SSH (`docker compose pull && up -d`). Pull request'ы проходят проверочную сборку образа без публикации. Для деплоя нужны секреты `SSH_HOST`, `SSH_USER`, `SSH_KEY`, `DEPLOY_DIR`.
