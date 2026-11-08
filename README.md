# Neural Language

Исследовательский **специализированный язык для разработки, обучения, проверки и оптимизации нейронных систем**. Цель владельца — единая технология «новый язык + встроенный помощник», способная создавать нейронные структуры и веса, генерировать исполняемый код и проверять предлагаемые улучшения.

**Статус: 0.3 Integrated / HOST-TESTED / CONDITIONAL.** Это не UNO и не готовая промышленная платформа ИИ. Работа на Android, преимущество над промышленными реализациями и самостоятельное улучшение исходного кода помощником **не доказаны**.

## Состав репозитория

- `language/compiler.py` — компилятор формата `.nir` в Neural IR JSON и C++17.
- `language/runtime/neural_runtime.hpp` — нативное вычислительное ядро и ограниченный помощник выбора режима вычислений.
- `language/examples/` — примеры на нашем языке. `integrated.nir` обучает и сохраняет модель, `inference.nir` загружает и выполняет предсказание без обучения.
- `language/tests/` — автоматические проверки компилятора, численной корректности и сохранения моделей.
- `oracle_integration.py` — независимый расчёт для узкой линейной демонстрации.
- `docs/RELEASE_0_3_README_RU.md` и `docs/RELEASE_0_3_FINAL_STATUS_RU.md` — подробности, ограничения и исторические результаты.
- `AGENTS.md` и `language/AGENTS.md` — обязательные инструкции для разработчиков и ИИ-агентов.

## Быстрый запуск

Требуется **Linux, Python 3 и GCC/Clang с C++17**. BLAS необязателен.

```bash
bash run.sh
```

Скрипт создаёт отдельную `runs/run_XXXXXX` и выполняет полный цикл: компиляция, обучение, сохранение весов, загрузка в другую программу и проверка предсказаний. Для Clang: `CXX=clang++ bash run.sh`.

```bash
cd language
python3 -m unittest -q tests.test_integration
python3 -m unittest -q tests.test_language
python3 -m unittest -q tests.test_quantization_fixes tests.test_verifier
python3 -m unittest -q tests.test_hardening
```

## Точная граница возможностей

Текущий путь: **`.nir` → Neural IR → C++17 → GCC/Clang → нативный бинарник**. Python нужен для компилятора, C++ — для вычислений. Помощник пока только сравнивает заранее заданные режимы на ограниченной нагрузке: он **не переписывает код и не является автономным исследователем**.

`budget` даёт статическую оценку, а не абсолютную гарантию RAM/времени. INT8-обучение и безопасность загрузки недоверенных моделей требуют дальнейшей работы. Никаких заявлений о превосходстве над C++ или OpenBLAS без честных независимых измерений.

## Работа с проектом

Сначала прочитать `AGENTS.md`, зафиксировать текущую ветку/коммит, воспроизводить тесты и сохранять исходники для отката. **GitHub Actions не запускать без отдельного разрешения владельца**. UNO, Jarvis и «Искра» — отдельные проекты; не переносить сюда их правила и код автоматически.

Эта версия загружена из ранее сохранённого интеграционного пакета 0.3. Исторические бинарные артефакты и исходный ZIP пакета **не включены** в Git: исходников репозитория достаточно для сборки на поддерживаемой Linux-среде.

## Экспериментальная ветка 0.4 — DAG

В ветке `feature/graph-dag-0.4-20261108` реализовано параллельное ветвление нейросети, конкатенация выходов и нативное обучение с проверкой градиентов. См. [`language/docs/GRAPH_DAG_0_4_RU.md`](language/docs/GRAPH_DAG_0_4_RU.md). Это исследовательский прототип; основная `main` остаётся контрольной точкой 0.3.

## Experimental engineering assistant 0.5

The separate branch `feature/verified-assistant-0.5-20261108` contains a bounded, **non-LLM** Neural IR architecture proposal and verification loop. It compiles both `.nir` candidates with GCC/Clang, compares validation errors, verifies native predictions with an independent Python model reader, and publishes a candidate only if the stated acceptance policy passes. This is a research experiment, **not** autonomous self-improving AI or proof of novelty versus existing frameworks. See [`docs/ENGINEERING_BASELINE_0_5_RU.md`](docs/ENGINEERING_BASELINE_0_5_RU.md), [`docs/ASSISTANT_0_5_USAGE_RU.md`](docs/ASSISTANT_0_5_USAGE_RU.md) and [`RELEASE_STATUS_0_5_RU.md`](RELEASE_STATUS_0_5_RU.md). The `main` branch stays at the 0.3 baseline.
