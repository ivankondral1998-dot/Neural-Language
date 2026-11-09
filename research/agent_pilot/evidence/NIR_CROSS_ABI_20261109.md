# Neural Language — .nir full-stack x86-64 / ARM64 QEMU parity gate (2026-11-09)

**Статус:** PASS на Linux x86-64 и ARM64 в пользовательском режиме QEMU; Android и физический ARM64 НЕ проверены.

## Проверенные исходники

- Репозиторий: `ivankondral1998-dot/Neural-Language`
- Исходная ветка: `research/pretrained-cartpole-agent-20261109`; изменения тестов сохранялись только в этой ветке.
- `language/compiler.py`: Git blob `65f4ef45ad4c144e3d6e07535c3dbce78b8adb6b`.
- `language/runtime/neural_runtime.hpp`: Git blob `faf78c47c60cf325152161e258978177be2e5cf9`.
- `language/examples/pretrained_cartpole.nir`: Git blob `57f9d55289f227c4619da1a7f11498ea81eb47af`.
- `research/agent_pilot/models/cartpole_pretrained.nlw`: Git blob `b70bcf774e1f6dbf32c22a366bc0c81350f551ed`, SHA-256 `a0738c97790d61dbd7b0fe2bde1ece59ff5041457ac4c77c8aae99c0be3cf8da`.
- 29 файлов скомпонованного тестового checkout были сверены с оригинальными Git blob SHA: **29/29 MATCH**. Исходники компилятора, runtime и моделей не менялись.

## Среда

Ubuntu 26.04.1 Linux x86-64; Python 3.14.4; GCC/G++ 15.2.0 (host и `aarch64-linux-gnu-g++`); QEMU user `qemu-aarch64` 10.2.1, ARM64 sysroot `/usr/aarch64-linux-gnu`. Нативную облачную ARM64-среду не удалось создать: `image_not_ready: No image with platform linux/arm64 found`. Использован явно обозначенный альтернативный метод кросс-компиляции + QEMU; никаких замеров мощности/производительности ARM64 здесь не следует делать. GitHub Actions не запускались.

## Фактические запуски

1. `bash run.sh` на x86-64: **END_TO_END_PASS**. Программа `integrated.nir` обучила линейную модель 1600 раз, провела 4 проверки, `verify 40`, `optimize 90`, сохранила `trained.nlw`. Отдельный бинарник `inference.nir` загрузил весовой файл, провёл 2 проверки, вернул `0.3`.
2. Тот же `.nir` компилятор собрал `pretrained_cartpole.nir` через `g++` и через `aarch64-linux-gnu-g++`. На x86-64 и под `qemu-aarch64 -L /usr/aarch64-linux-gnu`: **MODEL_LOADED**, `PREDICT version=1 values=-0.3736812805,0.3745496992`, **PASS**.
3. `integrated.nir` и `inference.nir` кросс-скомпилированы под ARM64 и выполнены QEMU. **ARM64_QEMU_FULL_NIR_PASS**: 1600 шагов обучения; 4 + 2 валидации; отдельная загрузка весов после сохранения; `VERIFY PASS`, `OPTIMIZE selected=FULL`, `PREDICT version=1 values=0.3`.
4. После добавления семи ранее не перенесённых тестовых примеров выполнен реальный полный отобранный набор: `python3 -m unittest -q tests.test_integration tests.test_language tests.test_quantization_fixes tests.test_verifier tests.test_hardening tests.test_shapes_changes_integration` из `language/`. **Ran 66 tests in 57.069s, OK (skipped=1)**. Первоначальные 8 ошибок были только FileNotFoundError отсутствовавших примеров при неполной локальной копии; устранены пополнением файлами из исходного GitHub, не изменением кода.
5. В отдельной исследовательской ветке добавлен `research/agent_pilot/check_nir_cross_abi.py`. Выполнено: `python3 research/agent_pilot/check_nir_cross_abi.py` → **CROSSABI_GATE PASS**. Проверяет три оригинальные .nir-программы на обеих ABI, численный oracle готовой модели, идентичность сгенерированных C++ и Neural IR, сохранение/загрузку обученного checkpoint, допустимость вычислений.

## Кросс-архитектурные контрольные результаты

- CartPole logits (обе архитектуры): `[-0.3736812805, 0.3745496992]`.
- Сгенерированные Neural IR и C++ совпали побайтово (не просто по численным значениям).
- Сохранённый после 1600 шагов обучения `trained.nlw` совпал побайтово между x86-64 и ARM64/QEMU.
- SHA-256 `trained.nlw`: `ef9c8a6ee12847bb785b3e822d3a115bd26122386812f7b5dc60f3f38a30cc0d`.
- В отчётах численные следы `VALIDATE` иногда различаются из-за FP64 округления в последнем бите (0 vs ~1.11e-16). Сырые события с runtime-таймингами также различаются: **JSON-отчёты в целом НЕ побайтово равны**. Сохранённые checkpoint-файлы — равны.

## Как повторить

Из полного checkout ветки, при наличии `python3`, `g++`, `aarch64-linux-gnu-g++`, `qemu-aarch64` и ARM64 sysroot:

```bash
python3 research/agent_pilot/check_nir_cross_abi.py
cd language && python3 -m unittest -q tests.test_integration tests.test_language tests.test_quantization_fixes tests.test_verifier tests.test_hardening tests.test_shapes_changes_integration
```

Сценарий создаёт отдельную папку `runs/crossabi_XXXXXX`, сохраняет stdout и `validation_result.json`; исходники не меняет, никаких workflow не инициирует. Основа аналогов для сравнения — стандартная независимая Python-модель и обычное C++17-исполнение.

## Границы

Это **подтверждённая воспроизводимость .nir → Neural IR → C++ → запуск** и существенный прогресс к переносимости, **но не** измеренное преимущество над ONNX/TFLite, не Android NDK APK, не запуск на физическом телефоне и не доказательство оригинальности оптимизации. CartPole при 4610 весах выбирает полный расчёт AUTO, значит ускорение этим workload не установлено. QEMU нельзя использовать для вывода о мобильном p95, энергии или температуре. Исследовательский механизм DAG/NLMODEL3 и отдельный мобильный JNI не сливались с main.
