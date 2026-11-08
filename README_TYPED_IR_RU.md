# Neural Language — статически проверяемый Neural IR (исследовательский этап)

**Назначение:** специализированный язык программирования `.nir` для описания, исполнения и обучения небольших нейронных систем. Это развитие языка/компилятора; никаких новых программных помощников и переноса кода из других проектов.

**Текущий путь исполнения:** `.nir → NeuralIR/0.2 + NeuralEffects/1 + NeuralShapes/1 → C++17 → GCC/Clang → native`.

Основной документ: `docs/NEURAL_IR_SHAPES_CONTRACT_RU.md`.

```bash
cd language
python3 -m unittest -q tests.test_typed_ir
python3 -m unittest -q tests.test_language tests.test_foundation tests.test_state_effects tests.test_ir_contract
python3 -m unittest -q tests.test_integration
python3 -m unittest -q tests.test_hardening
python3 -m unittest -q tests.test_quantization_fixes tests.test_verifier
cd ..
bash run.sh
```

Важные пределы: ранги только 1 для векторов, последовательные dense-слои, нет GPU/ARM64/Android-проверки, нет оптимизирующих IR passes и сравнительных бенчмарков против оптимизированных библиотек. Никакой гарантии предельного RSS. Обучение и математическая корректность произвольных сетей этим типовым контрактом не доказаны.

**База изменений:** GitHub `fix/neural-ir-effects-20261108` (`1bec6b678709c8bdb2ca6232779fd4669b311d23`). `main` и ветки 0.4/0.5 не меняются.

## Мобильное исполнение (изолированная исследовательская ветка)

В language/mobile_infer.cpp добавлено нативное inference-ядро для обученных последовательных моделей NLMODEL2, с компактным INT8-хранением и C ABI из language/mobile_infer.h. Проверка на хосте: python3 -m unittest -v language.tests.test_mobile_infer. Демонстрационная модель: language/examples/mobile_demo.nlw. Подробности: docs/MOBILE_NATIVE_INFERENCE_20261108_RU.md. Android NDK/ARM64 и реальный выигрыш ресурсов пока не проверены; main и отдельные DAG/assistant ветки не объединялись.

## Компактные модели и Android Java/JNI (NLMOBIN1)

Хостовый конвертер `python3 language/mobile_pack.py trained.nlw -o trained.nlmb` упаковывает FP64/INT8 и маски без изменения результатов inference. `mobile_infer.cpp` понимает оба формата; для Android предусмотрены `language/android/CMakeLists.txt`, JNI и `NeuralMobile.java`. Воспроизводимый запуск на Linux: `bash run_mobile.sh` (тесты и журналы сохраняются в новой папке `runs_mobile/`). Формат, границы и результаты: `docs/MOBILE_PACKED_MODELS_20261108_RU.md`; подключение JNI: `language/android/README_RU.md`. Android NDK, APK и испытания на устройстве пока отсутствуют.
