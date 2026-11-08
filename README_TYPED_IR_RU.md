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
