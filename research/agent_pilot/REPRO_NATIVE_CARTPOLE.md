# Neural Language — проверка нативного CartPole
Дата: 2026-11-09. Статус: isolated research; не Android-release.

## Задача

Независимо проверить C++ runtime Neural Language на предобученной PPO-модели 4→64→64→2 (4610 параметров). Сравниваются все состояния CartPole, логиты AUTO/FULL, выбранные действия и длины эпизодов со stdlib Python-эталоном, а не только AUTO vs FULL между собой.

## Воспроизведение (из корня полного checkout ветки)

Требуются Python 3 и GCC/Clang с C++17. Сеть, Gymnasium, NumPy и GitHub Actions не нужны.

    python3 -m unittest -v research/agent_pilot/test_verify_cartpole_reference.py research/agent_pilot/test_compare_native_cartpole.py
    python3 research/agent_pilot/verify_cartpole_reference.py research/agent_pilot/models/cartpole_pretrained.nlw --episodes 100
    python3 research/agent_pilot/compare_native_cartpole.py --compiler g++ --episodes 10
    python3 research/agent_pilot/compare_native_cartpole.py --compiler clang++ --episodes 10

Скрипт compare_native_cartpole.py компилирует новый agent_cartpole_trace.cpp против language/runtime/neural_runtime.hpp во временном каталоге, запускает реальную загруженную модель, сверяет данные шаг за шагом и печатает JSON status=PASS только при успехе. Трасса не является измерителем производительности; нельзя интерпретировать её время как ускорение нейронных вычислений.

## Проверено в этой итерации

- Локально: 11/11 Python-тестов проверяющих алгоритмов PASS, Python py_compile PASS.
- Локально: C++ исходник проходит GCC и Clang -std=c++17 -Wall -Wextra -Werror -fsyntax-only с *заглушкой интерфейса*, это НЕ сборка с настоящим runtime.
- GitHub: исходники теста и проверяющей программы сохранены в research/pretrained-cartpole-agent-20261109, прочитаны обратно; blob SHA сверены с локальными.

## Не проверено

В используемой локальной среде полный приватный checkout недоступен. Поэтому сборка и выполнение agent_cartpole_trace.cpp с настоящим neural_runtime.hpp и pretrained .nlw **ещё НЕ проводились**. В памяти имеется отдельная JavaScript-проверка 100 эпизодов реальных весов, но она не заменяет проверку C++.

Android/ARM64, оптимизированные TFLite/ONNX сравнения, RSS/энергия/температура и графовые NLMODEL3 модели — отдельные открытые вопросы. В main не сливать без независимой приёмки. Ничего не удалять; GitHub Actions не запускать без разрешения.
