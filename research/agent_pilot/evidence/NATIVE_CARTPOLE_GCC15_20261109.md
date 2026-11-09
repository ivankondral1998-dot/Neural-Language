# Neural Language — реальный нативный CartPole, 9 ноября 2026

## Provenance

- Исходники: `research/pretrained-cartpole-agent-20261109`; сравнивались файлы из приватного GitHub, без модификации.
- `language/runtime/neural_runtime.hpp`: Git blob `faf78c47c60cf325152161e258978177be2e5cf9` — совпал после переноса.
- `research/agent_pilot/models/cartpole_pretrained.nlw`: Git blob `b70bcf774e1f6dbf32c22a366bc0c81350f551ed` — совпал после переноса.
- SHA-256 оригинального `.nlw`: `a0738c97790d61dbd7b0fe2bde1ece59ff5041457ac4c77c8aae99c0be3cf8da`.
- Среда исполнения: временная Linux x86-64 Ubuntu 26.04.1, Python 3.14.4, GCC `g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0`, C++17, компиляция `-O2`. Внешняя сеть закрыта при тестировании. Сетевое зеркало временно открывалось только для установки компилятора.
- Комплект тестов существовал в репозитории до этого запуска. Оригинальные код и веса не заменялись заглушками.

## Фактически выполнено и прошло

1. `python3 -m unittest discover -s research/agent_pilot -p 'test_*.py' -v`: **14 tests, OK**.
2. `python3 research/agent_pilot/verify_cartpole_reference.py research/agent_pilot/models/cartpole_pretrained.nlw --episodes 100`:
   - `episodes=100`, `solved_500=100`, `steps=50000`, `min_episode=500`, `max_episode=500`;
   - `parameter_count=4610`, `golden_vectors=5`;
   - `min_action_margin=1.5299020041722598e-06`.
3. `python3 research/agent_pilot/compare_native_cartpole.py --compiler g++ --episodes 10`:
   ```json
   {"baseline_python_episodes":10,"compiler":"g++","episodes":10,"max_abs_cpp_vs_python":0.0,"oracle":"independent stdlib FP64","solved_500":10,"status":"PASS","steps":5000}
   ```
4. `python3 research/agent_pilot/compare_native_cartpole.py --compiler g++ --episodes 100`:
   ```json
   {"baseline_python_episodes":100,"compiler":"g++","episodes":100,"max_abs_cpp_vs_python":0.0,"oracle":"independent stdlib FP64","solved_500":100,"status":"PASS","steps":50000}
   ```
   Сверены все состояния, два выхода режима AUTO, два выхода FULL и действие на каждом шаге.
5. Оригинальный `research/agent_pilot/agent_cartpole.cpp` собран:
   ```
   g++ -std=c++17 -O2 -I language/runtime research/agent_pilot/agent_cartpole.cpp -o /tmp/agent_cartpole
   (cd research/agent_pilot/models && /tmp/agent_cartpole cartpole_pretrained.nlw 100)
   ```
   Единственная строка отчёта:
   ```
   PILOT_SUCCESS episodes=100 steps=50000 avg_episode=500.000000000 min_episode=500 max_episode=500 disagreements=0 max_abs_error=0.000000000 auto_us_per_step=7.435334000 full_us_per_step=7.389756280 auto_delta=0 auto_full=50000
   ```
   Времена единичного запуска **не являются доказательством скорости** или преимуществ языка.

## Границы результата

Это реальное выполнение C++ runtime с оригинальными весами на Linux x86-64. Оно подтверждает корректность данного предобученного inference-сценария, но не само выполнение компилятора `.nir → C++`, не ARM64, не Android, не мобильную производительность, не обучение и не поддержку LLM.

**AUTO НЕ даёт прироста в данном сценарии:** `auto_delta=0` и `auto_full=50000`; количество весов 4610 меньше консервативного порога `full_cost()<65536` для последовательной сети. Для доказательства ускорения нужны другие workloads и сравнение с релевантным baseline на физическом устройстве.

Ветка `main` и архитектура языка не менялись. Никакие GitHub Actions не запускались.
