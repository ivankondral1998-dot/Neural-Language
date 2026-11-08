# NL-COMP-003

См. [результат](RESULT_003_RU.md), [benchmark](bench_observe.cpp) и тесты [C++](../../language/tests/observe_reuse_stress.cpp), [Python](../../language/tests/test_observe_reuse.py).

Проверка из корня репозитория: `python3 -m unittest -v language.tests.test_observe_reuse language.tests.test_observe_delta`.

Полная серия: `python3 -m unittest -q language.tests.test_language language.tests.test_observe_delta language.tests.test_observe_reuse language.tests.test_integration language.tests.test_hardening language.tests.test_quantization_fixes language.tests.test_verifier`.

Сравнение с OpenBLAS требует локально настроенных путей к совместимому BLAS. Отдельный архив владельца содержит регистрацию экспериментов, сырые CSV и логи. Android пока не проверен.
