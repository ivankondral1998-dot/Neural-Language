# Neural Language — мобильное нативное исполнение NLMODEL2

Статус: реализованный и проверенный на Linux x86_64 inference-компонент исследовательской ветки. Не является готовым Android APK или доказательством превосходства по скорости и памяти на телефоне.

## Назначение

Существующий язык Neural Language компилирует .nir и сохраняет обученные веса командой save_model в формате NLMODEL2. Новый language/mobile_infer.cpp загружает такой checkpoint и выполняет прямой проход без Python, SGD, снимков, BLAS и обучающего Model. Отдельный легковесный C++17 backend подходит для встраивания через Android NDK и JNI. Исходный синтаксис языка не заменяется.

Поддерживается: последовательные dense-сети до восьми слоёв, FP64 и INT8 хранение весов с фиксированным масштабом 1/127, linear/relu/sigmoid, маски связей, до 4 млн весов. Не поддерживаются графы NLMODEL3/DAG, обучение в runtime, произвольные тензоры, GPU/NPU и недоверенные checkpoint.

## Быстрый запуск

Из корня репозитория на Linux или другой платформе с C++17:

    c++ -std=c++17 -O2 -Wall -Wextra -Werror language/mobile_infer.cpp -o mobile_infer
    ./mobile_infer --model language/examples/mobile_demo.nlw --input 0.25,0.75 --stats

Ожидаемая демонстрационная проверка: PREDICT values=0.30000000000000004

Файл mobile_demo.nlw содержит аналитически заданные веса, не выдаётся за обученную модель. Вместо него можно передать trained.nlw, полученный из прежней программы Neural Language через save_model.

Входы идут через --input с разделителем запятой. --max-tensor-bytes N ограничивает сумму байтов полезных числовых массивов модели и двух рабочих векторов. Это НЕ предел RSS: объекты STL, аллокатор, файловые буферы, стек и ОС не включены. Лимит по умолчанию 256 MiB. Файлы должны быть доверенными.

## Нативная библиотека для Android

Пример команды с предварительно установленным Android NDK, НЕ испытанный в этой сессии:

    $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++ -std=c++17 -O2 -fexceptions -fPIC -shared -DNL_MOBILE_LIBRARY_ONLY language/mobile_infer.cpp -o libneuralmobile.so

Публичный C ABI описан в language/mobile_infer.h:
nlmobile_open, nlmobile_input_count, nlmobile_output_count, nlmobile_tensor_payload_bytes, nlmobile_predict, nlmobile_last_error, nlmobile_close.

В Android-приложении дополнительно необходима JNI-обвязка и упаковка .so. APK сейчас нет. На реальном Android/ARM64 код ещё не запускался.

## Проверки

    python3 -m unittest -v language.tests.test_mobile_infer

На Linux x86_64 выполнены пять групп PASS: сборка GCC и Clang, 100 детерминированных случайных сетей с независимым Python-оракулом для обоих компиляторов, FP64/INT8/маски/активации, неправильные checkpoint/входы/лимиты, встроенная сборка и C ABI через ctypes. Отдельно GCC собран обычный C клиент abi_smoke.c и успешно связан с .so.

Не проверены качество обучения на Android, энергопотребление, производительность ARM64, сопоставимые бенчмарки PyTorch/BLAS, самохостинг компилятора и полный набор прежних тестов в этой сессии.

## Сопутствующее исправление

В этой же изолированной ветке compiler.py получил защиту от перезаписи исходного файла выбранными путями --output (включая генерируемые .cpp и .nir.json) и исправление допустимых физических номеров строк при большом количестве комментариев. Для этого добавлены language/tests/test_cli_safety.py; пока эти тесты не перезапущены против полного компилятора в локальной среде.

main, DAG 0.4 и помощник 0.5 не менялись. Автоматических слияний, GitHub Actions и переноса кода из других проектов не было.
