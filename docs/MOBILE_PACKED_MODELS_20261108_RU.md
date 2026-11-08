# Neural Language — компактные модели и JNI (NLMOBIN1/1)

**Исследовательский этап:** мобильная часть языка, ветка `feature/mobile-inference-runtime-20261108`.
Это реализация для локального инференса, а не самостоятельный завершённый язык или Android-релиз.

## Выходной файл

Хостовый `.nir` компилируется и обучается старым компилятором Neural Language. Его команда
`save_model "trained.nlw"` по-прежнему сохраняет доверенный текстовый `NLMODEL2`. Текстовый файл
переводится в переносимый, бинарный `NLMOBIN1` на хосте (Python 3, только stdlib):

    python3 language/mobile_pack.py trained.nlw -o trained.nlmb

Конвертер никогда не переписывает существующий целевой файл: применяется `O_EXCL`.
Для файлообмена используется новый `.nlmb`. Оригинальный `.nlw` остаётся без изменений.

## NLMOBIN1 — фиксированный контракт версии 1

Все целочисленные поля little-endian, вещественные — IEEE-754 binary64 little-endian.
Заголовок строго 40 байт:

    magic[8] = ASCII "NLMOBIN1"
    version: u32 = 1
    input_width: u32
    layer_count: u32
    total_weights: u32
    payload_bytes: u64
    payload_crc32: u32 (CRC32/ISO-HDLC, как zlib.crc32)
    reserved: u32 = 0

Следом подряд для каждого последовательного полносвязного слоя:

    input_width: u32
    output_width: u32
    activation: u32 (0=linear, 1=relu, 2=sigmoid)
    storage: u32 (0=FP64, 1=INT8 scale=1/127)
    bias: output_width × FP64
    weights: output_width*input_width × (FP64 либо signed INT8)
    masks: ceil(weights/8) байт, по 8 флагов в байте, младший бит первым

Последний неполный байт масок должен иметь нулевые старшие биты. Имена слоёв в бинарный
формат не входят — они не нужны для инференса, а графовые модели пока не поддерживаются.
Поля `mode`, `snapshots`, исходный `.nir` и состояние обучения также не сохраняются:
`NLMOBIN1` — **только inference checkpoint**. Восстановить обучаемое состояние по нему нельзя.

Декодер валидирует целостность CRC32, строгий размер файла, границы весов, формы,
числа и маски. CRC32 обнаруживает случайное повреждение, но **не является криптографической
подписью** и не превращает модель в безопасный недоверенный ввод. Оригинальные `.nlw` также
остаются читаемыми. Статический tensor budget — не измерение RSS.

## CLI на Linux

    c++ -std=c++17 -O2 -Wall -Wextra -Werror language/mobile_infer.cpp -o mobile_infer
    ./mobile_infer --model trained.nlmb --input 0.25,0.75 --stats

## Java/Android интеграция

Исходники: `language/android/neural_jni.cpp`, `language/android/CMakeLists.txt`,
`language/android/src/main/java/com/neurallanguage/mobile/NeuralMobile.java`.
Программный контракт Java:

    try (NeuralMobile model = new NeuralMobile(pathToPrivateCheckpoint, 64L*1024*1024)) {
        double[] result = model.predict(new double[] {0.25, 0.75});
    }

`pathToPrivateCheckpoint` — реальный путь к читаемому локальному файлу, не URI.
Для Android необходим CMake, включённый в Gradle externalNativeBuild, установленный NDK,
JNI-библиотека `libneuralmobile.so` внутри APK/AAB и копирование модели в приватные файлы
приложения. **Готового Android-пакета в этом этапе нет.**

При проверке на Linux/x86-64: сборка `cmake`, `g++` и `clang++`; Java 21 + JNI smoke;
10 групп Python unittest (200 сетей в двух независимых раундах по 100), сверка с независимым
Python-эталоном, ошибки целостности и no-overwrite. Также проведена проба ASan/UBSan на
валидном и повреждённом файлах. Android/ARM64/энергопотребление/физическая RAM ещё не проверены.

## Измеренное сокращение размера

FP64 и INT8 значения сохраняют арифметическое значение весов, включая маски и biases.
На детерминированной сети 128→256 с 32 768 INT8-весами:

- текстовый NLMODEL2: **730 190 байт**;
- NLMOBIN1: **38 968 байт**;
- сокращение: **94,66%** (только файл, на Linux).

На очень маленьких сетях бинарный заголовок может сделать файл больше — в примере из двух
весов он занимает 81 байт против 48 байт текстового оригинала. Никаких доказательств
ускорения/экономии RAM на Android данный тест не предоставляет.
