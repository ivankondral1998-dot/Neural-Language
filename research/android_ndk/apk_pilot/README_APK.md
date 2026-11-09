# Neural Language Android Pilot 0.1.0 — инструкция и приёмка

**Тип:** диагностический APK, не законченный ИИ-агент и не промышленный продукт.
**Статус:** BUILD/SIGNATURE/STRUCTURE PASS, **реальное исполнение на Android ещё не проверялось**.

## Скачать и установить

Приватный GitHub, исследовательская ветка `research/android-pilot-apk-20261109`:

[Neural Language Pilot APK](https://github.com/ivankondral1998-dot/Neural-Language/raw/research/android-pilot-apk-20261109/research/android_ndk/artifacts/NEURAL_LANGUAGE_PILOT_v0.1.0_ANDROID_ARM64_DEBUG.apk)

Открыть ссылку на ARM64-телефоне под аккаунтом GitHub с доступом к репозиторию.
Это APK, **не ZIP и не программа для ADB**. Пакет `dev.neurallanguage.pilot`,
версия `0.1.0`, `minSdk=26`, `targetSdk=34`. Root, GMS, доступ в Интернет и
разрешения к личным данным не нужны. При установке через браузер может
понадобиться обычное разрешение Android на установку из этого источника.
Подписано отдельным исследовательским debug-ключом, с другими приложениями
Neural Language/UNO/Jarvis не конфликтует.

## Что проверять на устройстве

При запуске доступны три кнопки:

- **CartPole — 10 эпизодов:** запускает оригинальный C++ CartPole agent pilot с
  реальными весами Neural Language. Ожидается `PASS`,
  `PILOT_SUCCESS episodes=10 steps=5000`, `disagreements=0`.
- **CartPole — 100 эпизодов:** такой же тест на 100/50000 шагов, несколько дольше.
  Ожидается `PASS`, `PILOT_SUCCESS episodes=100 steps=50000`,
  `disagreements=0`. Результат на реальном телефоне может отличаться;
  заявлять успех до испытания нельзя.
- **.nir: обучение → сохранение → загрузка:** выполняет скомпилированные
  `integrated.nir` и `inference.nir` в одной JNI-библиотеке, сохраняет
  checkpoint в приватной папке приложения и проверяет `PREDICT ... 0.3`.

Все действия офлайн. Экран показывает нативный журнал; копия результатов
сохраняется внутри private storage в отдельных каталогах `run_...`.
Приложение не удаляет результаты предыдущих испытаний.

## Что действительно уже выполнено

- Официальный Android NDK r30 / Clang 21: C++17 library `libneural_pilot.so`,
  ARM64, `-O2`, статическая C++ standard library, `llvm-strip`: PASS.
- Реальный `.nir` компилятор сгенерировал C++ для `pretrained_cartpole.nir`,
  `integrated.nir`, `inference.nir`; они встроены в JNI вместе с оригинальным
  `agent_cartpole.cpp`.
- Java 8 `MainActivity`, D8 `classes.dex`, Android AAPT2, zipalign: PASS.
- apksigner: подпись APK v2 и v3 **PASS**.
- `aapt2 dump badging`: package, versionCode=1, versionName=0.1.0,
  minSdk=26, targetSdk=34, `arm64-v8a` — PASS.
- `llvm-readelf`: экспорт JNI-функции
  `Java_dev_neurallanguage_pilot_MainActivity_nativeRun` обнаружен.
- APK размер **1 184 301 байт**; SHA-256:
  `b82db41ec6099eafea0903b3c9d1a61bae395df8a8ec5be1dd5e6182b854a508`.
- Сертификат debug-подписи SHA-256:
  `d7a8c00835ec5aa9d8d35b94b22a986ac189ccfd4e09c6d8e082c6e4d35a8e1d`.

## Ограничения

- **Нет проверки на физическом Android и реальных замеров CPU/энергии/RSS.**
- Внутри нет обучения PPO или LLM; весовая модель заранее обучена.
- На CartPole режим AUTO использует полный пересчёт; ускорение языка не доказано.
- Временный debug-ключ первоначальной сборки не сохранён в исходниках.
  Повторно подписанный другим ключом APK нельзя поставить поверх этой
  версии: потребуется удалить именно `dev.neurallanguage.pilot` либо
  получить обновление с тем же ключом. Не использовать подпись debug-APK для
  публичного релиза.
- Для воспроизводимой сборки с новым или собственным debug-ключом есть
  `build_debug_apk.sh`. Он принимает при необходимости путь к имеющемуся
  keystore первым аргументом и не удаляет исходники.

## Отдельные ветки

Никакого слияния с `main`, UNO, Jarvis или отдельным мобильным runtime не было.
GitHub Actions не запускались. Оригинальные исходники и веса сохранены.
