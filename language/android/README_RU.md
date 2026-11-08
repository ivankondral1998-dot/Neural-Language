# Neural Language — Java/JNI мост для Android

Нативный код требует C++17, NDK, JNI и локального файла модели NLMODEL2/NLMOBIN1.
В `CMakeLists.txt` уже задана библиотека `neuralmobile`. Исходник Java находится в
`src/main/java/com/neurallanguage/mobile/NeuralMobile.java`.

Минимальное подключение в существующий Android Gradle-модуль:

    android {
      defaultConfig { externalNativeBuild { cmake { cppFlags "-std=c++17" } } }
      externalNativeBuild { cmake { path "<путь>/language/android/CMakeLists.txt" } }
    }

Это пример DSL, не готовый build.gradle и не автоматически проверенная конфигурация Android.
Путь для модели должен быть обычным файлом (например, внутри getFilesDir()),
а не `content://` URI. Вызов `NeuralMobile.predict(double[])` синхронный;
в Android UI его надо выполнять вне главного потока.

Проверка Java/JNI без Android SDK на хосте Linux:

    JDK=/usr/lib/jvm/default-java
    g++ -std=c++17 -O2 -fPIC -shared -DNL_MOBILE_LIBRARY_ONLY \
      -I "$JDK/include" -I "$JDK/include/linux" \
      language/mobile_infer.cpp language/android/neural_jni.cpp -o libneuralmobile.so
    mkdir -p build_java
    javac -d build_java language/android/src/main/java/com/neurallanguage/mobile/NeuralMobile.java \
      language/android/tests/JniSmoke.java
    java -Djava.library.path=. -cp build_java \
      com.neurallanguage.mobile.JniSmoke language/examples/mobile_demo.nlmb

Наблюдавшийся результат: `JNI_SMOKE_PASS: 0.30000000000000004`.
На реальном телефоне не проверено. Не создавать публичного доступа к локальным моделям.
