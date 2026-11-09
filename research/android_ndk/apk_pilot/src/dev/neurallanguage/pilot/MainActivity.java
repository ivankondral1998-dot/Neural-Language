package dev.neurallanguage.pilot;
import android.app.Activity;
import android.os.Bundle;
import android.graphics.Color;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import java.io.File;
import java.io.InputStream;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;

public final class MainActivity extends Activity {
    static { System.loadLibrary("neural_pilot"); }
    private static native String nativeRun(String directory, int kind, int episodes);
    private TextView output;
    private Button ten, hundred, training;
    private volatile boolean busy = false;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        column.setPadding(22, 22, 22, 22);
        column.setBackgroundColor(Color.rgb(13, 21, 34));

        TextView title = new TextView(this);
        title.setText("Neural Language | Android Pilot");
        title.setTextSize(21);
        title.setTextColor(Color.WHITE);
        column.addView(title);

        TextView note = new TextView(this);
        note.setText("Офлайн-тест настоящего C++-ядра и .nir-программ. Без облака.");
        note.setTextColor(Color.rgb(194, 211, 228));
        note.setPadding(0, 12, 0, 14);
        column.addView(note);

        ten = button("CartPole — 10 эпизодов", column, 1, 10);
        hundred = button("CartPole — 100 эпизодов", column, 1, 100);
        training = button(".nir: обучение → сохранение → загрузка", column, 2, 0);

        output = new TextView(this);
        output.setTextColor(Color.rgb(221, 238, 255));
        output.setTextSize(13);
        output.setTextIsSelectable(true);
        output.setPadding(0, 20, 0, 20);
        output.setText("Нажми проверку. Результат сохранится во внутренней памяти приложения.");
        ScrollView scroll = new ScrollView(this);
        scroll.addView(output);
        column.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));
        setContentView(column);
    }

    private Button button(String text, LinearLayout parent, final int kind, final int episodes) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        parent.addView(b);
        b.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { start(kind, episodes); }
        });
        return b;
    }

    private void setBusy(boolean enabled) {
        ten.setEnabled(!enabled);
        hundred.setEnabled(!enabled);
        training.setEnabled(!enabled);
    }

    private static String exception(Throwable t) {
        StringWriter sw = new StringWriter();
        t.printStackTrace(new PrintWriter(sw));
        return sw.toString();
    }

    private void copyModel(File run) throws Exception {
        File target = new File(run, "cartpole_pretrained.nlw");
        try (InputStream in = getAssets().open("cartpole_pretrained.nlw");
             FileOutputStream out = new FileOutputStream(target)) {
            byte[] b = new byte[16384];
            int n;
            while ((n = in.read(b)) != -1) out.write(b, 0, n);
            out.getFD().sync();
        }
    }

    private void start(final int kind, final int episodes) {
        if (busy) return;
        busy = true;
        setBusy(true);
        output.setText("Выполняется локальный тест…");
        new Thread(new Runnable() {
            @Override public void run() {
                String result;
                try {
                    File dir = new File(getFilesDir(), "run_" + System.currentTimeMillis() + "_" + System.nanoTime());
                    if (!dir.mkdirs()) throw new IllegalStateException("Cannot create test directory");
                    copyModel(dir);
                    result = nativeRun(dir.getAbsolutePath(), kind, episodes);
                    try (FileOutputStream out = new FileOutputStream(new File(dir,"test_result.txt"))) {
                        out.write(result.getBytes("UTF-8"));
                        out.getFD().sync();
                    }
                    result += "\n\nЛоги: " + dir.getAbsolutePath();
                } catch (Throwable t) {
                    result = "TEST FAILED\n" + exception(t);
                }
                final String done = result;
                runOnUiThread(new Runnable() {
                    @Override public void run() {
                        output.setText(done);
                        busy = false;
                        setBusy(false);
                    }
                });
            }
        }, "NeuralLanguagePilotTest").start();
    }
}