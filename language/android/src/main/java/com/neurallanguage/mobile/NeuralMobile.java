package com.neurallanguage.mobile;

/** Small offline Neural Language inference API. Requires libneuralmobile.so. */
public final class NeuralMobile implements AutoCloseable {
    static { System.loadLibrary("neuralmobile"); }
    private long nativeHandle;

    public NeuralMobile(String checkpointPath, long maxTensorBytes) {
        if (checkpointPath == null || checkpointPath.isEmpty() || maxTensorBytes <= 0) {
            throw new IllegalArgumentException("checkpoint path or memory bound invalid");
        }
        nativeHandle = nativeOpen(checkpointPath, maxTensorBytes);
    }

    public synchronized int inputSize() {
        requireOpen();
        return nativeInputSize(nativeHandle);
    }

    public synchronized int outputSize() {
        requireOpen();
        return nativeOutputSize(nativeHandle);
    }

    public synchronized double[] predict(double[] input) {
        requireOpen();
        if (input == null) throw new IllegalArgumentException("input cannot be null");
        return nativePredict(nativeHandle, input);
    }

    @Override public synchronized void close() {
        if (nativeHandle != 0L) {
            nativeClose(nativeHandle);
            nativeHandle = 0L;
        }
    }

    private void requireOpen() {
        if (nativeHandle == 0L) throw new IllegalStateException("model already closed");
    }

    private static native long nativeOpen(String path, long memoryLimit);
    private static native int nativeInputSize(long model);
    private static native int nativeOutputSize(long model);
    private static native double[] nativePredict(long model, double[] input);
    private static native void nativeClose(long model);
}
