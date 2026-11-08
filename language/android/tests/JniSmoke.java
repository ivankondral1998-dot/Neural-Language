package com.neurallanguage.mobile;

public final class JniSmoke {
    public static void main(String[] args) {
        if (args.length != 1) throw new AssertionError("model path missing");
        try (NeuralMobile model = new NeuralMobile(args[0], 4096)) {
            if(model.inputSize()!=2 || model.outputSize()!=1)throw new AssertionError("bad shape");
            double[] prediction=model.predict(new double[]{0.25,0.75});
            if(prediction.length!=1 || Math.abs(prediction[0]-0.3)>1e-12)
                throw new AssertionError("prediction mismatch");
            boolean negative=false;
            try {model.predict(new double[]{1.0});}
            catch (IllegalArgumentException expected) {negative=true;}
            if(!negative)throw new AssertionError("bad shape was not rejected");
            System.out.println("JNI_SMOKE_PASS: "+prediction[0]);
        }
    }
}
