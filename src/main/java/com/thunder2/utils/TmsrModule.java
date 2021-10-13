package com.thunder2.utils;


import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.IValue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class TmsrModule {
    private Module module;
    private List<String> CLASSES = Arrays.asList(
            "Leiji", "Fanji", "Raoji", "Waipo", "Shanhuo", "Shigongpengxian", "Yiwuduanlu", "Binghai",
            "Bingshan", "Fubinguozai", "Tuobintiaoyue", "Wudong", "Fengpian", "Niaohai", "Wushan", "Qita",
            "Feileiji"
    );

    /**
     * @brief
     * @param modelPath .pt
     */
    public TmsrModule(String modelPath) {
        this.module = Module.load(modelPath);
    }

    /**
     * @brief
     * @param freq
     * @param input
     * @return
     * - `(String)listObject.get(0)`
     * - `(float)listObject.get(1)`
     */
    public List<Object> infer(int freq, float[] input) {
        if (freq != 5e5 && freq != 1e6 && freq != 2e6) {
            System.out.print("Input frequency inconsistent with training data.");
            System.exit(1);
        }

        Tensor inTensor = Tensor.fromBlob(this.samplePreprocess(freq, input), new long[] { 1, 1, 600 });
        Tensor outTensor = this.module.forward(IValue.from(inTensor)).toTensor();
        float[] output = outTensor.getDataAsFloatArray();

        // Performs Softmax.
        float outMax = output[0];
        for (float out : output) {
            if (out > outMax) {
                outMax = out;
            }
        }
        float expSum = 0;
        for (int i = 0; i < output.length; i++) {
            output[i] = (float)Math.exp((double)(output[i] - outMax));
            expSum += output[i];
        }
        for (int i = 0; i < output.length; i++) {
            output[i] /= expSum;
        }
        System.out.println(Arrays.toString(output));

        // Performs Argmax.
        float logitMax = output[0];
        int logitMaxIdx = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > logitMax) {
                logitMax = output[i];
                logitMaxIdx = i;
            }
        }
        List<Object> listObject = new ArrayList<Object>();
        listObject.add(this.CLASSES.get(logitMaxIdx));
        listObject.add(logitMax);
        return listObject;
    }

    private float[] samplePreprocess(int srcFreq, float[] input) {
        int dstFreq = (int)5e5;
        int numSamples = (int)(2400 * dstFreq / 2e6);
        int stride = srcFreq / dstFreq;
        float[] output = new float[input.length / stride];

        for (int i = 0; i < output.length; i++) {
            output[i] = input[i * stride];
        }

        if (output.length > numSamples) {
            int preambleIdx = 0;
            for (; preambleIdx < output.length; preambleIdx++) {
                if (Math.abs(output[preambleIdx]) > 40) break;
            }
            preambleIdx = preambleIdx != output.length ? preambleIdx : 0;
            int preamble200 = (int)(200 * dstFreq / 2e6);
            if (preambleIdx > preamble200) {
                preambleIdx = Math.min(preambleIdx - preamble200, output.length - numSamples);
                output = Arrays.copyOfRange(output, preambleIdx, output.length);
            }
            if (output.length > numSamples) {
                output = Arrays.copyOf(output, numSamples);
            }
        }

        if (output.length < numSamples) {
            output = Arrays.copyOf(output, numSamples);
        }
        return output;
    }
}
