package onnx_examples;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Map;

public class SentimentAnalysisWithoutTokenizer {
    public static void main(String[] args) throws OrtException {
        String onnxModelPath = ".onnx/distilbert-base-uncased-finetuned-sst-2-english/model.onnx";
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(onnxModelPath, sessionOptions);

        long[][] inputIdsData = {{101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172,
                2607, 2026, 2878, 2166, 1012, 102},
                {101, 1045, 5223, 2023, 2061, 2172, 999, 102, 0, 0,
                        0, 0, 0, 0, 0, 0}};

        long[][] attentionMaskData = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}};

        // Perform inference
        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsData);
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMaskData)) {

            OrtSession.Result result = session.run(
                    Map.of("input_ids", inputIdsTensor,
                            "attention_mask", attentionMaskTensor),
                    session.getOutputNames()
            );

            // Get the inference results
            float[][] outputData = (float[][]) result.get(0).getValue();

            // Print the output
            for (float[] row : outputData) {
                float[] probabilities = softmax(row);
                for (float value : probabilities) {
                    System.out.print(value + " ");
                }
                System.out.println();
            }
        }
    }

    // Softmax function
    public static float[] softmax(float[] logits) {
        float maxLogit = logits[0];
        float sumExps = 0.0f;

        // Find the maximum logit for numerical stability
        for (float logit : logits) {
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        // Compute exponentials and sum
        float[] exps = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExps += exps[i];
        }

        // Normalize to get probabilities
        float[] softmax = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            softmax[i] = exps[i] / sumExps;
        }

        return softmax;
    }
}

