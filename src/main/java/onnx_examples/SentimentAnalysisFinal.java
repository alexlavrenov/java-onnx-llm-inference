package onnx_examples;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;

public class SentimentAnalysisFinal {
    public static void main(String[] args) throws OrtException, IOException {
        String onnxModelPath = ".onnx/distilbert-base-uncased-finetuned-sst-2-english/model.onnx";
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(onnxModelPath, sessionOptions);

        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("src/main/resources/tokenizer.json"));

        String[] texts = {
                "I've been waiting for a HuggingFace course my whole life.",
                "I hate this so much!"
        };

        Encoding[] encodings = tokenizer.batchEncode(texts);

        long[][] inputIdsData = Arrays.stream(encodings)
                .map(Encoding::getIds)
                .toArray(long[][]::new);
        long[][] attentionMaskData = Arrays.stream(encodings)
                .map(Encoding::getAttentionMask)
                .toArray(long[][]::new);

        // Perform inference
        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsData);
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMaskData)) {

            OrtSession.Result result = session.run(
                    Map.of("input_ids", inputIdsTensor,
                            "attention_mask", attentionMaskTensor),
                    session.getOutputNames()
            );

            float[][] outputData = (float[][]) result.get(0).getValue();

            for (float[] row : outputData) {
                float[] probabilities = softmax(row);
                for (float value : probabilities) {
                    System.out.print(value + " ");
                }
                System.out.println();
            }
        }
    }

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
