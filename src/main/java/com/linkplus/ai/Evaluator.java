package com.linkplus.ai;



import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class Evaluator {
    public static void evaluate(Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        System.out.println("\n===== Rezultatet e Modelit =====");
        System.out.println(eval.toSummaryString());
        System.out.println("Precision (spam): " + eval.precision(1));
        System.out.println("Recall (spam): " + eval.recall(1));
        System.out.println("F1 (spam): " + eval.fMeasure(1));
        System.out.println("\nConfusion Matrix:");
        System.out.println(eval.toMatrixString());
    }
}
