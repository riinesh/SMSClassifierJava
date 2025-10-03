package com.linkplus.ai;


import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class Trainer {
    public static NaiveBayes trainNaiveBayes(Instances data) throws Exception {
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);
        return nb;
    }
}

