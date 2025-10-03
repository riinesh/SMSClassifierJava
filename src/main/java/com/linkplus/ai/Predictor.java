package com.linkplus.ai;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Predictor {

    private static StringToWordVector filter;

    public static void setFilter(StringToWordVector f) {
        filter = f;
    }

    public static String predict(String message, Classifier model, Instances trainingData) throws Exception {
        Instance newInst = new DenseInstance(trainingData.numAttributes());
        newInst.setDataset(trainingData);

        Attribute textAtt = trainingData.attribute("text");
        newInst.setValue(textAtt, message);

        Instances instForPred = new Instances(trainingData, 0);
        instForPred.add(newInst);
        instForPred = Filter.useFilter(instForPred, filter);

        double predIndex = model.classifyInstance(instForPred.instance(0));
        return trainingData.classAttribute().value((int) predIndex);
    }
}
