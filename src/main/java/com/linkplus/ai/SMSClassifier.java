package com.linkplus.ai;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class SMSClassifier {
    public static void main(String[] args) throws Exception {

        ArrayList<Attribute> attributes = new ArrayList<>();
        ArrayList<String> classVals = new ArrayList<>();
        classVals.add("ham");
        classVals.add("spam");
        attributes.add(new Attribute("class", classVals));
        attributes.add(new Attribute("message", (ArrayList<String>) null));

        Instances dataset = new Instances("SMSDataset", attributes, 0);
        dataset.setClassIndex(0);


        BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/SMSSpamCollection"));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split("\t", 2);
            if (parts.length < 2) continue;
            double[] vals = new double[dataset.numAttributes()];
            vals[0] = classVals.indexOf(parts[0]);
            vals[1] = dataset.attribute(1).addStringValue(parts[1]);
            dataset.add(new DenseInstance(1.0, vals));
        }
        reader.close();


        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataset);


        DenseInstance test = new DenseInstance(2);
        test.setDataset(dataset);
        test.setValue(dataset.attribute(1), dataset.attribute(1).addStringValue("Free entry! Claim your prize now!"));
        double result = nb.classifyInstance(test);

        System.out.println("SMS test: 'Free entry! Claim your prize now!'");
        System.out.println("Predikimi: " + classVals.get((int) result));
    }
}
