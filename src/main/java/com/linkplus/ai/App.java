package com.linkplus.ai;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class App {
    public static void main(String[] args) throws Exception {
        System.out.println("1. Po ngarkoj dataset-in...");
        Instances data = DataLoader.load("src/main/resources/SMSSpamCollection");
        System.out.println("Data instances: " + data.numInstances());
// i kem bo keto printa per me kuptu errorin e mosprocesimit
        System.out.println("2. Po bej preprocessing...");
        Instances processed = Preprocessor.apply(data);
        System.out.println("Processed instances: " + processed.numInstances());

        System.out.println("3. Po trajnoj modelin...");
        NaiveBayes nb = Trainer.trainNaiveBayes(processed);

        System.out.println("4. Po bej evaluimin...");
        Evaluator.evaluate(nb, processed);

        System.out.println("5. Po testoj predikimin...");
        String msg = "Congratulations! You've won a free ticket, claim now!";
        String result = Predictor.predict(msg, nb, processed);
        System.out.println("Mesazhi: " + msg);
        System.out.println("Predikimi: " + result);
    }
}
