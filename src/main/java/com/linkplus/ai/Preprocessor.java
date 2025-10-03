package com.linkplus.ai;



import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Preprocessor {
    public static Instances apply(Instances data) throws Exception {
        StringToWordVector filter = new StringToWordVector();
        filter.setTFTransform(true);
        filter.setIDFTransform(true);
        filter.setLowerCaseTokens(true);
        filter.setInputFormat(data);

        Instances newData = Filter.useFilter(data, filter);
        Predictor.setFilter(filter);
        return newData;
    }
}
