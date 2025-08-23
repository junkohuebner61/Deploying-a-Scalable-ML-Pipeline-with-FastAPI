# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a Random Forest model that predicts if a person makes more than $50K a year. It uses census data from 1994.

## Intended Use

The model is for predicting income levels based on personal and work information. It can be used for research or social studies.

## Training Data

The model was trained on 80% of the 1994 U.S. Census data, which includes features like workclass, education, and race.

## Evaluation Data

The remaining 20% of the census data was used to test the model and check how well it works.

## Metrics

On the test data, the model got:

- Precision: 0.74  
- Recall: 0.64  
- F1-Score: 0.69  

We also checked performance on different groups in the data to find any differences.

## Ethical Considerations

The data reflects past biases, so the model might also be biased.

## Caveats and Recommendations

Regularly check model performance on different groups to detect issues and retrain with updated data when possible.