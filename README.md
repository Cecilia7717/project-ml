# Final Project Report: [Project Title]

**Course**: CS383 - Introduction to Machine Learning  
**Semester**: Fall 2024  
**Team Members**: Amy Li, Cecilia Chen
**Instructor**: Adam Poliak

---

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Problem Statement](#problem-statement)
4. [Related Work](#related-work)
5. [Data Description](#data-description)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Discussion](#discussion)
9. [Conclusion and Future Work](#conclusion-and-future-work)
10. [References](#references)

---

## Abstract
Provide a brief summary of your project, including the problem tackled, the methodology used, and the key findings. This section should be concise and no more than 150-200 words.

---

## Introduction
Introduce the problem or question your project addresses. Explain its significance and relevance to machine learning. Include a brief overview of your approach and the objectives of the project.

---

## Problem Statement
Clearly define the problem you aimed to solve or the research question you sought to answer. Include any hypotheses you formulated and the scope of your project.

---

## Related Work
Summarize prior research or existing methods related to your project. Include citations or links to relevant papers, tools, or datasets. Discuss how your work builds upon or differs from these efforts.

---

## Data Description
Describe the dataset(s) you used, including:
- **Source(s)**: Where the data came from (e.g., Kaggle, UCI ML Repository, custom dataset).
- **Size and Format**: Number of rows, features, and data types.
- **Preprocessing**: Steps taken to clean or transform the data, including handling missing values or feature engineering.

---

## Methodology
Outline your approach, including:
1. The algorithms or models used (e.g., linear regression, neural networks, etc.).
2. Details of the training process (e.g., train-test splits, cross-validation).
3. Any hyperparameter tuning performed.
4. Tools and libraries employed (e.g., scikit-learn, PyTorch).

---

## Results
Present the results of your experiments, including:
- Key metrics (e.g., accuracy, precision, recall, F1 score, etc.).
- Comparisons between models or baselines.
- Visualizations (e.g., plots, confusion matrices).

### Hyperparameter tuning

Here we will include the results for hyperparameter tuning for both AdaBoost and Random Forest. We will use validation curve with both training accuracy and validation accuracy to show the sensitivity between changes in AdaBoost and Random Forest's accuracy with changes in hyperparameters of the model.

#### AdaBoost

**n_estimators**

The graph below shows the accuracy of the model on both the training and validation sets as the number of estimators increases. We varies n_estimators from 0 to 150. The training accuracy generally increases with more estimators at early stage, while the validation accuracy reaches a plateau and starts to fluctuate. This suggests that increasing the number of estimators beyond a certain point may lead to overfitting, as the model becomes too complex and starts to fit the noise in the training data. We suggests that setting n_estimators to be around 12 might be best. However, this might be influenced by the fact that we only use 1000 samples from the dataset.

![Alt text](https://github.com/Cecilia7717/project-ml/blob/main/Validation%20Curve%20for%20AdaBoost%20(n_estimators).png)

**learning rate**

The plot shows the validation curve for an AdaBoost model as the learning rate increase from 0.1 to 1.

The blue line represents the training accuracy, while the dashed orange line represents the validation accuracy. As the learning rate increases, the training accuracy generally improves, but the validation accuracy initially increases, reaches a peak around a learning rate of 0.53, and then starts to decline.
This pattern suggests that a learning rate around 0.52 might be a good choice for this model. A higher learning rate can lead to overfitting, where the model becomes too sensitive to the training data and performs poorly on unseen data.

![Alt text](https://github.com/Cecilia7717/project-ml/blob/main/Validation%20Curve%20for%20AdaBoost%20(learning_rate_1).png)

#### Random Forest
**n_estimators**

The plot illustrates the validation curve for a Random Forest model, showcasing the impact of the number of n_estimators on both training and validation accuracy. As the number of n_estimators increases, the training accuracy steadily improves, while the validation accuracy initially increases and then stabilizes around a value of 0.88. This suggests that increasing the number of estimators beyond a certain point (around 50) provides diminishing returns in terms of validation accuracy. Therefore, a model with approximately 45 estimators might be a good balance between model complexity and generalization performance.

![Alt text](https://github.com/Cecilia7717/project-ml/blob/main/Validation%20Curve%20for%20Random%20Forest%20(n_estimators).png)

**min_samples_split**

The plot displays the validation curve for a Random Forest model, focusing on the min_samples_split hyperparameter. As min_samples_split increases, the training accuracy decreases while the validation accuracy initially improves but then fluctuates. A min_samples_split value around 12 seems to offer a good balance between overfitting and underfitting, but further tuning might be necessary.

![Alt text](https://github.com/Cecilia7717/project-ml/blob/main/Validation%20Curve%20for%20Random%20Forest%20(min_samples_split).png)

**max_depth**

The figure shows the validation curve of the random forest model, illustrating the effect of max_depth on the training and validation accuracy. As max_depth increases, the training accuracy steadily increases and peaks around 20. However, the validation accuracy initially rises, peaks around the same max_depth value, and then begins to decline. This suggests a potential overfitting problem, where the model becomes too complex and begins to memorize the training data instead of generalizing to unseen data. max_depth values around 20 seem to be a good compromise between bias and variance.

![Alt text](https://github.com/Cecilia7717/project-ml/blob/main/Validation%20Curve%20for%20Random%20Forest%20(max_depth).png)

**criterion**

The plot illustrates the validation curve for a Random Forest model, comparing Gini impurity and Log Loss as splitting criteria. While Log Loss generally achieves higher training accuracy, Gini impurity often leads to better validation accuracy, suggesting a potential trade-off between model complexity and generalization.

![Alt text](https://github.com/Cecilia7717/project-ml/blob/main/Validation%20Curve%20for%20Random%20Forest%20(criterion).png)


**Example:**

| Model          | Accuracy | Precision | Recall | F1 Score |
|-----------------|----------|-----------|--------|----------|
| Logistic Reg.   | 0.85     | 0.84      | 0.83   | 0.84     |
| Random Forest   | 0.90     | 0.89      | 0.88   | 0.89     |

---

## Discussion
Interpret your results:

- What worked well?
- What challenges or limitations did you encounter?
- How do the results address your problem statement?

---

## Conclusion and Future Work

Summarize the key findings and discuss potential extensions of your work. What would you do differently with more time or resources?

---

## References

Include any citations for datasets, tools, libraries, or papers used in your project. Use a consistent citation format.
