This is a project meant to demonstrate the effectiveness of an adaptive ensemble model on detecting DDoS attacks.

The model has four key innovative components:
• Base Model Ensemble: Four different machine learning algorithms are used (SVM, Random Forest, Logistic Regression, and XGBoost). Initially equal weights are given to each model and are increased and reduced for the best and least performing model respectively during training iterations.
• Pattern Evolution Tracking: The architecture consists of a memory system that contains known attack patterns and uses the cosine similarity to detect new attacks.
• Drift Detection: We keep track of model performance using the sliding window technique. When there is a drift, rates of adaptation are incremented automatically, and new patterns are added to the memory for future use.
• Explainability Framework: Feature importance score is determined for each prediction, giving insights into model decision making.

The model is trained on a dataset of over 100 000 records of network flows and tested on its effectiveness in a simulated environment. The results of model performance are shown below:

Performance and Resource Usage Metrics

Accuracy 99.98%
Precision 100%
Recall 99.96%
F1 score 99.98%
Memory 10.06MB
CPU Usage 105.20MB
