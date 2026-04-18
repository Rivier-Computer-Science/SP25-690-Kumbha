# SP25-690-Kumbha
Learning When NOT to Predict: Selective Classification with Reject Option using CNN and Vision Transformers for Reliable Image Recognition
 

Problem Statement and Motivation: In many deep learning systems, models are forced to make predictions even when they are uncertain, which can lead to highly confident but incorrect outputs. This behavior is problematic in real-world applications where incorrect predictions may have serious consequences. Most existing approaches focus only on improving accuracy, without considering whether the model should abstain from making a prediction. This project focuses on selective classification, where the model is allowed to reject uncertain inputs instead of making unreliable predictions. The goal is to improve the reliability of deep learning systems by enabling them to recognize their own limitations and avoid making decisions when confidence is low.

 

Task Definition (Inputs and Outputs): The input to the system will be an image that may include noise, blur, or other real-world distortions. The output will either be a predicted class label or a reject decision when the model determines that the input is too uncertain. In addition, the model will produce a confidence score used to decide whether to accept or reject the prediction. The goal is to maximize classification accuracy on accepted samples while minimizing incorrect predictions. Success will be measured using accuracy at different coverage levels, risk-coverage curves, and Expected Calibration Error. A meaningful result would be a model that achieves lower risk than a baseline at the same coverage level.

 

Data Description: The project will use a standard dataset such as CIFAR-10. To simulate real-world uncertainty, additional corrupted versions of the dataset will be generated using techniques such as Gaussian noise, blur, and brightness variations. The dataset will include both clean and noisy samples to evaluate how well the model handles varying levels of difficulty and uncertainty.

 

Method and Model Design: The project will implement two main architectures: a convolutional neural network (CNN) and a Vision Transformer (ViT). Both models will be extended to support selective classification by incorporating a confidence-based rejection mechanism. The models will output class probabilities, and a threshold will be applied to determine whether to accept or reject a prediction. Training may include regularization techniques such as dropout to improve uncertainty estimation. The comparison between CNN and transformer architectures will help analyze how model design affects the ability to identify uncertain inputs.

 

Baseline and Comparison Plan: A standard CNN without a reject option will be used as the primary baseline, where the model is forced to make predictions on all inputs. The proposed models (CNN with reject option and Vision Transformer with reject option) will be compared against this baseline. Additional comparisons will evaluate performance under clean versus noisy conditions. Controlled experiments will also be conducted by varying the rejection threshold to study the trade-off between coverage and risk.

 

Experimental Setup: The dataset will be divided into training, validation, and test sets. Models will be trained using an optimizer such as Adam with standard hyperparameters. During training and evaluation, metrics such as accuracy, coverage, risk, and Expected Calibration Error will be tracked. Experiments will include multiple runs to ensure stability. Performance will be analyzed across different rejection thresholds to generate risk-coverage curves.

 

Expected Results and Evaluation: It is expected that the baseline model will achieve high overall accuracy but will make confident errors on difficult inputs. The selective models should reduce these errors by rejecting uncertain predictions, leading to lower risk at comparable coverage levels. The Vision Transformer may show improved performance in identifying uncertain inputs due to its global attention mechanism. Results will be presented using tables and plots, including risk-coverage curves and reliability diagrams.

 

Failure Analysis and Limitations: The model may reject too many inputs, reducing coverage and limiting usability in practice. It may also fail to reject certain difficult inputs if confidence estimation is imperfect. The choice of rejection threshold may significantly impact performance and require careful tuning. These limitations will be analyzed by examining rejected samples, incorrect predictions, and performance under different thresholds and noise levels.

 

Ethics, Limitations, and Responsible Use: While selective classification improves reliability, it does not eliminate errors. Rejected predictions may require human intervention, which introduces additional system complexity. There is also a risk that the model may perform unevenly across different types of inputs, leading to biased rejection behavior. This system should not be used as a fully autonomous decision-making tool in critical applications without human oversight. Proper evaluation and transparency in rejection decisions are important for responsible use.

 

Feasibility and Scope: This project is feasible within the given timeframe because it uses standard datasets and well-established architectures. Both CNN and Vision Transformer models can be implemented using existing deep learning libraries and trained on platforms such as Google Colab. The selective classification mechanism can be implemented with relatively simple modifications, making the project manageable while still allowing for meaningful experimentation and analysis.
