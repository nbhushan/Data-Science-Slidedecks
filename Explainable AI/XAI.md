autoscale: true
slidenumbers: false
slidecount: true
list: alignment(left)
slide-transition: true

# [fit] EVERYTHING IS

# [fit] **`XAI`**

---

# [fit] THIS PRESENTATION IS ABOUT

^ nice to see you all here! Rather than going through the slides, I would like to have a discussion about the topics. This will be a high level overview of XAI, and rather than overwhelm you with all of XAI, I would like to focus on the most important parts. If you are curious to learn more, I will be happy to share some resources with you, and you will have time and resources in Y3 and Y4 to go deeper into the topics.

- taxonomy
- model (un)certainty
- gradCAMs
- the Nigerian prince

---

# [fit] THIS PRESENTATION IS ABOUT

^ nice to see you all here! Rather than going through the slides, I would like to have a discussion about the topics. This will be a high level overview of XAI, and rather than overwhelm you with all of XAI, I would like to focus on the most important parts. If you are curious to learn more, I will be happy to share some resources with you, and you will have time and resources in Y3 and Y4 to go deeper into the topics.

- taxonomy
- model (un)certainty
- gradCAMs
- the Nigerian prince

> thoughts about the presentation this morning?

---

# [fit] TAXONOMY OF XAI

^ There is always a trade-off between interpretability and accuracy. For example, linear models are very interpretable, but they are not very accurate. On the other hand, deep neural networks are very accurate, but they are not very interpretable. XAI is a field that tries to bridge this gap between accuracy and interpretability. Talk about taxonomy .Always pick a model depending on the task at hand. For example, if you are building a model to predict the price of a house, you should start with a linear model. If you are building a model to predict whether a patient has cancer, you should use a deep neural network and then use XAI to address the trade-off between interpretability and accuracy.

![right filtered fit](images/txt.png)

- **by default**
    1. linear models
    2. tree-based models

- **black boxes**
  - global vs local methods
  - model-specific vs model-agnostic

---

# [fit] TAXONOMY OF XAI

^ There is always a trade-off between interpretability and accuracy. For example, linear models are very interpretable, but they are not very accurate. On the other hand, deep neural networks are very accurate, but they are not very interpretable. XAI is a field that tries to bridge this gap between accuracy and interpretability. Talk about taxonomy .Always pick a model depending on the task at hand. For example, if you are building a model to predict the price of a house, you should start with a linear model. If you are building a model to predict whether a patient has cancer, you should use a deep neural network and then use XAI to address the trade-off between interpretability and accuracy.

![right filtered fit](images/txt.png)

- **by default**
    1. linear models
    2. tree-based models

- **black boxes**
  - global vs local methods
  - model-specific vs model-agnostic

> classify gradCAM according to the taxonomy

---

# [fit] TAXONOMY OF XAI

^ gradCAM is a technique invented to explicitly address one of the most important problems in deep learning: interpretability. It is a local, model-specific method (convolutional neural networks) to explain the predictions of a CNN.

![right fit](images/txt.png)

- **by default**
    1. linear models
    2. tree-based models

- **black boxes**
  - global vs **local** methods
  - **model-specific** vs model-agnostic

> gradCAM is a local, model-specific method to explain the predictions of a CNN

---

# [fit] MODEL (UN)CERTAINTY

^ Gamblers and Astronomers — the two very distinct groups of people who were actually responsible for the origination of probability theory. The former wanted to better maximize luck (minimize risk) while the latter was trying to have accurate observations from their rudimentary tools.

![filtered](images/dice.jpeg)

> “It is remarkable that science, which originated in the consideration of games of chance, should have become the most important object of human knowledge.”
-- Pierre Simon Laplace

---

# [fit] MODEL (UN)CERTAINTY I

^ The likelihood function can help us to quantify the uncertainty of our models. Aleatoric uncertainty or statistical uncertainty is the uncertainty that is inherent in the data. Sources of aleatoric uncertainty include measurement errors, noise, and randomness.

![right filtered](images/mnisttwo.png)

## **aleatoric uncertainty**

statistical in nature, refers to random variation

:bell: $$ P(Y|X;\theta)$$

---

# [fit] MODEL (UN)CERTAINTY I

^ The likelihood function can help us to quantify the uncertainty of our models. Aleatoric uncertainty or statistical uncertainty is the uncertainty that is inherent in the data. Sources of aleatoric uncertainty include measurement errors, noise, and randomness.

![right filtered](images/mnisttwo.png)

## **aleatoric uncertainty**

statistical in nature, refers to random variation

:bell: $$ P(Y|X;\theta)$$

### more data is not a solution

> data quality, model averaging, and prediction intervals help.

---

# [fit] MODEL (UN)CERTAINTY II

^ Epistemic uncertainty or human uncertainty is the uncertainty that is inherent in the model. Uncertainity in $$\theta$$ is the source of epistemic uncertainty. Any step we take to improve the precision with which we estimate $$\theta$$ will reduce epistemic uncertainty.

![left filtered](images/mnisttwo.png)

## **epistemic uncertainty**

human in nature, refers to ignorance  

:bell: $$ P(Y|X;\theta)$$

---

# [fit] MODEL (UN)CERTAINTY II

^ Epistemic uncertainty or human uncertainty is the uncertainty that is inherent in the model. uncertainty in $$\theta$$ is the source of epistemic uncertainty. Any step we take to improve the precision with which we estimate $$\theta$$ will reduce epistemic uncertainty.

![left filtered](images/mnisttwo.png)

## **epistemic uncertainty**

human in nature, refers to ignorance  

:bell: $$ P(Y|X;\theta)$$

### more data is a solution

> hyperparameter tuning, regularization etc. help

---

# [fit] HOW IS THIS USEFUL?

[.build-lists: true]

^ In practice, every model building exercise is an attempt towards reducing uncertainty. And remember that your model score is not always a good measure of the quality of your model.

![right fit filtered](images/mnisttwo.png)

if we train a classifier on digits [0,8], what happens when we run..?

- model.predict([2])
- model.predict([7])
- model.predict([9])

---

# [fit] HOW IS THIS USEFUL?

^ In practice, every model building exercise is an attempt towards reducing uncertainty. And remember that your model score is not always a good measure of the quality of your model.

![right fit filtered](images/mnisttwo.png)

if we train a classifier on digits [0,8], what happens when we run..?

- model.predict([2])
- model.predict([7])
- model.predict([9])

> :exclamation: **never forget that the models you build are probabilistic in nature**

---

# [fit] :musical_note: MIND YOUR STEP

^ chatgpt and uncertainty. this is a nice example of how an AI developer uses UX to help the user understand the uncertainty of the model.

![right fit filtered](images/chatgpt.png)

## even the best models can be wrong, and with horrible consequences

---

# [fit] :musical_note: MIND YOUR STEP

^ chatgpt and uncertainty. this is a nice example of how an AI developer uses UX to help the user understand the uncertainty of the model.

![right fit](images/chatgpt.png)

## even the best models can be wrong, and with horrible consequences

> remember the end user!


---

# [fit] EMBRACE THE UNCERTAINTY

^ quantifying the uncertainty of deep learning models is a hot research topic. This is because statistical approaches such as confidence intervals and prediction intervals are not yet widely used in the deep learning community. :warning: **neural networks tend to be overconfident when being completely wrong**

![right](https://www.youtube.com/watch?v=2HMPRXstSvQ&t=29s)

### Successful decisions are built on

- minimizing our ignorance
- accepting inherent randomness
- knowing the difference between the two.

> think self-driving cars, medical diagnosis, or exploding rockets

---

# [fit] :taxi:

^ would you get into a self-driven taxi? Now what if I tell you that the taxi has a 50% chance of killing you? And what if I can reduce that to 10%? Would you get into the taxi then? And what if you are with your family? Would you still get into the taxi?

> `may occasionally kill passengers`

---

# [fit] EXPLAINING

# **`DEEP NEURAL NETWORKS`**

---

# [fit] EXPLAINING DEEP NEURAL NETWORKS

^ talk about the distinction between the 3 methods.

[.column]

## feature visualization

> visualize the activations of a deep neural network

[.column]

## feature attribution

[.column]

## adversarial examples

---

# [fit] EXPLAINING DEEP NEURAL NETWORKS

^ talk about the distinction between the 3 methods.

[.column]

## feature visualization

> visualize the activations of a deep neural network

[.column]

## feature attribution

> visualize the features that contribute strongly to the activations of CNN layers for a given input image and class label.

[.column]

## adversarial examples

---

# [fit] EXPLAINING DEEP NEURAL NETWORKS

^ talk about the distinction between the 3 methods.

[.column]

## feature visualization

> visualize the activations of a deep neural network

[.column]

## feature attribution

> visualize the features that contribute strongly to the activations of CNN layers for a given input image and class label.

[.column]

## adversarial examples

> understand the robustness of a deep neural network by generating adversarial examples giving insight into the decision boundaries of the model.

---

# [fit] GRAD-CAM

[.build-lists: true]

^ explain gradcam in 3 levels

![left fit](images/softmax.png)

- grad-CAM is a technique used in deep learning to highlight which parts of an image were important in predicting its classification.

- grad-CAM works by computing the gradients of the output class score with respect to the feature maps of the last convolutional layer in the CNN.

- grad-CAM works by computing the gradients of the output class score with respect to the feature maps of the last convolutional layer in the CNN. These gradients are then used to compute a weight map for each feature map, which are then multiplied together and summed to produce the final heatmap [^1].

[^1]: Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

---

# [fit] TF_EXPLAIN

^ we <3 open source. I understand things faster when I see an implementation of a method, and then I read the paper, and then back to the code.

[.column]
[.code-highlight: all]

```python

!pip install tf_explain
!pip install opencv-python

#load libraries
import numpy as np
import tensorflow as tf
import PIL

#load GradCAM
from tf_explain.core.grad_cam import GradCAM

IMAGE_PATH = "./assets/images/cat.jpg" 
class_index = 281

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)

model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

#get model summary
model.summary()

#first create the input in a format that the explainer expects (a tuple)
input_img = (np.array([img]), None)

#initialize the explainer as an instance of the GradCAM object
explainer = GradCAM()

# Obtain explanations for your image using VGG 16 and GradCAM
grid = explainer.explain(input_img,
                         model,
                         class_index=class_index
                         )

#save the resulting image
explainer.save(grid, "./outputs/explain/", "grad_cam_cat.png")
```

[.column]

![fit](images/cat.jpg)

---

# [fit] TF_EXPLAIN

^ hides away all the abstraction and gives you a user friendly API to use the methods.

[.column]
[.code-highlight: 1, 8-10, 26-35]

```python

!pip install tf_explain
!pip install opencv-python

#load libraries
import numpy as np
import tensorflow as tf
import PIL

#load GradCAM
from tf_explain.core.grad_cam import GradCAM

IMAGE_PATH = "./assets/images/cat.jpg" 
class_index = 281

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)

model = tf.keras.applications.vgg16.VGG16(weights="imagenet", include_top=True)

#get model summary
model.summary()

#first create the input in a format that the explainer expects (a tuple)
input_img = (np.array([img]), None)

#initialize the explainer as an instance of the GradCAM object
explainer = GradCAM()

# Obtain explanations for your image using VGG 16 and GradCAM
grid = explainer.explain(input_img,
                         model,
                         class_index=class_index
                         )

#save the resulting image
explainer.save(grid, "./outputs/explain/", "grad_cam_cat.png")
```

[.column]

![fit](images/grad_cam_cat.png)

---

# [fit] SOURCE CODE

^ the nice thing about open source is that you can see how it works. please get into the habit of reading the source code of the libraries you use. it will help you understand how they work and how to use them. and maybe motivate you to contribute to the community yourself (bug fixes etc)

![left filtered](images/tf-explain_source.png)

:computer: [https://github.com/sicara/tf-explain](https://github.com/sicara/tf-explain)

- [x] gradCAM
- [x] smoothGrad
- [x] guided smoothgrad
- [x] integrated gradients
- [x] occlusion sensitivity
- [x] activations visualization

---

# [fit] SOURCE CODE

^ the nice thing about open source is that you can see how it works. please get into the habit of reading the source code of the libraries you use. it will help you understand how they work and how to use them. and maybe motivate you to contribute to the community yourself (bug fixes etc)

![left filtered](images/tf-explain_source.png)

:computer: [https://github.com/sicara/tf-explain](https://github.com/sicara/tf-explain)

- [x] gradCAM
- [x] smoothGrad
- [x] guided smoothgrad
- [x] integrated gradients
- [x] occlusion sensitivity
- [x] activations visualization

> this is an active area of research, so expect more methods to be added in the future

---

# [fit] THE NIGERIAN PRINCE

^ does anyone remember the nigerian prince scam?

![filtered fit](images/scam.jpg)

---

# [fit] MOVING BEYOND FEATURE ATTRIBUTION

^ the nigerian prince scam is a classic example of this, where the scammer intentionally introduces small perturbations in the email to fool the spam filter into predicting that the email is not spam

![left filtered ](images/scam.jpg)

- deep neural networks are notoriously sensitive to small perturbations in the input
- by intentionally introducing small perturbations in the input, we can fool the model into making a different prediction

---

# [fit] MOVING BEYOND FEATURE ATTRIBUTION

^ the nigerian prince scam is a classic example of this, where the scammer intentionally introduces small perturbations in the email to fool the spam filter into predicting that the email is not spam

![left filtered ](images/scam.jpg)

- deep neural networks are notoriously sensitive to small perturbations in the input
- by intentionally introducing small perturbations in the input, we can fool the model into making a different prediction

> it's a cat and mouse game between the attacker and the defender

---

# [fit] ADVERSARIAL TRAINING vs ADVERSARIAL ATTACKS

^ related to white hat vs black hat hacking. What is the difference?

[.column]
:white_circle:

- Using **complete knowledge** of the model architecture, it's sources of uncertainty, it's parameters, and the training data, we can intentionally introduce adversarial examples to the model to test the robustness and reliability of the model. **adversarial training**. Used to improve the network's ability to make accurate predictions in real-world scenarios.

> An example of adversarial training is training a neural network to recognize handwritten digits by incorporating adversarial examples of slightly modified digits to the training data.

[.column]
:black_circle:

---

# [fit] ADVERSARIAL TRAINING vs ADVERSARIAL ATTACKS

^ related to white hat vs black hat hacking. What is the difference?

[.column]
:white_circle:

- Using **complete knowledge** of the model architecture, it's sources of uncertainty, it's parameters, and the training data, we can intentionally introduce adversarial examples to the model to test the robustness and reliability of the model. **adversarial training**. Used to improve the network's ability to make accurate predictions in real-world scenarios.

> An example of adversarial training is training a neural network to recognize handwritten digits by incorporating adversarial examples of slightly modified digits to the training data.

[.column]
:black_circle:

- Using **incomplete knowledge** of the model architecture, it's sources of uncertainty, it's parameters, and the training data, we can intentionally try to manipulate the input data to cause it to make incorrect predictions. **adversarial attacks**. Used to identify vulnerabilities in the network and to improve security.

> An example of an adversarial attack is adding a small amount of noise to an image of a stop sign in order to make it appear as a go sign to an autonomous vehicle's image recognition system.

---

# [fit] ADVERSARIAL ATTACKS

^ stop the video and raise discussion points.

![fit](https://www.youtube.com/watch?v=AOZw1tgD8dA&t=237s)

---

# [fit] SUMMARY 

[.build-lists: true]

^ read through summary, stress that this is just the tip of the iceberg.

- **XAI** is a field of research that aims to make machine learning models more interpretable and explainable
- **linear models** are intrinsically interpretable because they are linear functions of the input features
- **decision trees** are intrinsically interpretable because they are a series of if-then-else statements
- **neural networks** are not intrinsically interpretable because they are non-linear functions of the input features. They need to be **interpreted** for high-stakes decision making applications.
- **quantifying uncertainty** is a key component of XAI regardless of the model architecture
- methods like **grad-CAM** help us understand which parts of the input image are most important for the model to make a prediction
- **adversarial attacks** and **adversarial training** are two sides of the same coin to testing the reliability and robustness of machine learning models

---

# [fit] EXPLODING ROCKETS

![fit ](https://www.youtube.com/watch?v=bvim4rsNHkQ)

[watch](https://www.youtube.com/watch?v=bvim4rsNHkQ)

---

# [fit] THANK YOU

^ If you find such material useful, please follow me on Github, I will continue to add more slides and notebooks there.

![inline 50%](images/git.png)

> [@nbhushan](https://github.com/nbhushan)
