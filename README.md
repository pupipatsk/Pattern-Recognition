# Pattern Recognition
2110573 Pattern Recognition (Machine Learning)\
Instructor: Ekapol Chuangsuwanich\
https://github.com/ekapolc/pattern_2024

## Course Outline

- **Introduction**
  - Overview of Machine Learning
  - Key Concepts and Terminology

- **K-Means Clustering**
  - Introduction to K-Means
  - Algorithm Explanation
  - Applications and Use Cases

- **Regression Analysis**
  - Linear Regression
  - Logistic Regression
  - Maximum Likelihood Estimation (MLE)
  - Maximum A Posteriori Estimation (MAP)

- **Naive Bayes and Gaussian Mixture Models (GMM)**
  - Naive Bayes Classifier
  - GMM Overview
  - Expectation-Maximization (EM) Algorithm
  - Evidence Lower Bound (ELBO)

- **Dimensionality Reduction**
  - Principal Component Analysis (PCA)
  - Linear Discriminant Analysis (LDA)
  - Random Projections (RP)

- **Visualization Techniques**
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)
  - UMAP (Uniform Manifold Approximation and Projection)
  - PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding)

- **Support Vector Machines (SVM) and Neural Networks (NN)**
  - Introduction to SVM
  - Basic Neural Network Concepts

- **Advanced Neural Networks**
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN)

- **Neural Network Architectures & PyTorch**
  - Advanced NN Architectures
  - PyTorch Demonstration

- **Transformers and Self-Supervised Learning**
  - Introduction to Transformers
  - Self-Supervised Learning Techniques I
  - Self-Supervised Learning Techniques II

- **Generative Models**
  - Generative Adversarial Networks (GAN)
  - Variational Autoencoders (VAE)
  - Diffusion Models

- **Reinforcement Learning**
  - Basics of Reinforcement Learning
  - Key Concepts and Algorithms

## Directory Structure
```txt
.
├── HW
│   ├── HW01
│   │   ├── HW 1-240124.pdf
│   │   ├── HW 1.ipynb
│   │   ├── HW1_Regression_2024.pdf
│   │   ├── HW1_optional.pdf
│   │   ├── K-Means Clustering.py
│   │   ├── Logistic Regression.py
│   │   ├── T11 submission score.png
│   │   ├── archive
│   │   │   ├── HW 1-240123.pdf
│   │   │   ├── HW 1_Q1-240117.pdf
│   │   │   ├── HW 1_Q2-240119.pdf
│   │   │   ├── HW 1_Q3-240119.pdf
│   │   │   ├── HW 1_Q3-240123.pdf
│   │   │   ├── HW 1_Q3-240123_1600.pdf
│   │   │   └── HW 1_Q3-240124.pdf
│   │   ├── submission.csv
│   │   ├── titanic_test.csv
│   │   └── titanic_train.csv
│   ├── HW02
│   │   ├── HW2-T1_OT1.pdf
│   │   ├── HW2.pdf
│   │   ├── HW2_Attrition_Prediction_2024.pdf
│   │   ├── HW2_EmployeeAttritionPrediction.ipynb
│   │   ├── HW2_EmployeeAttritionPrediction.pdf
│   │   ├── HW2_SimpleBayesClassifier.ipynb
│   │   ├── HW2_SimpleBayesClassifier.pdf
│   │   ├── README.md
│   │   ├── SimpleBayesClassifier.py
│   │   ├── SimpleBayesPlot.jpg
│   │   ├── SimpleBayesPlot.py
│   │   ├── hr-employee-attrition-with-null.csv
│   │   └── starter_code
│   │       ├── Pattern_HW2_student_2024.ipynb
│   │       └── SimpleBayesClassifier.py
│   ├── HW03
│   │   ├── GMM_GaussianMixtureModel.py
│   │   ├── HW03.ipynb
│   │   ├── HW03.pdf
│   │   ├── HW3_Fisherfaces(student).ipynb
│   │   ├── HW3_Fisherfaces_2024.pdf
│   │   ├── facedata.mat
│   │   └── facedata_mat.zip
│   ├── HW04
│   │   ├── HW4_2024_Neural_Networks.pdf
│   │   ├── HW4_Pupipat_Singkhorn.pdf
│   │   ├── HW4_Simple_Neural_Network_Lab.pdf
│   │   ├── HW4_T1-T4
│   │   │   ├── main.aux
│   │   │   ├── main.bbl
│   │   │   ├── main.log
│   │   │   ├── main.pdf
│   │   │   ├── main.synctex.gz
│   │   │   └── main.tex
│   │   ├── HW4_T1-T4.pdf
│   │   ├── hw4_2024.zip
│   │   └── hw4_prob_part1
│   │       ├── Simple_Neural_Network_Lab.ipynb
│   │       ├── cattern
│   │       │   ├── __init__.py
│   │       │   ├── __pycache__
│   │       │   │   ├── __init__.cpython-39.pyc
│   │       │   │   ├── gradient_check.cpython-39.pyc
│   │       │   │   └── neural_net.cpython-39.pyc
│   │       │   ├── data_utils.py
│   │       │   ├── gradient_check.py
│   │       │   └── neural_net.py
│   │       └── mnist_data
│   │           ├── __init__.py
│   │           ├── __pycache__
│   │           │   ├── __init__.cpython-39.pyc
│   │           │   ├── load_mnist.cpython-39.pyc
│   │           │   └── vis_utils.cpython-39.pyc
│   │           ├── load_mnist.py
│   │           ├── t10k-images-idx3-ubyte.gz
│   │           ├── t10k-labels-idx1-ubyte.gz
│   │           ├── train-images-idx3-ubyte.gz
│   │           ├── train-labels-idx1-ubyte.gz
│   │           └── vis_utils.py
│   ├── HW05
│   │   ├── HW5_Precipitation_Nowcasting_PyTorch_Student_2024.html
│   │   └── HW5_Precipitation_Nowcasting_PyTorch_Student_2024.ipynb
│   ├── HW06
│   │   ├── HW6_Contrastive_learning_ver_2024_(student_version).ipynb
│   │   ├── HW6_Contrastive_learning_ver_2024_(student_version).pdf
│   │   └── weights
│   │       ├── best_infonce_weights.pth
│   │       ├── best_siamese_weights.pth
│   │       └── best_triplet_weights.pth
│   ├── HW07
│   │   ├── HW7_GANs_(student_version).ipynb
│   │   └── HW7_GANs_(student_version).pdf
│   └── HW08
│       ├── HW8_student.html
│       ├── HW8_student.ipynb
│       └── Model_tailoring_tutorial.ipynb
├── README.md
└── miscellaneous
    ├── Remove DS_Store file.md
    ├── Vector_Matrix_and_Tensor_Derivatives-Erik_Learned_Miller.pdf
    ├── introstats_normal_linear_combinations.pdf
    └── markdown-cheat-sheet.md

20 directories, 84 files
```