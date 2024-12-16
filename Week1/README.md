<!-- This week is going to be covering some basic concepts and get the hang of syntax of required libraries. 

Some basic fast learning tips:
You can always ask chatgpt to understand the syntax of any library/framework, you can give relevant prompts like:
"Explain me all the basic numpy syntax with relevant examples for each concept", and other variations of it would also work(Most of you would be expert in giving GPT prompts by now :) ).
Also you can ask chatgpt about the syntax that you don't understand from the resources/documentation and ask it to make some
complex practice example to get the hang of it in depth.

**We would recommend you to solve the assignments on your own without using any LLM's, if you don't know the syntax to make a particular part of problem work in particular framework/library you are working on,then you can just google the syntax required and implement it on your own, if you implement any stuff using LLM, make sure you understand each and every line of it and then implement that stuff by yourselves rather than copy pasting.**

**We would also recommend to go through optional readings and optional assignments** -->

### Week 1: Get Familiar with Python Libraries
- **Libraries**: Numpy, Matplotlib, PyTorch

#### References for Python
- [GeeksforGeeks: Python Programming Language Tutorial](https://www.geeksforgeeks.org/python-programming-language-tutorial/)
- [Python Documentation](https://docs.python.org/3.13/)

  Get Familier with basic python syntax, datatypes, functions and basics of object oriented programming

#### References for numpy
- [Numpy Documentation](https://numpy.org/doc/2.1/user/basics.html)
  - Just get to know about the working of numpy arrays, dimensions, axes, matrix operations, broadcasting, go through all examples thoroughly and try to go over more examples from internet for practice, try to create and experiment with your own examples.(You can also teach someone to enhance your understanding).
 
#### Some basic ML algorithms(to get the hang of machine learning)

-  Linear Regression(Reading): https://www.geeksforgeeks.org/ml-linear-regression/?ref=ml_lbp
- (Optional) https://www.coursera.org/learn/machine-learning. You can go through this course to understand regression better
- (Optional) https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LinearRegression.html, this link contains various regression based algorithms and their applications on various datasets, you can go through them to get more intuitive and application based understanding of linear regression and their derivatives.


#### Understanding the Basics of Neural Networks
- **Articles and Reading**:
  - [Medium: Introduction to Neural Networks](https://medium.com/deep-learning-demystified/introduction-to-neural-networks-part-1-e13f132c6d7e)
  - [Medium: A Beginner Intro to Neural Networks](https://purnasaigudikandula.medium.com/a-beginner-intro-to-neural-networks-543267bda3c8)
  - Read the introductory chapter from “Introduction to Machine Learning - Gurney et al.” and optionally the theory (highly recommended).
  

- **Implementation pytorch tutorials(Optional, but highly recommended to learn the working of tensors and basic feed forward neural networks)**:
  - https://youtu.be/c36lUUr864M?si=ipK9wX2L0EgOtZHV
  - Also get the understanding of some Commonly used loss functions (and get the intuition behind them) and optimizers(like SGD, Adam and their mathematical background)

> **Note**: We know that the articles we provided mostly cover the theory of neural networks; we also recommend you to get the basics concepts and syntax of pytorch and implement the basic neural networks and play with them(Also understand the loss functions and optimizer used, their mathematical aspects).

### Assignments(Deadline: 18th December)
- **Assignment 1**: Implement Linear and Logistic Regression from scratch using numpy.
  - Build a Linear regression model to predict Student performance Index using the dataset provided in the link below. You will have to load the data using the library of your choice and drop rows in which some cells are empty.You are expected to use Numpy to create the linear regression model. Student Performance Dataset is provided in this folder.
  - Build a logistic regression model to predict whether the patient has a 10-year risk of future coronary heart disease (CHD) using the dataset available here: https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression?resource=download
  - You are free to use any library of your choice to load the data and drop the rows which have a NA entry. You are expected to use numpy to build the logistic regression model. (Optional: Implement any imputation technique or outlier removal technique and observe their effect on model accuracies).
  - Use Gradient Descent algorithm or any other derivative of it to optimize the loss function, you can refer to various optimizers on google.
  - Evaluating your model: for linear regression you can print the RMSE loss and for logistic regression print the accuracy of your model (percentage of data points classified correctly) along with the percentage false negatives (numbers of false negatives /number of positives) and percentage false positives(number of false positives/number of negatives)

- **Assignment 2**: Implement K-means clustering algorithm to get used to Numpy operations. Problem statement, algorithm and relevant files are uploaded in assignment folder within this week1 folder.
  
- **Assignment 3 (Optional)**: Implement Feed Forward Neural Network on cat and dog dataset from Kaggle and improve the model accuracy using various preprocessing techniques and hyperparameter fine-tuning.

<!-- - Submission format: Make your github repository. You can clone our repo and add your work to it. We will float Google forms asking the repo links, please feel free to reach out in case you are new to GitHub. -->