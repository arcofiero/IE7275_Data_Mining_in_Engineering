
#PDF MALWARE Detection Based on Supervised Machine Learning Model
# IE7275_Data_Mining_in_Engineering

Problem Setting:
PDFs, or Portable Document Format files, have for long been the most common document format
due to their portability and reliability. It was created by Adobe in 1992 to present documents,
including text formatting and images, in a manner independent of software, hardware, or operating
system. Unfortunately, the popularity of this format and the advanced features it provides has made
it a target for hackers to use for malicious attacks. A PDF has several critical features that an
attacker can exploit to deliver a malicious payload. These malicious PDFs are designed to evade
security checks, therefore making them an efficient carrier for viruses.

Problem Definition:
The objective of this analysis is to use concepts of data mining and machine learning to identify
the best classification algorithm and predictors to correctly detect whether a document is benign
or contains malware. The intention is to seek information on how attributes such as metadata,
embedded data, external links, etc. impact the classification of documents as malicious or benign.

Data Sources:
The data is taken from the Evasive-PDFMal2022 dataset created in the Canadian institute of
Cybersecurity, based at the University of New Brunswick in Fredericton
(https://www.unb.ca/cic/datasets/pdfmal-2022.html). The data description is from the respective
publication [1].

Data Description:
The Evasive-PDFMal2022 dataset consists of 10,025 records with 5557 malicious and 4468
benign records that tend to evade the common significant features found in each class. It contains
37 static representative features, including 12 general features and 25 structural features extracted
from each PDF file.

Data Exploration and Visualization:
Looking at the dataframe read using the Pandas library, we see that the data has 33 attributes and
10026 records are split into 2 classes of interest, Benign and Malicious. We will designate the Malicious
class as the target class, so it will be denoted by 1, while the benign class will be denoted by 0.

We also visualized the data using the Plotly library. Some of the visualizations were:
1. We plotted a scatter plot between PDFSize and MetadataSize.
2. Counting the number of pages in the PDFs and plotting a histogram.
3. For images in the documents, we binned the number of images in 5 bins as shown in figure
below.
4. Correlation Analysis: Plotting the correlation plot between all the numerical variables we get
the following heatmap.
We can clearly see that the attributes Javascript and JS are redundant. Also, the attributes metadata
size and title characters are very positively correlated. Therefore, we decide to drop JS and title
characters.
Data Cleaning and Preprocessing:
We dropped the one record that has a NaN class because it also has a majority of attribute values
missing. We also converted the header values to categorical 1 and 0 using regular expressions. We
also dropped the File Name attribute because it served only as an identifier. Also, the StandardScaler
method was used the standardize the data.

Model Selection:
Given that this is a classification task, we needed to select the best classifier model in order to get
a better overall accuracy score. To this end, we split the data into training and validation sets,
reserving 25% of the data for the validation and using the rest as training. We narrowed down the
candidates for the best model to:

1. Logistic Regression:
Logistic regression is a method of modeling the probability of a discrete result given an input
variable. The most frequent logistic regression models have a binary outcome, which might be
true or false, yes or no, and so forth. Logistic regression is a handy analysis tool for determining
if a fresh sample fits best into a category in classification tasks.
The logistic function (also called the sigmoid) is used, which is defined as:

Implementation: The logistic regression model was implemented using the liblinear solver,
resulting in a training accuracy of 0.842042 and validation accuracy of 0.82287.
The equation for the model comes out to be:


2. K Nearest Neighbors:
The KNN algorithm assumes that objects that are similar are close together. By computing the
distance between points on a graph, KNN encapsulates the concept of similarity (also known
as distance, proximity, or closeness).
Implementation: The KNN model was run multiple times using different values for
n_neighbors, with 500 being the optimal parameter, resulting in a training accuracy of 0.86470
and validation accuracy of 0.86085.

3. Support Vector Classifier:
The support vector machine algorithm's goal is to find a hyperplane in an N-dimensional space
(N = the number of features) that distinguishes between data points. There are numerous
hyperplanes from which to choose to separate the two classes of data points. The goal is to
discover a plane with the greatest margin, or the greatest distance between data points from
both classes. Maximizing the margin distance gives some reinforcement, making it easier to
classify subsequent data points.
Implementation: The base SVM Classifier was implemented, yielding a training accuracy of
0.9996 and validation accuracy of 0.884846.

After comparing the models, we concluded that although SVC is overfitting the training data, it is
still yielding the best accuracy score for the validation data. Therefore, we chose to work on the
SVC model for the data mining task. Our objective is to remove the overfitting and tune the model
to result in a better accuracy score.

Implementation of Selected Model:
To combat the overfitting of the SVC model, we worked on balancing the dataset by oversampling
the minority Benign class using SMOTE (Synthetic Minority Oversampling Technique) [2].
SMOTE works by selecting instances in the feature space that are close together, drawing a line in
the feature space between the examples, and drawing a new sample at a location along that line. A
random case from the minority class is picked initially. Then, for that example, k of the closest
neighbors are found (usually k=5). A randomly determined neighbor is chosen, and a synthetic
example is constructed at a randomly chosen location in the feature space between the two examples.
After oversampling the minority class, we now have 11070 records in the dataset. To prepare this
data for the prediction model, we standardize the data. Also, to increase the accuracy score, we
performed feature selection using Recursive Feature Elimination (RFE) [3].

RFE is based on the idea of continually building a model and selecting either the best or worst
performing feature, setting the feature aside, and then repeating the process with the remaining
features. This procedure is repeated until the dataset's features have been exhausted. The purpose
of RFE is to pick features by considering fewer and smaller sets of features in a recursive manner.

Using RFE in tandem with SVC using a linear kernel, we narrow down the optimal number of
features to be 20. 

After feature selection, we use to feed the training data to a linear kernel SVC model, resulting in a
training accuracy of 0.9592869 and validation accuracy of 0.9573699. This is a clear step-up from
the overfitting base model.

We can also improve the accuracy of the model by tuning the hyperparameters of the model using
Grid Search Cross-Validation. Grid Search is a technique for determining the hyperparameters of
a model that produce the most ‘correct’ predictions. We tune the following hyperparameters:
1. C: The regularization parameter.
2. Kernel: Kernels are a way to solve non-linear problems with the help of linear classifiers. We
chose between linear and RBF kernels. Here, RBF is a non-linear kernel that stands for radial
basis function.
3. Gamma: Only used by non-linear kernels. The Gamma parameter of RBF controls the distance of
the influence of a single training point.
By running the GridSearchCV method, we get the best params to be {C: 10, gamma: auto, kernel:
rbf}. And these parameters yield a final training accuracy of 0.979523 and validation accuracy of
0.972543.

Performance Evaluation:
Using the Scikit-learn metrics module, we calculate the accuracy score to be 0.972543. The
classification report and confusion matrix for a 0.5 cutoff can be seen.


We also plotted some performance evaluation visualizations and compared them with the results
of the base SVC model.
1. ROC Curve: This graph shows the performance of a classification model at all classification
thresholds. It plots the True Positive Rate vs False Positive rate.

2. Precision-Recall Curve: The precision-recall curve shows the tradeoff between precision and
recalls for different thresholds. A high area under the curve represents both high recall and high
precision, where high precision relates to a low false-positive rate, and high recall relates to a
low false-negative rate.

3. Cumulative Gains Curve: The cumulative gains curve is an evaluation curve that assesses the
performance of the model and compares the results with the random pick. It shows the
percentage of targets reached when considering a certain percentage of the population with
the highest probability to be targeted according to the model.

4. Lift Chart: A lift chart graphically represents the improvement that a mining model provides
when compared against a random guess and measures the change in terms of a lift score.


Conclusion:
Even today, malicious PDFs pose a severe cyber threat. We may conclude, based on the PDF’s
complicated structure and powerful capabilities, that attackers can transmit malware in a variety
of methods. The ineffectiveness of common anti-viruses, along with a lack of user understanding,
has increased the risk even further. In this project, we tried to analyze the numerous features that
can be extracted from a PDF document and then used these features to build a prediction model.
Among the selected models, Support Vector Classifier with Recursive Feature Elimination
outperformed the others, given the high accuracy and f1-score.

References:
[1] Maryam Issakhani, Princy Victor, Ali Tekeoglu, and Arash Habibi Lashkari1, “PDF Malware
Detection Based on Stacking Learning”, The International Conference on Information Systems
Security and Privacy, February 2022
[2] N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, “SMOTE: Synthetic Minority
Over-sampling Technique”, Journal of Artificial Intelligence Research, Volume 16, June 2002
[3] Isabelle Guyon, Jason Weston, Stephen Barnhill and Vladimir Vapnik, “Gene Selection for
Cancer Classification using Support Vector Machines”, Machine Learning 46, Jamuary 2002
