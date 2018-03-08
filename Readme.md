# cs632
Deep Learning


Basic useful feature list:

 * Ctrl+S / Cmd+S to save the file
 * Ctrl+Shift+S / Cmd+Shift+S to choose to save as Markdown or HTML
 * Drag and drop a file into here to load it
 * File contents are saved in the URL so you can share files


This assignment is divided in two parts:-


 1) 1. Import Scikit learn iris data and divide into train and test data in the ratio 70:30.
    2. Create a custom classifier to train the training data and predict the accuracy.
    3. Implement fit , predict and constructor method similar to k nearest neighbour classifier
    4. Input number of nearest neighbour k in classifier method and display accuracy

 2) 1. Retireve 50 emails with labels as spam/ not spam.
    2. Use dataframe to clean the data and divide the data in 50:50 ratio of training data to test data.
    3  Implement classiifer developed in part one and train  models.
    4. put value of nearest neighbour as 3(or any) and test the training data.
    5. Display acurracy of training data.

Look, a list!

 IRIS training data:-
    array([[ 5.8,  4. ,  1.2,  0.2],
       [ 5.1,  2.5,  3. ,  1.1],
       [ 6.6,  3. ,  4.4,  1.4],
       [ 5.4,  3.9,  1.3,  0.4],
       [ 7.9,  3.8,  6.4,  2. ],
       [ 6.3,  3.3,  4.7,  1.6],
       [ 6.9,  3.1,  5.1,  2.3],
       [ 5.1,  3.8,  1.9,  0.4],
       [ 4.7,  3.2,  1.6,  0.2],.....

 IRIS testing data:-
     array([[ 5. ,  3.4,  1.6,  0.4],
       [ 6.8,  2.8,  4.8,  1.4],
       [ 5. ,  3.5,  1.6,  0.6],
       [ 4.8,  3.4,  1.9,  0.2],
       [ 6.3,  3.4,  5.6,  2.4],
       [ 5.6,  2.8,  4.9,  2. ],
       [ 6.8,  3.2,  5.9,  2.3],
       [ 5. ,  3.3,  1.4,  0.2],
       [ 5.1,  3.7,  1.5,  0.4],
       [ 5.9,  3.2,  4.8,  1.8],
       [ 4.6,  3.1,  1.5,  0.2],
       [ 5.8,  2.7,  5.1,  1.9],.......

 Spam email :- 
     Bodyfile:-
       	
            0	One of a kind Money maker! Try it for free!Fro...
            1	link to my webcam you wanted Wanna see sexuall...
            2	Re: How to manage multiple Internet connection...
            3	[SPAM] Give her 3 hour rodeoEnhance your desi...
            4	Best Price on the netf5f8m1 (suddenlysusan@Sto...
            5	linux.ie mailing list memberships reminderThis...
            6	Re: Apple Sauced...againAt 1:16 AM -0400 on 10...
            7	Re: results for giant mass-check (phew)I never...
            8	Re: RPM's %post, %postun etcHave you tried reb...    

      LabelFile:-
                    label	Name
            0	0	00000.txt
            1	0	00001.txt
            2	1	00002.txt
            3	0	00003.txt
            4	0	00004.txt
            5	1	00005.txt
            6	1	00006.txt
            7	1	00007.txt
            8	1	00008.txt
            9	1	00009.txt
            10	1	00010.txt
            11	0	00011.txt       

And here's some code! :+1:

Classifier:-
  class myCustomClassifier():
    def __init__(self,n_number = 3):
        self.n_number = n_number
    
    def fit(self,iris_X_train,iris_y_train):
        self.iris_X_train = iris_X_train
        self.iris_y_train = iris_y_train
        
    def closest(self,row):
        
        tempDist = []
        tempFull = []
       
        counter = 0
        
        for i in range(1,len(iris_X_train)):
            
            dist = eucledean(row,self.iris_X_train[i])
            tempDist.append((dist,self.iris_y_train[i]))
           
        
        tempFull = [i[1] for i in sorted(tempDist)[:self.n_number]]
        voteResult = Counter(tempFull).most_common(1)[0][0]
     
        #Take vote of k number of closest train data and return one with most vote      
        return voteResult
    
    
        
    def predict(self,x_test):
        
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions



----------------------------------------------------------------------------
Part 1 b)

1. In a Nearest Neighbor classifier, is it important that all features be on the same scale?
Think: what would happen if one feature ranges between 0-1, and another ranges
between 0-1000? If it is important that they are on the same scale, how could you
achieve this?
Ans:- All Features can be of different scale  but we should normalize the data for just point of view reference.
you should normalize data when your model is sensitive to magnitude, and the units of two different features are different, and arbitrary. This is like the case you suggest, in which something gets more influence than it should.
If One feature is in scale 1-10 and another in 0-1000, the second one will influence more on the result. Normalizing or scaling the data will be helpful.

2. What is the difference between a numeric and categorical feature? How might you
represent a categorical feature so your Nearest Neighbor classifier could work with it?
Ans:- 
Categorical:-
Values or observations that can be sorted into groups or categories.
Examples: Sex, Eye colour and Favourite colour.
Bar charts and pie graphs are used to graph categorical data.

Numerical
Values or observations that can be measured. And these numbers can be placed in ascending or descending order. Examples: Height, Arm Span and Weight.
Scatter plots and line graphs are used to graph numerical data.

In our case in part1.py , we could use color of the flower to categorize data by assigining bits to the color.

3. What is the importance of testing data?
Ans:-  In order to estimate how well your model has been trained (that is dependent upon the size of your data, the value you would like to predict, input etc) and to estimate model properties (mean error for numeric predictors, classification errors for classifiers, recall and precision for IR-models etc.)

4. What does “supervised” refer to in “supervised classification”?
Ans:- In supervised classification the user or image analyst “supervises” the pixel classification process. The user specifies the various pixels values or spectral signatures that should be associated with each class. This is done by selecting representative sample sites of known cover type called Training Sites or Areas. The computer algorithm then uses the spectral signatures from these training areas to classify the whole image.

5. If you were to include additional features for the Iris dataset, what would they be, and
why?
Ans:- If we can thin of additional dataset for iris, I can use color and image as a parameter. Using sequential modelling and softmax regression we can more accurately predict the flower type.


Part 2 b)
1. What are the strengths and weaknesses of using a Bag of Words? (Tip: would this
representation let you distinguish between two sentences the same words, but in a
different order?)
Ans:- Weakness -  Bag of words models encode every word in the vocabulary as one-hot-encoded vector i.e. for vocabulary of size |V||V|, each word is represented by a |V||V| dimensional sparse vector with 11 at index corresponding to the word and 00 at every other index. As vocabulary may potentially run into millions, bag of word models face scalability challenges.
While modeling phrases using bag-of-words the order of words in the phrase is not respected. 

Strengths:- In this model, a text such as a sentence or a document is represented as the bag i.e multiset of its words, disregarding grammar and even word order but keeping multiplicity.  

Bag of words does not understand grammer or the order of words in a sentence and hence we cannot distinguish 2 words in same sentences but in different order.

2. Which of your features do you think is most predictive, least predictive, and why?
Ans:-  Term frequency, namely, the number of times a term appears in the text is the most predictive feature. This list or vector representation does not preserve the order of the words in the original sentences, which is just the main feature of the Bag-of-words model.
Least predictive features are vowels like a,e,i,o,u as they are present in every sentence and also words like a ,the ,or, and etc.These can be reduced by normalizing the data.

Did your classifier misclassify any examples? Why or why not?
Ans:- Yes as the accuracy score is not 1.0, it did misclassify the data. 
As you can see, this is nearest neighbor votes for k = 3 with 1 as spam and 0 as non spam:-
[1, 1, 1]
[1, 1, 0]
[0, 0, 1]
[0, 0, 1]
[1, 1, 1]
[0, 1, 0]
[1, 1, 1]
[1, 0, 0]
[1, 0, 1]
[1, 0, 1]
[1, 1, 1]
[0, 0, 1]
[1, 1, 1]
[1, 1, 1]
[1, 1, 1]

If nearest neighbor gives out 1 i.e spam and other to neighbors tell it as non spam , it will display as non spam which might be wrong in many cases.


