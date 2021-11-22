# IIS-logistic-regression-model
ML-based prediction system to predict if the graduates will get "high salary" or not. The dataset (attached as ‘Data.xlsx’) provides anonymized biodata information for graduates along with their respective skill scores.

First do feature selection, so remove ‘ID’, ‘DOB’, ‘12graduation’, 'CollegeID', ‘CollegeCityID’. I decided to remove them as ‘ID’ and ‘DOB’ are not relevant to calculate if a person will get a high salary or not. ‘12graduation’ is like ‘GraduationYear’ and also it doesn’t affect our model much(the accuracy of the model is almost the same).  'CollegeID' and ‘CollegeCityID’ is sort of captured by ‘CollegeCityTier’ and ‘CollegeState’.
Also removed ‘10board’ as at 75-25 split accuracy improved from 72.10 to 72.40 after removing.
Did not remove ‘CollegeState’ as accuracy dropped to 72.40 from 72.10 after removing.

1)For columns ‘ComputerProgramming’, ‘ElectronicsAndSemicon’, ‘ComputerScience’, ‘MechanicalEngg’, ‘ElectricalEngg’, ‘TelecomEngg’, ‘CivilEngg’ only take the max of the IDs marks in all these subjects and normalised it to values between 0 and 1.

2) one-hot encoded columns ‘Gender’, ‘12board’, ‘Degree’, ‘Specialization’, ‘CollegeState’,

3) Normalised columns ‘10percentage’, ‘12percentage’, ‘CollegeTier’, ‘collegeGPA’, ‘CollegeCityTier’, ‘English’, ‘Logical’, ‘Quant’, ‘Domain’ to values between zero and one using MinMaxScaler().

4) Normalised columns ‘conscientiousness’, ‘agreeableness’, ‘extraversion’, ‘nueroticism’, ‘openess_to_experience’ to values between -1 and 1 using MinMaxScaler().

5) Normalised column ‘GraduationYear’ to values between 0 and 10 using MinMaxScaler().

5) Split the data into test and train and also shuffle it.

6) label-encoded the output(y) values.

7) Trained the model and checked its accuracy.

At train-test spit = 60-40 , accuracy = 71.81
At train-test spit = 70-30 , accuracy = 72.00
At train-test spit = 75-25 , accuracy = 72.40
At train-test spit = 80-20 , accuracy = 71.00
At train-test spit = 90-10 , accuracy = 71.00

Therefore 75-25 gives the best fit.

