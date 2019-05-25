
#### Introduction
- Data page: https://www.kaggle.com/uciml/pima-indians-diabetes-database

================================================================================
#### Explanation on data
- Pregnancies<br/>
- Glucose<br/>
- BloodPressure<br/>
- SkinThickness<br/>
- Insulin<br/>
- BMI<br/>
- DiabetesPedigreeFunction<br/>
- Age<br/>
- Outcome<br/>

================================================================================
#### Libraries
- Python 3.6
- PyTorch 1.0.1.post2
- CUDA V10.0.130
- CuDNN v7.4
- Scikit-Learn
- And others which you can install whenever you run into unmet-dependencies

================================================================================
#### Overview of entire dataset<br/>
```
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \
count  768.000000   768.000000  768.000000     768.000000     768.000000   
mean   3.845052     120.894531  69.105469      20.536458      79.799479    
std    3.369578     31.972618   19.355807      15.952218      115.244002   
min    0.000000     0.000000    0.000000       0.000000       0.000000     
25%    1.000000     99.000000   62.000000      0.000000       0.000000     
50%    3.000000     117.000000  72.000000      23.000000      30.500000    
75%    6.000000     140.250000  80.000000      32.000000      127.250000   
max    17.000000    199.000000  122.000000     99.000000      846.000000   

              BMI  DiabetesPedigreeFunction         Age     Outcome  
count  768.000000  768.000000                768.000000  768.000000  
mean   31.992578   0.471876                  33.240885   0.348958    
std    7.884160    0.331329                  11.760232   0.476951    
min    0.000000    0.078000                  21.000000   0.000000    
25%    27.300000   0.243750                  24.000000   0.000000    
50%    32.000000   0.372500                  29.000000   0.000000    
75%    36.600000   0.626250                  41.000000   1.000000    
max    67.100000   2.420000                  81.000000   1.000000    
```

================================================================================
#### Overview of entire dataset<br/>
```
Pregnancies                 768 non-null int64
Glucose                     768 non-null int64
BloodPressure               768 non-null int64
SkinThickness               768 non-null int64
Insulin                     768 non-null int64
BMI                         768 non-null float64
DiabetesPedigreeFunction    768 non-null float64
Age                         768 non-null int64
Outcome                     768 non-null int64
```

1.. Meaning<br/>
..(1) 768 number of data on each feature<br/>
..(2) There is no NaN in entire data<br/>
..(3) Datatype is int64 or float64<br/>

================================================================================
#### Preprocessing data
1.. I add label data by using a criterion of glucose>120<br/>
Before<br/>
```
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
0  6            148      72             35             0        33.6   
1  1            85       66             29             0        26.6   

   DiabetesPedigreeFunction  Age  Outcome  
0  0.627                     50   1        
1  0.351                     31   0        
```

After<br/>
```
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
0  6            148      72             35             0        33.6   
1  1            85       66             29             0        26.6   

   DiabetesPedigreeFunction  Age  Outcome  glucose_over_120  
0  0.627                     50   1        1.0               
1  0.351                     31   0        0.0               
```

================================================================================
#### Visualize train data  
1.. See the estimated probability distribution density function (PDF) of each feature data using 2 parameters; mean and variance<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_10_06%3A48%3A15.png" alt="drawing" width="1000" height="1000"/><br/>
..(1) Goal of this estimation for PDF: to see variance distributions of each feature data<br/>
..(2) Meaning: there are nagative and positive relationships between factors<br/>
....1) Large variance: Insulin, Glucose<br/>
....2) Small variance: Outcome, Pregnancies<br/><br/>

2.. See the actual probability distribution function by using histogram (bins=50)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_10_06%3A36%3A24.png" alt="drawing" width="1000" height="1000"/><br/>
..(1) Meaning<br/>
....1) Pregnancies<br/>
- It has most values between 0 to 5<br/>

....2) Glucosse<br/>
- 0 glucose may be incorrectly measure value<br/>
- You can predict this is data of people with potential or diagnosed diabetes <br/>
because mean of glucose is rather high (100 to 125)<br/>
Normaly, it should have 80-100 range in glucose<br/>
- And there are quite many ones which have high glucose (>125)<br/>

....3) BloodPressure<br/>
- 0 blood pressure may be incorrectly measure data (outlier)<br/>
- I'm not sure what blood pressure numerical figures represent<br/>
- Just guess it could represent the diastolic blood pressure<br/>
- Generally, blood pressure should be 120/80, +-10<br/>

....4) SkinTickness<br/>
- I'm not sure 0 can be meaningless outlier or not.<br/>
- I just guess 0 skinthickness can exist by some medical diagnosis criterion.<br/>
- And there are many samples which have 0 skinthinkness.<br/>
- So, it's hard to simply consider those 0 values as outlier.<br/>

....5) Insulin<br/>
- There are data which has 0 value<br/>
- I'm not sure 0 can be meaningless outlier or not just like "4) SkinTickness" case<br/>

....6) BMI<br/>
- It seems like distribution of BMI data follows Gaussian normal distribution<br/>
- And it makes sense in that noraml natural phenomenon generally follows this Gaussian normal distribution<br/>

....7) DiabetesPedigreeFunction<br/>

....8) Age<br/>
- This data is originated from young people (20 to 40)<br/>

....9) Outcome<br/>
- Binary feature (0 or 1)<br/>
- Distribution of binary feature data can be modeled by Bernoulli distribution (which is special case of categerical distribution)<br/>

3.. See correlations on features<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_09_21%3A27%3A16.png" alt="drawing" width="1000" height="1000"/><br/>
..(1) Meaning: there are nagative and positive relationships between factors<br/>
....1) Negative correlation: Insulin-Pregnancies, SkinThickness-Pregnancies, Age-SkinThickness,<br/>
....2) Positive correlation: Age-Pregnancies, BMI-SkinThickness, Insluin-SkinThickness, Insulin-Glucose<br/>

4.. See paired scatter plot<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_10_07%3A25%3A57.png" alt="drawing" width="1000" height="1000"/><br/>

5.. See paired scatter plot which is separated by labels (1 if glucose>120, 0 otherwise)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_10_07%3A28%3A16.png" alt="drawing" width="1000" height="1000"/><br/>
..(1) Meaning<br/>
....1) See Glucose-Insulin<br/>
- Less Glucose results in less Insulin<br/>
- Increase Glucose results in More Insulin<br/>

....2) See BMI-Glucose<br/>
- Even if low BMI, it can have high Glucose<br/>

....3) At least according this plotting, there is no strong correlation between all features and glucose level<br/>
- It means low BMI can have high glucose, high BMI can have high glucose.<br/>
- Low SkinThickness can have high glucose, high SkinThickness can have high glucose.<br/>

6.. See paired scatter plot with "estimated distribution via Kernel Density Estimation" + "separated distribution by labels (1 if glucose>120, 0 otherwise)"<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_10_07%3A48%3A47.png" alt="drawing" width="1000" height="1000"/><br/>

7.. See paired scatter plot with "linear regression line which can explain distributional pattern of each 2 feature data"<br/>
Even if I tried this, it looks like this is meaningless<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_10_08%3A05%3A39.png" alt="drawing" width="1000" height="1000"/><br/>

================================================================================
#### Analyze train data
1.. Conditional probability<br/>
```
# c nb_gt_120: number of sample which is greater than glucose 120
nb_gt_120=np.sum(glucose_data>120)
# print("nb_gt_120",nb_gt_120)
# 349

# c prob_val_of_glucose_gt_120: probability value of "glucose greater than 120" occuring
prob_val_of_glucose_gt_120=nb_gt_120/768
# print("prob_val_of_glucose_gt_120",prob_val_of_glucose_gt_120)
# 0.45

BMI_data=loaded_csv.iloc[:,5]
# c nb_gt_25_in_BMI: number of samples which are greater than 25 in BMI
nb_gt_25_in_BMI=np.sum(BMI_data>25.0)
# print("nb_gt_25_in_BMI",nb_gt_25_in_BMI)
# 645

# c prob_val_of_BMI_gt_25: probability value of "BMI greater than 25" ocurring
prob_val_of_BMI_gt_25=nb_gt_25_in_BMI/768
# print("prob_val_of_BMI_gt_25",prob_val_of_BMI_gt_25)
# 0.83984375

# Conditional probability (when BMI>25 is given, probability of "glucose greater than 120" occuring)
# P[B|A]=P[A \cap B] \times P[A]
# P[nb_gt_25_in_BMI|prob_val_of_BMI_gt_25]=P[prob_val_of_BMI_gt_25 \cap nb_gt_25_in_BMI] * P[prob_val_of_BMI_gt_25]
# P[nb_gt_25_in_BMI|prob_val_of_BMI_gt_25]=P[prob_val_of_BMI_gt_25 \cap nb_gt_25_in_BMI] * 0.83
# P[nb_gt_25_in_BMI|prob_val_of_BMI_gt_25]=(0.45*0.83984375)*0.83
# P[nb_gt_25_in_BMI|prob_val_of_BMI_gt_25]=0.313681640625
```

================================================================================
#### Importance weights of each feature to diabetes (high glucose>120) by using conditional probability values

1.. Pregnancies<br/>
..(1) Fomular: p(high_glucose|preg) = P(high_glucose \cap preg) / P(preg)<br/>
..(2) Result<br/>
```
interval information: [0,3]
P_high_glucose_AND_preg_each_interval 0.23
P_preg_each_interval 0.55
cond_prob_of_high_glucose_when_preg_0to3_is_given 0.42

interval information: [4,7]
P_high_glucose_AND_preg_each_interval 0.15
P_preg_each_interval 0.29
cond_prob_of_high_glucose_when_preg_0to3_is_given 0.52

interval information: [8,11]
P_high_glucose_AND_preg_each_interval 0.08
P_preg_each_interval 0.13
cond_prob_of_high_glucose_when_preg_0to3_is_given 0.62

interval information: [12,17]
P_high_glucose_AND_preg_each_interval 0.02
P_preg_each_interval 0.03
cond_prob_of_high_glucose_when_preg_0to3_is_given 0.67
```

2.. P(high_glucose|blood_pressure_of_each_interval)<br/>
..(1) Fomular: p(high_glucose|blood_pressure_of_each_interval) = P(high_glucose \cap blood_pressure_of_each_interval) / P(blood_pressure_of_each_interval)<br/>
..(2) Result:<br/>
```
interval information: [0,38]
P_high_glucose_AND_preg_each_interval 0.02
P_preg_each_interval 0.05
cond_prob_of_high_glucose_when_preg_0to3_is_given 0.4

interval information: [40,48]
P_high_glucose_AND_preg_each_interval 0.01
P_preg_each_interval 0.02
cond_prob_of_high_glucose_when_preg_0to3_is_given 0.5

interval information: [50,55]
P_high_glucose_AND_preg_each_interval 0.02
P_preg_each_interval 0.05
cond_prob_of_high_glucose_when_preg_0to3_is_given 0.4

interval information: [56,61]
P_high_glucose_AND_preg_each_interval 0.03
P_preg_each_interval 0.09
cond_prob_of_high_glucose_when_preg_0to3_is_given 0.33
```
3.. P(high_glucose|insulin_of_each_interval)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_10_14%3A06%3A37.png" alt="drawing" width="1000" height="300"/><br/>
Meaning:<br/>
- According to 1st plot, probability of "insulin" and "high glucose" occuring at the same time becomes lower as interval goes?<br/>
- According to 2nd plot, probability of "insulin" at each interval becomes lower?<br/>
- According to 3rd plot, high insulin feature must result in high glucose? (high importance weight to diabetes phenomenon?)<br/>

4.. P(high_glucose|BMI_of_each_interval)<br/>
<img src="https://raw.githubusercontent.com/youngminpark2559/bio_code/master/My_code/V_0001/prj_root/img_out/Analyze_train_data/2019_05_10_14%3A49%3A11.png" alt="drawing" width="1000" height="300"/><br/>

================================================================================
#### To do
[V] 1.. Plot entire data on the 2D plane to see how each feature gathers<br/>
[V] 2.. See "weight of importance" to the diabetes phenomenon by using conditional probability<br/>

- P(diabetes|age): when age prior is given, probability of diabetes occuring<br/>
- P(diabetes|pregnancies): when pregnancies prior is given, probability of diabetes occuring<br/>
- P(diabetes|BMI): when BMI prior is given, probability of diabetes occuring<br/>
- ...<br/>

[--] 3.. Inspect data in terms of Gaussian mixture model<br/>