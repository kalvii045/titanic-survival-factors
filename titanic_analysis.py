
# coding: utf-8

# Throughout this report we will be gathering data from a sample to answer one important question about the titanic. 
# - What factors affected survival? Did PClass determine whether the passenger was more likely to survive or not? 
# - Were there other factors too such as gender and age that influenced the rate of survivors on the Titanic? 
# 
# Hypothesis: Chances are PClass did affect survival because higher class means access to better resources, gender and age may also influence survival since there is a practice of "women and children first".
# The purpose here is to find what factors may have influenced the chances of one surviving. We will also look at the data from different perspectives such as survival of females and males (each gender) in each class, this can give us an overview of whether certain genders are more advantaged in some classes more than others. Thus it may not be one specific factor but factors within factors (multiple reasons) as to why an individual may have a higher chance of survivor. Perhaps I should also mention why I used "may have a higher chance", it's because all these statistics are just data and they literally show results, not concrete evidence that having these specific factors will guarantee survival. Yes some of these factors such as higher class, as mentioned earlier can increase chances of survival since upper class members have access to better resources but there are also many non-survivors in first class and many survivors in the third class.   
# 
# Throughout the data there are some values missing. For example cabin numbers are missing for many passengers and so are the ages. Cabin numbers are not relevant to answer our question however ages are. Since we are missing many ages, I will calculate the standard deviation of the age of the survivors and non-survivors and find the t-score to get an "overall" estimate and an "overall result" to come up with a conclusion on how age affected survival. This is discussed more on top of cell 20.   

# In[218]:

import unicodecsv
import pandas as pd
import numpy as np 
import math as mt 
get_ipython().magic(u'pylab inline')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 

## This cell tries to find a correlation between Pclass and survival
## Open csv file in list format to find specific variables related to each other

data = pd.read_csv('titanic-data.csv')
survived = np.array(data['Survived'])
pclass = np.array(data['Pclass'])

x1 =  data.groupby('Survived')['PassengerId'].count()
print x1
print " "

## Set parameters of the graphs 
fig_size = plt.figure(figsize=(18,6), dpi=1600)
## Set location of the first graph and plot the first graph (survivors and non-survivors)
graph1 = plt.subplot2grid((2,3),(0,0))
data.Survived.value_counts().plot(kind='bar')
plt.xlabel('Survivor or non-survivor, 1=survivor')
plt.ylabel('Amount')
plt.title("Survivors and non-survivors")

## Group survivors and non-survivors into three different classes 
df = pd.DataFrame({'survivor': survived == 1, 'first_class': pclass == 1,'second_class':pclass == 2,
                  'third_class':pclass == 3})
first_survivors = df.groupby('survivor')
x = first_survivors.sum()
print x

## Plot number of people in each class
plt.subplot2grid((2,3),(0,1))
data.Pclass.value_counts().plot(kind='bar')
plt.title('Number of people in each class')
plt.xlabel('Class')
plt.ylabel('Amount') 

## Plot survivors corresponding to class
x.plot(kind='bar')
plt.ylabel('# of Survivors')
plt.title('Survivors by class')


# The first cell of code looks into the relationship between PClass and survival.I grouped each individual by class, and then by 1 or 0 (whether they survived or not) and took the sum of each group. Out of the 891 board members there were 342 survivors and 549 non-survivors. The number of survivors in each class can be seen by executing the cell above. As the results show out of the 342 survivors 136 were first class making up majority of the survival portion,87 survivors in the second and 119 survivors in the third class. In addition, out of the 549 non survivors 80 were first class, the lowest number of non-survivors out of all three classes. Second class survivors are less than third class but this is because there are lower number of second class passengers in this sample. Looking at it proportionately out of the 184 second class members 87 survived standing at 47%, compared to third class members where 119 survived out of 491 standing at only 24%. Percentage of non survivors in the first class is the lowest at 14.57% whereas non-survival of third class is at 67.5%. This data makes sense because higher class means having access to boats, upper/better parts of the ship and having access to the best available resources. Not to mention higher class means higher possibility of having personal relationship with high ranking crew members who can arrange a safe and sound rescue for the passenger. Meanwhile third class members were put towards the bottom parts of the ship which made it harder to escape and they did not have access to the best resources such as boats. In the bar graph we can see the red bar for non-survivors (third class) is significantly taller than the rest of the bars indicating, that the highest number of non-survivors is among third class members. 

# In[251]:

## Find correlation between gender and survival rate


fig = plt.figure(figsize=(5,3))
gender = np.array(data['Sex'])

print data.groupby(['Sex','Survived'])['PassengerId'].count()
print " "
print data.pivot_table(index = ['Pclass', 'Sex'], columns='Survived', values='PassengerId', aggfunc=len)

## Plot graph for number of males and females on board
plt.subplot2grid((1,3),(0,1))
data.Sex.value_counts().plot(kind="bar")
plt.title('Number of males and females on board', size=8)

## Plot graph for male survivors and female survivors 
## Put all males, females and corresponding survivors in a data frame 
gender_frame = pd.DataFrame({'survivor':survived == 1, 'male':gender == 'male', 'female':gender=='female'})
## Group out those who survived by setting survived ==1 and using groupby('survivor')
gender_frame_plotter = gender_frame.groupby('survivor').sum() 
gender_frame_plotter.plot(kind='bar', alpha=0.44) 
plt.title('Male and Female survivors')



# In[281]:

## Plot each gender in upper and lower class based on whehter or not they survived

## Plot survival rates among first class males and females
plt.subplot2grid((2,8),(0,1))
first_class_male = data.Survived[data.Pclass!= 3][data.Sex == 'male'].value_counts().plot(kind='bar')
plt.title('Upper Class male survivors', size=6)
plt.xlabel('survived or not, survived =1')

plt.subplot2grid((2,8),(0,3))
first_class_female = data.Survived[data.Pclass!= 3][data.Sex == 'female'].value_counts().plot(kind='bar')
plt.title('Upper Class female survivors', size=6)

## Plot survival rates among third class males and females
plt.subplot2grid((2,8),(0,5))
third_class_female = data.Survived[data.Pclass==3][data.Sex == 'female'].value_counts().plot(kind='bar')
plt.title('Third Class female survivors', size=6)

plt.subplot2grid((2,8),(0,7))
third_class_male = data.Survived[data.Pclass==3][data.Sex == 'male'].value_counts().plot(kind='bar')
plt.title('Third Class male survivors', size=6)


# Next we will try to find a relationship between gender and survival. Execute the cell above to see the results. The statistics show a much higher rate in female survival than male. 68% of the survivors were female and 85% of the non-survivors were male. There are also more female survivors in each class than male survivors in each class. In addition there are more male non-survivors in each class than female non-survivors in each class. The highest non-survival is among male third class passengers. This sounds very reasonable because Titanic was a ship made in the West, travelling within the west and carrying people of the West and in the west we have the ideology "women and children first". Women whom are mothers are also likely to arouse some level of "sympathy" since they have children with them. We can see here that gender and class are both related when it comes to survival. If you're a passenger in the third class you have a higher chance of surviving if you're a woman. There are 72 female survivors while only 47 males in the third class, 70 females and 17 males survivors in the second class, 91 female survivors and 45 male survivors in the first class. 

# Last we will try to analyze any relationship between age and survival. I will take a different approach in doing this. There are a wide variety of ages within big ranges, from baby to senior from children to adult. Not only that there are babies in the non survival list and survival list, there are adults in survival and non survival list. It is impossible to find one age that survived more than another. Rather I will group them, children(0-12), teen(13-20), adult(20-65), senior(65+) and see how many survived and how many did not survive in each group. I will also take the mean age of survivors and non-survivors and calculate the t-statistic to see if the average ages are approximately equal or not. This will also be a good method since a lot of the ages are missing. It's a good way to calculate the "overall ages" and take into account any outliers and come up with an appropriate statistic. 
# 
# To calculate the t-statistic I will be using the two independent age samples(survivors and non-survivors). There are 290 ages in the survivor list and 424 in non-survivor. As mentioned earlier, some of the ages are missing. This makes the degrees of freedom 712. At an alpha level of 0.05 on a two tailed test, the t-critical value is 1.962. The null hypothesis here is the average of the two age groups will be approximately equal, that is the t-statistic is less than the critical value. The alternative hypothesis is the two age groups will not be equal, or the t-statistic will higher than the t-critical. 

# In[284]:

## Find relation between age and survival 

## Sort survivors into 4 age groups, children(0-12), teen(13-20), adult(20-65), senior(65+) and find out the number of 
## males and females in each group

age = np.array(data['Age'])
s = pd.Series(data['Age'].values)

children = np.array(s[s<13].values)
teen = np.array(s[(s >=13) & (s<21)].values)
gender = np.array(data['Sex'])

adult = np.array(s[(s >= 21) & (s < 65)].values)
senior = np.array(s[s >=65].values)
surv_original = pd.Series(data['Survived'].values)


def age_group_func(gender_type,age_group,age_range): 
    df2 = pd.DataFrame({'survivor': survived == 1,gender_type: gender == gender_type, age_group: s.isin(age_range)}) 
    group1 = df2.groupby(['survivor', gender_type])
    plotter = group1.sum()
    print plotter
    print " "
    
children = age_group_func('male','children',children)
teen = age_group_func('male','teen',teen)
adult = age_group_func('male','adult',adult)
senior = age_group_func('male','senior', senior)


survivor_ages = pd.DataFrame({'survivor': survived == 1, 'age':age})
x = survivor_ages.groupby('survivor').mean()
y= survivor_ages.groupby('survivor').std()
print "Mean ages:"
print x
print "Standard deviation:"
print y
print " "
        


# Out of total of 69 children (age 0-12) 40 or 58% survived, from a total of 111 teenagers 42 or 38% survived, from 523 adults 207 or 40% survived, from 188 seniors 53 or 28% survived. Children have the highest survival rate. However, once again this data can be misleading because as mentioned earlier many ages are missing, so for instance many of the passengers that are survivors with missing ages could have been seniors. But overall since we see so many senior non survivors over survivors we can "assume" that seniors are relatively not advantaged. Also many of the missing ages could have children non-survivors.  Putting children aside we also see more female survivors and more male non-survivors in each group. The t-statistic(2.046) was higher than the t-critical(1.962) so we have to reject the null hypothesis. 
# 
# Overall we see that the higher class passengers have a higher survival rate. Amongst all classes, there are higher number of survivors for females than males. Females amongst all ages also have a higher survival rate. This is only not true for children (age 0-12) and senior. However, there are a lot of ages missing and there are 0 senior female non-survivors and 0 female senior  survivors, so no data for females seniors were encountered. There are also many male non-survivors among seniors. Since we only have one sample and not the full population and many ages are missing, I calculated the t-statistic because t-statistic can make a presumption and an explanation from a sample about the population. T-statistic can also be used here to see if the average ages of the survivors and non-survivors are approximately equal. I found that they were not. Everything mentioned throughout the report are results and possible explanations for them. There is no "100% certainity" that the results were due to correlation. For example, we saw that higher classes has more survivors but this doesn't mean surival was solely based on wheter or not you were in first class or your gender or age, there were many third class survivors as well. It's important to know that the numbers don't necessarily show a correlation because it could cause a potential bias, for example people may think being first class means guaranteed survival or being female means guaranteed survival since survival rate among females and first class members are higher, but clearly this is not true as there many first class non-survivors and many female non survivors as well. Once again based on certain ideologies such as "women and children first" we can assume that this is the reason more women survived but this doesn't mean being a woman means you are going to survive. The data should perhaps include which passengers had access to boats, this can give a more straight forward explanation as to who has guaranteed survival. For example, contain a column called "access to boats" and if the passenger had access to boats we can input a 1 in that row or 0 otherwise. And then we can check each row to see if they're a survivor and if they had access to a boat and vice versa. 
# 
