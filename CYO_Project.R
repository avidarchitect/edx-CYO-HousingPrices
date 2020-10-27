#' ---
#' title: 'edX Data Science Capstone: Choose Your Own (CYO): Housing Prices'
#' author: "Komalkumar Tagdiwala"
#' date: "October 2020"
#' ---
options(scipen=999) # Disable scientific notations in all graphs

#' ## Dataset
#' This project explores the use of machine learning to predicting housing prices 
#' for a dataset available on www.kaggle.com at 
#' https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data. 
#' The underlying source of this data is the Ames Housing dataset (Ames, Iowa) 
#' compiled by Dean De Cook for use in data science education.  
#' It's an incredible alternative for data scientists looking for a modernized 
#' and expanded version of the often cited Boston Housing dataset.
#' 
#' ## Goal
#' The goal of this Capstone is to train a machine learning algorithm using 
#' the inputs in one subset to predict housing prices in the validation set. 
#' The data set already includes separate train.csv and test.csv files 
#' provided by Kaggle for training and validation sets. RMSE, known as the 
#' Root Mean Square Error, will be used to evaluate how close the predictions 
#' made by our model are to the actual/true values contained in the validation set. 
#' We will pick the model that yields the lowest RMSE.  

# Load the required libraries

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# Library for feature importance/selection
if(!require(Boruta)) install.packages("Boruta", repos = "http://cran.us.r-project.org")

# Library for comprehensive data exploration
if(!require(DataExplorer)) install.packages("DataExplorer", repos = "http://cran.us.r-project.org")

#favstats and utility functions for data exploration
if(!require(mosaic)) install.packages("mosaic", repos = "http://cran.us.r-project.org")

# For describe() for data exploration
#'Hmisc Contains many functions useful for data analysis, 
#'high-level graphics, utility operations, etc.
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")

#'Tools for Splitting, Applying and Combining Data. Example: ddply()
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")

#'To model using Random Forest
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

# For computing accuracy of predictions
if(!require(forecast)) install.packages("forecast", repos = "http://cran.us.r-project.org")

# For Classification and Regression Tree-based modeling (using cart)
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
# # To visualize results from cart-based model
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")

# To view tidy results
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")

#' Create training and validation set (final hold-out test set). 
#' The file train.csv was downloaded and made available in the local folder. 
#' Read it from the local file system.
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
training <- read_csv('train.csv') # Tibble of 1460 observations of 81 variables
validation <- read_csv('test.csv') # Tibble of 1459 observations of 80 variables

#' We begin by first examining what training and validation are.
class(training)
class(validation)

#' Let us now review the structure of the dataset to understand its composition. 
str(training)
#' We see that *our training set* has **1,460** observations of 81 variables/features.

# Validation Set = validation
colnames(validation)
#' Our *validation set* has **1,459** observations of the same 80 features 
#' with the exception of the SalePrice column, which is our outcome/dependent 
#' variable that must be predicted.

#' We are interested in the prediction of SalePrice (y). Consequently, we can 
#' make the following determination:
#' - SalePrice = Outcome/Dependent Variable "y"
#' - Remaining 80 variables are the predictors/independent variables that will be used for our analysis.

#' Let us see a few of the records to examine type different values in some of the rows
head(training)

#' The different features present in the dataset can be easily summarized using 
#' the *introduce()* function in the **dataexplorer** library as follows.
introduce(training)

#' We see that of the 81 features, 
#' **43** are **discrete** while **38** are **continuous** in nature. 

#' Given the volume of features available in this dataset, the provider of 
#' this dataset has included a separate **text** file, **description.txt**, 
#' that includes comprehensive descriptions for each field along with a 
#' detailed explanation of the encoded values contained in each field. 
#' For ease of reference, I am pasting the description for each field below.

#' Outcome (y)
#' SalePrice: Contains the SalePrice a buyer paid for a given house. 
#' This is what we want to predict using machine learning.
#' 
#' **FIELD DESCRIPTIONS from data_description.txt**

#'   1. Id: House ID
#'   2. 1stFlrSF: First Floor square feet
#'   3. 2ndFlrSF: Second floor square feet
#'   4. 3SsnPorch: Three season porch area in square feet
#'   5. Alley: Type of alley access to property
#'   6. Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
#'   7. BldgType: Type of dwelling
#'   8. BsmtCond: Evaluates the general condition of the basement
#'   9. BsmtExposure: Refers to walkout or garden level walls
#'  10. BsmtFinSF1: Type 1 finished square feet
#'  11. BsmtFinSF2: Type 2 finished square feet
#'  12. BsmtFinType1: Rating of basement finished area
#'  13. BsmtFinType2: Rating of basement finished area (if multiple types)
#'  14. BsmtFullBath: Basement full bathrooms
#'  15. BsmtHalfBath: Basement half bathrooms
#'  16. BsmtQual: Evaluates the height of the basement
#'  17. BsmtUnfSF: Unfinished square feet of basement area
#'  18. CentralAir: Central air conditioning
#'  19. Condition1: Proximity to various conditions
#'  20. Condition2: Proximity to various conditions (if more than one is present)
#'  21. Electrical: Electrical system
#'  22. EnclosedPorch: Enclosed porch area in square feet
#'  23. ExterCond: Evaluates the present condition of the material on the exterior
#'  24. Exterior1st: Exterior covering on house
#'  25. Exterior2nd: Exterior covering on house (if more than one material)
#'  26. ExterQual: Evaluates the quality of the material on the exterior 
#'  27. Fence: Fence quality
#'  28. FireplaceQu: Fireplace quality
#'  29. Fireplaces: Number of fireplaces
#'  30. Foundation: Type of foundation
#'  31. FullBath: Full bathrooms above grade
#'  32. Functional: Home functionality (Assume typical unless deductions are warranted)
#'  33. GarageArea: Size of garage in square feet
#'  34. GarageCars: Size of garage in car capacity
#'  35. GarageCond: Garage condition
#'  36. GarageFinish: Interior finish of the garage
#'  37. GarageQual: Garage quality
#'  38. GarageType: Garage location
#'  39. GarageYrBlt: Year garage was built
#'  40. GrLivArea: Above grade (ground) living area square feet
#'  41. HalfBath: Half baths above grade
#'  42. Heating: Type of heating
#'  43. HeatingQC: Heating quality and condition
#'  44. HouseStyle: Style of dwelling
#'  45. Kitchen: Kitchens above grade
#'  46. KitchenQual: Kitchen quality
#'  47. LandContour: Flatness of the property
#'  48. LandSlope: Slope of property
#'  49. LotArea: Lot size in square feet
#'  50. LotConfig: Lot configuration
#'  51. LotFrontage: Linear feet of street connected to property
#'  52. LotShape: General shape of property
#'  53. LowQualFinSF: Low quality finished square feet (all floors)
#'  54. MasVnrArea: Masonry veneer area in square feet
#'  55. MasVnrType: Masonry veneer type
#'  56. MiscFeature: Miscellaneous feature not covered in other categories
#'  57. MiscVal: $Value of miscellaneous feature
#'  58. MoSold: Month Sold (MM)
#'  59. MSSubClass: Identifies the type of dwelling involved in the sale.
#'  60. MSZoning: Identifies the general zoning classification of the sale.
#'  61. Neighborhood: Physical locations within Ames city limits
#'  62. OverallCond: Rates the overall condition of the house
#'  63. OverallQual: Rates the overall material and finish of the house
#'  64. PavedDrive: Paved driveway
#'  65. penPorchSF: Open porch area in square feet
#'  66. PoolArea: Pool area in square feet
#'  67. PoolQC: Pool quality
#'  68. RoofMatl: Roof material
#'  69. RoofStyle: Type of roof
#'  70. SaleCondition: Condition of sale
#'  71. SalePrice: Selling price of the house
#'  72. SaleType: Type of sale
#'  73. ScreenPorch: Screen porch area in square feet
#'  74. Street: Type of road access to property
#'  75. TotalBsmtSF: Total square feet of basement area
#'  76. TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#'  77. Utilities: Type of utilities available
#'  78. WoodDeckSF: Wood deck area in square feet
#'  79. YearBuilt: Original construction date
#'  80. YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#'  81. YrSold: Year Sold (YYYY)

#' ## Key Steps
#' We will be undertaking the following steps to achieve our goal of predicting the **SalePrice**:
#' * Data Wrangling/Cleaning/Pre-processing
#' * Data Exploration
#' * Data Visualization
#' * Insights Gained from the prior steps
#' * Build one or more Machine Learning Models
#' * Make Predictions using our model(s)
#' * Choose the model with lowest RMSE
#' 
#' We begin data exploration using the **DataExplorer** package, which allows us
#' to quickly explore key characteristics of the data set including several 
#' visualization plots with minimum lines of code. Similarly, **Hmisc** provides
#' a comprehensive summary of the various features of the data set including 
#' missing values, unique values, etc. Using the findings from both these packages, 
#' we will undertake **Feature Engineering** by dropping columns that contain 
#' several missing values while keeping others. 
#' 
#' Next, we will employ the **Boruta** package to further improve our feature engineering 
#' exercise to narrow down on those features that have the potential for maximum 
#' impact on our outcome variable, SalePrice.
#' 
#' For the model building, we will begin with **Random Forest**. 
#' Next, we will  build a **Classification and Regression Tree (CART)**-based model 
#' to analyze the splits and understand how decisions are made at the various nodes. 
#' We will then undertake **linear regression** to identify statistically significant features 
#' and compute RMSE. 
#' 
#' Finally, we will be **comparing the RMSEs** obtained from each of the models 
#' and **choose the one with the lowest RMSE**.


#' ``# Methods/Analysis
#' ## Process and Techniques
#' ### Data Cleaning/Pre-processing/Wrangling
#' We start with the summary() function to obtain the big picture on our dataset
#' for things, such as min, max, median, quantiles, class, etc.
summary(training)

#' Let us obtain a more comprehensive review of what each column possesses. 
#' We can employ the Hmisc::describe() to obtain this comprehensive view. 
hd<-Hmisc::describe(training)

#' Because the output of above function runs over 14 pages, I have commented out the
#' printing of the return value but we will be using this variable, **(hd)**, 
#' later in the analysis of missing values. The 14 pages contain comprehensive output 
#' for each of the 81 fields contained in the data set including information, 
#' such as count of missing values, unique values, mean, lowest values, highest values, etc.

#' For example, the Id field has the following characteristics
hd$Id

#' Similarly, the MSZoning field has the following characteristics
hd$MSZoning

#' **Unique Values**
# Extract unique values for each column in the dataset
df_unique_values <- lapply(training, unique)

#' Get a count of unique values in each column. Then print the count for unique 
#' values in each column sorted in ascending order of count
k <- lengths(df_unique_values)
sort.int(k)

#' **Missing Data**
plot_missing(training)

#' We see that most of the features have no missing values but there are some 
#' that do have some missing values while others have a huge amount of missing values. 
#' Because it hard to visualize the above graph with the one having 0 missing values, 
#' we will now focus only on the ones that do have missing values.
 
#' Plot only those features with missing values
plot_missing(training,missing_only = TRUE)

#' We find that the following features have a ton of missing values:
#' 1. PoolQC (99.52% missing)
#' 2. MiscFeature (96.3%)
#' 3. Alley (93.77%)
#' 4. Fence (80.75%)

#' We are better off dropping these columns from our analysis as these won't help much with our analysis.
final_training <- drop_columns(training,
                               c("PoolQC","MiscFeature","Alley","Fence"))
#' Let us plot the missing values again. 
plot_missing(final_training,missing_only = TRUE)

#' The **FireplaceQu** field has 47.26% missing values and a recommendation 
#' that it is "Bad" for analysis. Let us get more details. We will check the
#' Hmisc::describe done earlier for more details about **FireplaceQu**
hd$FireplaceQu

#' We find that of the 1460 records in the training set, 
#' **690 records have missing values for FireplaceQu** and it has categorical data. 
#' We are better off dropping this column as well from our analysis.
final_training <- drop_columns(final_training,"FireplaceQu")

#' We will make one final plot for missing values.
plot_missing(final_training,missing_only = TRUE)

#' The plot looks good to proceed now. 
#' Let us review the number of remaining columns in our final_training set 
#' which will be used for further analysis including the splitting for 
#' **training_set** and **test_set** for use against one or more models.
dim(final_training)
#' The final_training contains 1460 records of 76 features.   

#' Rename 2 of the features that begin with a number since certain functions 
#' cannot deal with variables that begin with a number.
names(final_training)[names(final_training) == "1stFlrSF"] <- "First_Floor_SF"
names(final_training)[names(final_training) == "2ndFlrSF"] <- "Second_Floor_SF"

# Perform the same change for validation set
names(validation)[names(validation) == "1stFlrSF"] <- "First_Floor_SF"
names(validation)[names(validation) == "2ndFlrSF"] <- "Second_Floor_SF"

#' **Data Pre-processing**  
#' To avoid the issue of overfitting by building different models and subjecting 
#' them to the same validation/final hold-out set 
#' (as confirmed with the TA/edx Staff in edX discussion forum), 
#' we will split our current University provided edx training set into a train and test set.

library(caTools)
library(dplyr)
library(tidyr)
set.seed(123)

############################################
# Generate the Training and Test set from pre-processed training set, 
# final_training, to avoid overfitting 
############################################

#We will go for 80-20 split of Train and Test data
split = sample.split(final_training$SalePrice, SplitRatio = 0.8) 

# New Training Set to be used by all models
training_set = subset(final_training, split == TRUE) #  1258 obs. of 76 variables (80% of training set)

# New Test Set to be used by all models
test_set = subset(final_training, split == FALSE)#  202 obs. of 76 variables (20% of training set)

#' 
#' Effectively, our new test set has 20% of original training data and new training set has 80% of training data
#' 
#' For **ALL** our models, we will be using these new training_set and test_set to make predictions
#' and compare RMSE values.

#' For the **CHOSEN** model, we will additionally subject that to the original validation set
#' to make predictions on validation data set that **does not contain our outcome variable, SalePrice**. 

#' ### Data Exploration and Visualization
#' We begin exploring the different features of the dataset including the outcome **SalePrice** that we want to predict. Understanding what the different dataset features contain helps us gain meaningful insights about how each attribute/feature contributes to our data analysis in addition to determining the appropriate modeling technique.
#' 
#' #### Categorical Features

#' Upon close inspection of the feature description provided in data_description.txt 
#' and examining the dataset values, we identify the following 38 categorical features.
categorical_features<-c("BldgType","BsmtCond","BsmtExposure","BsmtFinType1",
                        "BsmtFinType2","BsmtQual","CentralAir","Condition1",
                        "Condition2","Electrical","ExterCond","Exterior1st",
                        "Exterior2nd","ExterQual","Foundation","Functional",
                        "GarageCond","GarageFinish","GarageQual","GarageType",
                        "Heating","HeatingQC","HouseStyle","KitchenQual",
                        "LandContour","LandSlope","LotConfig","LotShape",
                        "MasVnrType","MSZoning","Neighborhood","PavedDrive",
                        "RoofMatl","RoofStyle","SaleCondition","SaleType",
                        "Street","Utilities")

categorical_features

#' **Three** of the above 38 features include **numeric values** that will need 
#' to be **one-hot encoded** later in the modeling and analysis section below. 
#' The 3 features involved are:
#' 1. MSSubClass
#' 2. OverallCond
#' 3. OverallQual

class(final_training$MSSubClass)
class(final_training$OverallCond)
class(final_training$OverallQual)

#' #### Visualize Histogram of Continuous Variables
#' Plotting the histogram for continuous variables, including our outcome **SalePrice** 
#' helps us visualize the distribution of these variables. We will use the 
#' plot_histogram() function in the DataExplorer package. 
#' 
#' Because we have nearly 38 continuous features in the data set, the output 
#' of this command will be 38 histograms that will run into multiple pages. 
options(repr.plot.width = 4, repr.plot.height = 3,scipen=999)
plot_histogram(training_set,nrow=3L,ncol=2L)


#' #### Insights on Outcome: SalePrice Distribution
#' One of the assumptions of Linear Regression is that for any fixed value of X,
#' outcome, Y, is normally distributed. Let us confirm if that is indeed the case
#' with our outcome variable, SalePrice. 

ggplot(training_set, aes(x = SalePrice, fill = ..count..)) +
  geom_histogram(binwidth = 10000) +
  ggtitle("Distribution of Outcome variable, SalePrice") +
  ylab("House count") +
  xlab("SalePrice (Outcome, Y)") + 
  theme(plot.title = element_text(hjust = 0.5))

#' We see that distribution for SalePrice is skewed. To account for this discrepancy, 
#' let us take a log of SalePrice to adjust the distribution for our further analysis 
#' and plot the distribution again.

training_set$logSalePrice <- log(training_set$SalePrice)
options(scipen=10000)

ggplot(training_set, aes(x = logSalePrice, fill = ..count..)) +
  geom_histogram(binwidth = 0.1) +
  ggtitle("Distribution of transformed Outcome variable, logSalePrice") +
  ylab("House count") +
  xlab("Log of SalePrice (Outcome, Y)") + 
  theme(plot.title = element_text(hjust = 0.5))

#' #### Insights on Categorical Feature: MSZoning

#' **MSZoning** identifies the general zoning classification of the sale using the following scheme.
#' -  A: Agriculture
#' -  C: Commercial
#' - FV: Floating Village Residential
#' -  I: Industrial
#' - RH: Residential High Density
#' - RL: Residential Low Density
#' - RP: Residential Low Density Park 
#' - RM: Residential Medium Density 

#' Let us see how the houses are distributed across these different zones.
ggplot(training_set, aes(x = MSZoning, fill = MSZoning )) + 
      geom_bar()+ 
      scale_fill_hue(c = 150)+
      ggtitle("Distribution of MSZoning")+
      xlab("MSZoning")+
      ylab("Number of Houses")+
      theme(plot.title = element_text(hjust = 0.5),
            legend.position="right", 
            legend.background = element_rect(fill="grey100", size=0.5, 
                                             linetype="solid",colour ="black"))+
      geom_text(stat='count',aes(label=..count..),vjust=-0.25)


options(repr.plot.width=9, repr.plot.height=6)

# boxplot of SalePrice by MSZoning
# Display average value of SalePrice as a BLUE dot
ggplot(training_set, aes(MSZoning,SalePrice, fill=MSZoning)) + 
  geom_boxplot(alpha=0.3,outlier.colour = "red") +
  stat_summary(fun=mean, geom="point", shape=20, size=4, color="blue", fill="red")+
  theme(legend.position="none")+
  ggtitle("SalePrice by MSZoning")+
  theme(plot.title = element_text(hjust = 0.5))

#' Looking at the above plot, we find that Floating Village (FV) has the highest average SalePrice 
#' followed by Residential Low Density (RL). The flexibility in development can be 
#' attributed to the higher cost. 
#' 
#' Another possibility is that the square footage available in FV might be 
#' more than its Residential counterparts making it more desirable and hence, more expensive. 
#' We also find that the lowest average SalePrice is in the Commercial Zone. 
#' **Is it possible that the square foot area available is indeed a contributing factor to price?**
 
#' Let's find out by checking the average square foot of houses in each Zone. 
options(repr.plot.width=9, repr.plot.height=6)
# boxplot of Home Area by MSZoning
# Display average square feet area as a RED asterisk

ggplot(training_set, aes(x=MSZoning,y=GrLivArea, fill=MSZoning)) + 
  geom_boxplot(alpha=0.3, outlier.colour = "red") +
  stat_summary(fun=mean, geom="point", shape=8, size=4, color="red", fill="red")+
  theme(legend.position="none")+
  ggtitle("Home Area in Square Feet by MSZoning")+
  theme(plot.title = element_text(hjust = 0.5))

#' Let us also see the actual values for average square foot per zone.

library(plyr)

x<-ddply(training_set, .(MSZoning), summarize,  average_area=mean(GrLivArea))
x[order(-x$average_area),] # Display in descending order of average area

#' The table above validates our finding that the average area of homes in 
#' Floating Village is higher (1,575 sq.ft.) than its Residential counterparts 
#' and the average area for homes in the Commercial zone is the lowest (1154 sq.ft.). 

#' #### Insights on Categorical Feature: Building Type (BldgType)
#' **BldgType** represents the type of dwelling and it can take any of the following five values:
#' 1. 1Fam: Single-family Detached	
#' 2. 2FmCon: Two-family Conversion; originally built as one-family dwelling
#' 3. Duplx: Duplex
#' 4. TwnhsE: Townhouse End Unit
#' 5. TwnhsI: Townhouse Inside Unit
 
#' Just as we determined the average SalePrice by MSZoning in the prior section, 
#' we will now determine the average, maximum, and minimum SalePrice, and number of houses 
#' for each of the dwelling types.

x<-ddply(training_set, .(BldgType), summarize, 
                                    average_price=mean(SalePrice),
                                    number_of_houses=length(BldgType),
                                    minimum_price=min(SalePrice),
                                    maximum_price=max(SalePrice))
x[order(-x$average_price),] # Display in descending order of average price

#' We note that **Single-Family Detached** houses outnumber the other dwelling types 
#' not only in count (1050) but also in terms of average price ($189,686) 
#' and maximum price ($755,000) paid for a house in that category of building. 
#' There are only 27 **Two-family Conversion** type houses and that category 
#' possesses the lowest average price of $129,033.
#' 
#' We can find additional details about the distribution of houses in each category by plotting a histogram.
ggplot(training_set, aes(SalePrice)) +
 geom_histogram(aes(fill = BldgType), 
                position = position_stack(reverse = TRUE), 
                binwidth = 50000) +
 coord_flip() +
 ggtitle("SalePrice by Building Type") +
 ylab("Number of Houses") +
 xlab("SalePrice") + 
 theme(plot.title = element_text(hjust = 0.5),
       legend.position="right", 
       legend.background = element_rect(size=0.5, 
                                        linetype="solid", 
                                        colour ="black"))

#' Combining the results from the table showing minimum and maximum price 
#' by dwelling type with the Histogram plotted above, we make the following inferences:

#' - Houses that sold in the higher price range of 400K to 755K were Single-Family Detached.
#' - Houses of other dwelling types were confined to the range of 55K to 393K
#' - While it makes sense that the most expensive house ($755,000) was a Single-Family Detached type, 
#'   it is interesting to see that the least expensive ($34,900) also happens to 
#'   be a Single-Family detached.
#' - In summary, Single-Family Detached spans the overall range of SalePrice, 
#'   indicating the higher spread (and hence choice for homeowners) from a financial standpoint. 
 

#' #### Insights on Categorical Feature: Overall Quality (OverallQual)  
#' While we reviewed the distribution of SalePrice a few sections earlier, 
#' we will now review that in the context of Overall Quality (OverallQual) of the house 
#' which represents the overall material and finish of the house with values 
#' ranging from 1 through 10 with 1 being Very poor and 10 being Very Excellent.

ggplot(training_set, aes(SalePrice)) +
 geom_histogram(aes(fill = as.factor(OverallQual)), 
                position = position_stack(reverse = TRUE), 
                binwidth = 10000) +
 coord_flip() +
 ggtitle("SalePrice by Overall Quality") +
 ylab("Number of Houses") +
 xlab("SalePrice") + 
 scale_fill_discrete(name="Overall Quality\n1 = Very Poor, \n10 = Very Excellent")+
 theme(plot.title = element_text(hjust = 0.5),
       legend.position="right", 
       legend.background = element_rect(size=0.5, 
                                        linetype="solid", 
                                        colour ="black"))

#' We make the following findings:
#' - Houses with higher Overall Quality (9-10) sold at much higher prices ($275,000 to $755,000) 
#'   than the rest (which makes logical sense).
#' - Houses with Above Average quality (6) but below Excellent (9) sold in the 
#'   mid-range of $100,000 to $350,000.
#' - Both of the above findings jive well with the common sense that higher the overall quality, 
#'   higher the price a homeowner should expect to pay for the house.
#' - The distribution of houses in each quality category is fairly even/symmetric.

#' #### Visualize ALL Continuous Features by SalePrice  

#' We will now make box plots for all the continuous features by SalePrice 
#' to see the data distribution, central values, and variability. 
plot_boxplot(training_set, by="SalePrice",
              geom_boxplot_args = list("outlier.color" = "red"),
              nrow = 4L, ncol=2L)

#' #### Feature Engineering: Determine Important Features for Analysis
#' To get a sense on some of the features that should be important for our analysis, 
#' we will employ **Boruta**, an all relevant feature selection wrapper algorithm, 
#' capable of working with any classification method that output variable importance measure (VIM);portance with importance achievable at random, estimated using their permuted copies, and progressively eliminating irrelevant features to stabilize that test. **For more details on Boruta**, check out: **https://www.rdocumentation.org/packages/Boruta/versions/7.0.0/topics/Boruta**.
#' 
#' Perform Boruta search and check the output.
boruta_output <- Boruta(SalePrice ~ ., data=na.omit(training_set), doTrace=0)  
names(boruta_output)

#' Get significant variables including tentative.
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif) # Display Boruta significant results.

#' If you are not sure about the tentative variables being selected for granted, 
#' you can choose a TentativeRoughFix on boruta_output.' 
#' Do a tentative rough fix.
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

#' Boruta has decided on the 'Tentative' variables on our behalf. 

#' Variable Importance Scores
imps <- attStats(roughFixMod)
confirmed_features = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(confirmed_features[order(-confirmed_features$meanImp), ])  # descending sort


#' **Plot variable importance**\  
plot(boruta_output, cex.axis=.7, las=2, 
     xlab="", main="Variable Importance")
legend('top', legend=c("Confirmed","Rejected","Tentative"), 
       col=c("Green","Red","Yellow"),horiz = TRUE,
       text.font=4, cex=0.8,pch=15)

#' This plot reveals the importance of each of the features. 
#' The columns in green are 'confirmed' and the ones in red are not. 
#' There are couple of blue bars representing ShadowMax and ShadowMin. 
#' They are not actual features, but are used by the boruta algorithm 
#' to decide if a variable is important or not.
 
#' Here's a list of all the features that were confirmed using Boruta 
#' in descending order of importance. We will be using a subset of these 
#' features in our **Modeling Approach** with some of the categorical features 
#' that were analyzed earlier.
rownames(confirmed_features[order(-confirmed_features$meanImp), ])

#' #### Insights using Correlation
#' Let us first undertake correlation analysis against the top 20 features 
#' recommended by Boruta in the prior section.

#' List the top 20 important features confirmed by Boruta. 
#' Here, we take 21 because the 1st one, logSalePrice is something we added
#' as part of prior data processing and actually represents the outcome. So we will 
#' take the first 21 and skip the logSalePrice for this correlation analysis.

top_20_boruta<-rownames(confirmed_features[order(-confirmed_features$meanImp), ])[2:21]
top_20_boruta<- c(top_20_boruta,"SalePrice")
top_20_boruta<- training_set[top_20_boruta]

#' Before we make correlation plot, we need to make sure the features are numeric. 
#' We see that five of the top_20_boruta features are categorical in nature. 
#' Let us first convert them to numeric.

#' The three quality features - BsmtQual, ExterQual, and KitchenQual are encoded 
#' such that Excellent=5 and Poor=1 with the rest of the values in diminishing value of quality.

#' The remaining 2 categorical features are encoded as shown in the code below.

top_20_boruta$BsmtQual <- as.numeric(factor(top_20_boruta$BsmtQual, 
                                  levels = c("Ex", "Gd","TA", "Fa","Po"),
                                  labels = c(5,4,3,2,1) ,ordered = TRUE))

top_20_boruta$ExterQual <- as.numeric(factor(top_20_boruta$ExterQual, 
                                  levels = c("Ex", "Gd","TA", "Fa","Po"),
                                  labels = c(5,4,3,2,1) ,ordered = TRUE))

top_20_boruta$KitchenQual <- as.numeric(factor(top_20_boruta$KitchenQual, 
                                  levels = c("Ex", "Gd","TA", "Fa","Po"),
                                  labels = c(5,4,3,2,1) ,ordered = TRUE))

top_20_boruta$MSZoning <- as.numeric(factor(top_20_boruta$MSZoning, 
                                  levels = c("A", "C","FV", "I","RH","RL","RP","RM"),
                                  labels = c(1,2,3,4,5,6,7,8) ,ordered = TRUE))

top_20_boruta$GarageType <- as.numeric(factor(top_20_boruta$GarageType, 
                                  levels = c("2Types", "Attchd","Basment",
                                             "BuiltIn","CarPort","Detchd","NA"),
                                  labels = c(1,2,3,4,5,6,7) ,ordered = TRUE))


#plot correlation heatmap for SalePrice for the top_20_boruta confirmed features
options(repr.plot.width=8, repr.plot.height=6)
library(ggplot2)
library(reshape2)
qplot(x=Var1, y=Var2, data=melt(cor(top_20_boruta, use="p")), fill=value, geom="tile") +
   scale_fill_gradient2(low = "green", high = "red", mid = "white", 
   midpoint = 0, limit = c(-1,1), space = "Lab", 
   name="Correlation") +
   theme_minimal()+ 
   theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 8, hjust = 1))+
   coord_fixed()+
   ggtitle("Top 20 Boruta - Correlation Heatmap") +
   theme(plot.title = element_text(hjust = 0.4))

#' We make the following observations:
#' 1. Red shows positive correlation, whereas Green shows negative correlation.
#' 2. Pretty much all the 20 features confirmed by Boruta as being important 
#'    show strong correlation with SalePrice. This is confirmed by looking 
#'    at the plot and focusing on the **top row** for **SalePrice** which 
#'    shows higher shades of Red or Green for almost all of the 20 features.

#' We can safely proceed with further analysis having validated the strong correlation 
#' for the top 20 Boruta confirmed features.

#' #### Insights using ScatterPlot by SalePrice
#' We make one final plot to visualize the relationship of these 
#' Boruta-recommended top 20 features with SalePrice. 
#' This is a Scatter Plot of the features by SalePrice.
plot_scatterplot(top_20_boruta, by="SalePrice",nrow = 4L, ncol=3L)

#' We can confirm that:
#' - First_Floor_SF, Second_Floor_SF, GarageArea, TotalBstmtSF, BsmtFinSF1, 
#'   and LotArea have a positive relationship with SalePrice. 
#' - Similarly, the newer the house (YearBuilt, GarageYrBlt) or the more recent 
#'   the house was remodeled (YearRemodAdd), the higher the expected price for the house.
#' - The plots for FullBath and TotRmsAbvGrd plot show that the more the number 
#'   of full baths and number of rooms available above the ground, the higher the SalePrice.

#' ### Modeling Approach   

#' #### Data Pre-processing
#' Let us now update our training_set and test_set with the following changes:
#' 1. Only include the top 20 Boruta confirmed features.
#' 2. Include the log transformed SalePrice for modeling.
#' 3. Apply the one-hot encoding for the categorical features as was done for correlation

# Add logSalePrice to test_set
test_set$logSalePrice <- log(test_set$SalePrice)

# Add the outcome variable to our Validation set because it does not contain that column.
validation["SalePrice"]=0
validation["logSalePrice"]=0

# Reduce the number of features used for analysis by limiting our feature set 
# to the ones confirmed by Boruta as being important.
training_set<-training_set[c(names(top_20_boruta),"logSalePrice")]
test_set<-test_set[c(names(top_20_boruta),"logSalePrice")]
validation<-validation[names(top_20_boruta)]


# Convert any character vectors to factors in both the training and test set
training_set$ExterQual<- as.factor(training_set$ExterQual)
training_set$KitchenQual<- as.factor(training_set$KitchenQual)
training_set$MSZoning<- as.factor(training_set$MSZoning)
training_set$GarageType<- as.factor(training_set$GarageType)
training_set$BsmtQual<- as.factor(training_set$BsmtQual)


test_set$ExterQual<- as.factor(test_set$ExterQual)
test_set$KitchenQual<- as.factor(test_set$KitchenQual)
test_set$MSZoning<- as.factor(test_set$MSZoning)
test_set$GarageType<- as.factor(test_set$GarageType)
test_set$BsmtQual<- as.factor(test_set$BsmtQual)

validation$ExterQual<- as.factor(validation$ExterQual)
validation$KitchenQual<- as.factor(validation$KitchenQual)
validation$MSZoning<- as.factor(validation$MSZoning)
validation$GarageType<- as.factor(validation$GarageType)
validation$BsmtQual<- as.factor(validation$BsmtQual)

#' 
#' 
#' #### Random Forest
#' We will run the Random Forest algorithm against the training_set for 
#' different values of several of its key parameters - ntree, nodesize, and mtry. 
#' We are undertaking supervised learning and for regression using Random Forest, 
#' we set **nodesize = 5**. The parameter **mtry** represents the number of variables 
#' randomly sampled as candidates at each split. 
#' 
#' Note that the default values are different for classification (sqrt(p) 
#' where p is number of variables in x) and regression (p/3). 
#' In our case, we are going to limit our analysis to the top 20 Boruta 
#' confirmed features so a value of 20/3 = 6.33 is a good start. 

#' After playing with different values, I settled for **mtry=5** to obtain the 
#' lowest RMSE in combination with **ntree=600** and **nodesize=5**.   

# Model 1: Random Forest
set.seed(500)
RF <- randomForest(logSalePrice ~.-SalePrice, 
                   data = training_set, 
                   na.action=na.roughfix,
                   importance =TRUE,
                   ntree=600,
                   nodesize=5,
                   mtry=5)

#' Now let us plot the Dotchart of variable importance as measured by a 
#' Random Forest using the **varImpPlot()** function.

# variable importance
options(repr.plot.width=9, repr.plot.height=6)
varImpPlot(RF, type=1)


#' Next, we will make predictions against the test_set and compute accuracy 
#' using the *accuracy()* function from the **forecast** library. 

#prediction
pred_rf <- predict(RF, newdata=test_set )
acc_rf<-accuracy(pred_rf, test_set$logSalePrice)
acc_rf
acc_rf<- as.data.frame(acc_rf)

#' We will now build a table to keep track of the RMSE results from each of our models.
rmse_results <- data.frame(Model_Method="Random Forest",RMSE_Values=acc_rf$RMSE)
rmse_results %>% knitr::kable()

# Visualize Predicted versus Actual
plot(pred_rf, test_set$logSalePrice, 
     main = "Visualize Predicted vs. Actual logSalePrice",
     xlab="Predictions using Random Forest Model",
     ylab="Actual SalePrice in test_set") 
abline(a=0,b=1,col="blue")


#' #### Classification and Regression Tree-based Model using CART
#' We will now build a  regression tree model using the **cart** package. 
#' We set the method to "anova" and run *rpart* against our training_set's 
#' Boruta recommended top 20 features.
set.seed(500)
# Generate regression tree using rpart
fit <- rpart(logSalePrice ~.-SalePrice, 
             data = training_set, 
             method="anova")

#' Print the results and make a plot to visualize the results.
cp_table<-as.data.frame(printcp(fit)) # display the results

rpart.plot(fit,
           fallen.leaves=TRUE,
           main="Regression Tree using rpart")

#' Plot a Complexity Parameter Table for our fitted model.
plotcp(fit,
       minline = TRUE, 
       lty = 5, 
       col.lab = "red",
       col=2,
       upper="splits") # visualize cross-validation results

#' Plot the Approximate R-Square for different Splits.
par(mfrow=c(1,2)) # two plots on one page
rsq.rpart(fit) # Plots the Approximate R-Square for different Splits 

#' Prune the tree and plot the pruned tree. We begin by first identifying 
#' the minimum value for **xerror** and looking for the corresponding **cp** 
#' value to prune the tree from.
min_xerror<-min(cp_table$xerror)
min_xerror # 0.2972265

cp_4_min_xerror<- cp_table$CP[which.min(cp_table$xerror)]

pfit<- prune(fit, cp=cp_4_min_xerror) # # prune the tree using the cp value from cptable   

# plot the pruned tree
rpart.plot(pfit,
           fallen.leaves=TRUE,
           main=paste("Pruned Tree for cp=",
                      toString(cp_4_min_xerror),
                      " and min_xerror=",
                      min_xerror))

#' We essentially get the same tree even after pruning.

#' Let us now make predictions using our rpart fitted model.
cart_pred<- predict(fit, test_set,na.action = na.roughfix) # Make Predictions

#' Compute accuracy of this CART-based model.
acc_cart<- accuracy(cart_pred, test_set$logSalePrice) # Compute Accuracy
acc_cart<-as.data.frame(acc_cart)

#' Let us add the resulting RMSE value to our RMSE_Values table.
rmse_results <- bind_rows(rmse_results,
                          data.frame(
                            Model_Method="CART-based Regression Tree with Pruning",  
                            RMSE_Values = acc_cart$RMSE)) # Add RMSE to our table
rmse_results %>% knitr::kable()


#' #### Linear Regression-based Models
#' We will now build a linear regression model using the lm() function using the training_set.
#' **logSalePrice** is going to be a linear combination of multiple independent variables, 
#' the top 20 Boruta recommended features.
regressor = lm(formula = logSalePrice ~.-SalePrice, data = training_set)

# Let us review the contents of regressor using summary()
summary(regressor)

# create a dataframe from model's output
tm = tidy(regressor)

# visualize dataframe of the model using non scientific notation of numbers
options(scipen = 999)
tm

#' Let us now identify statistically significant variables returned by our 
#' linear regression model. To do this, we will filter out the coefficients 
#' that possess a p-value <=0.05.

# get variables with p-value less than 0.05 (Statistically Significant)
signif_coeff<- tm %>% filter(tm$p.value <= 0.05)

#' Let us display these coefficients in ascending order of their p-values. 
#' The ones at the top are the most statistically significant. 
signif_coeff[order(signif_coeff$p.value),] # Display in ascending order of p.value

#' Here's a visual on these coefficients using ggplot.
ggplot(signif_coeff, aes(x=term, y=p.value)) +
  geom_point(stat="identity") +
  theme(axis.text.x = element_text(angle=45, hjust=1, vjust = 1))+
  labs(title = "Independent Variables having p-value<=0.05")

#' Let us make our initial prediction against the test_set using the predict() function.
lm_pred <- predict(regressor,test_set,type = "response") # Make predictions

#' Compute residuals.
residuals <- test_set$logSalePrice - lm_pred # Compute residuals

linreg_pred <- data.frame("Predicted" = lm_pred, 
                          "Actual" = test_set$logSalePrice, 
                          "Residual" = residuals)

plot(lm_pred, 
     test_set$logSalePrice, 
     main = "Test Set: Predicted vs. Actual log SalePrice") 
abline(0,1,col="red")

#' Compute accuracy for our linear regression model.
acc_lm<-as.data.frame(accuracy(lm_pred, test_set$logSalePrice))
acc_lm

#' Compute the RMSE on this initial model to get a sense on our model's ability to make good predictions.
rmse_results <- bind_rows(rmse_results,
                          data.frame(
                            Model_Method="Linear Regression",  
                            RMSE_Values = acc_lm$RMSE)) # Add RMSE to our table
rmse_results %>% knitr::kable()

#' #### Insights Gained so Far

#' We make the following findings:
#' 1. Of the 3 models we experimented with, Random Forest yielded the lowest RMSE of 0.1076720.
#' 2. The CART-based model yielded the highest RMSE and despite pruning for the lowest value of xerror, the Regression Tree remained the same. The CART model did allow us to visualize how decisions are made at the various splits for the important features included in our training_set.
#' 3. The features that were recommended by Boruta turned out to be very valuable and were validated to have significant importance as confirmed by the varImpPlot and the statistically significant variables identified in Linear Regression (p-value<0.05).

#' Using RMSE as our metric for model selection, we will now employ Random Forest against our Validation set.
 
# Results
#' 
#' ## Predictions using Random Forest on Validation Set
#' As a final test before choosing this model, we will run this model against 
#' our our final holdout validation set. As noted in the **Overview** section,
#' our validation set provided as *test.csv* file **does not** contain the 
#' outcome variable, SalePrice. Consequently, **we won't have 
#'  anything to compare our predictions against** or compute RMSE against the 
#'  validation set. **We will merely make predictions** against the validation set.

#' ########################################################################
#' FINAL TEST against our hold-out validation set
#' ########################################################################

final_pred_rf <- predict(RF, newdata=validation)

validation$logSalePrice<- final_pred_rf
validation$SalePrice <- exp(validation$logSalePrice)  

# Reset NA with 0. We want this to help compute the min and max.
validation["SalePrice"][is.na(validation["SalePrice"])] <- 0 
validation["logSalePrice"][is.na(validation["logSalePrice"])] <- 0

# Display the minimum SalePrice predicted for data in the validation set
validation_with_non_zero_SalePrice<- validation %>% filter(SalePrice>0)
min(validation_with_non_zero_SalePrice$SalePrice)

# Display the maximum SalePrice predicted for data in the validation set
max(validation_with_non_zero_SalePrice$SalePrice)

# Display the average SalePrice predicted for data in the validation set
mean(validation_with_non_zero_SalePrice$SalePrice)

#' 
#' ## Model Performance
#' The results documented in the above table demonstrate how we considered several
#' modeling techniques and compare their accuracy to yield our chosen metric, 
#' **RMSE**. The **Random Forest** model yielded the best performance and we 
#' employed it against the final hold-out validation set to make our predictions 
#' as it yielded the **lowest RMSE score of 0.1076720**. 

#' # Conclusion
#' The Random Forest-based model helped achieve a better performance as measured 
#' using the RMSE score of 0.1076720 and it was used to make predictions for our 
#' final hold-out validation set. 
#' 
#' Because our validation set did not include the outcome variable, we stopped 
#' our analysis at making predictions.
#' 
#' **Chosen Model**: We choose the **Random Forest-based model** for this project 
#' as it yielded the lowest RMSE score of 0.1076720.
#' 
#' ## Limitations
#' One of the biggest challenges with this dataset was the sheer number of 
#' independent variables that can require us to spend countless hours of data exploration. 
#' While Boruta helped us identify nearly 43 features, we limited our analysis 
#' to the top 20 of those recommended features for building our models. 
#' 
#' We chose to do so to avoid overfitting and the curse of dimensionality problem 
#' and try different modeling techniques. 
#' 
#' Also, because we did not have the outcome available in our validation set, 
#' we had to stop our analysis with making predictions. As such, we could not 
#' compute the accuracy of our model against the validation set but that is exactly 
#' how real world works. We use some level of supervised learning to train our model 
#' and put it to use against new data to make predictions.
#' 
#' ## Future Work
#' We can reexamine the remaining 23 features confirmed by Boruta for our 
#' future analysis and see if there are additional features that significantly 
#' impact our outcome variable, SalePrice. We can also explore other modeling 
#' techniques instead of limiting ourselves to the three that we tried in this project. 

