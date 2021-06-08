#!/usr/bin/env python
# coding: utf-8

# # Task 6 - Decision Tree Algorithm - Decision Tree Classifier
# # Task: For the given ‘Iris’ dataset, create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.
# 
# # Libraries/Datasets Used: Scikit Learn, Pandas, Pydotplus, Iris Dataset
# 
# # Task completed during Data Science & Analytics Internship @ The Sparks Foundation
# # By- Vivisha Singh

# # Importing libraries | Loading Iris datasets | Forming the iris dataframe into notebook

# In[8]:


# Importing libraries into notebook
import pandas as pd
import sklearn.datasets as datasets
from sklearn import tree


# In[9]:


# Loading iris dataset into the notebook
iris = datasets.load_iris()
print("Iris dataset loaded successfully")


# In[10]:



# Forming the iris dataframe
df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(df.head(15))

y=iris.target
print(y)


# # Defining Decision Tree Algorithm

# In[11]:



from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(df,y)

print("Decision Tree Classifier Created!!")


# # Visualizing the Decision Tree Created

# In[12]:


# installing the required libraries
get_ipython().system('pip install pydotplus')
get_ipython().system('pip install graphviz')


# In[13]:


# importing necessary libraries for Tree Visualization
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
print("Import Successful")


# In[14]:



# Visualizing the graph
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names = iris.feature_names, filled = True, rounded = True, special_characters = True, node_ids = True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

END OF CODE