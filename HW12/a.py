import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

x=pd.DataFrame([[40,160,'male',10],[60,170,'male',30],[50,160,'female',40],[40,150,'female',15]], columns=['weight','height','sex','age'])
class chosa:
   def __init__(self, y='male'):
       self.y=y
       self.z=x[x.sex==self.y]

   def a(self,w):
       v=self.z[self.z['age']>w]
       return(v.height.mean())

a1=smf.ols('weight~sex',data=x).fit().summary()
a2=smf.ols('weight~sex-1',data=x).fit().summary()
q=chosa('female')
a3=q.a(20)
q2=chosa()
a4=q2.a(10)
a5=q2.y


pass