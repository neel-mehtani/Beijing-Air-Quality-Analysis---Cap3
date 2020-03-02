import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from aqi_calc import aqi_level


#Load preprocessed data

df = pd.read_csv('beijing_grouped_daily.csv')

#categorize pollution levels
 
df['Pollution'] =  df['AQI'].apply(lambda x: aqi_level(x))


#create count vector for each category


pollution_counts = {'Good': len(df[df['Pollution'] == 'good']) ,
                          'Moderate': len(df[df['Pollution'] == 'moderate']) , 
                          'USG': len(df[df['Pollution'] == 'usg']), 
                          'Unhealthy': len(df[df['Pollution'] == 'unhealthy']), 
                          'Very Unhealthy': len(df[df['Pollution'] == 'very unhealthy']), 
                          'Hazardous': len(df[df['Pollution'] == 'hazardous'])}

labels = list(pollution_counts.keys())
sizes = list(pollution_counts.values())

df_counts = pd.DataFrame({'Count': sizes}, index = labels)


#Sns bar plot
sns.set(style="whitegrid")
ax = sns.barplot(x= labels, y="Count", data=df_counts)
plt.title('Frequency of Air Quality Days')
plt.savefig('pollution_counts.png')
plt.show()


