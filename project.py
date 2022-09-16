import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

# ---------------------------------------------------------------------------

df = pd.read_csv("project.csv")

toefl_score = df["TOEFL Score"].tolist()
result = df["Chance of admit"].tolist()


# ---------------------------------------------------------------------------

fig = px.scatter(x=toefl_score, y=result)
fig.show()

toefl_score = df["TOEFL Score"].tolist()
gre_score = df["GRE Score"].tolist()

results = df["Chance of admit"].tolist()
colors=[]
for data in results:
  if data == 1:
    colors.append("green")
  else:
    colors.append("red")


fig = go.Figure(data=go.Scatter(
    x=toefl_score,
    y=gre_score,
    mode='markers',
    marker=dict(color=colors)
))
fig.show()


# ---------------------------------------------------------------------------
scores = df[["GRE Score", "TOEFL Score"]]
results = df["Chance of admit"]


# ---------------------------------------------------------------------------------------------------------------------------------
score_train, score_test, results_train, results_test = train_test_split(scores, results, test_size = 0.25, random_state = 0)


# ---------------------------------------------------------------------------
classifier = LogisticRegression(random_state = 0) 
classifier.fit(score_train, results_train)


# ---------------------------------------------------------------------------

results_pred = classifier.predict(score_test)

print ("Accuracy : ", accuracy_score(results_test, results_pred)) 


# ---------------------------------------------------------------------------

sc_x = StandardScaler() 

score_train = sc_x.fit_transform(score_train)  

user_gre_score = int(input("Enter the GRE score -> "))
user_toefl_score = int(input("Enter the TOEFL Score -> "))

user_test = sc_x.transform([[user_gre_score, user_toefl_score]])

user_result_pred = classifier.predict(user_test)

if user_result_pred[0] == 1:
  print("This user may pass!")
else:
  print("This user may not pass!")






