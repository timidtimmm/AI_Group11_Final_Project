# AI_Group11_Final_Project_Read_Me

## Group Member


| 112550045 | 112550047 | 112550099 | 112550190 |
| -------- | -------- | -------- | -------|
| 曾歆喬|徐瑋晨  |蔡烝旭   |劉彥廷 |

## Requirements:
Python version: 3.10.12
```
pandas == 2.2.3
Matplotlib == 3.9.2
requests == 2.25.1
joblib == 1.5.0
torch == 2.7.0+cu126
numpy == 2.1.3
sklearn == 1.6.1
tqdm == 4.67.1
```

## Introduction:
We are students from the Department of Computer Science at National Yang Ming Chiao Tung University. This project is an application we developed as part of our coursework in the Introduction to Artificial Intelligence class. 

In this project, we primarily focus on the traffic flow in New York City. By training and utilizing AI models, including some factors like pedestrian demand, weather, and historical record, we aim to predict the approximate traffic volume on a given street, considering factors such as time of day, weather conditions, and other influencing variables.

==demo video:== 


## DataSet: total_dataset_final.csv
Our dataset is compiled by collecting and integrating data from sources such as Kaggle and publicly available government datasets. In this file, having roughly 377300 data, from 2016 to 2022. 

#### Columns:
ID,Boro,Date,Hour,weekday,volumn,segmentID,street,fromst,tost,Direction,temperature,precipitation,rain,cloudcover,windspeed,demand,Air_quality
#### data source:
[NYC traffic volumn counts from 2016 to 2022](https://data.cityofnewyork.us/Transportation/Traffic-Volume-Counts/btm5-ppia/about_data)
[weather from 2016 to 2022](https://www.kaggle.com/datasets/aadimator/nyc-weather-2016-to-2022)
[Pedestrian demand](https://data.cityofnewyork.us/Transportation/Pedestrian-Mobility-Plan-Pedestrian-Demand/fwpa-qxaf/about_data)
[Air quality](https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r/data_preview)
[Traffic volumn](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/data_preview)

## Model: Long Sort Term Memory(LSTM)
![image](https://hackmd.io/_uploads/SyHJdyMfgl.png)

## Model_detail: LSTM
In this project, we choose using the model -- "LSTM".
1. Before training model, we standardlized the input features.
2. We separated dataset into two parts, 80% for training, and 20% for testing.
3. We used training dataset to train LSTM model follow the below diagram.
4. We tested our model with testing dataset.

#### LSTM outline
The following diagram is the outline for our LSTM architecture. In the LSTM model, we have several cells(layer) to propagate the features and its loss.

![image](https://hackmd.io/_uploads/Sy2RYMQMee.png)

#### LSTM cell
Some details in LSTM cell:
* forget gate($f$): Determines what portion of the past information should be discarded.
* input gate($i$): Controls what new information should be added.
* cell state update($c$): Combines the forget gate and input gate to update the cell state.
* output gate($o$): Determines what part of the updated cell state should be output.
* hidden state($h$): Taking the element-wise product of the output gate and the cell state.

Formula:
$i$ $=$ $sigmoid(i)$
$f$ $=$ $sigmoid(f)$
$g$ $=$ $tanh(g)$
$o$ $=$ $sigmoid(o)$

$c_t = f。c_{t-1} + i。g$
$h = o * tanh(c)$

![image](https://hackmd.io/_uploads/rJtyxWfGee.png)

##### fraction of our implement:
```python
    def forward(self, x, h_prev, c_prev):
        gates = self.x2h(x) + self.h2h(h_prev)                      #linear transformation
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i_gate)              #input gate
        f_gate = torch.sigmoid(f_gate)              #forget gate
        g_gate = torch.tanh(g_gate)                 #cell gate
        o_gate = torch.sigmoid(o_gate)              #output gate 
        c = f_gate * c_prev + i_gate * g_gate       #cell state : c = f 。 c_prev + i 。 g
        h = o_gate * torch.tanh(c)                  #hidden state: taking the element-wise product of the output gate and the cell state
        return h, c                                 #pass to next cell
```
## experiments
//TODO

## Real World application:
In this part, we use API(Application Programming Interface) to capture the real time information that will be used in prediction of traffic. We get at must data from Google weather API, such as temperature, precipitation, rain, cloudcover, and windspeed. And get the Air quality from IQair. Thus, we can achieve the real world application to predict the traffic for the next 6 hours.

* API Source
    * https://developers.google.com/maps/documentation/weather/overview?hl=zh-tw
    * https://api-docs.iqair.com/?version=latest
    
Since one of API needs account's key. Please use own API key to run the real world application.

in the file "./application/api_application.py line 75"

The final result is like that
![image](https://hackmd.io/_uploads/ryjHizQGex.png)
