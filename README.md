# AI_Group11_Final_Project_ReadMe

## Group Member


| 112550045 | 112550047 | 112550099 | 112550190 |
| -------- | -------- | -------- | -------|
| 曾歆喬|徐瑋晨  |蔡烝旭   |劉彥廷 |

## Requirements
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
## File Structure
* AI_group11_Final_Project 
    * application : real world application code
    * model_baseline : linear regression baseline model
    (including the models, resulting csv files, plots of comparison of actual value and predicted value)
    * model_global : The LSTM model trained without any grouping
    (including the models, resulting csv files, plots of the training loss)
    * model_group : The LSTM model trained by grouping into (boro, street) pair
    (including the models, resulting csv files, plots of comparison of actual value and predicted value)
    * model_street : The LSTM model trained by grouping into each street
    (including the models, resulting csv files, plots of comparison of actual value and predicted value)
    * model_street_test : Modified model settings experiment 
    * plot_between_models : show the results and performance between different model

## Introduction
We are students from the Department of Computer Science at National Yang Ming Chiao Tung University. This project is an application we developed as part of our coursework in the Introduction to Artificial Intelligence class. 

In this project, we primarily focus on the automatic traffic flow in New York City. By training and utilizing AI models, including some factors like pedestrian demand, weather, and historical record, to provide user with the prediction of approximate traffic volume on a given street, considering real time factors such as time of day, weather conditions, and other influencing variables to make the prediction more close to the real world conditions.

#### Why is this problem important
There are numerous factors in the real world that can affect traffic conditions.For Example, in rainy day, or in poor air quality day, people may prefer driving cars than walking on the street. In areas with higher pedestrian demand, traffic light durations may be longer. Therefore, we suggest that wheather and pedestrian demand may influence the traffic condition.

In this project, we aim to explore three key factors — demand, historical traffic volume, and weather — to improve the accuracy of prediction.


#### How does your work differ from the original
Related work: [Traffic flow prediction under multiple adverse weather based on self-attention mechanism and deep learning models](https://www.sciencedirect.com/science/article/pii/S0378437123005435)
In the project above, they focused on highway and only use traffic and adverse weather data, e.g. snowstorm, and foggy. In constrast, we focused on the typical traffic flow on urban roads in NYC and usual weather condition. In addition, we incorporated pedestrian demand as a factor, which has not been considered in previous work. 

In summary, we think **we should spend time working on this project**.


### demo video link:
https://www.youtube.com/watch?v=eCv6sx0mzq0

## DataSet: total_dataset_final.csv
Our dataset is compiled by collecting and integrating data from sources such as Kaggle and publicly available government datasets. In this file, having roughly 377300 data, from 2016 to 2022. 

#### Columns in dataset:
| Attribute       | Description                                             |
|---------------|---------------------------------------------------------|
| **ID**        | Unique identifier for the record                        |
| **Boro**      | Borough or regional designation                         |
| **Date**      | Recorded date of the data point                         |
| **Hour**      | Time in hour                                           |
| **weekday**   | weekday or not                                         |
| **volumn**    | Traffic amount in each hour                        |
| **segmentID** | Unique identifier for a street segment                  |
| **street**    | Name of the street                                      |
| **fromst**    | Starting point of the segment                           |
| **tost**      | Ending point of the segment                             |
| **Direction** | Traffic or data collection direction                    |
| **temperature** | Atmospheric temperature at the recorded time          |
| **precipitation** | Precipitation in mm                        |
| **rain**      | Precipitation only contain rain                          |
| **cloudcover** | Percentage of cloud coverage                           |
| **windspeed** | Speed of wind at the time                              |
| **demand**    | Pedestrian demand                 |
| **Air_quality** | Measurement of pm2.5           |

#### Data source:
[NYC traffic volumn counts from 2016 to 2022](https://data.cityofnewyork.us/Transportation/Traffic-Volume-Counts/btm5-ppia/about_data) <br>
[weather from 2016 to 2022](https://www.kaggle.com/datasets/aadimator/nyc-weather-2016-to-2022) <br>
[Pedestrian demand](https://data.cityofnewyork.us/Transportation/Pedestrian-Mobility-Plan-Pedestrian-Demand/fwpa-qxaf/about_data) <br>
[Air quality](https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r/data_preview) <br>
[Traffic volumn](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/data_preview) <br>

## Model: Long Short Term Memory (LSTM)
![image](https://hackmd.io/_uploads/SyHJdyMfgl.png)

## Main Apporch: LSTM model
In this project, we choose the LSTM architecture to construct our model and we use the features as the input of the model and the output of the model will be the traffic prediction value.
1. Before training model, we grouped the dataset by different streets and standardlized the input features using `StandardScaler` .
2. We separated dataset into two parts, 80% for training, and remaining 20% for testing.
3. We used training dataset to train LSTM model follow the below diagram.
4. We tested our model with testing dataset.

#### LSTM outline
The following diagram is the outline for our LSTM architecture. In the LSTM model, we have several cells(layer) to propagate the features and its loss.
![image](https://hackmd.io/_uploads/Bkhic4QMeg.png)


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

##### Fraction of our implement:
```python
    def forward(self, x, h_prev, c_prev):
        gates = self.x2h(x) + self.h2h(h_prev)      #linear transformation
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i_gate)              #input gate
        f_gate = torch.sigmoid(f_gate)              #forget gate
        g_gate = torch.tanh(g_gate)                 #cell gate
        o_gate = torch.sigmoid(o_gate)              #output gate 
        c = f_gate * c_prev + i_gate * g_gate       #cell state : c = f 。 c_prev + i 。 g
        h = o_gate * torch.tanh(c)                  #hidden state: taking the element-wise product of the output gate and the cell state
        return h, c                                 #pass to next cell
```
## Evaluation Metrics
We use these three indexes to measure our model.
1. $RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 }$
    * It is used to measure the average magnitude of the differences between predicted values from our model and actual values in the dataset.
    * More sensitive to large errors than $MAE$.
2. $MAE = \frac{1}{n} \sum_{i=1}^{n} \left| \hat{y}_i - y_i \right|$
    * It is used to evaluate average difference between the predicted values and actual values.
3. $R^2 = 1 - \frac{ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }{ \sum_{i=1}^{n} (y_i - \bar{y})^2 }$
    * It is used to measure the goodness of fit of our model to the data.
## Results
### Baseline 
For each street, we plot the actual traffic value and the predicted value into a figure, below is one of the result of a street we randomly picked:
![1 AVENUE](https://hackmd.io/_uploads/ryHZ_vrzeg.png)

The following table shows the overall values of the three evaluation matrics:
| $RMSE$ | $MAE$ |   $R^2$     |
| -------- | -------- | ------ |
| 344.7538 | 217.9076 | 0.7011 |

Although the $R^2$ value has reached 0.7, we can see from the image above that there is still a significant discrepancy between our baseline traffic predictions and the actual traffic flow.


### Main approach
For each street, we plot the actual traffic value and the predicted value into a figure, below is one of the results of the same street we picked above:
![1 AVENUE](https://hackmd.io/_uploads/HkFquwSGgg.png)

The following table shows the overall values of the three evaluation matrics:

| $RMSE$ | $MAE$ |   $R^2$     |
| -------- | -------- | ------ |
| 241.9979 | 129.6413 | 0.8527 

### Comparisons of baseline and LSTM
* $RMSE$
![2models_all_RMSE](https://hackmd.io/_uploads/Byxhv3SGgl.png)
* $AME$
![2models_all_MAE](https://hackmd.io/_uploads/HyeMunrGxl.png)
* $R^2$
![2models_all_R2](https://hackmd.io/_uploads/Sy3cwhSGeg.png)

* Overall $R^2$, $RMSE$, and $MAE$
![2models_all_barchart](https://hackmd.io/_uploads/HJPaM6BGex.png)


## Experiments
In experiment part, we aim to **find some difference between those factors and traffic volume by using different grouping method**. For example, we built several models for different grouping method that takes different variables as featrue to train and test, then plot its $RMSE$, $MAE$, $R^2$ to show the difference between models under different situation.

Our models' name are as the format `A_B`
* for `A`, it is the model.
    * `baseline` using the baseline model (linear regression).
    * `group` : LSTM model trained by the same boro and street.
    * `street`: LSTM model trained by the same street.
    * `global`: LSTM model trained by all boro and street together.
* for `B`, it is a feature we used to train models.
    * `all`: model is trained with all feature
    * `demand`: model is trained only with demand
    * `historical`: model is trained only with historical data
    * `weather`: model is trained only with weather

That is to say, we have 16 models in total.

The following figure shows the results of the 16 models.
* $R^2$:
![16models_R2](https://hackmd.io/_uploads/HJSox1IMge.png)
* $RMSE$:
![16models_RMSE](https://hackmd.io/_uploads/H1y3lyIfll.png)
* $MAE$:
![16models_MAE](https://hackmd.io/_uploads/r152eyLGlx.png)

Furthermore, we also modified the model to check how the results will be affected by changing the settings of the model, including increase the epoch (20 to 30) and number of cell layers (3 to 4) and add dropout layer in fully-connected with rate = 0.1.
![LSTM_street_experiment_barchart](https://hackmd.io/_uploads/BJ8QBpSMel.png)

### 
## Real World application:
In this part, we use API(Application Programming Interface) to capture the real time information that will be used in prediction of traffic. We get some data from Google weather API, such as temperature, precipitation, rain, cloudcover, and windspeed, and get the Air quality from IQAir. After getting the data we need, input into the model to generate the corresponding traffic prediction for the next 6 hours.

* API Source
    * https://developers.google.com/maps/documentation/weather/overview?hl=zh-tw
    * https://api-docs.iqair.com/?version=latest
    
Since one of API needs account's key. Please use own API key to run the real world application.

```python
# current time is get from user's system time.
current_time = datetime.now(ZoneInfo("America/New_York"))

# replace {YOUR_APIKEY} with your api key.
# url for Google Weather API
url = "https://weather.googleapis.com/v1/currentConditions:lookup?key={YOUR_APIKEY}&location.latitude=40.7304&location.longitude=-74.0537"
"""
temperature, precipitation, rain, cloudcover, and windspeed from Google Weather API
"""
# url for IQAir API
url =  "https://api.airvisual.com/v2/city?city=New%20York%20City&state=New%20York&country=USA&key={YOUR_APIKEY}"
"""
 from Google Weather API
"""
```
For using our system, user must to enter the correct street. If user enter the street not in NYC, our system will print error. 
<details>
<summary> the valid street list </summary>
1 AVENUE<br>
10 AVENUE<br>
11 AVENUE<br>
111 STREET<br>
12 AVENUE<br>
14 AVENUE<br>
145 STREET BRIDGE<br>
188 STREET<br>
2 AVENUE<br>
21 STREET<br>
26 AVENUE<br>
3 AVENUE<br>
31 AVENUE<br>
34 AVENUE<br>
39 AVENUE<br>
4 AVENUE<br>
45 AVENUE<br>
47 AVENUE<br>
48 STREET<br>
5 AVENUE<br>
58 STREET<br>
65 STREET<br>
7 AVENUE<br>
73 PLACE<br>
8 AVENUE<br>
80 STREET<br>
86 STREET<br>
9 AVENUE<br>
9 STREET<br>
AMBOY ROAD<br>
AMSTERDAM AVENUE<br>
ARTHUR KILL ROAD<br>
ASTORIA BOULEVARD<br>
ATLANTIC AVENUE<br>
AUDUBON AVENUE<br>
AVENUE D<br>
AVENUE J<br>
AVENUE M<br>
AVENUE OF THE AMERICAS<br>
BAY PARKWAY<br>
BEACH 20 STREET<br>
BEDFORD AVENUE<br>
BEDFORD PARK BOULEVARD<br>
BELT PARKWAY<br>
BORDEN AVENUE BRIDGE<br>
BOSTON ROAD<br>
BROADWAY <br>
BROADWAY BRIDGE <br>
BRONXWOOD AVENUE <br>
BROOKLYN QUEENS EXPRESSWAY <br>
BRUCKNER BOULEVARD <br>
BUSHWICK AVENUE <br>
CENTRAL AVENUE <br>
CENTRAL PARK SOUTH <br>
CENTRAL PARK WEST <br>
CHRYSTIE STREET <br>
CHURCH AVENUE <br>
COLUMBIA STREET <br>
COLUMBUS AVENUE <br>
COOPER AVENUE <br>
CORTELYOU ROAD <br>
CRESCENT STREET <br>
CROPSEY AVENUE <br>
CROPSEY AVENUE BRIDGE <br>
CROSS BAY BOULEVARD <br>
DEKALB AVENUE <br>
DYCKMAN STREET <br>
EAST 138 STREET <br>
EAST 149 STREET <br>
EAST 161 STREET <br>
EAST 163 STREET <br>
EAST 174 ST BRIDGE <br>
EAST FORDHAM ROAD <br>
EAST GUN HILL ROAD <br>
EAST TREMONT AVENUE <br>
EMMONS AVENUE <br>
F D R DRIVE <br>
FLATBUSH AVENUE <br>
FLATLANDS AVENUE <br>
FLUSHING AVENUE <br>
FREDERICK DOUGLASS BOULEVARD <br>
FRESH KILLS BRIDGE <br>
FT HAMILTON PARKWAY <br>
FULTON STREET <br>
GRAND CONCOURSE <br>
GRAND STREET BRIDGE <br>
HAMILTON AVENUE BRIDGE <br>
HICKS STREET <br>
HILLSIDE AVENUE <br>
HOLLIS COURT BOULEVARD <br>
HOOK CREEK BRIDGE <br>
HUGUENOT AVENUE <br>
HUNTS POINT AVENUE <br>
HYLAN BOULEVARD <br>
JACKIE ROBINSON PARKWAY <br>
JACKSON AVENUE <br>
JAMAICA AVENUE <br>
JEROME AVENUE <br>
JJ BYRNE MEMORIAL BRIDGE <br>
JUNCTION BOULEVARD <br>
KINGS HIGHWAY <br>
KISSENA BOULEVARD <br>
LAFAYETTE AVENUE <br>
LAFAYETTE STREET <br>
LEXINGTON AVENUE <br>
LIBERTY AVENUE <br>
LINDEN BOULEVARD <br>
LONG ISLAND EXPRESSWAY <br>
LONGWOOD AVENUE <br>
MADISON AVENUE <br>
MADISON AVENUE BRIDGE <br>
MADISON STREET <br>
MAIN STREET <br>
MAJOR DEEGAN EXPRESSWAY <br>
MANHATTAN AVENUE <br>
MANHATTAN BRIDGE <br>
MANOR ROAD <br>
MARATHON PARKWAY <br>
MARTLING AVENUE <br>
MERRICK BOULEVARD <br>
METROPOLITAN AVENUE <br>
MYRTLE AVENUE <br>
NEPTUNE AVENUE <br>
NEW YORK AVENUE <br>
NORTH CHANNEL BRIDGE <br>
NORTH CONDUIT AVENUE <br>
NORTHERN BOULEVARD <br>
OCEAN AVENUE <br>
OCEAN PARKWAY <br>
OCEAN TERRACE <br>
PARK AVENUE <br>
PARSONS BOULEVARD <br>
PELHAM BRIDGE <br>
PELHAM PARKWAY <br>
PELHAM PARKWAY SOUTH <br>
PENNSYLVANIA AVENUE <br>
PROSPECT AVENUE <br>
PULASKI BRIDGE <br>
QUEENS BLVD <br>
QUEENS BOULEVARD <br>
RALPH AVENUE <br>
RANDALL AVENUE <br>
REMSEN AVENUE <br>
RICHMOND AVENUE <br>
RICHMOND ROAD <br>
RIKERS ISLAND BRIDGE <br>
RIVER AVENUE <br>
ROCKAWAY BEACH BOULEVARD <br>
ROCKAWAY BOULEVARD <br>
ROCKLAND AVENUE <br>
ROOSEVELT AVENUE <br>
ROOSEVELT AVENUE BRIDGE <br>
SEAGIRT BOULEVARD <br>
SEDGWICK AVENUE <br>
SMITH STREET <br>
SOUTH CONDUIT AVENUE <br>
SOUTHERN BLVD <br>
SOUTHERN BOULEVARD <br>
STATE STREET <br>
STEINWAY STREET <br>
SUNRISE HIGHWAY <br>
SURF AVENUE <br>
SUTTER AVENUE <br>
TODT HILL ROAD <br>
TOMPKINS AVENUE <br>
UNION TURNPIKE <br>
UTOPIA PARKWAY <br>
VAN CORTLANDT PARK EAST <br>
VERNON BOULEVARD <br>
VICTORY BOULEVARD <br>
WADSWORTH AVENUE <br>
WASHINGTON AVENUE <br>
WEBSTER AVENUE <br>
WEST 165 STREET <br>
WEST 42 STREET <br>
WEST END AVENUE <br>
WESTCHESTER AVENUE <br>
WESTCHESTER AVENUE BRIDGE <br>
WHITE PLAINS ROAD <br>
WOODHAVEN BOULEVARD <br>
YORK AVENUE <br>
</details>

The final result is like that
![image](https://hackmd.io/_uploads/ryjHizQGex.png)
