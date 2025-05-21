import requests
from datetime import datetime

# API have call limitation, be careful to use it! or you can store in the text file to avoid repeatly calling.

# get current time
current_time = datetime.now()
print("current time:" + str(current_time.hour))

# set header
headers = {
    'User-Agent': 'MyWeatherApp/1.0 tt121892185@gmail.com' 
}


# API 
url = "https://api.weather.gov/gridpoints/OKX/31,35/forecast"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()

    forecast_current = data['properties']['periods'][0]
    # forecast_further = data['properties']['periods'][1]

else:
    print("错误:", response.status_code)
    print("错误信息:", response.text)



# API 

url = "https://api.airvisual.com/v2/city?city=New%20York%20City&state=New%20York&country=USA&key=7fc1f886-e778-41c3-83c6-1daf90fb85a9"
response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    print(data)
    weather = data['data']['current']['weather']
    pollution = data['data']['current']['pollution']
else:
    print("错误:", response.status_code)
    print("错误信息:", response.text)

url = "https://www.meteosource.com/api/v1/free/point?place_id=new-york-city&sections=current%2Chourly&language=en&units=auto&key=8ppm5ny2p15frup7z246w12yylk4qsentwt2wrco"
response = requests.get(url, headers=headers)
if response.status_code == 200:
    cloud = response.json()
    cloud_coverage = cloud['current']["cloud_cover"]
else:
    print("错误:", response.status_code)
    print("错误信息:", response.text)





#features =           ['Hour',            'weekday',                               'temperature', 'precipitation'                                         , 'rain'                                                                        , 'cloudcover', 'windspeed', 'Air_quality', 'demand']
current_information = [current_time.hour, 'Y' if current_time.weekday()<5 else 'N', weather['tp'], forecast_current['probabilityOfPrecipitation']['value'], 'Y' if forecast_current['probabilityOfPrecipitation']['value'] == 100 else 'N',  cloud_coverage , weather['wd'],  pollution['aqius'], None]
print(current_information)