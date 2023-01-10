# Data Processing

NHL data and all codes used for data processing are stored in this folder.


## Data Acquisition
User can access the NHL raw data by using the `get_data` method. It uses the `NHL class` to hit the NHL REST API and download the raw_data. 

User needs to provide the `start_season` and the `end_season` to get the raw_data for all those seasons and `file_path` to indicate data dump location.
If the data is already present in that given path the `NHL class` skips the step of hitting the `REST` API and downloading the data. Instead, it directly loads the data from the given location. 

### Demo 

```python
from ift758.data import PlayByPlayWidget, get_data

#downloads the data for season(2017-2018) to seaon (2020-2021)
season = get_data(2017, 2020, 'file_path')
```
Use this command to download or locally load NHL play-by-play data for both the regular season and playoffs.

### Play-By-Play Widget
---
The **play-by-play widget** is an interactive widget created using [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/). It gives a detailed information about the events taking place in an NHL game. The users are provided with 2 sliders using which they can select any game in a particular season and go through all important events recorded during the game. The user is able to get a visual sense of the type of play and where the play is happening on the ice-rink.

#### Code

First we need to have a season's NHL data. This can be obtained using our `NHL` class and `get_data` method which was described in detail in the data-acquisition section. Once we have the whole season's data. We can create and deploy the widget using `PlayByPlayWidget` as show below:
```python
from ift758.data import PlayByPlayWidget, get_data

season = get_data(2017, 2017, 'file_path')
pbp = PlayByPlayWidget(season)

pbp.display()
```
![play-by-play widget](../../figures/play_by_play_widget.jpeg)


### Tidy Data
---