# Saki bot
Private Discord server bot for Project Sekai point estimation and prediction (for ranks 500 and 1000) during in-game events.   
- The points of rank 500/1000 at each timestep is estimated through using [1D monotonic cubic interpolation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).   
- Prediction of the point of rank 500/1000 when the event ends is done by comparing the current event's time series of points to that of all past events, to find the most similar past event's time series (having the lowest mean absolute percentage error). The current event's time series will then be extrapolated linearly then combined with the final part of the most similar event's time series.

## Bot commands
`$plot <rank> <start> <end>`   
Plot time series of points of previous events.   
*rank*: 500 | 1000   
*start*, *end*: start and end event number to be plotted   

`$accuracy`   
Display previous point prediction errors. 
