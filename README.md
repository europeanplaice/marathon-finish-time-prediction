# marathon-finish-time-prediction
It predicts your marathon's finishing time based on the split time using a method of machine learning. It can show not only the specific estimated time but also a range of possible finishing times. Inside this model, I employed Tensorflow and Tensorflow Probability.

## Features
- [x] Visualization
- [x] Outputs an estimation to stdout

## Get Started
You can train a model with your data. However, I've already prepared a trained model, and you can estimate the time without your data by using the trained weight in this repository.  

To predict a finishing time, for example, run  

```python main.py --elapsed_time "0:25:00, 0:50:00"```  

Then it shows
```
**Estimation**
15Km        lower_95 => 1:12:45   lower_50 => 1:14:36   median => 1:15:34   upper_50 => 1:16:32   upper_95 => 1:18:23
20Km        lower_95 => 1:35:26   lower_50 => 1:39:37   median => 1:41:48   upper_50 => 1:44:00   upper_95 => 1:48:09
Half        lower_95 => 1:40:46   lower_50 => 1:45:05   median => 1:47:21   upper_50 => 1:49:36   upper_95 => 1:53:55
25Km        lower_95 => 1:58:04   lower_50 => 2:04:59   median => 2:08:36   upper_50 => 2:12:13   upper_95 => 2:19:05
30Km        lower_95 => 2:21:17   lower_50 => 2:31:48   median => 2:37:17   upper_50 => 2:42:53   upper_95 => 2:53:23
35Km        lower_95 => 2:43:22   lower_50 => 2:59:17   median => 3:07:35   upper_50 => 3:15:57   upper_95 => 3:31:52
40Km        lower_95 => 3:05:47   lower_50 => 3:26:45   median => 3:37:38   upper_50 => 3:48:29   upper_95 => 4:09:10
Finish      lower_95 => 3:16:59   lower_50 => 3:39:20   median => 3:51:06   upper_50 => 4:02:52   upper_95 => 4:25:17
```
and a graph.

![Figure_1](https://user-images.githubusercontent.com/38364983/129431606-f888b98b-0731-47f2-8d1d-1ef0dcd578a5.png)

To predict, you must specify at least one time, but you can start prediction at any position where you are running.


For example, either  
```python main.py --elapsed_time "0:27:00"``` 
or  
```python main.py --elapsed_time "0:27:00, 0:55:00, 1:15:00, 1:40:00"```  
is ok.

