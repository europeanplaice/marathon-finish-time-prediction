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


## What if mode
You can also compare two estimations that are different from each other. 


```python main.py --elapsed_time "0:26:00" --elapsed_time_what_if "0:27:00,0:55:00"``` 

The results are
```
**Estimation**
5Km        0:26:00    lower_95 => ******* lower_50 => ******* median => ******* upper_50 => ******* upper_95 => *******
10Km                  lower_95 => 0:50:03 lower_50 => 0:51:25 middle => 0:52:08 upper_50 => 0:52:52 upper_95 => 0:54:14
15Km                  lower_95 => 1:14:22 lower_50 => 1:17:12 middle => 1:18:41 upper_50 => 1:20:10 upper_95 => 1:23:00
20Km                  lower_95 => 1:38:13 lower_50 => 1:43:21 middle => 1:46:04 upper_50 => 1:48:47 upper_95 => 1:53:58
Half                  lower_95 => 1:43:16 lower_50 => 1:48:38 middle => 1:51:27 upper_50 => 1:54:18 upper_95 => 1:59:40
25Km                  lower_95 => 2:01:24 lower_50 => 2:09:31 middle => 2:13:42 upper_50 => 2:17:56 upper_95 => 2:26:01
30Km                  lower_95 => 2:25:38 lower_50 => 2:37:14 middle => 2:43:20 upper_50 => 2:49:28 upper_95 => 3:01:08
35Km                  lower_95 => 2:49:40 lower_50 => 3:06:04 middle => 3:14:37 upper_50 => 3:23:11 upper_95 => 3:39:28
40Km                  lower_95 => 3:14:03 lower_50 => 3:34:50 middle => 3:45:45 upper_50 => 3:56:34 upper_95 => 4:17:12
Finish                lower_95 => 3:24:44 lower_50 => 3:46:36 middle => 3:58:09 upper_50 => 4:09:29 upper_95 => 4:31:30
**Estimation**
5Km        0:27:00    lower_95 => ******* lower_50 => ******* median => ******* upper_50 => ******* upper_95 => *******
10Km       0:55:00    lower_95 => ******* lower_50 => ******* median => ******* upper_50 => ******* upper_95 => *******
15Km                  lower_95 => 1:19:36 lower_50 => 1:21:44 middle => 1:22:52 upper_50 => 1:23:59 upper_95 => 1:26:08
20Km                  lower_95 => 1:42:46 lower_50 => 1:48:11 middle => 1:51:01 upper_50 => 1:53:53 upper_95 => 1:59:16
Half                  lower_95 => 1:49:13 lower_50 => 1:54:28 middle => 1:57:15 upper_50 => 2:00:01 upper_95 => 2:05:15
25Km                  lower_95 => 2:08:09 lower_50 => 2:16:22 middle => 2:20:38 upper_50 => 2:24:54 upper_95 => 2:33:00
30Km                  lower_95 => 2:33:48 lower_50 => 2:45:45 middle => 2:52:00 upper_50 => 2:58:16 upper_95 => 3:10:08
35Km                  lower_95 => 2:59:14 lower_50 => 3:15:56 middle => 3:24:53 upper_50 => 3:33:43 upper_95 => 3:50:33
40Km                  lower_95 => 3:24:20 lower_50 => 3:46:02 middle => 3:57:31 upper_50 => 4:09:05 upper_95 => 4:30:45
Finish                lower_95 => 3:34:20 lower_50 => 3:57:37 middle => 4:09:51 upper_50 => 4:22:05 upper_95 => 4:45:01
```
also a graph is

![estimation](https://user-images.githubusercontent.com/38364983/129465386-2ee26a0a-b44f-4b90-b18e-da68410357b0.jpg)
