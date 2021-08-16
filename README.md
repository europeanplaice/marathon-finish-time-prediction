# marathon-finish-time-prediction
It predicts your marathon's finishing time based on the split time using a method of machine learning. It can show not only the specific estimated time but also a range of possible finishing times. Inside this model, I employed Tensorflow and Tensorflow Probability.

To make a prediction, it doesn't use any features such as race conditions, runners' age, past results, and so on, but it uses only the time elapsed.

## Features
- [x] Visualization
- [x] Outputs an estimation to stdout

## Requirements
* Tensorflow
* Tensorflow Probability
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Tqdm

## Get Started
You can train a model with your data. However, I've already prepared a trained model, and you can estimate the time without your data by using the trained weight in this repository.  

To predict a finishing time, for example, run  

```python main.py --elapsed_time "0:25:00,0:50:00"```  

Then it shows
```
**Estimation**
5Km        0:25:00    lower_95 => ******* lower_50 => ******* median => ******* upper_50 => ******* upper_95 => *******
10Km       0:50:00    lower_95 => ******* lower_50 => ******* median => ******* upper_50 => ******* upper_95 => *******
15Km                  lower_95 => 1:12:57 lower_50 => 1:14:49 median => 1:15:48 upper_50 => 1:16:46 upper_95 => 1:18:39
20Km                  lower_95 => 1:34:49 lower_50 => 1:39:16 median => 1:41:36 upper_50 => 1:43:57 upper_95 => 1:48:22
Half                  lower_95 => 1:40:32 lower_50 => 1:44:56 median => 1:47:16 upper_50 => 1:49:35 upper_95 => 1:53:59
25Km                  lower_95 => 1:57:48 lower_50 => 2:04:51 median => 2:08:34 upper_50 => 2:12:19 upper_95 => 2:19:26
30Km                  lower_95 => 2:21:04 lower_50 => 2:31:39 median => 2:37:12 upper_50 => 2:42:48 upper_95 => 2:53:24
35Km                  lower_95 => 2:43:38 lower_50 => 2:59:02 median => 3:07:04 upper_50 => 3:15:01 upper_95 => 3:30:14
40Km                  lower_95 => 3:05:54 lower_50 => 3:26:20 median => 3:36:53 upper_50 => 3:47:29 upper_95 => 4:07:26
Finish                lower_95 => 3:16:49 lower_50 => 3:38:30 median => 3:49:56 upper_50 => 4:01:20 upper_95 => 4:23:15
```
and a graph.

![estimation](https://user-images.githubusercontent.com/38364983/129465869-c1d2c398-41dd-4fab-97c3-3f8f15e67bb9.jpg)

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
10Km                  lower_95 => 0:50:03 lower_50 => 0:51:25 median => 0:52:08 upper_50 => 0:52:52 upper_95 => 0:54:14
15Km                  lower_95 => 1:14:22 lower_50 => 1:17:12 median => 1:18:41 upper_50 => 1:20:10 upper_95 => 1:23:00
20Km                  lower_95 => 1:38:13 lower_50 => 1:43:21 median => 1:46:04 upper_50 => 1:48:47 upper_95 => 1:53:58
Half                  lower_95 => 1:43:16 lower_50 => 1:48:38 median => 1:51:27 upper_50 => 1:54:18 upper_95 => 1:59:40
25Km                  lower_95 => 2:01:24 lower_50 => 2:09:31 median => 2:13:42 upper_50 => 2:17:56 upper_95 => 2:26:01
30Km                  lower_95 => 2:25:38 lower_50 => 2:37:14 median => 2:43:20 upper_50 => 2:49:28 upper_95 => 3:01:08
35Km                  lower_95 => 2:49:40 lower_50 => 3:06:04 median => 3:14:37 upper_50 => 3:23:11 upper_95 => 3:39:28
40Km                  lower_95 => 3:14:03 lower_50 => 3:34:50 median => 3:45:45 upper_50 => 3:56:34 upper_95 => 4:17:12
Finish                lower_95 => 3:24:44 lower_50 => 3:46:36 median => 3:58:09 upper_50 => 4:09:29 upper_95 => 4:31:30
**Estimation**
5Km        0:27:00    lower_95 => ******* lower_50 => ******* median => ******* upper_50 => ******* upper_95 => *******
10Km       0:55:00    lower_95 => ******* lower_50 => ******* median => ******* upper_50 => ******* upper_95 => *******
15Km                  lower_95 => 1:19:36 lower_50 => 1:21:44 median => 1:22:52 upper_50 => 1:23:59 upper_95 => 1:26:08
20Km                  lower_95 => 1:42:46 lower_50 => 1:48:11 median => 1:51:01 upper_50 => 1:53:53 upper_95 => 1:59:16
Half                  lower_95 => 1:49:13 lower_50 => 1:54:28 median => 1:57:15 upper_50 => 2:00:01 upper_95 => 2:05:15
25Km                  lower_95 => 2:08:09 lower_50 => 2:16:22 median => 2:20:38 upper_50 => 2:24:54 upper_95 => 2:33:00
30Km                  lower_95 => 2:33:48 lower_50 => 2:45:45 median => 2:52:00 upper_50 => 2:58:16 upper_95 => 3:10:08
35Km                  lower_95 => 2:59:14 lower_50 => 3:15:56 median => 3:24:53 upper_50 => 3:33:43 upper_95 => 3:50:33
40Km                  lower_95 => 3:24:20 lower_50 => 3:46:02 median => 3:57:31 upper_50 => 4:09:05 upper_95 => 4:30:45
Finish                lower_95 => 3:34:20 lower_50 => 3:57:37 median => 4:09:51 upper_50 => 4:22:05 upper_95 => 4:45:01
```
also a graph is

![estimation](https://user-images.githubusercontent.com/38364983/129465880-ff55b09c-d9cc-41c4-8e63-342ffe531c68.jpg)

