## Solution for NUS DSC HP acumen chips

Slides (restricted):
https://docs.google.com/presentation/d/1-fbW70QaAzujOYgounphAEMuB3IpGgnSAs2zULIHpyY/edit

Method 1:
Yolov4 via Darknet
- Took so long to converge
- Precision was high on flaw chip but recall but not really
- Counting of chip is not as good as yolov5 for now
- Detecting if handwriting is actually persent is not really good

Method 2: Yolov5
- Took around 90 minutes to converge on 200 epochs
- Very good in predicting number of chips given an image (maybe 80% accuracy)
- Not good in predicting flaw chip
- Detecting if handwriting is actually persent is not really good

Yolov4 setup:

You will need to load to colab notebook, then you will need the required darknet files (which presents in my drive folder `yolov4` for my case :b). Then run the code all, and there will be results in term of `.txt` (for my case `coordinate_results_zzz.txt`) in your darknet folder in google colab. Then run the yolov4 extractor on this text file.