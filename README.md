# SensorFlow: Sensor and Image Fused Video Stabilization

This is the evaluation metric code proposed in the WACV 2025 paper SensorFlow: Sensor and Image Fused Video Stabilization.website [[website](https://jiyangyu.github.io/sensorflow/)][[paper](https://jiyangyu.github.io/sensorflow/files/sensorflow.pdf)]


Stability, initially proposed in [Liu et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/Stabilization_SIGGRAPH13.pdf) measures the proportion of the low frequency motion in the overall motion of the stabilized video, which is widely used in evaluating video stabilization algorithms. 
However, in our experiment under the standard definition of stability metric, we observe that 1) most methods have only subtle differences compared to the scale of the metric score and 2) significant discrepancy exists in visual stability and the metric score ranking. 

To this end, we improve the stability metric by considering the inter-frame motion instead of the accumulated motion in the frequency spectrum analysis.
Our modified metric shows good consistency with the actual user study (Sec. 5.3).
Please refer to Sec. 5.2 in our [paper](https://jiyangyu.github.io/sensorflow/files/sensorflow.pdf) for details and reasoning of this improved metric.

## Usage

The metric script requires two videos, 1) the unstable input video and 2) the stabilized video for evaluation.

Run the script as follows, the Stability, Distortion and Cropping score will be reported when process ends:

```bash
python metric.py --input_video INPUT_VIDEO_PATH --stable_video STABLE_VIDEO_PATH
```

## Citing

If you find this work useful, please consider citing our paper.
```BibTeX
@article{yu2025sensorflow
  author    = {Yu, Jiyang and Zhang, Tianhao and Shi, Fuhao and He, Lei and Liang, Chia-Kai},
  title     = {SensorFlow: Sensor and Image Fused Video Stabilization},
  journal   = {WACV},
  year      = {2025},
}
```