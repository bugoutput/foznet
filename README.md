# foznet
17-parameter neural net to diagnose misfiring Subaru engines from accelerometer data. <br>
Based on Shawn Hymel's TinyML anomaly detection tutorial, as seen here: https://github.com/ShawnHymel/tinyml-example-anomaly-detection <br>

Normal Idle (top) vs Cylinder 2 Misfire (bottom) acceleration data recorded from center of dashboard <br>
<img width="413" alt="fozData_filt" src="https://github.com/user-attachments/assets/49214c15-f97b-485c-b550-f51541f1bd46" /><br>
Experimental setup showing accelerometer and ESP32 data logger on dashboard, data acqusition laptop on seat. <br>
![foznetpic](https://github.com/user-attachments/assets/84b57be8-f12a-4bfb-8b99-6c5ddef34b66)<br>

Mean Squared Error (MSE) histogram showing separation between normal (blue) and anomaly (red) data samples. <br>
<img width="372" alt="hist_1e-5" src="https://github.com/user-attachments/assets/cb8f263c-c4e7-4e41-82bc-8c93410a1a23" />


