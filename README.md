# LRC

A PyTorch implementation of LRC based on the paper
[Weakly-supervised Temporal Action Localization with Local Response Consistency]().

![Network Architecture](structure.png)

## Usage

Git clone the corresponding repos and replace the files provided by us, then run the code according to `readme` of 
corresponding repos.

For example, to train HAM-Net on THUMOS14 dataset:
```
git clone https://github.com/asrafulashiq/hamnet.git
mv AGCT/hamnet/* hamnet/
python main.py
```

To evaluate HAM-Net on THUMOS14 dataset:
```
python main.py --test --ckpt [checkpoint_path]
```

## Benchmarks

The models are trained on one NVIDIA GeForce TITAN X GPU (12G). All the hyper-parameters are the default values.

### THUMOS14

<table>
<thead>
  <tr>
    <th rowspan="3">Method</th>
    <th colspan="8">THUMOS14</th>
    <th rowspan="3">Download</th>
  </tr>
  <tr>
    <td align="center">mAP@0.1</td>
    <td align="center">mAP@0.2</td>
    <td align="center">mAP@0.3</td>
    <td align="center">mAP@0.4</td>
    <td align="center">mAP@0.5</td>
    <td align="center">mAP@0.6</td>
    <td align="center">mAP@0.7</td>
    <td align="center">mAP@AVG</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center"><a href="https://github.com/asrafulashiq/hamnet">HAM-Net</a></td>
    <td align="center">66.8</td>
    <td align="center">60.9</td>
    <td align="center">52.2</td>
    <td align="center">42.9</td>
    <td align="center">33.4</td>
    <td align="center">22.7</td>
    <td align="center">12.2</td>
    <td align="center">41.6</td>
    <td align="center"><a href="https://1drv.ms/u/s!AtyHkt-GdJtIilloJ6Uo867V9yr8?e=aWvLyY">OneDrive</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/zhang-can/CoLA">CoLA</a></td>
    <td align="center">67.5</td>
    <td align="center">60.6</td>
    <td align="center">51.9</td>
    <td align="center">43.2</td>
    <td align="center">34.2</td>
    <td align="center">24.2</td>
    <td align="center">13.9</td>
    <td align="center">42.2</td>
    <td align="center"><a href="https://1drv.ms/u/s!AtyHkt-GdJtIilhzTkTD-uZvy4ya?e=PFdCWR">OneDrive</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/harlanhong/MM2021-CO2-Net">CO<sub>2</sub>-Net</a></td>
    <td align="center">70.8</td>
    <td align="center">64.7</td>
    <td align="center">55.7</td>
    <td align="center">46.8</td>
    <td align="center">39.8</td>
    <td align="center">26.5</td>
    <td align="center">13.8</td>
    <td align="center">45.4</td>
    <td align="center"><a href="https://1drv.ms/u/s!AtyHkt-GdJtIilfZvzmjkg9BcFTx?e=0ptemo">OneDrive</a></td>
  </tr>
</tbody>
</table>

mAP@AVG is the average mAP under the thresholds 0.1:0.1:0.7.

### ActivityNet

<table>
<thead>
  <tr>
    <th rowspan="3">Method</th>
    <th colspan="4">ActivityNet 1.2</th>
    <th rowspan="3">Download</th>
  </tr>
  <tr>
    <td align="center">mAP@0.5</td>
    <td align="center">mAP@0.75</td>
    <td align="center">mAP@0.95</td>
    <td align="center">mAP@AVG</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center"><a href="https://github.com/asrafulashiq/hamnet">HAM-Net</a></td>
    <td align="center">41.3</td>
    <td align="center">25.2</td>
    <td align="center">5.5</td>
    <td align="center">25.4</td>
    <td align="center"><a href="https://1drv.ms/u/s!AtyHkt-GdJtIilTBE-4dzLgc3Okw?e=naUsTl">OneDrive</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/zhang-can/CoLA">CoLA</a></td>
    <td align="center">41.0</td>
    <td align="center">27.5</td>
    <td align="center">4.2</td>
    <td align="center">26.4</td>
    <td align="center"><a href="https://1drv.ms/u/s!AtyHkt-GdJtIilYYT3fWJqCg76g5?e=nSzmUr">OneDrive</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/harlanhong/MM2021-CO2-Net">CO<sub>2</sub>-Net</a></td>
    <td align="center">44.4</td>
    <td align="center">27.0</td>
    <td align="center">5.4</td>
    <td align="center">27.1</td>
    <td align="center"><a href="https://1drv.ms/u/s!AtyHkt-GdJtIilVBJFZ7mhdg1uM4?e=rx7BKz">OneDrive</a></td>
  </tr>
</tbody>
</table>

mAP@AVG is the average mAP under the thresholds 0.5:0.05:0.95.
