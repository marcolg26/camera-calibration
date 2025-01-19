# Camera Calibrator
Camera calibrator made implementing [Zhang's procedure](https://ieeexplore.ieee.org/document/888718)


To run the calibrator with default images and settings:
```bash
  python main.py
```
Additional parameters:
```bash
  python main.py personal_device n_rows n_columns square_size
```
where `personal_device` = 1 if you want to calibrate your personal device, otherwise `personal_device` = 0.

For example: 
```bash
  python main.py 1 8 11 11
```
