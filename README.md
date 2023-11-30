## **运行(环境ubuntu20)**
单目2d和3d目标检测和跟踪以及bev可视化。qd-3dt相比常见的目标检测器，引入基于lstm的运动模型，同时实现跨帧间目标的信息共享，进一步优化目标检测器的结果，使得检测更鲁棒。           
参考官方代码的基础下做了以下调整：
- 简化代码，保留核心部分
- 支持任意分辨率的图像输入
- 只支持cpu(gpu需要调整部分代码)
- 添加bev可视化

值得注意的是，该模型的泛化性不一定好，因而使用自己的数据集前，需要重新训练，请参照官方的教程。运行bev模式时，需要提供Twc，若数据没Twc，可以通过slam等获取。


```
# 安装pypanglin
git clone -b v0.8 --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

# See what package manager and packages are recommended
./scripts/install_prerequisites.sh --dry-run recommended

# Install dependencies (as described above, or your preferred method)
./scripts/install_prerequisites.sh recommended

# Configure and build
cmake -B build -DPython_EXECUTABLE=`which python3` 
# cmake -B build -DCMAKE_INSTALL_PREFIX=/home/spurs/installed -DPython_EXECUTABLE=`which python3`
cmake --build build -t pypangolin_pip_install

# 模型参数下载
百度链接(kitti): https://pan.baidu.com/s/1-ElUcDVL-YOOxrLCgdDmBA 提取码: bt3g

官方链接：https://github.com/SysCV/qd-3dt/blob/main/readme/MODEL_ZOO.md

# kitti测试数据下载
链接：https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip
相关参数：https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip

更多的kitti数据见https://zhuanlan.zhihu.com/p/664386718

# 下载代码
git clone https://github.com/lturing/qd_3dt_modified

# 运行
cd qd_3dt_modified
python run_kitti.py --config quasi_dla34_dcn_3dmatch_multibranch_conv_dep_dim_cen_clsrot_sep_aug_confidence_mod_anchor_ratio_small_strides_GTA.py --checkpoint path_to_latest_kitti.path --lstm_checkpoint path_to_batch8_min10_seq10_dim7_train_dla34_regress_pretrain_VeloLSTM_kitti_100_linear.pth --data path_to_kittii_data --pose path_to_kitti_data_pose -- cali path_to_kitti_cali 
```

## demo(b站)
- [单目2d和3d目标检测与跟踪(基于qd-3dt)](https://www.bilibili.com/video/BV1Jv411F7gW) 
- [单目3d目标检测(基于qd-3dt)](https://www.bilibili.com/video/BV14G411S7Vt/)
- [单目3d目标检测跟踪以及bev可视化](https://www.bilibili.com/video/BV1594y1E7cT/)

## 感谢

- [qd-3dt](https://github.com/SysCV/qd-3dt)
- [pykitti](https://github.com/utiasSTARS/pykitti)
