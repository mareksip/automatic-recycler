# Automatic recycler

Raspberry Pi Automatic Recycler with Tensorflow object detection. 

# Introduction

To boost motivation of recycling we introduce a prototype of waste detector with capability of deciding which items to recycle. 

Youtube video https://www.youtube.com/watch?v=Tpj1u7IZ9A0

# Software and Hardware
- Raspberry Pi 3 B+ 
- Raspberry Camera Module V2
- NEMA 17 stepper motors model: 17HS4401S 

3D components designed by Timothée Gerliner with Catia software and printed on Makerbot REPLICATOR+


Sample images for training mode were gathered from http://image-net.org/

For following categories:
- Beercan: n02823510
- Bottle: n02876657
- Box: n02883344
- Coffee can: n03062985
- Paper cup: n03216710
- Pen: n03906997
- Pop bottle: n03983396
- Soda Can: n04255586
- Paper: n06255613
- Apple: n07739125
- Banana: n07753592
- Egg: n07840804

# Installation 

Downloading images for research purpose from http://image-net.org/ can be automated with ImageNet downloadutils.py https://github.com/tzutalin/ImageNet_Utils

```
py downloadutils.py --downloadImages --wnid n02084071
```

# Further improvements

# Team

- Thomas Polus
- Marek Šíp
- Louis Yonatan
- 3D component design by Timothée Gerlinger
- Components and laboratory provided by KSU

2018, In cooperation with Mechatronics department of 경성대학교 / Kyungsung University in Busan, Republic of Korea.