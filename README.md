## Image Processing for Basic Depth Completion (IP-Basic, C++ version)

### 1. Introduction

**Related Paper: ** Ku J, Harakeh A, Waslander S L. **In defense of classical image  processing: Fast depth completion on the cpu**[C]//2018 15th Conference on Computer and Robot Vision (CRV). IEEE, 2018: 16-22. [[PDF]](https://arxiv.org/abs/1802.00036) [[Code]](https://github.com/kujason/ip_basic)

**Not the original author**. The original version was written in Python, but in this work, I rewrite it in C++. Because I only rewrite what I need, the code is **not complete**. A depth image captured by an rgbd camera is cited as an example for testing.

### 2. Run examples

```bash
mkdir build && cd build
cmake ..
make -j4
 ../bin/main ../data/rgb.png ../data/depth.png ../config/config.yaml 
```

