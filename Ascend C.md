## TODO List：

- [x] Add Custom

- [x] Reduce Custom；参考算子开发进阶篇;
  - [x] 实现
  - [x] 测试
  
- [x] Kernel launch: Reduce Custom; 

  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlas_ascendc_10_0023.html

  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0055.html

- [ ] 单算子调用：
  - [ ] 使用算子清单中的算子（应用开发-单算子调用-单算子API执行）
  - [ ] 调用自定义算子

- [ ] 算子开发高级篇

- [ ] 算子调试

- [ ] PQ实现（先寻找现成的算子）

  - [ ] kmeans（train）
  - [ ] encode
  - [ ] AQD


**QA**

问：\_\_aicore\_\_核函数中的标量计算在哪进行？

答：标量计算单元。标量指令处理队列解码指令后，将标量指令留标量指令处理队列中，其他指令如矩阵运算则发送到相应队列。

问：如何调试？如何打印log？

答：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlas_ascendc_10_0044.html

问：既然需要手动进行device和host之间的memcopy，那为什么核函数还要接收global指针？

答：

问：`/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/include/`只有Framework算子工程中的算子才会被注册到该路径中吗？

答：

**学习资料**

CANN7文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlas_ascendc_10_0007.html

进阶篇：AscendCL单算子调用

硬件架构：https://3ms.huawei.com/documents/docinfo/481900699022405632



## 开发环境准备

### 开发环境安装

1. 检查设备在位

   ```shell
   lspci | grep d500 #310p
   ```

2. 安装firmwork和driver

   https://www.hiascend.com/hardware/firmware-drivers/community?product=5&model=25&cann=7.0.0.beta1&driver=1.0.0.alpha

3. 安装toolkit

### 开发环境配置

```shell
# server1(96.10.9.166)
cd /home/HwHiAiUser/cuixk_test/samples-7.0.RC1/operator
ls /usr/local/Ascend7.0/
# 切换到开发用户
su HwHiAiUser
# 激活Ascend环境
source /usr/local/Ascend7.0/ascend-toolkit/set_env.sh
# 运行示例(cann 7.0)；执行run.sh脚本之前，请将run.sh中ASCEND_HOME_DIR环境变量修改为CANN软件包的安装路径。
bash run.sh ascend310p cpu
bash run.sh ascend310p npu_onboard
# 运行示例(cann 8.0)
bash run.sh -v ascend310p -r cpu
bash run.sh -v ascend310p -r npu

# server2(10.185.221.38)
cd /home/HwHiAiUser/cuixk/opSamples/
su HwHiAiUser
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 硬件架构

### 概述

<img src="Ascend C.assets/1712626351296_image.png" alt="1712626351296_image" style="zoom:80%;" />

> cache: 缓存，用于应对两端速度不匹配，将部分常用数据放到访存更快的位置；
>
> buffer: 缓冲区，是内存的一部分，用于流量整型，即先将某时刻的大量读写暂存以避免带宽不足导致的阻塞；

+ **CPU:** 昇腾 AI 处理器集成了多个 CPU 核心, 每个核心都有独立的 L1 和 L2 Cache, 所有核心共享一个片上 L3 Cache。分为AI CPU 和Control CPU 两类
+ **AI Core:** 除了 CPU 之外,该处理器真正的算力担当是采用了达芬奇架构的 AI Core（配备8 MB 片上 L2 Cache）。
+ **任务调度器 (Task Scheduler, TS):** 实现计算任务在 AI Core 上的高效分配和调度的专用 CPU。该 CPU 专门服务于 AI Core 和 AI CPU, 而不承担任何其他的事务和工作

> **ascend310p:** 8 \* AICore(DaVinci Arch), ? \* AICPU



### 达芬奇架构

<img src="Ascend C.assets/1712626362379_image.png" alt="1712626362379_image" style="zoom:80%;" />

在 AI Core 中，存储单元为各个计算单元提供转置过并符合要求的数据，计算单元返回运算的结果给存储单元，控制单元为计算单元和存储单元提供指令控制，三者相互协调合作完成计算任务。

#### 计算单元

AI Core拥有矩阵（cube和累加器）、向量和标量（负责标量数据计算和程序流程控制）四种计算单元。

<img src="Ascend C.assets/20210519225743223.png" alt="在这里插入图片描述" style="zoom:50%;" />

**1.矩阵计算单元**

Cube用256个矩阵计算子电路（如下图，由16个乘法器+归并电路构成）实现了一个时钟周期内完成两个16^2矩阵相乘（16^3次乘加）

<img src="Ascend C.assets/20210519231008902.png" alt="在这里插入图片描述" style="zoom:60%;" />

累加器实现在cube完成矩阵乘法后的矩阵加法，即C=A×B+C。

处理超过16\*16的矩阵时，需要实现按照特定的格式进行矩阵存储，并在计算过程中按特定顺序分块(16\*16)读取。矩阵A的切割和排布方式称为“大Z小Z”，矩阵B为“大Z小N”，结果矩阵C为“大N小Z”。tile_col 1 of C = tile_row 1 of A \* tile_row 1 of B，其中A是行主序，而B以inter-tile视角看为列主序、以inner-tile视角看是行主序。

<img src="Ascend C.assets/20210519231249649.png" alt="在这里插入图片描述" style="zoom:50%;" />

在利用矩阵计算单元进行大规模的矩阵运算时，由于矩阵计算单元的容量有限，往往不能一次存放下整个矩阵，所以也需要对矩阵进行分块并采用分步计算的方式.(与上一段中面对的情况有何区别？)



**2.向量计算单元**

实现向量与标量之间、向量与向量之间的计算（加法、点乘）。单元的源操作数和目的操作数通常都保存在输出缓冲区中。支持连续寻址、固定间隔寻址和不规则寻址。

另外，矩阵计算后的数据可以通过向量计算单元进行计算（激活函数、池化和格式转换等），然后保存到输出缓冲区或再次返回矩阵计算单元。

**3.标量计算单元**

该单元相当于一个cpu，控制整个AI Core（which相当于一个片上系统SOC）的运行——例如对循环进行控制、实现分支判断、其结果可以通过事件同步模块控制其他单元的执行流水。他还为矩阵或向量计算单元提供数据地址和相关参数的计，并能够实现基本的算术运算。

周围配备多个通用寄存器GPR和专用寄存器SPR。GPR用于变量或地址的寄存以及为算术逻辑运算提供源操作数和存储中间计算结果。SPR则用于支持指令集中的特殊功能。代表性的SPR有CoreID、向量地址寄存器VA和AICore运行状态寄存器等。



#### 存储系统

AI Core的片上存储单元和相应的数据通路构成了存储系统。

**1. 存储单元**

存储单元由存储控制单元、缓冲区和寄存器构成。存储单元即为AI Core的内部存储，统称为Local Memory。

<img src="Ascend C.assets/20210520214901808.png" alt="在这里插入图片描述" style="zoom:50%;" />

- **存储控制单元：**通过`总线接口`可以直接读写 AI Core 之外的更低层级的 L2 缓存, 并且也可以直通到 DDR 或 HBM, 从而可以直接读写内存；存储控制单元中还设置了`存储转换单元（MTE）`，其目的是将输入数据转换成 AI Core 中各类型计算单元所兼容的数据格式。
  - **总线接口单元：**是AI Core的大门、与系统总线交互的窗口。总线接口在读写数据的过程中可以将 AI Core 内部发出的读写请求转换为符合总线要求的外部读写请求, 并完成协议的交互和转换等工作。
  - **存储转换单元：**输入数据从总线接口读入后就会经由存储转换单元进行处理（但不是格式转换，达芬奇架构要求源数据必须被存放于输入缓冲区中，才能够进行格式转换）。存储转换单元作为 AI Core 内部数据通路的传输控制器，负责 AI Core 内部数据在不同缓冲区之间的读写管理，以及完成一系列的格式转换操作，如补零，Img2Col，转置、解压缩等。存储转换单元还可以控制 AI Core 内部的输入缓冲区，从而实现局部数据的缓存。
- **缓冲区：**包括了用于暂存原始图像特征数据的`输入缓冲区`，以及处于中心的`输出缓冲区`来暂存各种形式的中间数据和输出数据。
  - **输入缓冲区：**有利于将大量用于计算的数据一次性的被搬移到 AI Core 内部，由`输入缓冲控制器`负责控制数据流入；由于存储转换单元进行数据的格式转换操作时会产生巨大的带宽需求，达芬奇架构要求源数据必须先被存放于输入缓冲区中，才能够进行格式转换。
  - **输出缓冲区：**存放计算的中间结果（如神经网络中间层的输出）到距离计算单元更近的地方。
- **寄存器：**主要是标量计算单元在使用（GPR、SPR）。另外，在矩阵计算单元还包含有直接的供数寄存器，提供当前正在进行计算的大小为 16\*16的左、右输入矩阵。在矩阵计算单元之后，累加器也含有结果寄存器，用于缓存当前计算的大小为 16\*16 的结果矩阵。



**2. 数据通路**

数据通路指的是 AI Core 在完成一个计算任务时，数据在 AI Core 中的流通路径。下图既包含了核外存储（L2、DDR/HBM）也包含了核内存储系统。**达芬奇架构数据通路的特点是多进单出**。

<img src="Ascend C.assets/20210520223730748.png" alt="在这里插入图片描述" style="zoom:50%;" />

+ **核外存储系统向AI Core传输数据：**核外存储系统中的数据可以通过 LOAD 指令被直接搬运到矩阵计算单元中进行计算，输出的结果会被保存在输出缓冲区中；反复使用的数据可以通过 LOAD 指令先行传入输入缓冲区（以加速下次访存），再通过其它指令传输到矩阵计算单元中。
+ **输出缓冲区与计算单元之间的数据传输：**输出缓冲区和向量计算单元、标量计算单元以及核外存储系统之间都有一条独立的双向数据通路；输出缓冲区中的数据可以通过专用寄存器或通用寄存器进出标量计算单元。
+ **AI Core向外部存储系统传输数据：**AI Core 中的所有数据如果需要向外部传输，都必须经过输出缓冲区，才能够被写回到核外存储系统中。



#### 控制单元

AI Core的控制单元主要包括系统控制模块、指令缓存、标量指令处理队列、指令发射模块、矩阵运算队列、向量运算队列、存储转换队列和事件同步模块。

<img src="Ascend C.assets/20210521135831903.png" alt="在这里插入图片描述" style="zoom:50%;" />

多条指令从系统内存通过总线接口进入到 AI Core 的**指令缓存**中并等待后续硬件快速自动解码或运算。指令被解码后便会被导入**标量指令处理队列**中，实现地址解码与运算控制。这些指令包括矩阵计算指令、向量计算指令以及存储转换指令等。在进入**指令发射模块**之前，所有指令都作为普通标量指令被逐条顺次处理。**标量指令处理队列**将这些指令的地址和参数解码配置好后，由**指令发射模块**根据指令的类型分别发送到对应的**(矩阵、向量或存储转换)指令执行队列**中，而标量指令会驻留在**标量指令处理队列**中进行后续执行。

当指令执行队列中的指令到达队列头部时就进入真正的指令执行环节，并被分发到相应的执行单元中。不同的执行单元可以并行的按照指令来进行计算或处理数据，同一个指令队列中指令执行的流程被称作为指令流水线。

对于指令流水线之间可能出现的数据依赖，达芬奇架构的解决方案是通过设置**事件同步模块**来统一协调各个流水线的进程。事件同步模块时刻控制每条流水线的执行状态，并分析不同流水线的依赖关系，暂停未满足前置依赖的队列的指令发射。

**系统控制模块：**

+ AI Core运行前，需要外部的任务调度器来控制和初始化AI Core的各种配置接口，如指令信息、参数信息以及任务块信息等；
+ 控制任务块的执行进程；
+ 任务块执行完成后，进行中断处理和状态申报；



#### 指令集（ISA）设计

指令集（Instruction Set Architecture，ISA）是计算机程序能够调用的处理器全部功能的集合，是处理器功能的抽象模型，也是作为计算机软件与硬件的接口。指令集包含了数据类型、基本操作、寄存器、寻址模式、数据读写方式、中断、异常处理以及外部 IO 等。

昇腾 AI 芯片的指令集包括**标量指令、向量指令、矩阵指令**和**控制指令**等。标量指令类似于精简指令集（Reduced Instruction Set Computer，RISC），而矩阵、向量和数据搬运指令类似于复杂指令集（Complex Instruction Set Computer，CISC）。





## 算子开发
### 基本概念

+ **TBE**（Tensor Boost Engine）负责执行昇腾AI处理器中运行在AI Core上的算子，TBE提供了基于TVM（Tensor Virtual Machine）框架的自定义算子开发能力，通过TBE提供的API可以完成相应神经网络算子的开发。
+ **Ascend C：**面向算子开发场景的编程语言，原生支持C和C++标准规范，拥有多层接口抽象、自动并行计算、孪生调试等关键技术。



<img src="Ascend C.assets/zh-cn_image_0000001744361226.png" alt="img" style="zoom:50%;" />

AI Core中包含**计算单元、存储单元、搬运单元**等核心组件。计算单元包括了三种基础计算资源：**Cube计算单元、Vector计算单元和Scalar计算单元**。存储单元即为AI Core的内部存储，统称为Local Memory，与此相对应，AI Core的外部存储称之为Global Memory。DMA搬运单元负责在Global Memory和Local Memory之间搬运数据。



#### **核函数**

核函数是Ascend C算子设备侧实现的入口，用户在核函数中进行算子类对象的创建和其成员函数的调用，由此实现该算子的所有功能。在核函数中，需要为在一个核上执行的代码规定要进行的数据访问和计算操作，当核函数被调用时，多个核（AI Core）都执行相同的核函数代码，具有相同的参数，并行执行。

```c++
__global__ __aicore__ void kernel_name(argument list);
```

+ **函数类型限定符：**包含\_\_global\_\_和\_\_aicore\_\_。使用\_\_global\_\_函数类型限定符来标识它是一个核函数，可以被<<<...>>>调用；使用\_\_aicore\_\_函数类型限定符来标识该核函数在设备端AI Core上执行。
+ **变量类型限定符：**指针入参变量需要增加变量类型限定符\_\_gm\_\_。表明该指针变量指向Global Memory上某处内存地址。为了统一表达，建议使用GM_ADDR宏来修饰入参。
+ **规则：**必须返回void；仅支持入参为指针或C/C++内置数据类型；
+ **调用：**`kernel_name<<<blockDim, l2ctrl, stream>>>(argument list);`；核函数的调用是异步的，核函数的调用结束后，控制权立刻返回给主机端；
+ **同步：**强制主机端程序等待所有核函数执行完毕，需调用`aclError aclrtSynchronizeStream(aclrtStream stream);`

核函数与host侧执行函数、device侧执行函数（除核函数之外）的调用关系如下图：

<img src="Ascend C.assets/zh-cn_image_0000001791440449.png" alt="img" style="zoom:75%;" />

#### 编程API

[API文档地址](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0005.html)

Ascend C API的计算操作数都是Tensor类型：GlobalTensor和LocalTensor。

类库API的分类如下：

+ **高阶API：**提供Matmul、SoftMax等高阶API，封装常用算法逻辑，可减少重复开发，提高开发者开发效率。
+ **基础API：**
  - **计算类API**，包括标量计算API、向量计算API、矩阵计算API，分别实现调用Scalar计算单元、Vector计算单元、Cube计算单元执行计算的功能。
  - **数据搬运API**，计算API基于Local Memory数据进行计算，所以数据需要先从Global Memory搬运至Local Memory，再使用计算接口完成计算，最后从Local Memory搬出至Global Memory。执行搬运过程的接口称之为数据搬移接口，比如DataCopy接口。
  - **内存管理API**，用于分配管理内存，比如AllocTensor、FreeTensor接口。
  - **任务同步API**，完成任务间的通信和同步，比如EnQue、DeQue接口。



#### **并行方法**

SPMD对待处理数据切分，把切分后数据分片分发给不同进程处理，每个进程完成全部任务；而流水线并行的每个进程只会专注于一个任务的处理，并将处理完的数据分片传递给负责下一个任务的进程处理。

+ **SPMD数据并行：** *将需要处理的数据拆分并同时在多个计算核心上运行。*

  多个AI Core共享相同的指令代码（即核函数），每个核上的运行实例唯一的区别是block_idx不同，每个核通过不同的block_idx来识别自己的身份。

+ **流水线并行：** *在核函数内部，可以通过流水任务实现数据的并行处理，进一步提升性能。*

  把算子核内的处理程序，分成多个流水任务，通过队列（Queue）完成任务间通信和同步，并通过统一的内存管理模块（Pipe）管理任务间通信内存。

  + **Ascend C分别针对Vector、Cube编程设计了不同的流水任务：**Vector编程范式把算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。;Cube编程范式把算子的实现流程分为5个基本任务：CopyIn，Split，Compute，Aggregate，CopyOut。其中，CopyIn负责搬入，Split负责数据切分，Compute负责计算，Aggregate负责数据汇聚，CopyOut负责搬出。
  + **Ascend C中使用Queue队列完成任务之间的数据通信和同步。**Queue队列管理不同层级的物理内存时，用一种抽象的逻辑位置（QuePosition）来表达各级别的存储，Queue类型包括：VECIN、VECCALC、VECOUT、A1、A2、B1、B2、CO1、CO2。

  <img src="Ascend C.assets/zh-cn_image_0000001744520446-17132374283233.png" alt="img" style="zoom:50%;" />                     <img src="Ascend C.assets/zh-cn_image_0000001791480141.png" alt="img" style="zoom:50%;" />

  <img src="Ascend C.assets/zh-cn_image_0000001791480129.png" alt="img" style="zoom:80%;" />

  > **矢量编程：**CopyIn任务中将输入数据从Global内存搬运至Local内存后，需要使用[EnQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0033.html)将LocalTensor放入VECIN的Queue中；Compute任务等待VECIN的Queue中LocalTensor出队之后才可以完成矢量计算，计算完成后使用[EnQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0033.html)将计算结果LocalTensor放入到VECOUT的Queue中；CopyOut任务等待VECOUT的Queue中LocalTensor出队，再将其拷贝到Global内存。这样 ，Queue队列就完成了三个任务间的数据通信和同步。
  >
  > **矩阵编程：**相似。五个任务都先用DeQue取数，处理后用EnQue放数据。

  + **任务间数据传递使用到的内存统一由内存管理模块Pipe进行管理。**调用[AllocTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0031.html)来为LocalTensor分配内存；调用[FreeTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0032.html)来回收LocalTensor的内存。编程过程中使用到的临时变量内存同样通过Pipe进行管理。临时变量可以使用TBuf数据结构来申请指定QuePosition上的存储空间。使用TBuf申请的内存空间只能参与计算，无法执行Queue队列的入队出队操作。





### 基本流程

+ 算子定义

+ 核函数开发

    + 核函数定义：核函数中调用算子类的Init和Process函数。

    ```c++
    // 实现核函数
    extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        // 初始化算子类，算子类提供算子初始化和核心处理等方法
        KernelAdd op;
        // 初始化函数，获取该核函数需要处理的输入输出地址，同时完成必要的内存初始化工作
        op.Init(x, y, z);
        // 核心处理函数，完成算子的数据搬运与计算等核心逻辑
        op.Process();
    }
    
    // 调用核函数
    void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z)
    {
        add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z);
    }
    ```

    > 使用\_\_global\_\_函数类型限定符来标识它是一个核函数，可以被<<<...>>>调用；使用\_\_aicore\_\_函数类型限定符来标识该核函数在设备端AI Core上执行。指针入参变量需要增加变量类型限定符\_\_gm\_\_，表明该指针变量指向Global Memory上某处内存地址。为了统一表达，使用GM_ADDR宏来修饰入参
    + 实现算子类:
        + 一个空的构造函数
        + 初始化函数 Init()
        + 核心处理函数 Process()：分tiles调用一些private函数（流水任务：搬入、计算、搬出）；

+ **核函数调用（运行验证）**⭐

    需要：调用算子的应用程序、数据[或数据生成脚本]、验证脚本、编译工程文件CMakeLists.txt、编译运行算子的脚本run.sh

    + 应用程序框架
      + #include "data_utils.h"
      + 内置宏\_\_CCE\_KT\_TEST\_\_ 是区分运行CPU模式或NPU模式逻辑的标志
      + 根据运行硬件(\_\_CCE\_KT\_TEST\_\_ )不同：extern算子定义、包含头文件(acl/acl.h **or** tikicpulib.h)、编写调用程序；

    + CPU侧运行验证

    + NPU侧运行验证

+ **基础调用（Kernel Launch）**：为了简化上文描述的算子kernel开发流程，提供更易用的调试调优功能。

    ```c++
    // ACLRT_LAUNCH_KERNEL(kernel_name)(blockDim, stream, argument list);调用方式对内核调用符方式进行了功能加强
    ACLRT_LAUNCH_KERNEL(add_custom)(8, stream, x, y, z); //8个核上调用了add_custom核函数,参数列表为x，y，z
    aclError aclrtSynchronizeStream(aclrtStream stream); //同步
    ```

    

### Kernel Launch算子工程

开发者仅需提供kernel侧实现，基于工程框架可以快速实现Kernel Launch。工程支持PRINTF功能、DumpTensor功能和msprof命令等

参考[cann8文档:算子调用-kernel直调](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0059.html)

**工程目录：**

`cp -r /home/HwHiAiUser/cuixk/opSamples/ReduceSumSample/KernelLaunch/* KernelLaunch/`

<img src="Ascend C.assets/image-20240522095208805.png" alt="image-20240522095208805" style="zoom:50%;" /><img src="Ascend C.assets/image-20240522112043687.png" alt="image-20240522112043687" style="zoom:60%;" />

**1.算子kernel侧实现**

+ 核函数参数改为xxxTilingData（值传递代替针）；xxx_tiling.h中直接用c语法定义该类；

**2.算子调用(主程序实现)**

分为cpu调用和npu调用：

+ cpu侧

<img src="Ascend C.assets/zh-cn_image_0000001924936041.png" alt="img" style="zoom:60%;" />

+ npu侧

> 调用接口：`ACLRT_LAUNCH_KERNEL(kernel_name)(blockDim, stream, argument list);`
>
> 同步函数(强制主机端程序等待所有核函数执行完毕)：`aclError aclrtSynchronizeStream(aclrtStream stream);`

包含相应的头文件，并按下列流程编写主函数：

<img src="Ascend C.assets/zh-cn_image_0000001744361302.png" alt="img" style="zoom:75%;" />

**3.CMake配置文件编写**

+ 配置处理器型号、CANN软件包安装路径、编译模式等(SOC_VERSION、ASCEND_CANN_PACKAGE_PATH、CMAKE_BUILD_TYPE)
+ file(GLOB KERNEL_FILES)处添加kernel实现文件
+ ？配置编译宏，按需使能PRINTF/DumpTensor功能：`ascendc_compile_definitions(kernels PRIVATE -DASCENDC_DUMP)`



**4.数据生成和验证脚本编写**



**5.修改并执行一键式编译运行脚本**

？使能profiling功能：`msprof --ai-core=on --ascendcl=on --model-execution=on --runtime-api=on --task-time=on --application="./build/main" `

`bash run.sh --run-mode=npu  --soc-version=Ascend310P3 --install-path=/usr/local/Ascend/ascend-toolkit/latest  --build-type=Debug`



### Framework Launch算子工程

> 算子工程样例所在路径为：CANN路径中的“tools/msopgen/template/operator_demo_projects/ascendc_operator_sample”

<img src="Ascend C.assets/zh-cn_image_0000001791480217.png" alt="img" style="zoom:75%;" />

与基础调用（Kernel Launch）的核函数直调不同，该小节介绍使用框架调用（Framework Launch）的方式调用自定义算子的工程开发流程。

**0. 算子分析**

+ 算子输入输出：name, shape, type, [format](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlas_ascendc_10_0092.html)
+ 核函数名称
+ 使用的主要接口
+ 算子实现文件名称

<img src="Ascend C.assets/image-17126303151291.png" alt="image" style="zoom:40%;" />

**1. 算子工程创建**

```shell
# prepare
su HwHiAiUser
touch /home/HwHiAiUser/cuixk/opSamples/add_custom.json #as op analysis
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# create
/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen \
 -i /home/HwHiAiUser/cuixk/opSamples/add_custom.json \
 -c ai_core-Ascend310P \
 -lan cpp \
 -out /home/HwHiAiUser/cuixk/opSamples/AddCustom
```

<img src="Ascend C.assets/image-20240416103351591.png" alt="image-20240416103351591" style="zoom:50%;" />

**2. 代码编写**

**(a) \<projectDir\>/op_kernel/xxx.cpp**

+ 头文件

  ```c++
  #include "kernel_operator.h"
  using namespace AscendC;
  constexpr int32_t BUFFER_NUM = 2; 
  ```

+ 算子类：public[ 空构造函数、Init()、Process() ]、 private[CopyIn、Compute、CopyOut、内存、队列等]

+ 核函数：

  + `GET_TILING_DATA(tilingData, tiling);`获取Host侧传入的Tiling参数，再基于Tiling参数计算得到singleCoreSize（每个核上总计算数据大小）、tileNum（每个核上总计算数据分块个数）、singleTileLength（每个分块大小）等变量。

    > 对应的算子host实现中需要定义`TilingData`结构体(xx_tiling.h),实现并注册计算`TilingData`的`TilingFunc`函数(xx.cpp);

  + 调用算子类的Init函数：设置输入输出数据在Global Memory的内存地址（GlobalTensor.SetGlobalBuffer()），通过Pipe内存管理对象为输入输出Queue分配内存（TPipe.InitBuffer()）；每个AI Core视为一个Block，通过处理地址偏移GetBlockIdx() * BLOCK_LENGTH的数据实现inter-block并行；对单核上的数据切片（tile）；每片再切分为两片（开启double buffer），实现inner-block数据的流水线并行； 

  + 调用算子类的Process函数：分tiles调用一些private函数（流水任务：搬入、计算、搬出）

    + CopyIn：为输入数据分配Local内存地址；使用[DataCopy](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0163.html)接口将GlobalTensor数据拷贝到LocalTensor；使用[EnQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0033.html)将LocalTensor放入VecIn的Queue中
    + Compute：使用[DeQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0034.html)从VecIn中取出LocalTensor；为输出数据分配Local内存地址；使用Ascend C接口[Add](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0066.html)完成矢量计算；使用[EnQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0033.html)将计算结果LocalTensor放入到VecOut的Queue中；使用[FreeTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0032.html)将释放不再使用的LocalTensor。
    + CopyOut：使用[DeQue](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0034.html)接口从VecOut的Queue中取出LocalTensor；使用[DataCopy](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0163.html)接口将LocalTensor拷贝到GlobalTensor上；使用[FreeTensor](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlasascendc_api_07_0032.html)将不再使用的LocalTensor进行回收。

  + other issues: 

    + 核函数内推导输入数据类型和格式(算子工程在核函数内提供了DTYPE\_\<Arg\>、ORIG_DTYPE\_\<Arg\>、FORMAT\_\<Arg\>三种宏用于推导核函数入参的数据类型、原始数据类型和数据格式。其中\<Arg\>会自动大写)
    + 非对齐shape算子的kernel侧实现

+ 包装函数：调用核函数



**(b)  \<projectDir\>/op_host/xxx_tiling.h**

+ TilingData结构体设计，包含了切分算法相关参数。

  + 增加#ifndef...的判断条件，防止头文件的重复包含；包含register/tilingdata_base.h头文件
  + BEGIN_TILING_DATA_DEF(xxx)：注册一个名为xxx的类
  + TILING_DATA_FIELD_DEF(uint32_t, totalLength);：添加名为totalLength的uint32_t类型字段
  + REGISTER_TILING_DATA_CLASS(ReduceSumCustom, ReduceSumCustomTilingData)：注册tilingdata类到对应算子
  
  <img src="Ascend C.assets/zh-cn_image_0000001791440445.png" alt="img" style="zoom:50%;" />

**(c)  \<projectDir\>/op_host/xxx.cpp**

+ Tiling实现：根据算子的shape等信息来确定数据切分算法相关参数

  <img src="Ascend C.assets/zh-cn_image_0000001744361318.png" alt="img" style="zoom:75%;" />

  + 相关概念：

    + TilingData：切分算法相关参数结构体，在头文件中定义
    + block_dim：一般将block_dim设置为硬件平台的核数，根据核数进行数据切分
    + TilingKey（可选）：不同的kernel实现分支可以通过TilingKey来标识
    + workspace size（可选）：workspace是设备侧Global Memory上的一块内存。

  + 具体步骤：

    + 获取TilingContext的上下文（就是入参）,context->GetInputTensor(0)得到shape等信息

    + 设置TilingData（头文件中定义的类）：通过调用`set_*`接口来设置TilingData的字段值，通过调用TilingData类的SaveToBuffer接口完成TilingData的序列化和保存。

    + 设置context的blockdim、TilingKey和workspace大小。

    + ```c++
      return ge::GRAPH_SUCCESS;
      ```

  + other issues：

    + 非对齐shape
    + 属性信息通过TilingData传递：context->GetAttrs();

+ Shape推导等函数实现：根据算子的输入张量描述、算子逻辑及算子属性，推理出算子的输出张量描述，包括张量的Shape、数据类型及数据排布格式等信息。

  + static graphStatus InferShape(gert::InferShapeContext* context)
  + 

+ 算子原型注册：原型定义描述了算子的输入输出、属性等信息以及算子在AI处理器相关实现信息。

  + 算子原型定义：描述了算子的输入输出，属性等信息
  + 注册算子支持的AI处理器型号以及相关的配置信息
  + 关联tiling实现和shape推导等函数



**3. 编译部署**

+ 修改CMakePresets.json中**ASCEND_CANN_PACKAGE_PATH**为CANN软件包安装路径
+ ./build.sh
+ ./build_out/custom\_opp\_\<target os\>\_\<target architecture\>.run

部署后的目录结构：

<img src="Ascend C.assets/image-20240506143036888.png" alt="image-20240506143036888" style="zoom:80%;" />

**4. 运行验证**

+ 算子UT（Unit Test）测试：

*Ascend C动态shape算子暂不支持UT测试功能*



+ 算子ST（System Test）测试：

  + 生成算子测试用例定义文件并修改：`/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopst create -i ReduceSumCustom/op_host/reduce_sum_custom.cpp -out ./ReduceSumCustom_st/`

  + （可选）自定义期望数据生成函数：

    + 在Python文件中实现算子期望数据生成函数，文件目录和文件名称可自定义；测试用例定义文件中的全部Input、Output、Attr的name作为算子期望数据生成函数的输入参数，若Input是可选输入，请将该参数指定默认值为None。

    + 在ST测试用例定义文件“*OpType*_case.json”中增加比对函数。配置算子测试用例定义文件。

      ` "calc_expect_func_file": "/home/test/test_add_st.py:calc_expect_func"`

  + 配置环境变量并生成、执行测试用例

    ```shell
    # 已创建算子ST测试用例定义文件“AddCustom_case.json”，例如存储到跟算子工程目录“AddCustom”同级别的“AddCustom_st”路径下
    
    # 配置ST测试用例执行时依赖的环境变量
    export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub
    
    # 生成测试用例文件并执行
    /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopst run \
    -i /home/HwHiAiUser/cuixk/opSamples/ReduceSumSample/FrameworkLaunch/ReduceSumCustom_st/ReduceSumCustom_case.json \
    -soc Ascend310P3 \
    -out /home/HwHiAiUser/cuixk/opSamples/ReduceSumSample/FrameworkLaunch/ReduceSumCustom_st/
    ```
  
  + 测试结果记录在st_report.json中

**5. 算子调用**



### 矢量编程



<img src="Ascend C.assets/zh-cn_image_0000001744348642-17155682486492.png" alt="img" style="zoom:60%;" />

矢量单元分为8个block(与表示AI Core的block不同)各32字节，因此一次能够处理8*32=256字节。

 

### 矩阵编程

高阶算子：Matmul、Matmul Tiling；基础算子：Gemm

<img src="Ascend C.assets/zh-cn_image_0000001793118549.png" alt="img" style="zoom:55%;" />

如图，A1/B1用于存放整块矩阵；A2/B2用于存放切分后的小块矩阵；CO1用于存放小块结果矩阵；CO2用于存放整块结果矩阵；





## 算子调用

+ **内建算子**
  + **算子清单**中算子对应的头文件位于：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/`，包含了若干函数的注册（原型）
  + **二进制算子包**安装位置：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`，包含了若干数学函数（激活函数、三角函数、范数、最大最小值等）

+ **自定义算子**



算子调用的三种方式：kernel直调、单算子调用、第三方框架调用

**单算子调用**有API执行和模型执行两种方式：

+ API执行：**基于C语言的API执行算子**，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。
+ 模型执行：**基于图IR执行算子**，先编译算子（例如，使用ATC工具将Ascend IR定义的单算子描述文件编译成算子离线模型文件(*.om)），再调用AscendCL接口加载算子模型，最后调用AscendCL接口执行算子。



### 单算子API执行

#### 自定义算子

自定义算子编译部署(见*算子开发-Framework Launch算子工程-3*)后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：

```c++
aclnnStatus aclnnXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnXxx(void* workspace, int64 workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

其中aclnn*Xxx*GetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小后，按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnXxx执行计算。

**流程**

+ 前置步骤：算子开发完成，编译部署时开启二进制编译功能，算子工程目录下`build_out/auto_gen`正确包含aclnn_XXX.cpp和.h文件

+ 创建调用算子的工程文件

  <img src="Ascend C.assets/image-20240514114255531.png" alt="image-20240514114255531" style="zoom:50%;" />

+ 编写算子调用的代码

  <img src="Ascend C.assets/zh-cn_image_0000001791458793.png" alt="img" style="zoom:75%;" />

+ 编写CMakeLists文件：在头文件的搜索路径include_directories中增加build_out/autogen；在生成可执行文件规则add_executable中增加aclnn_add_custom.cpp；同时需要链接nnopbase链接库；

  ```makefile
  set(CUST_PKG_PATH "../../AddCustom/build_out/autogen")
  include_directories(
      ${INC_PATH}/runtime/include
      ${INC_PATH}/atc/include
      ../inc
      ${CUST_PKG_PATH}
  )
  add_executable(execute_add_op
      ${CUST_PKG_PATH}/aclnn_add_custom.cpp
      operator_desc.cpp
      op_runner.cpp
      main.cpp
      common.cpp
  )
  target_link_libraries(execute_add_op
      ascendcl
      acl_op_compiler
      nnopbase
      stdc++
  )
  ```

+ 生成测试数据：`python3 scripts/gen_data.py`

+ 编译与运行：

  ```shell
  export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
  export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
  
  # 编译
  mkdir -p build
  cd build
  cmake ../src
  make
  
  # 执行
  #以运行用户（例如HwHiAiUser）拷贝开发环境中样例工程output目录下的execute_add_op到运行环境任一目录。
  chmod +x execute_add_op
  ./execute_add_op
  
  # 比较真值文件
  python3 scripts/verify_result.py output/output_z.bin output/golden.bin
  ```

  

#### 内建算子





## 算子调试调优

**孪生调试：**相同的算子代码可以在CPU域调试精度，NPU域调试性能。编写Ascend C算子kernel侧源码后：1）通过通用的GCC编译器进行编译，编译生成通用的CPU域的二进制，可以通过gdb通用调试工具等调试手段进行调试；2）通过毕昇编译器进行编译，编译生成NPU域的二进制文件，可以通过仿真打点图或者Profiling工具进行上板数据采集等方式进行调试。

<img src="Ascend C.assets/zh-cn_image_0000001744520382.png" alt="img" style="zoom:65%;" />



**CPU域调试**

+ gdb调试

  

+ printf()打印：在代码中直接编写printf(...)来观察数值的输出。注意：NPU模式下目前不支持打印语句，所以需要添加内置宏**\_\_CCE_KT_TEST\_\_**予以区分。

  ```c++
  #ifdef __CCE_KT_TEST__
  printf("xLocal size: %d\n", xLocal.GetSize()); 
  #endif
  ```



**NPU域上板调试**

仅支持Kernel Launch、单算子API执行和间接调用单算子API(aclnnxxx)接口。





**性能分析工具？**

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/developmenttools/devtool/atlasprofiling_16_0001.html





## AICPU 算子开发

参考资料：[cann文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/opdevg/atlasopdev_10_0072.html)、[代码示例](https://gitee.com/ascend/samples/tree/master/cplusplus/level1_single_api/4_op_dev/2_verify_op/acl_execute_addblockcust)、[博客](https://blog.csdn.net/weixin_42172676/article/details/126595055)

### 算子开发、编译与部署

```sh
cd /home/algo/xdu/demoAdd/ops/ascend_c/CpuOps/
./build.sh 310P
build_out/custom_opp_centos_aarch64.run #运行run文件部署
```



### 算子生成（om）和调用

<img src="Ascend C.assets/zh-cn_image_0000001744508762.png" alt="img" style="zoom:67%;" />

[算子ST测试](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/opdevg/atlasopdev_10_0093.html)：ST测试的主要功能是：基于算子测试用例定义文件*.json生成单算子的om文件；使用AscendCL接口加载并执行单算子om文件，验证算子执行结果的正确性。

**1.利用ATC工具，根据单算子描述文件（json）生成单算子的离线模型（.om）**

编写json文件；

cd /home/algo/xdu/demoAdd/ops/ascend_c/AclOfflineInvocation/

`atc --singleop=config/aicpu_my_cpu_add_kernel.json --soc_version=Ascend310P3 --output=op_models`

om文件命名规范为：序号+opType + 输入的描述(dateType_format_shape)+输出的描述

**2.编写cpu侧调用代码，编译运行**

```sh
python3 scripts/myCpuAdd/gen_data.py

export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64

mkdir build
cd build
cmake ..
make

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0
../output/execute_op_test
```







# Debug

### 端到端算子开发

**a. 调用时函数名错误**

已解决：改为正确拼写

<img src="Ascend C.assets/image-20240416142714776.png" alt="image-20240416142714776" style="zoom:80%;" />

**b. 文件名称不匹配：** 

已解决：算子原型注册 this->AICore().AddConfig("ascendxxx");替换为实际型号

<img src="Ascend C.assets/image-20240416141749599.png" alt="image-20240416141749599" style="zoom:80%;" />

<img src="Ascend C.assets/image-20240416141822189.png" alt="image-20240416141822189" style="zoom:80%;" />

`cmake/func.cmake: function(opbuild), line 37`

/usr/local/Ascend7.0/ascend-toolkit/latest/toolkit/tools/opbuild/op_build \ /home/HwHiAiUser/cuixk_test/mySamples/AddCustom/build_out/autogen/libascend_all_ops.so \ /home/HwHiAiUser/cuixk_test/mySamples/AddCustom/build_out/autogen



`cmake/func.cmake: function(opbuild), line 17`

/usr/bin/c++ -g -fPIC -shared -std=c++11  -D_GLIBCXX_USE_CXX11_ABI=0 -I /usr/local/Ascend7.0/ascend-toolkit/latest/include -L \ /usr/local/Ascend7.0/ascend-toolkit/latest/lib64 -lexe_graph -lregister -ltiling_api -o /libascend_all_ops.so



**c. build.sh时找不到头文件或目录**

已解决：修改CMakePresets.json中**ASCEND_CANN_PACKAGE_PATH**为CANN软件包安装路径。



**d. ATC run failed**

已解决：kernel文件中加入`using namespace AscendC;`和`constexpr int32_t BUFFER_NUM = 2; `

<img src="Ascend C.assets/image-20240506162157405.png" alt="image-20240506162157405" style="zoom:50%;" />



### 完整流程

/usr/local/Ascend/ascend-toolkit/latest/include/exe_graph/runtime/shape.h

<img src="Ascend C.assets/image-20240513171041127.png" alt="image-20240513171041127" style="zoom:80%;" />



### Kernel Launch

用ACLRT_LAUNCH_KERNEL调用时参数列表不匹配。

`/home/HwHiAiUser/cuixk/opSamples/ReduceSumSample/KernelLaunch/build/include/kernels/aclrtlaunch_reduce_sum_custom.h`中的定义从何而来？为何与我的定义不同

<img src="Ascend C.assets/image-20240521112207564.png" alt="image-20240521112207564" style="zoom:80%;" />

![image-20240521112233162](Ascend C.assets/image-20240521112233162.png)



### Framework Launch

**ReduceSum算子，st时结果与预期不符：**

结果全为0，核函数删除op.process()外的if后，结果如下：

<img src="Ascend C.assets/image-20240527171312268.png" alt="image-20240527171312268" style="zoom:80%;" />

而同样的核函数在kernel launch时则结果正确：

<img src="Ascend C.assets/image-20240527171803905.png" alt="image-20240527171803905" style="zoom:80%;" />

改变测试数据的shape：4532->4096，则结果通过测试。

推测问题原因可能在于：API使用不正确，如repate、worksize等参数设置错误等



### 单算子调用

**简化版：**`operator/AddCustomSample/FrameworkLaunch/AclNNInvocationNaive`

**完整版：**`operator/AddCustomSample/FrameworkLaunch/AclNNInvocation`

```shell
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
```

设置环境变量后：

<img src="Ascend C.assets/image-20240528145732153.png" alt="image-20240528145732153" style="zoom:80%;" />

根据[文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/operatordevelopment/ascendcopdevg/atlas_ascendc_10_0041.html#ZH-CN_TOPIC_0000001744520278__section1930615371323)，修改cmakelists文件后，成功编译。

运行时出现错误代码161001：ACLNN_ERR_PARAM_NULLPTR；

<img src="Ascend C.assets/image-20240528164421423.png" alt="image-20240528164421423" style="zoom:80%;" />

<img src="Ascend C.assets/image-20240528171324450.png" alt="image-20240528171324450" style="zoom:60%;" />

然而根据debug信息，传入指针非空。



### 单算子OM模型调用

![image-20240612185310728](Ascend C.assets/image-20240612185310728.png)

错误代码100024是ACL_ERROR_OP_NOT_FOUND; 代表没找到对应算子

解决方案：main文件CreateOpDesc函数中定义要与算子一致

<img src="Ascend C.assets/image-20240613092958843.png" alt="image-20240613092958843" style="zoom:80%;" />

错误代码507018是ACL_ERROR_RT_AICPU_EXCEPTION ；代表AI CPU execution error

