# x64平台cpu端使用ncnn部署yolox模型
- 工作环境：Visual Studio 2019
- 技术路线说明：在x64（本地windows设备）上使用ncnn作为推理框架部署模型时，需要源码编译出ncnn三方库和头文件，而ncnn依赖于opencv和protobuf。因此，本项目的技术路线如下：
```
    源码编译opencv----->源码编译protobuf(libprotobuf.so为编译caffe2ncnn.exe模型生成工具所需的依赖)----->编译ncnn
```
## 1. 编译ncnn
```
如上所述，在编译ncnn之前需要先编译opencv和protobuf，且如果ncnn需要调试或者使用ncnn框架的推理工程需要debug调试模型，则opencv、protobuf和ncnn也需要编译debug版本。
```
### 1.1 编译opencv
- 编译环境：Visual Studio 2019 IDE
- 编译版本：opencv4.5.4
- 注意问题：opencv源码安装项目应该放在全英文目录下，不能放在中文目录下进行编译
- 编译方式： Release and Debug
- 编译方法：
```
step1：登录opencv官网，下载安装包；opencv官网链接：https://opencv.org/releases/；
step2：修改bat脚本。下载好的opencv保存在\ncnn\opencv-4.5.4中，配置三个bat脚本文件（env.bat、vs_create_projectname.bat和vs_run_projectnameu.bat）。其中env.bat存放了Visual Studio 2019的安装路径，须根据自身设备环境进行修改；vs_create_projectname.bat存放编译方法，通过修改cmake后的内容来控制编译哪些库（设置安装路径和各开关量）；vs_run_projectnameu.bat放置了可执行sln，根据环境不同须修改VC_DIR路径；
step3：开始编译。首先点击vs_create_projectname.bat进行mkdir build和cmake操作；然后点击vs_run_projectnameu.bat便用Visual Studio 2019打开了此项目；
step4：选择Debug模式还是Release模式（我两个版本都编译了）；右键点击"解决方案"OpenCV""--->"CMakeTargets"--->"ALL_BUILD"--->"重新生成"，等待编译完成；右键点击"解决方案"OpenCV""--->"CMakeTargets"--->"INSTALL"--->"仅用于项目/仅生成INSTALL"，等待安装完成；
step5：安装完毕的lib和include位于opencv-4.5.4\build\install目录下，可以拷贝到其他地方使用。
```
- 说明：我在编译的时候打开了BUILD_opencv_world开关，表示将所有的库都包在libopencv_world这个库中，使用时直接调用这一个库就好。

### 1.2 编译protobuf
- 编译版本：protobuf3.4.0
- 编译环境：Visual Studio 2019命令行工具
- 编译方式： Release and Debug
- 官网文档地址：https://developers.google.com/protocol-buffers/
- 官网github地址：https://github.com/google/protobuf/releases
- 编译方法：查看ncnn\protobuf-3.4.0\src\README.md文件，其中包含了windows和unix下安装protobuf的方法--->ncnn\protobuf-3.4.0\cmake\README.md
```
首先需要确保编译平台存在cmake和Visual Studio
1.2.1 Environment Setup
    step1：从开始菜单打开"x64 Native Tools Command Prompt for VS2019"命令行工具，然后更改到工作目录；
    比如：
        D:\Program Files (x86)\Microsoft Visual Studio\2019\Community>E:
        E:\>cd E:\_bak\algorithm\ncnn\protobuf-3.4.0
        E:\_bak\algorithm\ncnn\protobuf-3.4.0>
    step2：在工作目录创建一个文件夹，headers/libraries/binaries将均放在此文件夹下：mkdir install；
    step3：如果cmake命令在命令行无法找到，那么将其添加到环境变量中：
        set PATH=%PATH%;xxxxxx(path)\CMake\bin
1.2.2 Getting Sources
    step1：首先我通过github链接下载的zip源码包，但是包里没有用于单元测试的gmock（这个必须下载，如果不下载则1.2.3CMakeLists.txt会报错，到时候只能修改CMakeLists.txt了），所以这里需要下载gmock；
    step2：执行命令下载gmock：git clone -b release-1.7.0 https://github.com/google/googlemock.git gmock （这行命令的意思是克隆googlemock.git项目的release-1.7.0分支到protobuf-3.4.0\gmock目录）。我本地克隆报错了，所以我手动下载好放进此目录；
    step3：进入gmock目录，继续下载gtest，执行命令：git clone -b release-1.7.0 https://github.com/google/googletest.git gtest（如报错则继续按照step2手动操作）；
    step4：回到protobuf-3.4.0根目录。

注意：如果1.2.2不想操作，则在1.2.3执行cmake ..操作的时候需要增加参数，如下参考：
    If the *gmock* directory does not exist, and you do not want to build protobuf unit tests,you need to add *cmake* command argument `-Dprotobuf_BUILD_TESTS=OFF` to disable testing.

1.2.3 CMake Configuration(Debug同理，创建build/debug在该目录下操作即可)
    step1：进入cmake目录（ncnn\protobuf-3.4.0\cmake），创建build目录，再创建build/release目录，并进入cmake\build\release目录；
        如下：E:\_bak\algorithm\ncnn\protobuf-3.4.0\cmake\build\release>
    step2：在release目录下执行如下命令：
        cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../../../install ../..
    step3：按照step2的操作则已经在ncnn\protobuf-3.4.0\cmake\build\release目录下生成了nmake Makefile文件。
1.2.4 Compiling and install
    step1：依旧在E:\_bak\algorithm\ncnn\protobuf-3.4.0\cmake\build\release>目录，执行：nmake，即等待编译完成
    step2：依旧在同一目录下执行安装命令：nmake install，等待安装完成即可。headers/libraries/binaries被保存到ncnn\protobuf-3.4.0\install目录下，可以拷贝到其他地方使用。
```

### 1.3 编译ncnn
- 编译环境：Visual Studio 2019命令行工具
- 官网学习链接：https://github.com/Tencent/ncnn/wiki/use-ncnn-with-alexnet.zh
- how to build： https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017
- github网址：https://github.com/Tencent/ncnn
- 编译方式： Release and Debug
- 编译方法：
    - step1：首先将1.1和1.2小节中得到的库和头文件都放在ncnn\ncnn-master\3rdparty目录下（作为生成ncnn三方库和模型转换工具的依赖项）
    - step2：编译：
        Release版本库
        ```
        （1）从开始菜单打开"x64 Native Tools Command Prompt for VS2019"命令行工具，然后更改到工作目录为ncnn-master;
        （2）在此目录下创建build_tools文件夹，并进入此文件夹；
        （3）执行编译指令：cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=E:/_bak/algorithm/ncnn/ncnn-master/3rdparty/protobuf/include -DProtobuf_LIBRARIES=E:/_bak/algorithm/ncnn/ncnn-master/3rdparty/protobuf/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=E:/_bak/algorithm/ncnn/ncnn-master/3rdparty/protobuf/bin/protoc.exe -DNCNN_VULKAN=OFF -DNCNN_BUILD_WITH_STATIC_CRT=ON ..
        （4）等待执行完成后，执行命令：nmake；
        （5）等待完成后执行命令：nmake install；
        （6）等待执行完毕后在ncnn-master\build_tools\install目录下生成bin(包含转模型工具)、lib和include文件。
        ```
        Debug版本库
        ```
        （1）从开始菜单打开"x64 Native Tools Command Prompt for VS2019"命令行工具，然后更改到工作目录为ncnn-master;
        （2）在此目录下创建build_tools文件夹，并进入此文件夹；
        （3）执行编译指令：cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=E:/_bak/algorithm/ncnn/ncnn-master/3rdparty/protobuf/include -DProtobuf_LIBRARIES=E:/_bak/algorithm/ncnn/ncnn-master/3rdparty/protobuf/lib/libprotobufd.lib -DProtobuf_PROTOC_EXECUTABLE=E:/_bak/algorithm/ncnn/ncnn-master/3rdparty/protobuf/bin/protoc.exe -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_WITH_STATIC_CRT=ON ..  (把模型转换工具开关关掉)
        （4）等待执行完成后，执行命令：nmake；
        （5）等待完成后执行命令：nmake install；
        （6）等待执行完毕后在ncnn-master\build_tools\install目录下生成lib和include文件。
        ```

## 2. 模型转换caffe->ncnn生成ncnn离线模型
- 用途：生成ncnn需要的.bin文件和.param文件，用于推理
- 运行环境："x64 Native Tools Command Prompt for VS2019"命令行工具
- 操作流程：
    - step1：修改.prototxt文件内容。（1）修改文件头，即输入，按照ncnn解析的模式进行修改；（2）type: "Pooling"算子的round_mode: FLOOR参数去掉，因为ncnn模型转换工具没有这个参数，加上会报错；（3）type: "Upsample"算子的upsample_param参数去掉，因为ncnn模型转换工具没有这个参数，加上会报错；（4）最后的type: "Permute"算子去掉，ncnn在转模型和推理时候加上这个层会报错，索性用c++开发；（5）type: "Flatten"和（4）同样的问题。
    - step2：将修改好了.prototxt和之前得到的.caffemodel文件都统一打包放在ncnn\3rdparty\models目录下；
    - step3: 到ncnn\3rdparty\ncnn\bin目录下，执行如下命令分别生成phonedet和pearl检测模型的ncnn文件。（使用"x64 Native Tools Command Prompt for VS2019"命令行工具生成，执行caffe2ncnn.exe --help查看传参说明）
        - 执行命令：caffe2ncnn.exe ../../models/phonedet.prototxt ../../models/phonedet.caffemodel ../../models/phonedet.param ../../models/phonedet.bin
        - 生成文件：phonedet.param和phonedet.bin
        - 执行命令：caffe2ncnn.exe ../../models/pearl.prototxt ../../models/pearl.caffemodel ../../models/pearl.param ../../models/pearl.bin
        - 生成文件：pearl.param和pearl.bin
    - step3：生成的.bin文件和.param文件便可用于推理工程推理了。

```
一些思考：如果想要开发"Pooling"和"Upsample"这些算子的传参，则需要修改ncnn-master\tools\caffe中的.proto和.cpp文件，同时也需要修改ncnn-master\src\layer中的推理API。其过程比较繁琐，不建议这么操作。
```

## 3. 使用ncnn进行yolox模型推理的示例工程（含推理工程编译）
- 用途：使用ncnn库进行模型的推理测试,以珍珠检测为例
- 编译运行环境：Visual Studio 2019IDE
- 目录结构:
```
        ├─3rdparty----------------------依赖的三方库
        │  ├─ncnn-----------------------编译好的x64的开发库(包括debug和release)
        │  └─opencv---------------------编译好的opencv开发库(包括debug和release)
        ├─build-------------------------编译目录
        ├─images------------------------测试的输入图片
        ├─include-----------------------头文件
        ├─lib---------------------------推理依赖的bin文件
        │  └─common
        ├─src
        │  ├─main.cpp---- --------------测试推理和使用推理api的样例
        │  └─inference.cpp--------------sdk源码
        ├─models------------------------模型存放文件夹
        ├─results----------------------检测结果存放文件夹
        ├─pearl_api_cmake---------------存放生成推理库的cmake文件
        ├─env.bat-----------------------设置环境路径
        ├─vs_create_projectname.bat-----进行cmake编译
        ├─vs_run_projectname.bat--------进入Visual Studio 2019
        └─CMakeLists.txt
```
- 操作流程：
    - step1：准备工作:
        - (1)将1小节中生成的三方库(为例方便,我将三方库整理存放到了ncnn\3rdparty目录下)拷贝到ncnn\x64_cpu_ncnn_inference_pearldet\3rdparty目录下,包含opencv(opencv我把所有库都包在了libopencv_world.dll这一个库中了)和ncnn; 
        - (2)将第2小节中转出的ncnn离线模型(.param和.bin文件)存放在ncnn\x64_cpu_ncnn_inference_pearldet\models目录下;
        - (3)如果模型结构发生变化,则通过pytorch工程重新获取bin文件,并将其存放在..\lib\common目录下;
        - (4)修改env.bat文件中的路径名为操作者环境的安装路径;
        - (5)同样地,修改vs_run_projectname.bat中的三方库路径和Visual Studio 2019安装路径;
        - (6)根据自身需求,修改vs_create_projectname.bat中的DCMAKE_BUILD_TYPE编译项(Debug或Release).
    - step2：开始编译:
        - 点击vs_create_projectname.bat执行脚本,输出如下结果表示编译正确
        ```
        ...
        -- Configuring done
        -- Generating done
        -- Build files have been written to: E:/_bak/algorithm/ncnn/x64_cpu_ncnn_inference_pearldet/build
        ```
    - step3：进入Visual Studio 2019
        - 点击vs_run_projectname.bat进入Visual Studio 2019;
        - 选择解决方案配置(Debug或Release)与vs_create_projectname.bat中保持一致;
        - 鼠标右击选择"解决方案'pearldet_test'"--->"属性"--->"启动项目",改为pearl_test;
        - 鼠标右击选择"解决方案'pearldet_test'"下的pearldet_test--->鼠标右击"属性"--->"C/C++"--->"代码生成"--->"运行库"，将"多线程DLL(/MD)"修改为"多线程(/MT)"，然后重新生成解决方案(这是以Release为例，Debug也需要对应修改);
        - 编译通过,可以正常运行;
        - 通过更改vs_create_projectname.bat中的DCMAKE_BUILD_TYPE编译项来重新开始编译debug版本;
        - 通过设置-DBuild_Example=OFF或ON，可以切换到生成inference.dll调用库或编译示例(默认是编译示例)。
### 注意点
```
Debug模型时报错"User Error 1001: argument to num_threads clause must be positive"信息直接忽略即可。
```

## 4. 关于ncnn模型加密
- 用途：对于某些场景的应用，需要对ncnn模型(.bin及.param文件)进行加密后转换成二进制文件进行推理（同时为了适配对加密模型的解析，故对源码ncnn进行了修改，增加了对加密模型的解密接口int load_param_encrypt，这一部分的应用需要参考x64_cpu_ncnn_inference_phonedet推理工程，此套工程是对phone检测的加密ncnn推理示例）；
- 编译运行环境：Visual Studio 2019IDE
- 加密工程对应项目名：EncodeNcnnModel
- 使用加密模型进行推理的示例：x64_cpu_ncnn_inference_phonedet工程
- 加密工具所对应工程的目录结构（EncodeNcnnModel项目）
```
        ├─build-------------------------编译目录
        ├─include-----------------------头文件
        │  └─tools
        ├─model-------------------------模型存放文件夹(存放需要转换的ncnn模型)
        ├─src---------------------------源文件
        ├─env.bat-----------------------设置环境路径
        ├─vs_create_projectname.bat-----进行cmake编译
        ├─vs_run_projectname.bat--------进入Visual Studio 2019
        └─CMakeLists.txt
```
- 操作流程：
    - step1：准备工作：
        - (1)将需要加密的第2小节中转出的ncnn离线模型(phonedet.param和phonedet.bin文件)存放在ncnn\EncodeNcnnModel\model目录下;
        - (2)修改env.bat文件中的路径名为操作者环境的安装路径;
        - (3)同样地,修改vs_run_projectname.bat中的三方库路径（这里没有三方依赖）和Visual Studio 2019安装路径。
    - step2：开始编译:
        - 点击工程目录下的vs_create_projectname.bat执行脚本,输出如下结果表示编译正确
        ```
        ...
        -- Configuring done
        -- Generating done
        -- Build files have been written to: E:/_bak/algorithm/ncnn/EncodeNcnnModel/build
        ```
    - step3：进入Visual Studio 2019
        - 点击vs_run_projectname.bat进入Visual Studio 2019;
        - 鼠标右击选择"解决方案'encode_model'"--->"属性"--->"启动项目",改为encode_file;
        - 鼠标右击选择"解决方案'encode_model'"--->"重新生成解决方案";
        - 点击菜单栏"调试"--->"encode_file调式属性"--->"配置属性"--->"调试"，修改"工作目录"为项目所在绝对路径、"命令参数"为模型存放和生成存放地址，如：model/phonedet.param model/phonedet.enprm；
        - 点击运行，运行完毕可生成加密后的模型。
        - 加密获得结果：phonedet.param--->phonedet.enprm、phonedet.bin--->phonedet.enbin，phonedet.enprm和phonedet.enbin可以用在x64_cpu_ncnn_inference_phonedet工程进行加密模型解密推理。
    - step4：将加密后的模型放在ncnn\x64_cpu_ncnn_inference_phonedet\models目录下即可进行推理，x64_cpu_ncnn_inference_phonedet工程的使用方法同第3小节，只不过增加了解密模块的接口，不予赘述。

### 注意点
```
>>1. 在windows上推理的工程必须要在windows上对模型进行加密，如果模型加密平台和解密平台不一致可能会导致模型无法正确解析的问题（我猜测是因为不同平台模型编码方式不同所导致）。换言之，EncodeNcnnModel和x64_cpu_ncnn_inference_phonedet必须在同一平台下进行编译执行；
>>2. 加密和解密必须配合使用，且完全一致。
```

## 5. yolox的pytorch工程模型转换说明（pytorch->caffe）
- 用途：获取caffe模型，即pytorch->caffe
- 对应工程示例名称：pytorch_phonedet_and_pearl（以手机检测和珍珠检测为例）
- 执行环境：yolox容器，获取对应镜像的方法在下节
- 使用方法：（也可以参考Tengine\pytorch-phonedet\README.md）
    - step1：明确目录结构，Caffe目录为pytorch2caffe工具（同tengine工具一致），images下存放了pearl和phonedet的测试图片、models下存放了训练所得的pth文件、图片测试的结果存放在YOLOX_outputs目录下；
    - step2：修改根目录下demo测试脚本中的权重路径，测试图片路径，torch2caffe等开关量，修改生成的caffe模型名字；
    - step3：执行python demo_phonedet.py转出phonedet.prototxt和phonedet.caffemodel文件；
    - step4：执行python demo_pearl.py转出pearl.prototxt和pearl.caffemodel文件；（demo_pearl.py和demo_pearl_nms.py的区别是后者的nms操作与训练保持一致，测出来的效果更好，而前者测试小图效果较好，大图则存在漏检，但转模型时都不受影响）
- 注意点：pytorch_phonedet_and_pearl中包含了pytorch2caffe工具，此工具命名为Caffe，此工具可以迁移到其他工程中使用，使用方法参考pytorch_phonedet_and_pearl/demo.py中用法

```
此时已经得到caffe结构的模型（.prototxt和.caffemodel文件），可以按照第2小节的方法生成ncnn所需的模型文件了。
```

## 6. 工具-镜像
- 调试pytorch训练工程的镜像
```
获取工具的方式：
本人的dockerhub直接pull：命令为docker pull fanacio/yolo_yolox
或者直接找一个包含python依赖库的docker即可。
```
