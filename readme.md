# Truth inference in WuzhangaiAPP
Implementation of the Multi-task Streaming Truth Inference algorithm (MSTI) in the 'Wuzhangai APP'. 

Made from Beihang SDP-Crowdintelligence Group.

### Note that:
This algorithm is a variant of the classic truth inference algorithm DS [1].

[1] Dawid A P, Skene A M. Maximum likelihood estimation of observer error‐rates using the EM algorithm[J]. Journal of the Royal Statistical Society: Series C (Applied Statistics), 1979, 28(1): 20-28.





## 中文介绍与使用说明
本算法功能：在多任务流标注数据场景下的众包标注结果汇聚("结果汇聚"即"真值推理")  

### 仿真标注数据场景下的使用说明：  
1) 生成仿真的众包标注数据: `python wuzhangai_label_generate.py` 生成`simulation_annotations_init.npy`和`truth_init.npy`.  

2) 运行主程序以进行首次的结果汇聚：`python main.py --mode train_init --num_classes 3 3 --load_truth_information True` 
或者 `python main.py --mode train_init --num_classes 3 3`  
  将会生成如下文件:
  - `final_prediction_on_okinstances_probability.npy`: 算法认为已经推理完成的样本（即不用再进行进一步的审核标注的样本）的推理结果信息，其中为概率值.  
  - `final_prediction_on_okinstances.npy`: 算法认为已经推理完成的样本的推理结果信息，其中为具体的类别.  
  - `tobesolved_instances_annotations.npy`: 算法认为目前还没有得到可信赖的结果的样本（即还需再进行进一步的审核标注的样本）的标注信息，它作为历史信息被存起来为了在下一步的结果汇聚过程中再次使用.    
  - `worker_normalizer_all.npy`: 关于工人能力的信息，它作为历史信息被存起来以为了在下一步的结果汇聚过程中被再次使用.     
  - `worker_pi_all.npy`: 同样，关于工人能力的信息，它作为历史信息被存起来以为了在下一步的结果汇聚过程中被再次使用.   
  
3) 生成仿真的"首次流数据"的众包标注数据: `python wuzhangai_label_generate.py --new_stream_data True` 生成`simulation_annotations_stream_1.npy`和`truth_stream_1.npy`.  
4) 运行主程序以进行第二次的结果汇聚，即对新的流数据进行结果推理：`python main.py --mode train_stream --num_classes 3 3 --load_truth_information True` 
或者 `python main.py --mode train_stream --num_classes 3 3`  
   

### 真实标注场景下的使用说明：
首次运行结果汇聚主程序时，将simulation_annotations_init.npy替换成真实的标注数据，并运行`python main.py --mode train_int --num_classes 3 3` (或者不替换simulation_annotations_init.npy，
而是更改相对应的命令);
随后，在每一次的流数据场景下，将simulation_annotations_stream_1.npy替换成真实的标注数据，并运行`python main.py --mode train_stream --num_classes 3 3` (或者不替换simulation_annotations_stream_1.npy，
而是更改相对应的命令). 
另外，注意：1）替换数据时，请注意数据格式. 2）需要注意主程序parser中所可能需要改变的参数，尤其是`num_classes`这一属性.

##### 比如：当我们想要跑数据`./data/initdata.npy`时:
`python main.py --init_data_path './data/initdata.npy' --num_classes 3 3 10`
