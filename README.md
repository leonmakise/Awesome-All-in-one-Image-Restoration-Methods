
# Awesome-All-in-one-Image-Restoration-Methods
<p align="center">
  <img src="./figures/logo.jpg" alt="image" style="width:200px;">
</p>

A list of awesome all-in-one image restoration methods. Updating...!

Reference: Diffusion Models for Image Restoration and Enhancement--A Comprehensive Survey

Link: https://github.com/lixinustc/Awesome-diffusion-model-for-image-processing




**Purpose**: We aim to provide a summary of all-in-one image processing techniques, including restoration, enhancement, coding, and quality assessment. More papers will be summarized.

[Jiaqi Ma](https://scholar.google.com/citations?user=BJUlpoMAAAAJ&hl=zh-CN)<sup>1,✢</sup>, [Xu Zhang](https://scholar.google.com/citations?user=xDDy-DwAAAAJ&hl=zh-CN)<sup>1,✢</sup>, [Guoli Wang](https://scholar.google.com/citations?user=z-25fk0AAAAJ&hl=zh-CN)<sup>2,</sup>, [Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>2,</sup>, [Lefei Zhang](https://scholar.google.com/citations?user=BLKHwNwAAAAJ&hl=zh-CN)<sup>1,📧</sup>, [Bo Du](https://scholar.google.com/citations?user=Shy1gnMAAAAJ&hl=zh-CN)<sup>1</sup>, [Liangpei Zhang](https://scholar.google.com/citations?user=vzj2hcYAAAAJ&hl=zh-CN)<sup>1</sup>, [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=zh-CN)<sup>3</sup>

<sup>1</sup> Wuhan University\
<sup>2</sup> Horizon Robotics\
<sup>3</sup> Nanyang Technological University

(✢) Equal contribution.
(📧) corresponding author.

**Brief intro**: The survey for all-in-one IR has been released.

<!-- [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2308.09388v1.pdf)   ![visitors](https://visitor-badge.laobi.icu/badge?page_id=lixinustc/Awesome-diffusion-model-for-image-processing) -->

## 🔖 News!!!
- [x] 2024-04-25: **Updated new related works before 25/04/2024 in this GitHub.**
- [ ] 2024-06-25: **Updated new related works before 25/06/2024 in this GitHub.**

**📌 About new works.** If you want to incorporate your studies (e.g., the link of paper or project) on diffusion model-based image processing in this repository. Welcome to raise an issue or email us. We will incorporate it into this repository and our survey report ASAP. 


## 🌟  Features
- [x] **Survey for diffusion model-based Image Restoration** ([Arxiv version](https://arxiv.org/pdf/2308.09388v1.pdf) is released) 
- [x] **Summary for diffusion model-based Image/Video Compression**
- [x] **Summary for diffusion model-based Quality Assessment**



## Diffusion model-based Image Restoration/Enhancement
### Table of contents
<!-- - [Survey paper](#survey-paper)
- [Table of contents](#table-of-contents) -->
- [All-in-one for Natural Image Restoration](#image-super-resolution)
- [All-in-one for Medical Image Restoration](#image-restoration)
- [All-in-one for Remote Sensing Image Restoration](#image-inpainting)
- [All-in-one for Raw Image Restoration](#image-shadow-removal)
- [All-in-one for Under-water Image Restoration](#image-denoising)
- [All-in-one for Thermal Infrared Image Restoration](#image-dehazing)
- [All-in-one with New Techniques](#image-deblurring)
- [Benchmark Datasets](#benchmark-datasets)

<!-- - [Diffusion model for Image/video compression](#compression) -->
  <!-- - [Recommended Datasets](#recommended-datasets)
  - [All Datasets](#all-datasets) -->


### Adverse Weather Removal
| Model | Title | First Author | Type | Venue | Project |
| :-- | :---: | :--: | :--: |:--:|:--:|
|All-in-one| [All in One Bad Weather Removal using Architectural Search](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_All_in_One_Bad_Weather_Removal_Using_Architectural_Search_CVPR_2020_paper.pdf) | Ruoteng Li | Image |CVPR 2020 | |
|TKL|[Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model](https://arxiv.org/abs/2104.14951) | Wei-Ting Chen | Image | CVPR 2022 | [![Stars](https://img.shields.io/github/stars/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal.svg?style=social&label=Star)](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal) |
|TransWeather| [TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions](https://openaccess.thecvf.com/content/CVPR2022/papers/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.pdf) | Jeya Maria Jose Valanarasu | Image |CVPR 2022| [![Stars](https://img.shields.io/github/stars/jeya-maria-jose/TransWeather.svg?style=social&label=Star)](https://github.com/jeya-maria-jose/TransWeather)|
|UVRNet| [Unified Multi-Weather Visibility Restoration](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9966834) | Ashutosh Kulkarni | Image |TMM 2022|[![Stars](https://img.shields.io/github/stars/AshutoshKulkarni4998/UVRNet.svg?style=social&label=Star)](https://github.com/AshutoshKulkarni4998/UVRNet)|
|AIRFormer| [Frequency-Oriented Efficient Transformer for All-in-One Weather-Degraded Image Restoration](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10196308) | Tao Gao | Image |TCSVT 2023||
|WeatherDiffusion| [Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models](https://doi.org/10.1109/TPAMI.2023.3238179) | Ozan Özdenizci | Image |TPAMI 2023 |[![Stars](https://img.shields.io/github/stars/IGITUGraz/WeatherDiffusion.svg?style=social&label=Star)](https://github.com/IGITUGraz/WeatherDiffusion) |
|ADMS| [All-in-one Image Restoration for Unknown Degradations Using Adaptive Discriminative Filters for Specific Degradations](https://openaccess.thecvf.com/content/CVPR2023/papers/Park_All-in-One_Image_Restoration_for_Unknown_Degradations_Using_Adaptive_Discriminative_Filters_CVPR_2023_paper.pdf) | Dongwon Park | Image |CVPR 2023 | |
|SmartAssign| [SmartAssign: Learning A Smart Knowledge Assignment Strategy for Deraining and Desnowing](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_SmartAssign_Learning_a_Smart_Knowledge_Assignment_Strategy_for_Deraining_and_CVPR_2023_paper.pdf) | Yinglong Wang | Image |CVPR 2023 | |
|WGWS-Net| [Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_CVPR_2023_paper.pdf) | Yurui Zhu | Image |CVPR 2023 | [![Stars](https://img.shields.io/github/stars/zhuyr97/WGWS-Net.svg?style=social&label=Star)](https://github.com/zhuyr97/WGWS-Net)|
|WeatherStream| [WeatherStream: Light Transport Automation of Single Image Deweathering](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_WeatherStream_Light_Transport_Automation_of_Single_Image_Deweathering_CVPR_2023_paper.pdf) | Howard Zhang | Image |CVPR 2023 | [![Stars](https://img.shields.io/github/stars/UCLA-VMG/WeatherStream.svg?style=social&label=Star)](https://github.com/UCLA-VMG/WeatherStream)|
|RAHC| [Learning to Restore Arbitrary Hybrid Adverse Weather Conditions in One Go](https://arxiv.org/abs/2305.09996) | Ye-Cong Wan | Image |arXiv 2023 | [![Stars](https://img.shields.io/github/stars/Jeasco/RAHC.svg?style=social&label=Star)](https://github.com/Jeasco/RAHC)|
|WM-MoE| [WM-MoE: Weather-aware Multi-scale Mixture-of-Experts for Blind Adverse Weather Removal](https://arxiv.org/abs/2303.13739v2) | Yulin Luo | Image |arXiv 2023 | |
|MetaWeather| [MetaWeather: Few-Shot Weather-Degraded Image Restoration](https://arxiv.org/abs/2308.14334) | Youngrae Kim | Image |arXiv 2023 | |
|UtilityIR| [Always Clear Days: Degradation Type and Severity Aware All-In-One Adverse Weather Removal](https://arxiv.org/abs/2310.18293) | Yu-Wei Chen | Image |arXiv 2023 | [![Stars](https://img.shields.io/github/stars/fordevoted/UtilityIR.svg?style=social&label=Star)](https://github.com/fordevoted/UtilityIR)|
|DDCNet| [Decoupling Degradation and Content Processing for Adverse Weather Image Restoration](https://arxiv.org/abs/2312.05006) | Xi Wang | Image |arXiv 2023 ||
|PMDA| [Multi-weather Image Restoration via Domain Translation](https://openaccess.thecvf.com/content/ICCV2023/papers/Patil_Multi-weather_Image_Restoration_via_Domain_Translation_ICCV_2023_paper.pdf) | Prashant W. Patil | Image |ICCV 2023 | [![Stars](https://img.shields.io/github/stars/pwp1208/Domain_Translation_Multi-weather_Restoration.svg?style=social&label=Star)](https://github.com/pwp1208/Domain_Translation_Multi-weather_Restoration)|
|AWRCP| [Adverse Weather Removal with Codebook Priors](https://openaccess.thecvf.com/content/ICCV2023/papers/Ye_Adverse_Weather_Removal_with_Codebook_Priors_ICCV_2023_paper.pdf) | Tian Ye | Image |ICCV 2023 | |
|ViWS-Net| [Video Adverse-Weather-Component Suppression Network via Weather Messenger and Adversarial Backpropagation](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_Video_Adverse-Weather-Component_Suppression_Network_via_Weather_Messenger_and_Adversarial_Backpropagation_ICCV_2023_paper.pdf) | Yijun Yang | Video |ICCV 2023 | [![Stars](https://img.shields.io/github/stars/scott-yjyang/ViWS-Net.svg?style=social&label=Star)](https://github.com/scott-yjyang/ViWS-Net)|
|GridFormer| [GridFormer: Residual Dense Transformer with Grid Structure for Image Restoration in Adverse Weather Conditions](https://arxiv.org/abs/2305.17863) | Tao Wang | Image |IJCV 2024 | |
|CLIP-SRD| [Exploring the Application of Large-scale Pre-trained Models on Adverse Weather Removal](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10453462) | Zhentao Tan | Image |TIP 2024 | |
|DeformDeweatherNet| [Residual Deformable Convolution for better image de-weathering](https://www.sciencedirect.com/science/article/pii/S0031320323007902) | Huikai Liu | Image |PR 2024 | [![Stars](https://img.shields.io/github/stars/IntelligentDrivingCoding/DeformDeweatherNet.svg?style=social&label=Star)](https://github.com/IntelligentDrivingCoding/DeformDeweatherNet)|
|MoFME| [Efficient Deweather Mixture-of-Experts with Uncertainty-aware Feature-wise Linear Modulation](https://arxiv.org/abs/2312.16610) | Rongyu Zhang | Image |AAAI 2024 | [![Stars](https://img.shields.io/github/stars/RoyZry98/MoFME-Pytorch.svg?style=social&label=Star)](https://github.com/RoyZry98/MoFME-Pytorch)|
|imperfect-deweathering| [Learning Real-World Image De-Weathering with Imperfect Supervision](https://arxiv.org/abs/2310.14958) | Xiaohui Liu | Image |AAAI 2024 | [![Stars](https://img.shields.io/github/stars/1180300419/imperfect-deweathering.svg?style=social&label=Star)](https://github.com/1180300419/imperfect-deweathering)|
|DiffTTA| [Genuine Knowledge from Practice: Diffusion Test-Time Adaptation for Video Adverse Weather Removal](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Genuine_Knowledge_from_Practice_Diffusion_Test-Time_Adaptation_for_Video_Adverse_CVPR_2024_paper.pdf) | Yijun Yang | Video |CVPR 2024 |[![Stars](https://img.shields.io/github/stars/scott-yjyang/DiffTTA.svg?style=social&label=Star)](https://github.com/scott-yjyang/DiffTTA)|
|LDR| [Language-driven All-in-one Adverse Weather Removal](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Language-driven_All-in-one_Adverse_Weather_Removal_CVPR_2024_paper.pdf) | Hao Yang | Image |CVPR 2024 |[![Stars](https://img.shields.io/github/stars/noxsine/LDR.svg?style=social&label=Star)](https://github.com/noxsine/LDR)|



### Applycation in Adverse Weather Conditions
| Model | Title | First Author | Type | Venue | Project |
| :-- | :---: | :--: | :--: |:--:|:--:|
|WAS| [Semantic Segmentation under Adverse Conditions: A Weather and Nighttime-aware Synthetic Data-based Approach](https://arxiv.org/abs/2210.05626) | Abdulrahman Kerim | Image |BMVC 2022 |[![Stars](https://img.shields.io/github/stars/lsmcolab/Semantic-Segmentation-under-Adverse-Conditions.svg?style=social&label=Star)](https://github.com/lsmcolab/Semantic-Segmentation-under-Adverse-Conditions) |
|ROD-Weather| [Robust Object Detection in Challenging Weather Conditions](https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Robust_Object_Detection_in_Challenging_Weather_Conditions_WACV_2024_paper.pdf) | Himanshu Gupta | Image |WACV 2024 |[![Stars](https://img.shields.io/github/stars/hgupta01/Weather_Effect_Generator.svg?style=social&label=Star)](https://github.com/hgupta01/Weather_Effect_Generator) |
|IDD-AW| [IDD-AW: A Benchmark for Safe and Robust Segmentation of Drive Scenes in Unstructured Traffic and Adverse Weather](https://iddaw.github.io/static/pdfs/IDDAW_WACV24_final.pdf) | Furqan Ahmed Shaik | Image |WACV 2024 |[![Stars](https://img.shields.io/github/stars/Furqan7007/IDDAW_kit.svg?style=social&label=Star)](https://github.com/Furqan7007/IDDAW_kit) |
|SD4VS| [Leveraging Synthetic Data to Learn Video Stabilization Under Adverse Conditions](https://openaccess.thecvf.com/content/WACV2024/papers/Kerim_Leveraging_Synthetic_Data_To_Learn_Video_Stabilization_Under_Adverse_Conditions_WACV_2024_paper.pdf) | Abdulrahman Kerim | Image |WACV 2024 |[![Stars](https://img.shields.io/github/stars/SyntheticData4VideoStabilization_WACV_2024.svg?style=social&label=Star)](https://github.com/SyntheticData4VideoStabilization_WACV_2024) |
|CFMW| [CFMW: Cross-modality Fusion Mamba for Multispectral Object Detection under Adverse Weather Conditions](https://arxiv.org/abs/2404.16302) | Haoyuan Li | Image |arXiv 2024 |[![Stars](https://img.shields.io/github/stars/lhy-zjut/CFMW.svg?style=social&label=Star)](https://github.com/lhy-zjut/CFMW) |
|ControlUDA| [ControlUDA: Controllable Diffusion-assisted Unsupervised Domain Adaptation for Cross-Weather Semantic Segmentation](https://arxiv.org/abs/2402.06446) | Fengyi Shen | Image |arXiv 2024 | |
|DKR| [Semantic Segmentation in Multiple Adverse Weather Conditions with Domain Knowledge Retention](https://arxiv.org/abs/2401.07459) | Xin Yang | Image |AAAI 2024 ||
|PASS| [Parsing All Adverse Scenes: Severity-Aware Semantic Segmentation with Mask-Enhanced Cross-Domain Consistency](https://ojs.aaai.org/index.php/AAAI/article/view/29251) | Fuhao Li | Semantic Segmentation |AAAI 2024 ||
|Vehicle-weather| [Perception and sensing for autonomous vehicles under adverse weather conditions: A survey](https://www.sciencedirect.com/science/article/pii/S0924271622003367) | Yuxiao Zhang | Survey |ISPRS JPRS 2023 | |
|Fire-Detection| [An Effective Attention-based CNN Model for Fire Detection in Adverse Weather Conditions](https://www.sciencedirect.com/science/article/pii/S0924271623002940) | Hikmat Yar | Fire-Detection |ISPRS JPRS 2023 | [![Stars](https://img.shields.io/github/stars/Hikmat-Yar/ISPRS-Fire-Detection.svg?style=social&label=Star)](https://github.com/Hikmat-Yar/ISPRS-Fire-Detection)|


### Universal Image Restoration 
| Model | Title | First Author | Type | Venue | Project |
| :-- | :---: | :--: | :--: |:--:|:--:|
|AirNet| [All-in-one image restoration for unknown corruption](https://pengxi.me/wp-content/uploads/2022/03/All-In-One-Image-Restoration-for-Unknown-Corruption.pdf) | Boyun Li | Natural Image |CVPR 2022 |[![Stars](https://img.shields.io/github/stars/XLearning-SCU/2022-CVPR-AirNet.svg?style=social&label=Star)](https://github.com/XLearning-SCU/2022-CVPR-AirNet) |
|IDR| [Ingredient-oriented Multi-Degradation Learning for Image Restoration](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Ingredient-Oriented_Multi-Degradation_Learning_for_Image_Restoration_CVPR_2023_paper.pdf) | Jinghao Zhang | Natural Image |CVPR 2023 |[![Stars](https://img.shields.io/github/stars/JingHao99/IDR-Ingredients-oriented-Degradation-Reformulation.svg?style=social&label=Star)](https://github.com/JingHao99/IDR-Ingredients-oriented-Degradation-Reformulation) |
|PromptIR| [PromptIR: Prompting for All-in-One Blind Image Restoration](https://papers.nips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html) | Vaishnav Potlapalli | Natural Image |NeurIPS 2023 |[![Stars](https://img.shields.io/github/stars/va1shn9v/PromptIR.svg?style=social&label=Star)](https://github.com/va1shn9v/PromptIR) |
|ProRes| [ProRes: Exploring Degradation-aware Visual Prompt for Universal Image Restoration](https://arxiv.org/abs/2306.13653) | Jiaqi Ma | Natural Image |arXiv 2023 |[![Stars](https://img.shields.io/github/stars/leonmakise/prores.svg?style=social&label=Star)](https://github.com/leonmakise/prores) |
|MPerceiver| [Multimodal Prompt Perceiver: Empower Adaptiveness Generalizability and Fidelity for All-in-One Image Restoration](https://openaccess.thecvf.com/content/CVPR2024/papers/Ai_Multimodal_Prompt_Perceiver_Empower_Adaptiveness_Generalizability_and_Fidelity_for_All-in-One_CVPR_2024_paper.pdf) | Yuang Ai | Natural Image |CVPR 2024 | |
|AMIR| [All-In-One Medical Image Restoration via Task-Adaptive Routing](https://arxiv.org/html/2405.19769v1) | Zhiwen Yang | Medical Image |MICCAI 2024 |[![Stars](https://img.shields.io/github/stars/Yaziwel/All-In-One-Medical-Image-Restoration-via-Task-Adaptive-Routing.svg?style=social&label=Star)](https://github.com/Yaziwel/All-In-One-Medical-Image-Restoration-via-Task-Adaptive-Routing) |



### Medical Restoration (MRI,CT)
|Model| Paper | First Author | Training Way | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: | :--: |
|MCG| [Improving diffusion models for inverse problems using manifold constraints](https://arxiv.org/abs/2206.00941) | Hyungjin Chung | Zero-shot | NeurIPS 2022 | CT Reconstruction | [![Stars](https://img.shields.io/github/stars/HJ-harry/MCG_diffusion.svg?style=social&label=Star)](https://github.com/HJ-harry/MCG_diffusion) |
|--| [Solving inverse problems in medical imaging with score-based generative models](https://arxiv.org/abs/2111.08005) | Yang Song | Zero-shot | ICLR 2022 | CT Reconstruction | |
|AdaDiff|[Adaptive diffusion priors for accelerated mri reconstruction](https://arxiv.org/abs/2207.05876) | Alper Güngör | Supervised | Preprint'23 | MRI Reconstruction | [![Stars](https://img.shields.io/github/stars/icon-lab/AdaDiff.svg?style=social&label=Star)](https://github.com/icon-lab/AdaDiff) |


### Other tasks
|Model|Paper | First Author | Training Way | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: | :--: |
|RainDiffusion|[RainDiffusion: When Unsupervised Learning Meets Diffusion Models for Real-world Image Deraining](https://arxiv.org/abs/2301.09430) | Mingqiang Wei | Supervised | Preprint'23 | Image Deraining| |
|DiffGAR|[DiffGAR: Model-agnostic restoration from generative artifacts using image-to-image diffusion models](https://arxiv.org/abs/2210.08573) | Yueqin Yin | Supervised | Preprint'22 | Generative Artifacts Removal | |
|HSR-Diff|[HSR-Diff: Hyperspectral Image Super-Resolution via Conditional Diffusion Models](https://arxiv.org/abs/2306.12085) | Chanyue Wu | Supervised | Preprint'23 | hyperspectral images super-resolution | |

### Benchmark Datasets
|Dataset|Task|Usage|Year|
|:----:|:----:|:----:|:----:|
|[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K)|Image Super-resolution|Training,Testing|2017|
|[Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k)|Image Super-resolution|Training|2017|


## Diffusion model-based Image/Video Compression
<a id="compression"></a>
|Model| Paper | First Author | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: |
|DiffC| [Lossy Compression with Gaussian Diffusion](https://arxiv.org/abs/2206.08889) | Lucas Theisl | Preprint'22 | Lossy Image Compression | |
|CDC| [Lossy Image Compression with Conditional Diffusion Models](https://arxiv.org/abs/2209.06950) | Ruihan Yang | NeurIPS 2023 | Lossy Image Compression | [![Stars](https://img.shields.io/github/stars/buggyyang/cdc_compression.svg?style=social&label=Star)](https://github.com/buggyyang/cdc_compression)|


# Diffusion model-based Image quality assessment
|Model| Paper | First Author | Venue | Topic | Project |
| :--- | :---: | :---: | :--: | :--: |:--: |
|DifFIQA| [DifFIQA: Face Image Quality Assessment Using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2305.05768) | Žiga Babnik | Preprint'23 | Image quality assessment | |
|PFD-IQA| [Feature Denoising Diffusion Model for Blind Image Quality Assessment](https://arxiv.org/abs/2401.11949) | Xudong Li | Preprint'24 | Image quality assessment | |

## Cite US

If this work is helpful to you, we expect you can cite this work and star this repo. Thanks.

```
@article{li2023diffusion,
  title={Diffusion Models for Image Restoration and Enhancement--A Comprehensive Survey},
  author={Li, Xin and Ren, Yulin and Jin, Xin and Lan, Cuiling and Wang, Xingrui and Zeng, Wenjun and Wang, Xinchao and Chen, Zhibo},
  journal={arXiv preprint arXiv:2308.09388},
  year={2023}
}
```

 <p align="center">
  <a href="https://star-history.com/#lixinustc/Awesome-diffusion-model-for-image-processing&Date">
    <img src="https://api.star-history.com/svg?repos=lixinustc/Awesome-diffusion-model-for-image-processing&type=Date" alt="Star History Chart">
  </a>
</p>
