#  Edge Reptile Classifier Project
Author name: Zeyu Zhao  

Github repo:https://github.com/DawnDJ/CASA0018_Assignment2

Edge Impulse projects:https://studio.edgeimpulse.com/public/980344/live

Video presentation:(To     be     pasted      here)

## Introduction
In the suburban and rural areas around London, three reptile species — Adder (Vipera berus), Grass Snake (Natrix helvetica), and Slow Worm (Anguis fragilis) — share overlapping habitats. Adders are the only venomous snake native to the UK, while Grass Snakes are harmless and Slow Worms are legless lizards that look like small snakes (Woodland Trust, 2018; Froglife, 2025). Visually, these species are often confused by hikers, gardeners and outdoor workers, presenting a potential safety risk, especially when an Adder is misidentified as a non-venomous snake.

Traditional identification relies on expert knowledge, which is not accessible to the general public. Recent advances in TinyML and transfer learning have made it possible to deploy lightweight deep learning models directly on mobile devices, enabling real‑time species recognition in the field (David et al., 2021). This project applies transfer learning with pre‑trained lightweight architectures: MobileNetV1, MobileNetV2, EfficientNet to distinguish among the three species using a small dataset of publicly available images. The model is then deployed to an Android phone via Edge Impulse, demonstrating how edge‑based AI can help improve outdoor safety and ecological awareness.

<img width="690" height="388" alt="image" src="https://github.com/user-attachments/assets/7577c6c6-7510-4197-a155-57be1fb1a75d" />
<p align="left"><em>Adder (Viper berus)</em></p>
 <br>

<img width="690" height="388" alt="image" src="https://github.com/user-attachments/assets/a988eeef-d706-492f-95e8-95a384122af6" />
<p align="left"><em>Grass snake (Natrix helvetica)</em></p>
 <br>

<img width="690" height="388" alt="image" src="https://github.com/user-attachments/assets/951fe44d-4508-4ab4-85c1-064afa598573" />
<p align="left"><em>Slow-worm (Anguis fragilis)</em></p>
 <br>
 
## Application Overview
This project is an end-to-end TinyML application designed to classify three visually similar reptile species that co-inhabit the suburban areas of London. The system operates in two distinct phases—training and inference—as illustrated in the Application Diagram

<img width="1354" height="800" alt="image" src="https://github.com/user-attachments/assets/4af13c84-ab3b-4b7f-815c-9863552c0cf1" />
<p align="center"><em>Flow Chart</em></p>
 <br>
 
## Data
All image data were sourced from iNaturalist, a citizen science platform with geotagged wildlife photographs. The collection and construction of the dataset followed a three‑step process.

Step 1 – Collecting species images
For each of the three target species – Adder (Vipera berus), Grass Snake (Natrix helvetica), and Slow Worm (Anguis fragilis) – 40 high‑quality images were selected. Selection criteria included: the animal occupies a substantial portion of the frame, the image is in focus, lighting is natural (not studio), and background is typical of UK suburban/rural habitats (grass, leaves, soil, etc.). This resulted in 120 species images in total.

Step 2 – Constructing the ‘Unknown’ class from species images
The 40 images for the ‘Unknown’ class were cropped directly from the 120 species photographs collected in Step 1. Using Windows screenshot tools (Win + Shift + S), rectangular regions containing no reptile parts were carefully extracted. This process yielded exactly 40 clean environmental crops.

Step 3 – Uploading
The final dataset therefore consists of four balanced classes, each with 40 images:

Table1: Dataset Conponents 
<img width="1180" height="504" alt="image" src="https://github.com/user-attachments/assets/4d7833ba-87c2-4d6c-ad75-946d5a30e29e" />

All 160 images were manually reviewed to remove duplicates, then uploaded to Edge Impulse Studio. Each image was assigned one label: adder, grass_snake, slow_worm, or unknown. The dataset was automatically split into training (80 %, 128 images) and testing (20 %, 32 images) subsets.

Pre‑processing: image resizing and colour space
Once uploaded, Edge Impulse automatically pre‑processes all images through the Image processing block. For this project, two input resolutions were used across experiments: 96×96 pixels (for baseline lightweight models) and 160×160 pixels (to evaluate the impact of higher resolution). The colour space was kept as RGB (3 channels), because subtle colour differences between species (e.g., the brownish‑grey Adder vs. the greenish Grass Snake) may aid discrimination. The resizing mode was set to Fit shortest axis, which preserves the original aspect ratio of each image. This avoids distorting the body proportions of the reptiles.

<img width="638" height="510" alt="image" src="https://github.com/user-attachments/assets/c4118daf-cc02-4977-a126-dd4e9cd499e3" />
<p align="center"><em>Image Data Configuration</em></p>

<img width="1348" height="222" alt="image" src="https://github.com/user-attachments/assets/ef83b53c-4ddc-453f-a97e-d634bb796f90" />
<p align="center"><em>RGB Configuration</em></p>

To improve model generalisation and mitigate overfitting given the relatively small dataset , on‑the‑fly data augmentation was enabled during training. Edge Impulse applies random transformations including rotation, horizontal flip, brightness adjustment and zoom to each training image in every epoch to improve generalisation.

<img width="1352" height="1484" alt="image" src="https://github.com/user-attachments/assets/cb718d34-68c1-4e05-9977-66c8bcccfe4d" />
<p align="center"><em>Transfer Learning Configuration</em></p>

After configuring the Image processing block, Edge Impulse generates a feature vector for each image and then projects it into a 2D space using t‑SNE. The resulting Feature Explorer plot provides a visual check of whether the dataset is linearly separable before any model training.

<img width="1300" height="616" alt="image" src="https://github.com/user-attachments/assets/b5a13167-474d-457d-8413-6534b10726ed" />
<p align="center"><em>An Example of Feature Explorer</em></p>

The resulting plot shows that the vast majority of samples from all four classes (Adder, Grass Snake, Slow Worm, Unknown) are sparsely scattered within a small region, without forming any clear or dense clusters. A few individual points lie slightly outside this region, but there is no meaningful separation between the classes – points of different labels are thoroughly intermingled. In particular, the Unknown class (environmental crops) does not stand apart from the animal classes; all classes overlap substantially.

This low‑separability result is expected for a fine‑grained classification task with strong visual similarities. The three reptile species share similar body shapes, colour palettes and natural backgrounds; the Unknown crops were taken from the exact same images, thus sharing identical texture and lighting statistics. Under a simple linear projection (t‑SNE of raw features), these subtle differences are not captured, which demonstrates why non-linear feature learning, namely deep learning is necessary. 

## Model
All models were built using Edge Impulse’s Transfer Learning (Image) block. The input images were resized using Fit shortest axis to preserve the original aspect ratio, which is essential for distinguishing the body shape differences between the stocky Adder and the slender Grass Snake. Three pre‑trained architectures were selected for comparison: MobileNetV1, MobileNetV2 and EfficientNet‑B0. All were initialised with ImageNet weights and fine‑tuned on our dataset.

The following fixed hyperparameters were used across all training runs unless stated otherwise:

Learning rate: 0.0005 for all models

Batch size: 32 for MobileNetV1/V2, 16 for EfficientNet (All Edge Impulse default).

For MobileNetV1 we also experimented with two width multipliers (alpha: 0.25 and 0.1) to explore the trade‑off between model size and accuracy. For MobileNetV2 the default alpha (0.35) was used. EfficientNet‑B0 has no alpha parameter.


Table2: Models Configuration
<img width="912" height="498" alt="image" src="https://github.com/user-attachments/assets/08223fea-551f-4101-b318-ef6aa24c0613" />


## Experiments
To identify the most suitable model for real‑time mobile recognition of three reptile species (Adder, Grass Snake, Slow Worm) plus an ‘Unknown’ background class, we conducted six systematic experiments. Table 1 summarises the configuration and performance of each experiment. Validation accuracy is the final epoch accuracy on the held‑out validation set (20% of training data). Test accuracy is from the separate test set (20% of total data, never seen during training). On‑device metrics (RAM, Flash, inference time) were measured using the EON compiler targeting a mobile phone.

Experiment 1 (Baseline).
We started with the lightest possible model: MobileNetV1, 96×96, alpha=0.25, trained for 20 epochs. The validation accuracy reached 57.7%, but the test accuracy was only 12.5% – a clear sign of severe overfitting. The loss curves showed that validation loss did not decrease after the first few epochs. This suggested that 20 epochs were insufficient for the model to generalise.

<img width="1036" height="808" alt="accuracy loss_curve_1" src="https://github.com/user-attachments/assets/6c768ff3-2207-486b-bbdc-acb435c8c1dd" />
<p align="center"><em>Accuracy&Loss Curves results on Training Set of Model 1</em></p>

<img width="1350" height="1512" alt="train_results_1" src="https://github.com/user-attachments/assets/c3d74388-89f8-4a5b-8183-ad920391df48" />
<p align="center"><em>Other Results of Metrics on Training Set of Model 1</em></p>

<img width="1356" height="1296" alt="Test_results_1" src="https://github.com/user-attachments/assets/d3ee104a-c43c-4712-8c4f-d5c5c8d1c737" />
<p align="center"><em>Results on Test Set of Model 1</em></p>

Experiment 2 (Increase epochs).
To address under‑training, we increased epochs to 50 while keeping all other parameters identical. Validation accuracy remained 57.7% (no improvement), but test accuracy rose to 25.0% – still far from usable. The gap between train and test remained large, indicating that MobileNetV1 96×96 simply lacked enough capacity to capture the fine‑grained differences between the reptiles.

<img width="1050" height="816" alt="accuracy loss_curve_2" src="https://github.com/user-attachments/assets/5a8fa0e3-1d70-4fe4-b901-29ab902724f6" />
<p align="center"><em>Accuracy&Loss Curves results on Training Set of Model 2</em></p>

<img width="1350" height="1512" alt="train_results_1" src="https://github.com/user-attachments/assets/c3d74388-89f8-4a5b-8183-ad920391df48" />
<p align="center"><em>Other Results of Metrics on Training Set of Model 2</em></p>

<img width="1342" height="1506" alt="train_results_2" src="https://github.com/user-attachments/assets/6cc0cf1c-51e9-4714-b8d6-ccf7e1cbd390" />
<p align="center"><em>Results on Test Set of Model 2</em></p>

Experiment 3 (Reduce model size – alpha=0.1).
We then tested whether a smaller model could even work. Using MobileNetV1 with alpha=0.1 (50 epochs) drastically reduced RAM to 58.5 KB, but test accuracy collapsed to 3.1%. This confirmed that shrinking the model further only worsened performance. Therefore, we abandoned the alpha=0.1 configuration.

<img width="1044" height="810" alt="accuracy loss_curve_3" src="https://github.com/user-attachments/assets/0401e6c8-a37f-4965-8b8e-24eae3356e4f" />
<p align="center"><em>Accuracy&Loss Curves results on Training Set of Model 3</em></p>

<img width="1340" height="1494" alt="train_results_3" src="https://github.com/user-attachments/assets/6dc5231f-7c9f-4252-96f1-748aeb6a1624" />
<p align="center"><em>Other Results of Metrics on Training Set of Model 3</em></p>

<img width="1336" height="1418" alt="Test_results_3" src="https://github.com/user-attachments/assets/b855dfbd-6b9d-4fc5-b0fa-b8ab1b6076dc" />
<p align="center"><em>Results on Test Set of Model 3</em></p>


Experiment 4 (Switch to MobileNetV2).
Given that MobileNetV1 was too weak, we moved to a more powerful architecture: MobileNetV2 (96×96, alpha=0.35, 50 epochs). Validation accuracy jumped to 69.2%, test accuracy to 43.8% – a substantial improvement. However, test accuracy was still below 50%, and the confusion matrix showed that Grass Snake was often misclassified as Adder (42.9% of Grass Snake test images). This suggested that the 96×96 resolution might be discarding fine details such as the Adder’s zigzag stripe or the Grass Snake’s yellow collar.

<img width="1044" height="818" alt="accuracy loss_curve_4" src="https://github.com/user-attachments/assets/5e5b9456-5e06-4e16-993a-14da108f48f5" />
<p align="center"><em>Accuracy&Loss Curves results on Training Set of Model 4</em></p>

<img width="1336" height="1504" alt="train_results_4" src="https://github.com/user-attachments/assets/1793c366-c5b3-4e32-9aa2-8a918e6c3089" />
<p align="center"><em>Other Results of Metrics on Training Set of Model 4</em></p>

<img width="1356" height="1418" alt="Test_results_4" src="https://github.com/user-attachments/assets/9c9766cb-efc5-467e-9acb-7a58309e1d7e" />
<p align="center"><em>Results on Test Set of Model 4</em></p>


Experiment 5 (Increase resolution).
To preserve more detail, we kept MobileNetV2 but increased input resolution to 160×160 (still 50 epochs). Test accuracy soared to 75.0% – a dramatic gain. The confusion matrix improved: Adder 75% correct, Slow Worm 100%, Unknown 87.5%. The only remaining weakness was Grass Snake (only 37.5% correct, often confused with Adder). Although inference time increased to 3.5 seconds and RAM to 721.5 KB, the accuracy gain justified the trade‑off for a point‑and‑identify safety tool.

<img width="1032" height="810" alt="accuracy loss_curve_5" src="https://github.com/user-attachments/assets/54afab59-3a35-4036-9024-467a6ea953d5" />
<p align="center"><em>Accuracy&Loss Curves results on Training Set of Model 5</em></p>


<img width="1362" height="1506" alt="train_results_5" src="https://github.com/user-attachments/assets/ad9e7356-9865-4582-a663-1f42a384a3fb" />
<p align="center"><em>Other Results of Metrics on Training Set of Model 5</em></p>

<img width="1342" height="1298" alt="Test_results_5" src="https://github.com/user-attachments/assets/39206379-1233-4ea0-8132-77ea65cfca9e" />
<p align="center"><em>Results on Test Set of Model 5</em></p>


Experiment 6 (Try EfficientNet).
Finally, we tested a state‑of‑the‑art architecture: EfficientNet‑B0 (96×96, 50 epochs). Validation accuracy was 76.9%, but test accuracy was only 56.3% – lower than Exp5. Worse, its inference time was 60 seconds and RAM usage 1.3 MB, making it completely impractical for a mobile application. Hence, EfficientNet was discarded.

<img width="1036" height="808" alt="accuracy loss_curve_1" src="https://github.com/user-attachments/assets/6c768ff3-2207-486b-bbdc-acb435c8c1dd" />
<p align="center"><em>Accuracy&Loss Curves results on Training Set of Model 6</em></p>

<img width="1326" height="1492" alt="train_results_6" src="https://github.com/user-attachments/assets/bcb5804d-f9d0-4549-bd8e-80b4073670e2" />
<p align="center"><em>Other Results of Metrics on Training Set of Model 6</em></p>

<img width="1338" height="1290" alt="Test_results_6" src="https://github.com/user-attachments/assets/5270d0d2-411c-4365-8367-550f77886c2f" />
<p align="center"><em>Results on Test Set of Model 6</em></p>


Final model selection.
Based on the iterative experiments, Exp5 (MobileNetV2, 160×160, alpha=0.35, 50 epochs) was chosen as the final deployed model. It achieves the best balance of test accuracy (75.0%), acceptable memory (721.5 KB RAM) and reasonable inference time for a non‑real‑time safety aid.

<img width="1230" height="764" alt="image" src="https://github.com/user-attachments/assets/2e3d8efb-fa5e-4220-abdb-a3b6647f6a64" />
<p align="center"><em>Experimental results. Test accuracy are calculated by Quantized (int8) model . On‑device metrics from the EON compiler on a mobile phone target.</em></p>


## Results and Observations
After iterative experiments, the final model (MobileNetV2, 160×160, 50 epochs) achieved 75.0% test accuracy – a 62.5 percentage point improvement over the baseline (Exp1). The single most effective change was increasing input resolution, which boosted accuracy far more than switching to EfficientNet, a more complex architecture.

The experiments revealed two unexpected findings. First, EfficientNet‑B0, despite its state‑of‑the‑art reputation, was impractical for this task: 56.3% test accuracy, 60‑second inference time, and 1.3 MB RAM. Second, the ‘Unknown’ class (cropped environmental patches) worked surprisingly well, achieving 87.5% test accuracy without confusing Adders – validating our cropping strategy.

However, several limitations remain. The dataset (40 images per class from iNaturalist) lacks diversity in lighting, occlusion, and real‑field conditions. Both training and test data came from the same web source, so the 75% accuracy likely overestimates real‑world performance. I did not collect any first‑hand photographs or test the model on live video in an actual outdoor setting.

If more time were available, I would collect 20‑30 additional images per class under varied weather and lighting conditions, including partially occluded shots. Moreover, I would go to suburban areas or a zoo to test the live mobile app in a park with printed reptile images at different distances – providing a realistic performance check beyond the static test set.


## Bibliography
David, R., Duke, J., Jain, A., et al. (2021). TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems. In: Proceedings of Machine Learning and Systems (MLSys), 3, pp. 800-811. Available at: https://arxiv.org/abs/2010.08678 (Accessed: 4 May 2026).

Edge Impulse Documentation. (n.d.). Transfer learning (Images). Available at: https://edge-impulse.gitbook.io/docs/edge-impulse-studio/learning-blocks/transfer-learning-images (Accessed: 4 May 2026).

Froglife. (2025). UK Reptile ID. Available at: https://www.froglife.org/2025/04/01/uk-reptile-id (Accessed: 4 May 2026).

Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M. and Adam, H. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861 [cs.CV]. Available at: https://arxiv.org/abs/1704.04861 (Accessed: 4 May 2026).

Nordic Semiconductor / Edge Impulse Documentation. (n.d.). Image classification. Available at: https://docs.nordic.edgeimpulse.com/tutorials/end-to-end/image-classification (Accessed: 4 May 2026).

Sandler, M., Howard, A.G., Zhu, M., Zhmoginov, A. and Chen, L.C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. arXiv:1801.04381 [cs.CV]. Available at: https://arxiv.org/abs/1801.04381 (Accessed: 4 May 2026).

Tan, M. and Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In: Proceedings of the 36th International Conference on Machine Learning (ICML 2019), pp. 6105-6114. Available at: https://arxiv.org/abs/1905.11946 (Accessed: 4 May 2026).

Woodland Trust. (2018). Grass snake or adder? How to tell the difference between UK reptiles. Available at: https://woodlandtrust.org.uk/blog/2018/02/grass-snake-or-adder (Accessed: 4 May 2026).


## Declaration of Authorship

I, Zeyu Zhao, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


*Zeyu Zhao*

ASSESSMENT DATE: 4/5/2026
Word count: 1750(within 20% margin)
