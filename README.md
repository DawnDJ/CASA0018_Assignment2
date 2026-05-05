#  Edge Reptile Classifier Project
Author name: Zeyu Zhao  

Github repo:https://github.com/DawnDJ/CASA0018_Assignment2

Edge Impulse projects:https://studio.edgeimpulse.com/public/980344/live

Video presentation:(To     be     pasted      here)

## Introduction
In the suburban and rural areas around London, three reptile species — Adder (Vipera berus), Grass Snake (Natrix helvetica), and Slow Worm (Anguis fragilis) — share overlapping habitats. Adders are the only venomous snake native to the UK, while Grass Snakes are harmless and Slow Worms are legless lizards that look like small snakes (Woodland Trust, 2018; Froglife, 2025). Visually, these species are often confused by hikers, gardeners and outdoor workers, presenting a potential safety risk, especially when an Adder is misidentified as a non-venomous snake.

Traditional identification relies on expert knowledge, which is not accessible to the general public. Recent advances in TinyML and transfer learning have made it possible to deploy lightweight deep learning models directly on mobile devices, enabling real‑time species recognition in the field (David et al., 2021). This project applies transfer learning with pre‑trained lightweight architectures (MobileNetV1, EfficientNet) to distinguish among the three species using a small dataset of publicly available images. The model is then deployed to an Android phone via Edge Impulse, demonstrating how edge‑based AI can help improve outdoor safety and ecological awareness.

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
 

Hardware Setup: All inference and testing are performed on a standard Android smartphone (the author's personal device), utilising its built-in camera as the sole sensor to capture real-time image data of the target species. The training process runs entirely in the cloud via the Edge Impulse Studio on a standard PC.

Software Workflow: The software pipeline is divided into two parts:

Training Pipeline: A balanced dataset of four classes was compiled from iNaturalist user‑uploaded images. For each of the three target species (Adder, Grass Snake, Slow Worm), 40 images were selected where the animal occupied a significant portion of the frame. For the ‘Unknown’ class, 40 images were created by cropping pure environmental regions (e.g., the ground, leaf litter, or nearby background) directly from the 120 species photographs already collected. Every crop was carefully inspected to ensure that no reptile body parts (scales, eyes, body contours, etc.) remained. This strategy guarantees that the ‘Unknown’ class shares the exact same lighting, habitat type and image quality as the target species, forcing the model to learn genuine reptile features rather than relying on generic background cues. All 160 images were uploaded to Edge Impulse, labelled accordingly, and on‑the‑fly data augmentation (rotation, flip, brightness, zoom) was enabled. The dataset was split into training (80 %) and testing (20 %) subsets.

Inference Pipeline: The trained model (after optimisation and conversion to TensorFlow Lite) is deployed to an Android phone via the Edge Impulse ‘Deploy’ App. The user launches the application and points the phone’s camera at a reptile or an ambiguous outdoor scene. The app captures a frame, pre‑processes it, and runs the model. The inference result (e.g., “Adder – Venomous” or “Unknown – No reptile clearly visible”) is displayed on the screen in real‑time.

System Output: The final output is a real‑time classification label on the device’s screen. By including a dedicated ‘Unknown’ class derived from the same environmental context, the system is more robust to partially visible or distant animals and avoids forced misclassification.





## Data
All image data were sourced from iNaturalist, a citizen science platform with geotagged wildlife photographs. The collection and construction of the dataset followed a three‑step process.

Step 1 – Collecting species images
For each of the three target species – Adder (Vipera berus), Grass Snake (Natrix helvetica), and Slow Worm (Anguis fragilis) – 40 high‑quality images were selected. Selection criteria included: the animal occupies a substantial portion of the frame, the image is in focus, lighting is natural (not studio), and background is typical of UK suburban/rural habitats (grass, leaves, soil, etc.). This resulted in 120 species images in total.

Step 2 – Constructing the ‘Unknown’ class from species images
The 40 images for the ‘Unknown’ class were cropped directly from the 120 species photographs collected in Step 1. Using Windows screenshot tools (Win + Shift + S), rectangular regions containing no reptile parts were carefully extracted. Priority was given to foreground areas immediately adjacent to the animal (e.g., the ground the snake was resting on, nearby leaves or twigs), because these patches share the same camera distance, lighting and resolution as the target species. Any crop that accidentally included a scale, eye or body contour was discarded. This process yielded exactly 40 clean environmental crops.

Step 3 – Uploading
The final dataset therefore consists of four balanced classes, each with 40 images:

Table1: Dataset Conponents 
<img width="1180" height="504" alt="image" src="https://github.com/user-attachments/assets/4d7833ba-87c2-4d6c-ad75-946d5a30e29e" />

All 160 images were manually reviewed to remove duplicates, then uploaded to Edge Impulse Studio. Each image was assigned one label: adder, grass_snake, slow_worm, or unknown. The dataset was automatically split into training (80 %, 128 images) and testing (20 %, 32 images) subsets.

Pre‑processing: image resizing and colour space
Once uploaded, Edge Impulse automatically pre‑processes all images through the Image processing block. For this project, two input resolutions were used across experiments: 96×96 pixels (for baseline lightweight models) and 160×160 pixels (to evaluate the impact of higher resolution). The colour space was kept as RGB (3 channels), because subtle colour differences between species (e.g., the brownish‑grey Adder vs. the greenish Grass Snake) may aid discrimination. The resizing mode was set to Fit shortest axis, which preserves the original aspect ratio of each image. This avoids distorting the body proportions of the reptiles (e.g., the stout Adder vs the slender Grass Snake), which is critical for fine‑grained classification.

<img width="638" height="510" alt="image" src="https://github.com/user-attachments/assets/c4118daf-cc02-4977-a126-dd4e9cd499e3" />
<p align="center"><em>Image Data Configuration</em></p>

<img width="1348" height="222" alt="image" src="https://github.com/user-attachments/assets/ef83b53c-4ddc-453f-a97e-d634bb796f90" />
<p align="center"><em>RGB Configuration</em></p>

To improve model generalisation and mitigate overfitting given the relatively small dataset (40 images per class), on‑the‑fly data augmentation was enabled during training. Edge Impulse applies random transformations to each training image in every epoch (random rotation, horizontal flip, brightness adjustment, zoom) to improve generalisation.

All augmentations were applied only to the training set; validation and test sets remained unaltered to provide an unbiased performance estimate. This strategy effectively increased the diversity of the training data without requiring additional manual collection.

<img width="1352" height="1484" alt="image" src="https://github.com/user-attachments/assets/cb718d34-68c1-4e05-9977-66c8bcccfe4d" />
<p align="center"><em>Transfer Learning Configuration</em></p>

After configuring the Image processing block, Edge Impulse generates a feature vector for each image (e.g., 96×96×3 = 27,648 dimensions) and then projects it into a 2D space using t‑SNE. The resulting Feature Explorer plot provides a visual check of whether the dataset is linearly separable before any model training.

<img width="1300" height="616" alt="image" src="https://github.com/user-attachments/assets/b5a13167-474d-457d-8413-6534b10726ed" />
<p align="center"><em>An Example of Feature Explorer</em></p>

The resulting plot shows that the vast majority of samples from all four classes (Adder, Grass Snake, Slow Worm, Unknown) are sparsely scattered within a small region, without forming any clear or dense clusters. A few individual points lie slightly outside this region, but there is no meaningful separation between the classes – points of different labels are thoroughly intermingled. In particular, the Unknown class (environmental crops) does not stand apart from the animal classes; all classes overlap substantially.

This low‑separability result is expected for a fine‑grained classification task with strong visual similarities. The three reptile species share similar body shapes, colour palettes and natural backgrounds; the Unknown crops were taken from the exact same images, thus sharing identical texture and lighting statistics. Under a simple linear projection (t‑SNE of raw features), these subtle differences are not captured.

This demonstrates why non‑linear feature learning (i.e., deep neural networks) is necessary. The later sections will show that after fine‑tuning pre‑trained models (MobileNetV1, V2, EfficientNet), the networks learned a discriminative embedding and achieving improvements on test accuracy. The Feature Explorer result thus serves as a baseline justification for using transfer learning on this challenging task.


## Model
For continuous motion recognition, Edge Impulse offers DSP modules including Spectral Analysis and IMU (Syntiant). Additionally, the learning blocks provided include Classification and Anomaly Detection (K-means). These tools are integral for effectively processing and learning from sensor data, facilitating the development of robust models for recognizing diverse motion patterns.  
![image](https://github.com/zczqxc5/casa0018/assets/146037962/7d265477-62ea-4af0-aefc-f96b850b5913)
<p align="center"><em>DSP modules</em></p>
 <br>
 
![image](https://github.com/zczqxc5/casa0018/assets/146037962/6034dd8f-2e45-419d-8ee8-ba3ef0b48fb8)
<p align="center"><em>learning blocks</em></p>
 <br>

Under conditions where other variables (training cycles-1, learning rate-0.0005, batch size-32, etc.) were held constant, four tests were conducted with different combinations of the four modules. The experimental results showed that the best performance was achieved using the Spectral Analysis processing block and the Classification Learning Block. When these were applied, the model training reached a test accuracy of 97.18%.  
<br>
Table2: Blocks experiment 
![image](https://github.com/zczqxc5/casa0018/assets/146037962/97c5e744-1324-4c6d-ac0e-4830e5467631)
<br>

The parameter settings for the Spectral Analysis processing block are shown in the diagram below. It includes the setup of a low-pass filter, which helps to remove high-frequency noise from the signal. Subsequently, spectral features are extracted using FFT, and the extracted features undergo logarithmic transformation and frame overlapping to enhance their representational capability. This preprocessing step provides input features for the subsequent machine learning or deep learning models. The DSP results demonstrate a clear attenuation of the signal after 8Hz, indicating that the filter has successfully removed high-frequency noise.  
The Classification Learning Block offers automated model selection, which is particularly suitable for resource-constrained edge devices (IoT devices). This feature streamlines the deployment process, allowing for efficient model operation within limited hardware capabilities.  
![image](https://github.com/zczqxc5/casa0018/assets/146037962/e39a2bb9-7a52-4d70-9142-d3102f9c952f)
<p align="center"><em>DSP Result</em></p>
<br>

The final neural network architecture is depicted in the diagram below.
![image](https://github.com/zczqxc5/casa0018/assets/146037962/51eb2970-caa3-42f8-a5ed-28964ed599d6)
![image](https://github.com/zczqxc5/casa0018/assets/146037962/dd034384-25a9-4f97-93fe-03ee3290b778)




## Experiments
The project conducted three different types of experiments:   
1. Attempting to distinguish between left and right hand dumbbell movements.  
2. Adding noise (other) data.  
3. Modifying training parameters to find the most suitable model.  
<br>
Initially, each movement label was further subdivided into left-hand and right-hand actions to try to distinguish between them. However, subsequent tests revealed that the current model's capability to differentiate such subtle differences between the left and right hands was insufficient. Particularly for the dumbbell roll movement, the accuracy was even less than 50%, which is no better than random guessing. Consequently, the project abandoned the distinction between left and right hands and instead added more fitness movements for differentiation.  

![image](https://github.com/zczqxc5/casa0018/assets/146037962/b3004e2a-3985-4ad4-9031-eec6fc028e6d)
<p align="center"><em> left-hand and right-hand Model testing result</em></p>
<br>

The experiments found that without the addition of noise data, the model's accuracy was exceptionally high at 99.4%. However, during actual testing, some non-fitness movements were also misidentified as "fitness movements." This high accuracy rate was misleading due to the lack of noise data, which meant that any captured data was inevitably classified into one of the four fitness movements, even if they were not very similar. After incorporating other data (static, rolling, shaking, non-standard movements, and other fitness activities), the accuracy decreased to 97.18%. However, in practical tests, there were almost no instances of inaccurate recognition. This more realistic setting provided a significant improvement in the model's practical application.  
![image](https://github.com/zczqxc5/casa0018/assets/146037962/90e924da-c33f-4105-b2ae-0f983bb18191)
<p align="center"><em> Model testing result without noise data</em></p>
<br>

After selecting the model modules, the experiments also involved adjusting the neural network settings, including training cycles, learning rate, batch size, input layer features, and the number of neurons in the dense layer, in an effort to find the most efficient and accurate model configuration. A total of 12 experiments were conducted as follows, resulting in the most satisfactory data settings, achieving a test accuracy of 99.65%:  

Table3: Experiment
![image](https://github.com/zczqxc5/casa0018/assets/146037962/978fa0ef-ebd6-411a-bce3-f600405e0ec8)
<br>
Increasing the number of training epochs, although it extends the training time, significantly enhances accuracy. However, prolonged training can lead to overfitting, where the model learns the training data too well to the extent that it cannot generalize to unseen data. Therefore, a dropout layer was added to avoid this issue.  
![image](https://github.com/zczqxc5/casa0018/assets/146037962/b16af577-efda-464f-8b38-1aa8414742bf)
<br>

## Results and Observations
After extensive experimentation and testing, the model's test accuracy improved from the recommended settings provided by Edge Impulse at 97.18% to 99.65% through designing and fine-tuning the model parameters. Although it is not perfect, it is sufficiently accurate for practical applications. 
Table4: Final Result
![image](https://github.com/zczqxc5/casa0018/assets/146037962/78146e9e-9960-4d2d-af11-5c339ecc64e8)
<br>
![image](https://github.com/zczqxc5/casa0018/assets/146037962/17127bf8-ab39-43d8-a938-61f09eca35bf)
<br>
![image](https://github.com/zczqxc5/casa0018/assets/146037962/7d95efa6-c84f-4efd-88bf-7e1bfc188899)
<br>

The experiments revealed that the inclusion of 'other' data is crucial for enhancing the practical application capabilities of the AI model. Moreover, various parameters, including training cycles, learning rate, batch size, input layer features, and the number of neurons in the dense layer, are not necessarily better when they are higher or lower. Each change in data can affect training stability, generalization ability, risk of overfitting, convergence speed, and training time. These factors require experimental evaluation and holistic consideration during model training. Furthermore, the practical applicability of a model cannot be solely assessed by its test accuracy. Even with a very high accuracy, there is a risk that the model may not generalize well to unseen data.  

During the data collection phase, I neglected to differentiate the sources of the training and test data. They were both derived from the same ten individuals and randomly allocated proportionally. If two participants who were not involved in the training data collection could be added, and their data used to test the model, the actual application data of the model would be more realistically reflected. This is because the ultimate users of the model do not originate from the participants in the data collection. Although I invited a friend to test the dumbbell in real-time, and there were no inaccuracies, this test sample size is too small to prove the model's capabilities.  


If I had more time, I would first focus on collecting and expanding more data to enhance model stability, while also adding 200 test data points from different participants. Secondly, I would revisit the model approaches that were abandoned during the experiments: distinguishing whether each action is performed by the left hand or the right hand. Due to time and resource constraints, this approach was set aside in the early stages of the project. However, in reality, it could significantly enhance the applicability of the model, providing users with a more customized and interactive experience.  

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

I, Xin Cheng, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


*XIN CHENG*

ASSESSMENT DATE: 4/20/2024
