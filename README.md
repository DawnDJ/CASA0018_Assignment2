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
Training Pipeline: A balanced dataset of four classes was compiled from iNaturalist user‑uploaded images. For each of the three target species (Adder, Grass Snake, Slow Worm), 40 images were selected where the animal occupied a significant portion of the frame. For the ‘Unknown’ class, 40 images were created by cropping out purely environmental regions (e.g., grass, leaves, soil, bark) from the same set of species photographs, ensuring that no reptile body parts remained in the crop. This strategy trains the model to recognise “no target species present” without introducing irrelevant external backgrounds. All 160 images were uploaded to Edge Impulse, labelled accordingly, and on‑the‑fly data augmentation (rotation, flip, brightness, zoom) was enabled. The dataset was split into training (80 %) and testing (20 %) subsets.

Inference Pipeline: The trained model (after optimisation and conversion to TensorFlow Lite) is deployed to an Android phone via the Edge Impulse ‘Deploy’ App. The user launches the application and points the phone’s camera at a reptile or an ambiguous outdoor scene. The app captures a frame, pre‑processes it, and runs the model. The inference result (e.g., “Adder – Venomous” or “Unknown – No reptile clearly visible”) is displayed on the screen in real‑time.

System Output: The final output is a real‑time classification label on the device’s screen. By including a dedicated ‘Unknown’ class derived from the same environmental context, the system is more robust to partially visible or distant animals and avoids forced misclassification.


<img width="1354" height="792" alt="image" src="https://github.com/user-attachments/assets/ea9a54cf-9788-40a9-a64b-224dbf57a329" />
<p align="left"><em>Flow Chart</em></p>
 <br>
 


## Data
Gesture data are collected by the onboard accelerometer connected to the Arduino Nano (accX, accY, accZ, gyrX, gyrY, gyrZ, magX, magY, magZ). The experiment gathered 966 samples from 10 participants, totaling 34 minutes and 24 seconds, of which 81% and 19% were used for model training and test, respectively. This sample distribution was chosen to maximize the quantity of data and improve model accuracy. Fitness data were categorized into five labels: "Dumbbell Row," "Hammer Curls," "Upright Row," "Deep Squat," and "other". The first four fitness movement data were derived from ten participants of varying genders, heights, and fitness levels, using both left and right hands. They performed the movements by mimicking a standard reference video ("Top dumbbell exercises for your shoulders, back and arms | Technogym United Kingdom," n.d.). Data under the "other" label included potential dumbbell movements during fitness activities such as shaking, lying flat, rolling, and Interference movements. Additionally, the orientation of the dumbbell was considered during data collection, with arrows affixed to the dumbbell to indicate the direction of grip.

Table1: Data Infomation
![image](https://github.com/zczqxc5/casa0018/assets/146037962/29d13847-f937-4689-aca9-b1bdf5c6c787)
 <br>

![image](https://github.com/zczqxc5/casa0018/assets/146037962/4c88ed8a-7670-44b8-8c1f-3c561d5391bb)

<p align="center"><em>Dumbbell exercises I chose</em></p>
 <br>

![image](https://github.com/zczqxc5/casa0018/assets/146037962/cf50bb06-277a-4673-a17a-dc58422b7cb5)

<p align="center"><em>Dumbbell</em></p>
 <br>

Data processing involves both pre-processing and post-processing stages. In the pre-processing phase, 30 seconds of continuous motion data are collected, as continuous motion captures transitions between movements and natural variations in the movements, which is closer to real-world usage scenarios ("Continuous motion recognition | Edge Impulse Documentation," 2024). Subsequently, by observing data characteristics, windows are manually segmented into equal 2-second intervals, eliminating intervals between movements and data with indistinct features.

![image](https://github.com/zczqxc5/casa0018/assets/146037962/34760eaf-3497-4584-a7dd-e8d7b877dde0)
![image](https://github.com/zczqxc5/casa0018/assets/146037962/76d69fda-6a06-458b-ab43-f8b391df74f4)
![image](https://github.com/zczqxc5/casa0018/assets/146037962/bdfedefd-17e9-4f71-8726-e12512b6f849)
<p align="center"><em>Data of each action completed by cutting</em></p>
 <br>

After the initial training of the model, erroneous data are marked with red dots. Among these, disruptive data—flaws present due to manual trimming—are identified and removed. For example, as shown in the incorrect data graph below, the selection of the cropping window is clearly incorrect, featuring nearly a second of stagnant gesture. After removing these data, the model is retrained.


![image](https://github.com/zczqxc5/casa0018/assets/146037962/b974a5cc-6eaf-46b1-a9a9-8186ea50d044)

![image](https://github.com/zczqxc5/casa0018/assets/146037962/34def87f-6455-4191-baf7-fbedeeb6fc35)
<p align="center"><em>Flaw Data</em></p>
 <br>


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
