# A Comprehensive Survey of Blind Image Quality Assessment: Evolution from Hand-Crafted Features to Content-Adaptive and Vision-Language Architectures

> Summary
> Blind Image Quality Assessment (BIQA) has evolved from traditional, content-agnostic methods based on Natural Scene Statistics (e.g., BRISQUE) to deep learning approaches. A pivotal paradigm shift introduced content-adaptivity, exemplified by HyperNetwork-based models like HyperIQA, which dynamically generate assessment parameters based on image semantics to better handle diverse "in-the-wild" distortions. Subsequent advancements include Transformer-based architectures (e.g., MANIQA), which leverage self-attention for global context modeling, and Vision-Language Model (VLM) methods (e.g., LIQE, CLIP-IQA), which integrate multimodal pre-training for enhanced semantic understanding and zero-shot capability. The field continues to progress by balancing the trade-offs between accuracy, computational efficiency, and generalization across these architectural paradigms.

## 1. Introduction to Blind Image Quality Assessment: Fundamentals and Real-World Challenges

Blind Image Quality Assessment (BIQA) is defined as the automatic prediction of the perceptual quality of a digital image without access to its pristine reference [^2]. Formally, a BIQA model is a mapping ( Q: X \rightarrow \mathbb{R}, x \rightarrow \hat{q} = Q(x) ), where (\hat{q}) is the predicted quality score for the input image (x) [^2]. This capability is essential for modern imaging systems, as reference images are typically unavailable in real-world applications.

### 1.1 Significance in Real-World Applications

BIQA is a critical component in numerous practical systems deployed across consumer, industrial, and cloud-based domains [^2]. In consumer electronics and social media platforms, it enables automatic quality filtering and content curation for the billions of user-generated images processed daily. Within industrial imaging pipelines, such as those for manufacturing quality control and medical diagnostics, BIQA ensures consistent input quality for reliable automated analysis. Cloud-based multimedia services and content delivery networks also rely on BIQA for intelligent compression and adaptive streaming, optimizing vast quantities of visual data while maintaining perceptual standards.

### 1.2 Core Challenge: Assessing "In-the-Wild" Images

The fundamental difficulty in BIQA arises from the need to evaluate "in-the-wild" images—those captured in uncontrolled, real-world environments. These images present a stark contrast to synthetically distorted images created in laboratory settings. The primary challenges are characterized by distortion diversity, strong content dependency, and severe data scarcity [^2].

The table below summarizes these key challenges:

| Challenge                         | Characteristics                                                                                                                               | Impact on BIQA                                                                                                               |
| :-------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **Distortion Diversity**          | Multiple, simultaneous, and unknown degradations (e.g., blur, noise, compression artifacts combined); complex, authentic distortion patterns. | Models trained on isolated, synthetic distortions fail to generalize to real-world scenes [^1].                              |
| **Content Dependency**            | Human perception of quality is intrinsically linked to image semantics; the same distortion affects different content types differently.      | Models must adapt quality criteria based on recognized content to align with human judgment [^109]  [^111].                  |
| **Data Scarcity**                 | Extremely limited availability of large-scale, reliably annotated datasets with subjective Mean Opinion Scores (MOS).                         | Deep learning approaches face overfitting and poor generalization due to insufficient training data [^113].                  |
| **Domain Shift & Generalization** | Significant discrepancy between synthetic training data and authentic test images; constant emergence of novel, unseen distortion types.      | Models exhibit performance degradation when deployed in real scenarios and struggle with robustness to new distortions [^2]. |

### 1.3 The Authentic vs. Synthetic Distortion Divide

A critical understanding for BIQA is the fundamental difference between authentic and synthetic distortions. Synthetic distortions, such as controlled JPEG compression or Gaussian blur, are straightforward and easily manipulated during dataset generation. In contrast, authentic distortions are far more complex, arising organically during image acquisition, compression, and processing in real-world scenarios [^1]. They exhibit distinct features like uneven brightness, color impairment, and diverse noise types, which are often absent in synthetic counterparts. Consequently, BIQA methods that perform well on synthetic databases frequently struggle when applied to authentically distorted images [^1].

### 1.4 Technical Hurdles and Evolutionary Drivers

These challenges create specific technical hurdles that have driven the evolution of BIQA methodologies:

1. **Modeling Complex Degradation:** Capturing the intricate and variable characteristics of authentic distortions requires moving beyond hand-crafted features designed for controlled scenarios.
2. **Incorporating Semantic Context:** To address content dependency, models must integrate mechanisms for understanding image semantics to adapt their quality perception rules [^110].
3. **Overcoming Annotation Poverty:** The scarcity of MOS labels necessitates innovative learning strategies, such as unsupervised or semi-supervised approaches, and the generation of pseudo-labels [^108]  [^114].
4. **Ensuring Efficiency and Deployment:** Practical applications, especially on resource-constrained edge devices, demand models that balance high accuracy with low computational cost and memory footprint [^115].

The ongoing effort to solve these problems has propelled the field from early hand-crafted methods to deep learning models and, most recently, toward content-adaptive and multimodal paradigms. The following sections will detail this evolutionary journey, examining how each stage has sought to address the persistent challenges of assessing image quality in the wild.

## 2. Historical Evolution: From Hand-Crafted Features to Early Deep Learning

The methodological trajectory of Blind Image Quality Assessment (BIQA) reflects a broader shift in computer vision, moving from expert-driven, hand-crafted feature engineering to data-driven, end-to-end deep learning. This evolution was driven by the need to overcome the limitations of traditional models, particularly their struggle with the diverse and complex distortions found in real-world, "in-the-wild" images.

### 2.1 Traditional Hand-Crafted BIQA Methods

Early BIQA methodologies relied on feature extraction techniques derived from expert knowledge and engineering experience. These methods are broadly categorized into distortion-specific and general-purpose approaches[^1].

**Distortion-Specific BIQAs** are designed to quantify quality by considering the specific degradation patterns of particular applications, such as screen content, low-light imaging, High Dynamic Range (HDR), encryption, and omnidirectional stereo[^1]. While effective within their narrow domains, these methods lack generalizability and perform poorly when encountering distortions outside their design scope.

**General-Purpose BIQAs** aim to assess quality across a wide range of distortions without being tailored to a specific application. They primarily fall into two sub-categories:

* **Natural Scene Statistics (NSS)-based methods**: These operate on the assumption that high-fidelity natural images obey specific statistical regularities that are systematically altered by distortions\[\[1]\[5]]. They quantify these alterations in domains like the wavelet transform or directly in the spatial domain using features such as normalized luminance and gradient magnitude.

* **Human Visual System (HVS)-guided methods**: These attempt to incorporate perceptual characteristics of the HVS, leveraging concepts like the free-energy principle or visual sensitivity to attributes like luminance, structure, and texture[^1].

A conceptual model underlying these approaches can be expressed as measuring the deviation between the statistical distribution of a distorted image and that of natural scenes:

$$
Q \propto D(P_{\text{distorted}} \| P_{\text{natural}})
$$

where (Q) is the predicted quality, (P) denotes a feature distribution, and (D) is a divergence measure.

Despite their foundational role, these hand-crafted methods faced significant limitations. They were largely **content-agnostic**, applying the same quality rules regardless of image semantics. More critically, they exhibited a "significant performance degradation on real world authentic distorted images"[^18], which are far more complex and variable than the controlled synthetic distortions these models were often designed for.

### 2.2 The Paradigm Shift to Deep-Learned BIQA

The advent of deep learning marked a fundamental paradigm shift, leveraging neural networks to automatically learn quality-relevant features directly from data in an end-to-end manner\[\[1]\[5]]. This shift also introduced a key distinction between learning paradigms:

* **Supervised Learning-based (Opinion-Aware) BIQAs**: These methods minimize the discrepancy between predicted scores and subjective Mean Opinion Scores (MOS), explicitly learning from human judgments[^1].

* **Unsupervised Learning-based (Opinion-Unaware) BIQAs**: These extract latent embedding features without ground-truth MOS labels, measuring quality differences based on metrics like cosine similarity or domain alignment[^1].

Early implementations of this shift utilized Convolutional Neural Networks (CNNs). Pioneering CNN-based BIQA models established several key characteristics: they enabled joint end-to-end learning of features and regression from raw image data, often used a **patch-based input** strategy (processing small patches like 32×32 pixels) to manage computational cost, and employed relatively shallow architectures\[\[3]\[4]]. This approach eliminated the need for manual feature design, a major bottleneck of traditional methods.

### 2.3 Limitations of Early Deep Learning Approaches

While representing a major advance, early deep learning BIQA methods inherited and exposed new challenges. A core remaining issue was **content insensitivity**; these models still failed to adapt their assessment based on the semantic content of an image, which is crucial as the perceptual impact of a distortion varies dramatically with content[^18]. Furthermore, supervised methods faced the **data dependency** challenge, requiring large, expensively annotated datasets\[\[1]\[2]]. Generalization remained problematic, with models struggling on unseen distortion types or authentic distortions not well-represented in training data[^1]. Finally, the patch-based processing of early CNNs often failed to capture the global context and long-range dependencies important for holistic quality judgment.

This evolutionary phase from hand-crafted features to early deep learning laid the technical groundwork but clearly highlighted the need for more sophisticated architectures. The critical shortcomings—particularly the inability to handle content-dependent quality perception and complex authentic distortions—directly motivated the subsequent development of dynamic, content-adaptive models and more advanced learning paradigms.

## 3. The Content-Adaptivity Paradigm Shift: From Static to Dynamic Quality Assessment

The evolution of Blind Image Quality Assessment (BIQA) has been fundamentally shaped by a critical conceptual transition: the move from static, content-agnostic models to dynamic, content-adaptive architectures. This paradigm shift represents more than just a technical improvement—it embodies a fundamental rethinking of how computational systems should perceive and evaluate visual quality, mirroring the adaptive nature of human visual perception itself[^74].

### 3.1 Theoretical Motivations: Why Static Models Fail

Traditional BIQA models operate on a static principle expressed mathematically as $\phi(x, \theta) = q$, where fixed weight parameters $\theta$ are applied uniformly to all input images $x$ to produce quality scores $q$[^11]. This approach suffers from several fundamental limitations when confronted with the complexity of real-world images.

**Content-Distortion Interdependence**: Image quality perception emerges from complex, high-order interactions between semantic content and distortion characteristics[^124]. Human perception of quality varies with image content, meaning the impact of a distortion can differ based on what the image depicts[^124]. For example, blur might be more damaging in a portrait's facial features than in a textured background, yet static models apply identical quality rules regardless of content.

**Diversity of "In-the-Wild" Images**: Traditional BIQA methods were primarily designed for synthetic distortions with known characteristics, but they exhibit significant performance degradation on real-world authentic distorted images[^18]. Authentic distortions are far more complex, variable, and arise organically during image acquisition, compression, and processing, rendering fixed feature extraction strategies inadequate[^1].

**Spatial and Semantic Variability**: Distortions affect images non-uniformly across spatial positions and semantic regions. Static models that apply global quality metrics fail to capture this spatial dependency, where distortions typically occur in multiple local regions with varying perceptual significance[^124].

**Cognitive Mismatch with Human Perception**: The Human Visual System (HVS) is the final receiver of visual signals in BIQA tasks, and human perception is essentially the joint action of multiple sensory information[^74]. Static models lack the adaptive mechanisms that allow humans to adjust quality judgments based on contextual understanding and content recognition.

### 3.2 The HyperNetwork Revolution: Content-Adaptive Architectures

The introduction of HyperNetwork-based approaches, exemplified by HyperIQA, marked a watershed moment in BIQA evolution. These architectures fundamentally reconceptualized the quality assessment process by introducing dynamic parameter generation based on image content[^11].

**Architectural Innovation and Mathematical Formulation**: HyperIQA separates the IQA procedure into three distinct stages: content understanding, perception rule establishment, and quality prediction. This division mirrors the hypothesized top-down flow of human visual perception, where quality judgment follows content recognition[^11]. The content-adaptive approach is mathematically expressed as $\phi(x, \theta_x) = q$, where $\theta_x = H(S(x), \gamma)$. Here, $H$ represents a hyper network that learns a mapping from the image's semantic features $S(x)$ to the quality perceiving rules $\theta_x$. This allows the model to adaptively learn the rule for perceiving quality according to its recognized content[^11].

**Key Components**:

1. **Semantic Feature Extraction Network**: Utilizes backbone networks (typically ResNet50 pretrained on ImageNet) to extract semantic features $S(x)$ that are fed to the hyper network for weight generation[^11].
2. **Local Distortion Aware Module**: Processes multi-scale content features $S_{ms}(x)$ from different network layers, dividing them into non-overlapping patches to capture local distortions in addition to holistic content[^11].
3. **Hyper Network**: Comprises 1x1 convolution layers and weight-generating branches that dynamically produce parameters (weights and biases) for the target network based on semantic content[^11].
4. **Target Network**: A compact network (typically four fully connected layers) whose weights and biases are dynamically generated by the hyper network, producing the final quality score[^11].

**Adaptive Mechanism**: The core innovation lies in making network parameters $\theta_x$ dependent on the input image itself. For instance, when assessing a clear blue sky image, the model can learn to discount certain texture-based quality indicators that would incorrectly penalize the image for lacking high-frequency details, thus avoiding the misclassification of intentional flat regions as blurry artifacts[^11]. This explicit formulation of content-specific perception rules allows the model to handle diverse images and distortions more consistently with human visual perception[^11].

### 3.3 Comparative Analysis: Content-Adaptive vs. Static Approaches

| **Aspect**                 | **Static Models**                     | **Content-Adaptive Models**                  | **Implications**                       |
| -------------------------- | ------------------------------------- | -------------------------------------------- | -------------------------------------- |
| **Parameter Generation**   | Fixed weights $\theta$ for all images | Dynamic weights $\theta_x = H(S(x), \gamma)$ | Enables content-specific quality rules |
| **Content Understanding**  | Implicit or absent                    | Explicit semantic feature extraction $S(x)$  | Mimics human top-down perception       |
| **Distortion Handling**    | Uniform across image types            | Content-dependent distortion sensitivity     | Better captures perceptual variations  |
| **Generalization**         | Limited to training distortion types  | Adapts to diverse "in-the-wild" content      | Improved real-world applicability      |
| **Computational Overhead** | Lower inference cost                  | Higher due to dynamic weight generation      | Trade-off for perceptual accuracy      |
| **Human Alignment**        | Statistical correlation with MOS      | Mimics cognitive quality judgment process    | More psychologically plausible         |

**Practical Example**: Consider the assessment of medical images versus social media photographs. Static models would apply identical quality metrics to both, potentially over-penalizing compression artifacts in social media images while under-penalizing subtle noise in medical scans. Content-adaptive models, however, can learn that diagnostic utility in medical images requires different quality criteria than aesthetic appeal in social photos.

### 3.4 Theoretical Foundations: Bridging Computational and Perceptual Systems

The content-adaptivity paradigm is grounded in several interdisciplinary principles that bridge computational models with perceptual understanding.

**Perceptual Hierarchy**: Human visual perception operates hierarchically. Content-adaptive models mirror this through multi-scale feature extraction and hierarchical interaction modeling, as demonstrated in methods like CoDI-IQA which integrates internal, coarse, and fine interactions within a Progressive Perception Interaction Module (PPIM)[^124].

**Cognitive Flexibility**: Unlike static algorithms, human quality judgment adapts based on task context and image purpose. Content-adaptive architectures encode this flexibility through their dynamic parameter generation mechanisms.

**Semantic-Statistical Balance**: Effective quality assessment requires balancing semantic content understanding with statistical distortion analysis. Static models often overemphasize statistical regularities while neglecting semantic context, whereas content-adaptive approaches explicitly model the interplay between these factors[^124].

### 3.5 Limitations and Challenges of the Content-Adaptive Approach

Despite its theoretical advantages, the content-adaptivity paradigm introduces new challenges that must be addressed.

**Computational Complexity**: Dynamic weight generation increases inference time and memory requirements compared to static models, potentially limiting deployment in resource-constrained environments[^2].

**Training Stability**: Learning stable mappings from semantic features to quality rules requires careful architectural design and training strategies to prevent mode collapse or overfitting to specific content types.

**Interpretability Trade-offs**: While content-adaptive models better mimic human perception, their dynamic nature can make decision processes less transparent than fixed-rule systems[^2].

**Content Ambiguity**: Images with ambiguous or novel semantic content may not trigger appropriate quality rules, highlighting the need for robust out-of-distribution handling mechanisms.

The transition from static to content-adaptive BIQA represents a fundamental alignment with the principles of human visual perception. By acknowledging that quality is not an absolute property but emerges from the interaction between content, context, and distortion characteristics, this paradigm shift has enabled more psychologically plausible and practically effective quality assessment systems.

## 4. HyperNetwork Architectures: Pioneering Content-Adaptive BIQA with HyperIQA

The evolution of Blind Image Quality Assessment (BIQA) reached a pivotal milestone with the introduction of HyperNetwork-based architectures, most notably exemplified by HyperIQA[^11]. This paradigm represents a fundamental departure from traditional static models, introducing a dynamic, content-adaptive framework that directly addresses the core challenge of assessing "in-the-wild" images, where quality perception is inherently intertwined with semantic content[^18].

### 4.1 The Three-Stage Architecture: Mimicking Human Perception

HyperIQA's core innovation is its explicit separation of the IQA procedure into three distinct stages that mirror the top-down flow of human perceptual judgment[^11]. This tripartite design moves beyond the monolithic, end-to-end approach of early CNNs, which applied the same fixed rules to all images regardless of their content.

**Stage 1: Content Understanding (Semantic Feature Extraction)**\
The process begins with a backbone network (ResNet50 pretrained on ImageNet) that extracts semantic features from the input image. This stage is dedicated to understanding *what* the image depicts before any quality judgment is made. The backbone produces two parallel feature streams: semantic features $S(x)$ for the hyper network and multi-scale content features $S_{ms}(x)$ for the target network[^11].

**Stage 2: Perception Rule Learning (Hyper Network)**\
A hyper network component dynamically generates adaptive weight parameters $\theta_x$ based on the semantic features $S(x)$. This mechanism, expressed as $\theta_x = H(S(x), \gamma)$, allows the model to learn different "quality perceiving rules" tailored to the specific content of image $x$. This is the key to content adaptivity, enabling the model to assess, for instance, a blurred landscape differently from a blurred document[^11].

**Stage 3: Quality Prediction (Target Network)**\
A lightweight target network receives the multi-scale content features $v_x = S_{ms}(x)$ as input. Crucially, the weights and biases of this network's layers are the dynamically generated parameters $\theta_x$. The final quality score $q$ is produced according to $\phi(v_x, \theta_x) = q$, meaning the prediction is made using rules specifically adapted to the image's content[^11].

### 4.2 Technical Components Enabling Adaptivity

**Local Distortion Aware Module**\
To handle the diverse and often localized distortions in real-world images, HyperIQA incorporates a module that extracts features from multiple intermediate layers of the ResNet backbone (e.g., conv2\_10, conv3\_12, conv4\_18). These features are divided into non-overlapping patches, stacked, and processed to capture both local distortion patterns and global semantic context[^11].

**Hyper Network Weight Generation**\
The hyper network itself consists of convolutional layers and specialized branches. It transforms the semantic features into the specific weights and biases for the target network's fully connected layers through a series of convolution, reshape, pooling, and linear operations. Visualization confirms that this process generates distinct parameters for different image categories, solidifying its content-aware nature[^11].

### 4.3 Content Adaptivity: Addressing Core Limitations

The primary contribution of HyperIQA is its operationalization of content adaptivity. Traditional CNN-based BIQA models, described by $\phi(x, \theta) = q$, use a fixed parameter set $\theta$ for all inputs. This fails to account for how human quality judgment depends on context; for example, a large uniform area (like a clear sky) might be misjudged as blur by a static model[^11]. HyperIQA overcomes this by making parameters input-dependent ($\theta_x$), allowing it to learn that different contents warrant different perceptual rules. This leads to superior generalization on authentic image databases[^11].

### 4.4 Comparative Analysis and Performance Advantages

HyperIQA represents a clear advancement over previous methodologies:

| **Aspect**               | **Traditional CNN BIQA**         | **HyperIQA**                        |
| ------------------------ | -------------------------------- | ----------------------------------- |
| **Parameter Adaptation** | Static, content-agnostic         | Dynamic, content-adaptive           |
| **Perceptual Modeling**  | Implicit, monolithic             | Explicit, three-stage human-like    |
| **Content Awareness**    | Limited or absent                | Central to architecture             |
| **Generalization**       | Limited to training distribution | Improved through content adaptation |

This architectural innovation translated to significant performance gains. HyperIQA demonstrated state-of-the-art results on challenging authentic databases like LIVEC, BID, and KonIQ-10k[^11]. Its content-adaptive principle also proved effective in specialized applications, such as assessing the quality of high-resolution 4K content, where it outperformed traditional CNN-based methods by better integrating texture complexity and quality-aware features[^137].

### 4.5 Limitations and Evolutionary Context

Despite its pioneering role, the HyperNetwork approach has limitations that subsequent architectures have sought to address. The process of dynamically generating parameters for each input introduces computational overhead. Furthermore, while the ResNet backbone provides strong semantic features, its convolutional inductive bias may not capture the long-range dependencies and complex content-quality interactions as effectively as the self-attention mechanisms in later Transformer-based models[^138].

### 4.6 Conclusion

The HyperIQA architecture's lasting impact lies in establishing content adaptivity as a fundamental design principle for BIQA. By explicitly modeling the interaction between image semantics and quality perception, it bridged a critical gap between static computational models and dynamic human judgment. This principle, first crystallized in the hyper network paradigm, continues to underpin and inspire more recent advancements in transformer-based and vision-language BIQA methods.

## 5. Post-HyperNetwork Evolution: Transformer-Based Architectures and Attention Mechanisms

The advent of HyperNetwork-based models like HyperIQA represented a paradigm shift in Blind Image Quality Assessment (BIQA) by introducing content adaptivity through dynamically generated network parameters [^11]. However, the subsequent emergence of transformer-based architectures constitutes another fundamental advancement, leveraging self-attention mechanisms to model complex, long-range dependencies across entire feature maps [^140]. This transition addresses limitations in capturing global contextual relationships and multi-dimensional quality features that were challenging for both convolutional neural networks (CNNs) and the earlier HyperNetwork approach.

### 5.1 Architectural Shift: From Dynamic Weights to Self-Attention

The HyperIQA architecture pioneered content adaptivity via a three-stage framework: content understanding, perception rule learning, and quality prediction [^11]. Its core innovation was a hyper network that generated image-specific weight parameters $\theta_x$ based on extracted semantic features $S(x)$, enabling the target network to make content-aware predictions. While effective for adaptation, this architecture primarily focused on generating rules based on holistic content understanding, with inherent limitations in explicitly modeling intricate relationships between all spatial regions of an image.

In contrast, transformer-based BIQA models are built upon the self-attention mechanism, which computes attention scores across all spatial positions simultaneously [^149]. The fundamental operation is expressed as:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

where $Q$, $K$, and $V$ are query, key, and value matrices projected from the input features, and $d_k$ is the key dimension. This mechanism allows the model to establish direct relationships between any two patches in an image, enabling it to capture both local distortions and global semantic contexts that collectively influence perceptual quality. This represents a shift from adapting to content for rule generation (as in HyperNetworks) to directly modeling the relational structure within the content itself.

### 5.2 MANIQA: A Multi-Dimensional Attention Framework

A prominent example of this evolution is the MANIQA (Multi-dimension Attention Network for No-Reference Image Quality Assessment) architecture, which is specifically optimized for BIQA [^141]. MANIQA integrates several innovative components to address the multi-faceted nature of image quality:

| Component                               | Primary Function                                                | Key Innovation                                                                                                                  |
| --------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Feature Extractor**                   | Extracts multi-scale features using a Vision Transformer (ViT). | Concatenates features from specific intermediate ViT layers for a comprehensive representation.                                 |
| **Transposed Attention Block (TAB)**    | Applies self-attention across the channel dimension.            | Computes cross-covariance across channels to encode global context and re-weight channels based on their importance to quality. |
| **Scale Swin Transformer Block (SSTB)** | Enhances local interactions among image patches.                | Combines Swin Transformer layers with convolutional operations, using a scale factor to stabilize training.                     |
| **Dual-Branch Structure**               | Performs patch-weighted quality prediction.                     | Separates the tasks of predicting a quality score and an importance weight for each image patch.                                |

The TAB module is particularly innovative, applying attention across channels rather than spatial locations. Its operation is defined as:

$$
\hat{X} = W_p \text{Attn}(\hat{Q}, \hat{K}, \hat{V}) + X
$$

where $\text{Attn}(\hat{Q}, \hat{K}, \hat{V}) = \hat{V} \cdot \text{Softmax}(\hat{K} \cdot \hat{Q}/\alpha)$, and $\alpha$ is a scaling factor related to the spatial dimension [^141]. This channel-wise attention complements traditional spatial attention, allowing the model to understand which feature channels are most relevant for assessing quality.

The dual-branch structure mimics human visual attention by assigning different importance weights to various image regions. The final image-level quality score $\tilde{q}$ is computed as a weighted average of patch scores:

$$
\tilde{q} = \frac{\sum_{0<i<N} \omega_i \times s_i}{\sum_{0<i<N} \omega_i}
$$

where $N$ is the number of patches, $s_i$ is the quality score for patch $i$, and $\omega_i$ is its learned importance weight [^141].

### 5.3 Performance and Comparative Advantages

Empirical evaluations demonstrate that transformer-based BIQA models achieve superior performance compared to HyperNetwork approaches like HyperIQA. Benchmark results indicate consistent improvements across both synthetic and authentic distortion datasets \[\[155]\[156]]. For instance, on the challenging LIVEC dataset containing authentic "in-the-wild" distortions, a transformer-based model achieved a Spearman Rank Order Correlation Coefficient (SRCC) of 0.875, outperforming HyperIQA's SRCC of 0.859 [^156]. Similar gains are observed on synthetic databases like LIVE, where transformer models reach an SRCC of 0.980 versus 0.969 for HyperIQA [^156].

The performance advantage stems from several architectural strengths of transformers:

1. **Global Context Modeling:** Self-attention allows the model to directly relate any two image regions, capturing long-range dependencies that affect holistic quality perception, which is difficult for models with localized receptive fields.
2. **Multi-Dimensional Feature Interaction:** By employing both spatial and channel-wise attention (as in MANIQA), transformers can model the complex interplay between different types of quality-degrading factors.
3. **Content-Agnostic Processing:** Unlike HyperNetworks that generate unique parameters per image, transformers apply shared attention mechanisms, which can improve generalization across diverse and unseen image contents.

### 5.4 Further Architectural Innovations

The transformer paradigm has inspired several other architectural variants tailored for BIQA:

* **ADTRS (Attention Down-Sampling Transformer, Relative Ranking and Self-Consistency):** This model employs a hybrid CNN-Transformer encoder to extract features, followed by a transformer encoder to model dependencies. It incorporates relative ranking within batches and a self-consistency loss on horizontally flipped images to enhance robustness [^149].

* **ASCAM-Former:** This approach explicitly addresses the limitation of spatial-only attention by integrating **channel-wise self-attention** mechanisms alongside spatial attention, enabling more effective aggregation of quality information across feature dimensions \[\[147]\[152]].

* **Local Distortion Aware Transformers:** To compensate for the Vision Transformer's lack of innate bias for local image structure, some methods inject local distortion features extracted from a pre-trained CNN into the transformer architecture, combining the strengths of both paradigms [^150].

### 5.5 Challenges and Future Directions

Despite their strengths, transformer-based BIQA models face certain challenges:

* **Computational Complexity:** The self-attention mechanism has quadratic complexity with respect to the number of input patches, which can be prohibitive for very high-resolution images.

* **Data Requirements:** Transformers typically benefit from large-scale training data, though recent "data-efficient" methods aim to mitigate this [^155].

* **Interpretability:** The complex attention maps can be difficult to interpret, making it challenging to understand the exact reasoning behind a quality prediction.

The evolution from HyperNetwork to transformer-based architectures signifies a conceptual shift in BIQA: from adapting model parameters to the content, to modeling the relational structure within the content itself. This has unlocked new capabilities for capturing the intricate, global dependencies that define human perception of image quality, setting a new direction for the field \[\[140]\[141]].

## 6. Vision-Language Model Integration: A New Paradigm for Semantic Quality Assessment

The most recent and transformative paradigm shift in Blind Image Quality Assessment (BIQA) is the integration of large-scale pre-trained Vision-Language Models (VLMs). This evolution represents a fundamental leap from unimodal, purely visual analysis toward a framework that leverages rich semantic understanding and cross-modal reasoning[^5]. By exploiting the comprehensive visual-textual correspondences learned from massive datasets, VLM-based methods offer a powerful mechanism to address the intricate interplay between image content, distortion characteristics, and human perceptual quality, particularly for the complex authentic distortions found "in-the-wild"[^2].

### 6.1 The Rationale for VLM-Based BIQA

The transition to VLMs is a natural progression from earlier content-adaptive architectures like HyperIQA, which conceptually separated the assessment process into stages of content understanding, perception rule learning, and quality prediction[^11]. VLMs inherently possess a pre-trained, unified understanding of visual concepts and their linguistic descriptions, providing several distinct advantages for BIQA[^74]:

1. **Rich Semantic Grounding**: Models like CLIP have learned to associate visual patterns with semantic text, enabling quality assessment informed by high-level content understanding rather than just low-level statistical features.
2. **Zero-Shot Potential**: The pre-trained knowledge allows for quality estimation without task-specific fine-tuning, offering a promising solution for assessing novel or emerging distortion types where labeled data is scarce[^2].
3. **Multimodal Reasoning**: These models can integrate and reason over information from both visual and textual modalities, better mimicking the human perceptual process where quality judgment is influenced by semantic context[^1].

This aligns with the identified trend toward multimodal quality assessment, which seeks to enhance accuracy and robustness by leveraging complementary information from multiple sources[^1].

### 6.2 Pioneering VLM Approaches: CLIP-IQA and LIQE

The exploration of VLMs for BIQA has crystallized around two seminal methodologies, each representing a different adaptation paradigm.

**CLIP-IQA** pioneered the zero-shot use of the CLIP model for BIQA[^159]. Its core methodology involves computing the cosine similarity between a test image's visual embedding and the textual embeddings of carefully designed antonym prompt pairs (e.g., "This photo is of good quality" vs. "This photo is of bad quality"). A relative quality score is derived from these similarities, effectively assessing quality through text-image alignment without any fine-tuning on IQA data. While demonstrating the feasibility of zero-shot quality assessment, its performance is constrained by the fixed, hand-crafted prompt design, which may not optimally capture the nuances of diverse distortions[^159].

**LIQE (Language-Image Quality Evaluator)** advanced the paradigm by formulating BIQA as a multitask learning problem that explicitly leverages vision-language correspondence\[\[85]\[92]]. LIQE employs a textual template (e.g., "a photo of a(n) {scene} with {distortion} artifacts, which is of {quality} quality") to generate hundreds of candidate label combinations spanning scene category, distortion type, and quality level. A pre-trained CLIP model encodes both image patches and these textual descriptions. The model computes a joint probability distribution over the three tasks by measuring the alignment between visual and textual embeddings, formally expressed through cosine similarity logits and a softmax function with a learnable temperature parameter[^92]. Predictions for each task are obtained by marginalizing this joint distribution. This automated multitask framework allows LIQE to learn richer, more context-aware representations, leading to superior performance compared to earlier methods[^85].

### 6.3 Advanced Adaptation and Prompt Learning Strategies

To overcome the limitations of fixed prompts and enhance the alignment between VLM capabilities and the BIQA task, researchers have developed sophisticated prompt learning and adaptation strategies.

**MP-IQE (Multi-Modal Prompt Image Quality Evaluator)** introduces a dual-prompt scheme within the text branch of CLIP, employing separate learnable prompts to capture scene context and distortion-specific information[^159]. Concurrently, it implements deep visual prompting by injecting learnable tokens at each layer of the vision transformer encoder. A dedicated multimodal encoder then facilitates cross-attention between the adapted textual and visual features to predict the final quality score. This comprehensive prompting approach significantly boosts performance by ensuring the model extracts task-relevant semantic information from both modalities[^159].

**CLIP-AGIQA** represents a specialized adaptation for the emerging domain of AI-Generated Image (AGI) quality assessment. It utilizes multi-category learnable prompts and concatenates the adapted visual and textual features before passing them through regression layers to predict a quality score, effectively tailoring the CLIP framework for this novel application[^41].

### 6.4 Toward Self-Evolving and Unsupervised VLM BIQA

A groundbreaking direction seeks to eliminate dependency on human-annotated Mean Opinion Scores (MOS) altogether. **EvoQuality** is a fully self-supervised framework that enables a VLM to autonomously refine its quality perception[^40]. It operates through an iterative loop: first, the model generates pseudo-labels by performing pairwise comparisons on unlabeled images and applying majority voting to establish a consensus ranking. These pseudo-rankings then formulate a fidelity reward to guide the model's iterative evolution via Group Relative Policy Optimization (GRPO). This approach demonstrates that VLMs can achieve competitive, and sometimes superior, performance compared to supervised models without any ground-truth quality labels, offering a compelling solution to the perennial challenge of label scarcity\[\[2]\[40]].

### 6.5 Comparative Analysis of VLM Integration Paradigms

The evolution of VLM-based BIQA reveals distinct paradigms with complementary strengths and limitations, as summarized below:

| Paradigm                        | Representative Method | Key Innovation                                                                                   | Strengths                                                 | Limitations                                              |
| :------------------------------ | :-------------------- | :----------------------------------------------------------------------------------------------- | :-------------------------------------------------------- | :------------------------------------------------------- |
| **Zero-Shot**                   | CLIP-IQA[^159]        | Antonym prompt pairing for direct similarity scoring.                                            | No training required; highly generalizable.               | Performance limited by fixed prompts; less accurate.     |
| **Multitask Fine-tuning**       | LIQE\[\[85]\[92]]     | Joint probability modeling of quality, scene, and distortion via vision-language correspondence. | Comprehensive semantic understanding; high accuracy.      | Requires supervised training; higher computational cost. |
| **Multi-Modal Prompt Learning** | MP-IQE[^159]          | Learnable dual (scene/distortion) text prompts and deep visual prompts.                          | Enhanced feature alignment; state-of-the-art performance. | Complex training procedure; risk of overfitting.         |
| **Self-Evolving**               | EvoQuality[^40]       | Pairwise voting for pseudo-labels and policy optimization.                                       | No human labels needed; autonomous improvement.           | Iterative process is computationally intensive.          |

### 6.6 Semantic Understanding as the Core Advantage

The fundamental advancement of VLM-based BIQA lies in its semantic grounding. Unlike traditional models that regress quality from low-level features, VLMs can reason about quality in relation to understood concepts. For instance, they can contextualize that "blur" is more detrimental to the quality of a text document than to an impressionist painting, or that low "colorfulness" might be an artistic choice rather than a distortion. This ability to integrate content, distortion type, and appearance properties (e.g., brightness, contrast) into a coherent judgment closely approximates the contextual and semantic nature of human quality perception[^74].

### 6.7 Challenges and Future Directions

Despite remarkable progress, several challenges define the frontier of VLM-based BIQA research:

1. **Computational Efficiency**: The large size of foundation models conflicts with the need for deployable, real-time solutions in edge and mobile scenarios[^2].
2. **Reasoning Reliability**: Studies indicate that VLMs can produce contradictory textual descriptions and unstable score predictions, highlighting a gap between their semantic knowledge and reliable, human-like reasoning for IQA[^163].
3. **Generalization and Specialization**: Balancing the strong zero-shot generalization of VLMs with the need for high accuracy in specialized domains (e.g., medical imaging, underwater photography) remains an open problem[^1].
4. **Explainability**: The "black-box" nature of these complex models makes it difficult to interpret how specific image attributes lead to a particular quality score, limiting trust in critical applications[^2].

Future research will likely focus on creating more efficient and robust adaptation techniques, developing better frameworks for unsupervised and self-supervised learning, and deepening the integration of multimodal reasoning to fully harness the potential of foundation models for perceptual quality assessment.

## 7. Comparative Analysis and Future Directions: Architectural Trade-offs and Emerging Trends

The evolution of Blind Image Quality Assessment (BIQA) methodologies reflects a paradigm shift from expert-driven feature engineering to data-driven learning, culminating in the current exploration of multimodal foundation models. This progression has introduced significant architectural trade-offs that require systematic analysis to inform future research directions.

### 7.1 Architectural Evolution: From Static Models to Content-Adaptive Systems

The BIQA landscape can be broadly characterized by three evolutionary phases, each addressing core limitations of its predecessors.

**Phase 1: Hand-crafted Feature Engineering (Pre-2015)**
Early BIQA methods relied on domain expertise to design quality-aware features. This category includes distortion-specific methods for applications like screen content or low-light imaging, and general-purpose methods. The latter are primarily divided into Natural Scene Statistics (NSS)-based approaches, such as BRISQUE and NIQE, which quantify deviations from the statistical regularities of natural images, and Human Visual System (HVS)-guided methods that incorporate perceptual characteristics[^1]  [^74]. These methods are computationally efficient and easy to deploy but struggle to generalize across the diverse and complex distortions found in real-world, "in-the-wild" images[^1]  [^2].

**Phase 2: Deep Learning Revolution (2015-2020)**
The advent of deep neural networks enabled end-to-end learning of quality features directly from distorted images. Early CNN-based models, such as the pioneering work by Kang et al., used patch-based inputs and shallow architectures to predict quality scores[^3]. These supervised learning approaches minimized the discrepancy between predicted scores and subjective Mean Opinion Scores (MOS). While demonstrating superior performance over hand-crafted methods, these deep models were largely content-agnostic, applying the same fixed parameters regardless of an image's semantic content, which limited their effectiveness for authentic distortions[^18].

**Phase 3: Content-Adaptive Architectures (2020-Present)**
The recognition that human quality perception is inherently content-dependent led to the development of architectures that adapt their processing based on image semantics. HyperIQA pioneered this approach with a self-adaptive hypernetwork architecture[^11]. Its core innovation is separating the IQA procedure into three stages: content understanding, perception rule learning, and quality prediction. A hypernetwork, $H(S(x), \gamma)$, dynamically generates the weight parameters $\theta_x$ for a target prediction network based on the input image's semantic features $S(x)$. This allows the model, expressed as $\phi(v_x, \theta_x) = q$, to apply content-specific "perception rules," mimicking the human top-down perception flow where quality is judged after content is understood[^11].

### 7.2 Comparative Analysis of Modern BIQA Paradigms

The post-HyperIQA era has seen the emergence of several dominant architectural paradigms, each with distinct characteristics and trade-offs.

| **Architectural Paradigm**      | **Key Innovations**                                                                          | **Strengths**                                                                                    | **Limitations**                                                              | **Representative Models**                           |
| :------------------------------ | :------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------- | :-------------------------------------------------- |
| **HyperNetwork-based**          | Self-adaptive parameters, content-specific perception rules, local distortion-aware modules. | Excellent content adaptivity, strong performance on authentic distortions.                       | Computational overhead, complex training dynamics.                           | HyperIQA[^11]                                       |
| **Transformer-based**           | Global self-attention mechanisms, transposed attention blocks, multi-scale processing.       | Superior global context modeling, effective for complex synthetic distortions (e.g., GAN-based). | High computational cost, large parameter counts.                             | MANIQA[^21]  [^24], TReS[^171]                      |
| **Vision-Language Model (VLM)** | Multimodal feature fusion, semantic-textual correspondence, zero/few-shot capability.        | Strong generalization, leverages rich semantic understanding, human-aligned reasoning.           | Dependence on large-scale pre-training, potential reasoning inconsistencies. | LIQE[^85]  [^92], CLIP-IQA[^30]  [^34], SLIQUE[^39] |

**HyperNetwork vs. Transformer Architectures**
HyperIQA's adaptive mechanism excels at handling the unpredictable, local distortions common in authentic images by tailoring its analysis to the content[^11]. In contrast, transformer-based methods like MANIQA leverage global attention mechanisms to capture long-range dependencies across the entire image, which is particularly effective for assessing distortions in synthetically generated content[^21]. MANIQA introduces architectural innovations like Transposed Attention Blocks (TAB) to strengthen feature interactions[^24].

**Vision-Language Integration**
Methods like LIQE and CLIP-IQA represent a significant shift by leveraging pre-trained vision-language models such as CLIP. LIQE employs a multitask learning scheme that uses a textual template to describe combinations of quality levels, scene categories, and distortion types. It computes a joint probability distribution from the similarity between visual and textual embeddings, from which the quality score is derived[^85]  [^92]. This approach benefits from the rich, high-level semantic knowledge embedded in VLMs, enabling better generalization and alignment with human judgment that often relies on semantic description[^1].

### 7.3 Performance Trade-offs: Accuracy, Efficiency, and Generalization

The architectural evolution highlights fundamental, often competing, priorities in BIQA system design.

**Accuracy vs. Computational Efficiency**

* **Traditional CNNs & Hand-crafted Methods:** Offer moderate accuracy with good efficiency, suitable for resource-constrained environments[^3].

* **HyperNetworks:** Achieve high accuracy for authentic distortions but introduce overhead for dynamic weight generation[^11].

* **Transformers:** Deliver state-of-the-art accuracy for many tasks but at the cost of significantly higher computational and memory requirements[^21].

* **VLMs:** Provide competitive accuracy and exceptional generalization by leveraging massive pre-trained models, though inference can be heavy[^39].

**Generalization Capability**

* **Content-agnostic models:** Exhibit limited generalization across diverse image contents and unseen distortion types[^18].

* **Content-adaptive models (HyperNetworks):** Improve generalization by adapting to image semantics[^11].

* **Multimodal models (VLMs):** Offer superior cross-domain generalization by grounding quality assessment in shared semantic-textual understanding[^1]  [^85].

**Data Requirements**

* **Supervised methods:** Require extensive, costly datasets annotated with human MOS labels, which are scarce for authentic distortions[^2].

* **Unsupervised/Opinion-unaware methods:** Reduce dependency on labeled data but may face challenges in achieving optimal accuracy[^1].

* **Foundation models (VLMs):** Utilize large-scale pre-training to enable effective few-shot or zero-shot learning, mitigating data scarcity[^39]  [^40].

### 7.4 Current Research Gaps and Emerging Trends

Despite remarkable progress, several research gaps point toward promising future directions.

**1. Advanced Learning for Authentic Distortions**
The complexity and lack of labeled data for authentic "in-the-wild" distortions remain a core challenge. Emerging trends include sophisticated self-supervised and unsupervised learning paradigms. For example, frameworks like EvoQuality enable VLMs to self-evolve their quality perception through iterative self-consistency and ranking-based rewards, entirely without ground-truth labels[^40]. Other methods explore leveraging synthetic data or multiple full-reference models to generate pseudo-labels for training[^7].

**2. Lightweight and Deployable Architectures**
The computational demands of modern transformers and VLMs hinder deployment on edge devices (e.g., smartphones, cameras). Research is increasingly focused on efficiency, exploring knowledge distillation, neural architecture search, and the design of inherently lightweight models that maintain competitive performance[^3]  [^75].

**3. Expansion of Multimodal Integration**
While current multimodal BIQA primarily fuses vision and language, future systems could integrate additional sensory modalities to better mimic holistic human perception. This includes audio-visual quality assessment for video content and potentially leveraging other contextual metadata (e.g., EXIF data, user preferences)[^1].

**4. Robust Domain Adaptation and Continual Learning**
Real-world distortion types are dynamic and evolving. Future models need robust mechanisms to adapt to new domains and learn new distortion types continuously without catastrophic forgetting of previous knowledge. Meta-learning and domain adaptation techniques are key research avenues[^1]  [^91].

**5. Explainable and Human-Aligned Assessment**
Moving beyond "black-box" predictions, there is a growing need for models that provide interpretable reasoning for their quality scores, aligning more closely with human perceptual judgments and even accounting for individual or contextual variations in quality perception[^2]  [^96].

### 7.5 Toward Next-Generation BIQA Systems

The evolutionary trajectory suggests several principles for next-generation systems:

* **Hybrid Architectures:** Combining the content adaptivity of hypernetworks, the global modeling of transformers, and the semantic grounding of VLMs.

* **Efficiency by Design:** Architectures that inherently balance accuracy with computational cost for practical deployment.

* **Cross-Modal Foundation Models:** Leveraging knowledge from large-scale pre-training across vision, language, and other modalities for comprehensive assessment.

* **Adaptive and Personalized Assessment:** Systems that can adapt to specific application domains, device constraints, or even individual user preferences.

The field is at an inflection point where architectural innovations must balance the competing demands of accuracy, efficiency, generalization, and human alignment. The convergence of adaptive mechanisms, transformer architectures, and multimodal foundation models offers a promising path toward robust BIQA systems capable of addressing the complex challenges of real-world image quality assessment.

## 8. Reference

[^1]: Blind Image Quality Assessment: A Brief Survey - arXiv, <a href="https://arxiv.org/html/2312.16551v1" target="_blank"><https://arxiv.org/html/2312.16551v1></a>

[^2]: Blind Image Quality Assessment (BIQA) - Emergent Mind, <a href="https://www.emergentmind.com/topics/blind-image-quality-assessment-biqa" target="_blank"><https://www.emergentmind.com/topics/blind-image-quality-assessment-biqa></a>

[^3]: Blind image quality assessment based on hierarchical dependency ..., <a href="https://www.sciencedirect.com/science/article/abs/pii/S0925231224003928" target="_blank"><https://www.sciencedirect.com/science/article/abs/pii/S0925231224003928></a>

[^4]: Blind Image Quality Assessment Using Convolutional Neural Networks, <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC12656157/" target="_blank"><https://pmc.ncbi.nlm.nih.gov/articles/PMC12656157/></a>

[^5]: \[2312.16551] Blind Image Quality Assessment: A Brief Survey - arXiv, <a href="https://arxiv.org/abs/2312.16551" target="_blank"><https://arxiv.org/abs/2312.16551></a>

[^6]: Unsupervised blind image quality assessment via joint spatial and ..., <a href="https://www.nature.com/articles/s41598-023-38099-5" target="_blank"><https://www.nature.com/articles/s41598-023-38099-5></a>

[^7]: \[PDF] Deep Opinion-Unaware Blind Image Quality Assessment by ... - IJCAI, <a href="https://www.ijcai.org/proceedings/2025/0227.pdf" target="_blank"><https://www.ijcai.org/proceedings/2025/0227.pdf></a>

[^8]: \[PDF] Blind Image Quality Assessment Based on Geometric Order Learning, <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Shin_Blind_Image_Quality_Assessment_Based_on_Geometric_Order_Learning_CVPR_2024_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/CVPR2024/papers/Shin_Blind_Image_Quality_Assessment_Based_on_Geometric_Order_Learning_CVPR_2024_paper.pdf></a>

[^9]: \[PDF] Semi-Supervised Blind Image Quality Assessment through ..., <a href="https://ojs.aaai.org/index.php/AAAI/article/view/28236/28467" target="_blank"><https://ojs.aaai.org/index.php/AAAI/article/view/28236/28467></a>

[^10]: A Metric for Evaluating Image Quality Difference Perception Ability in ..., <a href="https://dl.acm.org/doi/10.1145/3689093.3689182" target="_blank"><https://dl.acm.org/doi/10.1145/3689093.3689182></a>

[^11]: \[PDF] Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive ..., <a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf></a>

[^12]: A Comprehensive Approach for Image Quality Assessment Using ..., <a href="https://www.sciencedirect.com/science/article/pii/S0031320325015535" target="_blank"><https://www.sciencedirect.com/science/article/pii/S0031320325015535></a>

[^13]: SSL92/hyperIQA - GitHub, <a href="https://github.com/SSL92/hyperIQA" target="_blank"><https://github.com/SSL92/hyperIQA></a>

[^14]: No-Reference Quality Assessment of Extended Target Adaptive ..., <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10781174/" target="_blank"><https://pmc.ncbi.nlm.nih.gov/articles/PMC10781174/></a>

[^15]: GSBIQA: Green Saliency-guided Blind Image Quality Assessment ..., <a href="https://arxiv.org/html/2407.05590v1" target="_blank"><https://arxiv.org/html/2407.05590v1></a>

[^16]: Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive ..., <a href="https://ieeexplore.ieee.org/iel7/9142308/9156271/09156687.pdf" target="_blank"><https://ieeexplore.ieee.org/iel7/9142308/9156271/09156687.pdf></a>

[^17]: Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive ..., <a href="https://www.researchgate.net/publication/343463381_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_Self-Adaptive_Hyper_Network" target="_blank"><https://www.researchgate.net/publication/343463381_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_Self-Adaptive_Hyper_Network></a>

[^18]: Progress in Blind Image Quality Assessment: A Brief Review - MDPI, <a href="https://www.mdpi.com/2227-7390/11/12/2766" target="_blank"><https://www.mdpi.com/2227-7390/11/12/2766></a>

[^19]: \[PDF] Blind Quality Assessment for in-the-Wild Images via ..., <a href="https://www.semanticscholar.org/paper/Blind-Quality-Assessment-for-in-the-Wild-Images-via-Sun-Min/848a5aded64c1ddaea2dd5471d56d4d5dbbe7987" target="_blank"><https://www.semanticscholar.org/paper/Blind-Quality-Assessment-for-in-the-Wild-Images-via-Sun-Min/848a5aded64c1ddaea2dd5471d56d4d5dbbe7987></a>

[^20]: Dual-branch vision transformer for blind image quality assessment, <a href="https://www.sciencedirect.com/science/article/abs/pii/S1047320323001001" target="_blank"><https://www.sciencedirect.com/science/article/abs/pii/S1047320323001001></a>

[^21]: Blind CT Image Quality Assessment Using DDPM-derived Content ..., <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11560125/" target="_blank"><https://pmc.ncbi.nlm.nih.gov/articles/PMC11560125/></a>

[^22]: Multi-scale Transformer with Decoder for Image Quality Assessment, <a href="https://dl.acm.org/doi/abs/10.1007/978-981-99-8850-1_18" target="_blank"><https://dl.acm.org/doi/abs/10.1007/978-981-99-8850-1_18></a>

[^23]: Transformer-based No-Reference Image Quality Assessment ... - arXiv, <a href="https://arxiv.org/abs/2312.06995" target="_blank"><https://arxiv.org/abs/2312.06995></a>

[^24]: MANIQA: Multi-dimension Attention Network for No-Reference ..., <a href="https://www.researchgate.net/publication/362884664_MANIQA_Multi-dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment" target="_blank"><https://www.researchgate.net/publication/362884664_MANIQA_Multi-dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment></a>

[^25]: A Survey of Transformer Architectures in Perceptual Image Quality ..., <a href="https://ieeexplore.ieee.org/iel8/6287639/10380310/10767243.pdf" target="_blank"><https://ieeexplore.ieee.org/iel8/6287639/10380310/10767243.pdf></a>

[^26]: Blind Image Quality Assessment via Transformer Predicted Error ..., <a href="https://www.semanticscholar.org/paper/Blind-Image-Quality-Assessment-via-Transformer-Map-Shi-Gao/9afbfe6f89412cc61e345558bff1da3dbbfe11cb" target="_blank"><https://www.semanticscholar.org/paper/Blind-Image-Quality-Assessment-via-Transformer-Map-Shi-Gao/9afbfe6f89412cc61e345558bff1da3dbbfe11cb></a>

[^27]: \[PDF] Boosting Image Quality Assessment through Efficient Transformer ..., <a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Boosting_Image_Quality_Assessment_through_Efficient_Transformer_Adaptation_with_Local_CVPR_2024_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Boosting_Image_Quality_Assessment_through_Efficient_Transformer_Adaptation_with_Local_CVPR_2024_paper.pdf></a>

[^28]: Dual-branch vision transformer for blind image quality assessment, <a href="https://dl.acm.org/doi/10.1016/j.jvcir.2023.103850" target="_blank"><https://dl.acm.org/doi/10.1016/j.jvcir.2023.103850></a>

[^29]: Few-Shot Image Quality Assessment via Adaptation of Vision ..., <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Few-Shot_Image_Quality_Assessment_via_Adaptation_of_Vision-Language_Models_ICCV_2025_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Few-Shot_Image_Quality_Assessment_via_Adaptation_of_Vision-Language_Models_ICCV_2025_paper.pdf></a>

[^30]: arXiv:2409.05381v1 \[cs.CV] 9 Sep 2024, <a href="https://arxiv.org/pdf/2409.05381?" target="_blank"><https://arxiv.org/pdf/2409.05381>?</a>

[^31]: arXiv:2405.19996v4 \[cs.CV] 17 Aug 2024, <a href="https://arxiv.org/pdf/2405.19996" target="_blank"><https://arxiv.org/pdf/2405.19996></a>

[^32]: Exploring CLIP for Assessing the Look and Feel of Images, <a href="https://www.researchgate.net/publication/371925434_Exploring_CLIP_for_Assessing_the_Look_and_Feel_of_Images" target="_blank"><https://www.researchgate.net/publication/371925434_Exploring_CLIP_for_Assessing_the_Look_and_Feel_of_Images></a>

[^33]: Video-Quality-Assessment-A-Comprehensive-Survey, <a href="https://github.com/taco-group/Video-Quality-Assessment-A-Comprehensive-Survey" target="_blank"><https://github.com/taco-group/Video-Quality-Assessment-A-Comprehensive-Survey></a>

[^34]: Boosting CLIP Adaptation for Image Quality Assessment ..., <a href="https://arxiv.org/html/2409.05381v1" target="_blank"><https://arxiv.org/html/2409.05381v1></a>

[^35]: Blind Image Quality Assessment With Multimodal Prompt ..., <a href="https://www.researchgate.net/publication/377825541_Blind_Image_Quality_Assessment_With_Multimodal_Prompt_Learning" target="_blank"><https://www.researchgate.net/publication/377825541_Blind_Image_Quality_Assessment_With_Multimodal_Prompt_Learning></a>

[^36]: 3D Indoor Scene Assessment via Layout Plausibility, <a href="https://papers.ssrn.com/sol3/Delivery.cfm/470b107e-c9e5-4a6b-a69a-0462cc8b3ca5-MECA.pdf?abstractid=4935975&mirid=1" target="_blank"><https://papers.ssrn.com/sol3/Delivery.cfm/470b107e-c9e5-4a6b-a69a-0462cc8b3ca5-MECA.pdf?abstractid=4935975&mirid=1></a>

[^37]: Analysis of Video Quality Datasets via Design ..., <a href="https://www.computer.org/csdl/journal/tp/2024/11/10499199/1WbKyQR2V3y" target="_blank"><https://www.computer.org/csdl/journal/tp/2024/11/10499199/1WbKyQR2V3y></a>

[^38]: Blind Image Quality Assessment via Vision-Language ..., <a href="https://www.researchgate.net/publication/373316627_Blind_Image_Quality_Assessment_via_Vision-Language_Correspondence_A_Multitask_Learning_Perspective" target="_blank"><https://www.researchgate.net/publication/373316627_Blind_Image_Quality_Assessment_via_Vision-Language_Correspondence_A_Multitask_Learning_Perspective></a>

[^39]: Vision Language Modeling of Content, Distortion and Appearance for Image Quality Assessment, <a href="https://arxiv.org/pdf/2406.09858" target="_blank"><https://arxiv.org/pdf/2406.09858></a>

[^40]: Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking, <a href="https://arxiv.org/pdf/2509.25787" target="_blank"><https://arxiv.org/pdf/2509.25787></a>

[^41]: CLIP-AGIQA: Boosting the Performance of AI-Generated Image Quality Assessment with CLIP, <a href="https://arxiv.org/pdf/2408.15098" target="_blank"><https://arxiv.org/pdf/2408.15098></a>

[^42]: A hybrid learning-based framework for blind image quality assessment, <a href="http://link.springer.com/10.1007/s11045-017-0475-y" target="_blank"><http://link.springer.com/10.1007/s11045-017-0475-y></a>

[^43]: A Multiscale Approach to Deep Blind Image Quality Assessment, <a href="https://ieeexplore.ieee.org/document/10056825?" target="_blank"><https://ieeexplore.ieee.org/document/10056825>?</a>

[^44]: Blind image quality assessment, <a href="http://ieeexplore.ieee.org/document/1038057/" target="_blank"><http://ieeexplore.ieee.org/document/1038057/></a>

[^45]: \[PDF] COMPARATIVE ANALYSIS OF UNIVERSAL METHODS NO ..., <a href="http://www.jatit.org/volumes/Vol99No9/5Vol99No9.pdf" target="_blank"><http://www.jatit.org/volumes/Vol99No9/5Vol99No9.pdf></a>

[^46]: (PDF) Comparative analysis of universal methods no reference ..., <a href="https://www.researchgate.net/publication/351812003_Comparative_analysis_of_universal_methods_no_reference_quality_assessment_of_digital_images" target="_blank"><https://www.researchgate.net/publication/351812003_Comparative_analysis_of_universal_methods_no_reference_quality_assessment_of_digital_images></a>

[^47]: No-Reference Quality Assessment, <a href="http://live.ece.utexas.edu/research/Quality/nrqa.htm" target="_blank"><http://live.ece.utexas.edu/research/Quality/nrqa.htm></a>

[^48]: No-Reference Image Quality Assessment Using the Statistics ... - MDPI, <a href="https://www.mdpi.com/2079-9292/12/7/1615" target="_blank"><https://www.mdpi.com/2079-9292/12/7/1615></a>

[^49]: Modified-BRISQUE as no reference image quality assessment for ..., <a href="https://www.sciencedirect.com/science/article/abs/pii/S0730725X17301340" target="_blank"><https://www.sciencedirect.com/science/article/abs/pii/S0730725X17301340></a>

[^50]: \[PDF] Blind Image Quality Assessment Based on Perceptual Comparison, <a href="https://web.xidian.edu.cn/wjj/files/66474f6d13a73.pdf" target="_blank"><https://web.xidian.edu.cn/wjj/files/66474f6d13a73.pdf></a>

[^51]: HVS and Contrast Sensitivity to Assess Image Quality ..., <a href="https://encyclopedia.pub/entry/45160" target="_blank"><https://encyclopedia.pub/entry/45160></a>

[^52]: \[PDF] Blind Quality Assessment of Image and Speech Signals - DR-NTU, <a href="https://dr.ntu.edu.sg/bitstreams/0620b1b3-165e-4f9a-bd48-e5b91ff1d1b1/download" target="_blank"><https://dr.ntu.edu.sg/bitstreams/0620b1b3-165e-4f9a-bd48-e5b91ff1d1b1/download></a>

[^53]: \[PDF] Learning a No-Reference Quality Assessment Model of Enhanced ..., <a href="https://kegu.netlify.app/PDF/Learning%20a%20no-reference%20quality%20assessment%20model%20of%20enhanced%20images%20with%20big%20data.pdf" target="_blank"><https://kegu.netlify.app/PDF/Learning%20a%20no-reference%20quality%20assessment%20model%20of%20enhanced%20images%20with%20big%20data.pdf></a>

[^54]: Study of no-reference image quality assessment algorithms on ..., <a href="https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-23/issue-6/061106/Study-of-no-reference-image-quality-assessment-algorithms-on-printed/10.1117/1.JEI.23.6.061106.full" target="_blank"><https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-23/issue-6/061106/Study-of-no-reference-image-quality-assessment-algorithms-on-printed/10.1117/1.JEI.23.6.061106.full></a>

[^55]: Blind image quality assessment for in-the-wild ..., <a href="https://www.sciencedirect.com/science/article/abs/pii/S0950705124014060" target="_blank"><https://www.sciencedirect.com/science/article/abs/pii/S0950705124014060></a>

[^56]: Toward a blind image quality evaluator in the wild by ..., <a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320322007750" target="_blank"><https://www.sciencedirect.com/science/article/abs/pii/S0031320322007750></a>

[^57]: DP-IQA: Utilizing Diffusion Prior for Blind Image Quality ..., <a href="https://arxiv.org/html/2405.19996v5" target="_blank"><https://arxiv.org/html/2405.19996v5></a>

[^58]: Blind Image Quality Assessment for Authentic Distortions ..., <a href="https://www.researchgate.net/publication/361040847_Blind_Image_Quality_Assessment_for_Authentic_Distortions_by_Intermediary_Enhancement_and_Iterative_Training" target="_blank"><https://www.researchgate.net/publication/361040847_Blind_Image_Quality_Assessment_for_Authentic_Distortions_by_Intermediary_Enhancement_and_Iterative_Training></a>

[^59]: Assessing Image Quality Issues for Real-World Problems, <a href="https://api.semanticscholar.org/CorpusID:214693119" target="_blank"><https://api.semanticscholar.org/CorpusID:214693119></a>

[^60]: QualityNet: A multi-stream fusion framework with spatial ..., <a href="https://www.nature.com/articles/s41598-024-77076-4" target="_blank"><https://www.nature.com/articles/s41598-024-77076-4></a>

[^61]: Computational Analysis of Degradation Modeling in Blind ..., <a href="https://dl.acm.org/doi/full/10.1145/3720547" target="_blank"><https://dl.acm.org/doi/full/10.1145/3720547></a>

[^62]: \[PDF] Opinion Unaware Image Quality Assessment via Adversarial ..., <a href="https://openaccess.thecvf.com/content/WACV2024/papers/Shukla_Opinion_Unaware_Image_Quality_Assessment_via_Adversarial_Convolutional_Variational_Autoencoder_WACV_2024_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/WACV2024/papers/Shukla_Opinion_Unaware_Image_Quality_Assessment_via_Adversarial_Convolutional_Variational_Autoencoder_WACV_2024_paper.pdf></a>

[^63]: Deep opinion-unaware blind image quality assessment by learning ..., <a href="https://dl.acm.org/doi/10.24963/ijcai.2025/227" target="_blank"><https://dl.acm.org/doi/10.24963/ijcai.2025/227></a>

[^64]: Opinion-unaware blind quality assessment of AI-generated ..., <a href="https://www.sciencedirect.com/science/article/abs/pii/S1047320325000756" target="_blank"><https://www.sciencedirect.com/science/article/abs/pii/S1047320325000756></a>

[^65]: Opinion-Unaware Blind Image Quality Assessment using Multi ..., <a href="https://arxiv.org/html/2405.18790v1" target="_blank"><https://arxiv.org/html/2405.18790v1></a>

[^66]: Convolutional Neural Network for Blind Image Quality Assessment, <a href="https://www.researchgate.net/publication/337287870_Convolutional_Neural_Network_for_Blind_Image_Quality_Assessment" target="_blank"><https://www.researchgate.net/publication/337287870_Convolutional_Neural_Network_for_Blind_Image_Quality_Assessment></a>

[^67]: Deep Activation Pooling for Blind Image Quality Assessment - MDPI, <a href="https://www.mdpi.com/2076-3417/8/4/478" target="_blank"><https://www.mdpi.com/2076-3417/8/4/478></a>

[^68]: \[PDF] Opinion-Unaware Blind Quality Assessment of AI-Generated ... - SSRN, <a href="https://papers.ssrn.com/sol3/Delivery.cfm/5dd9d253-3762-429d-b02b-8aeb2979a4c9-MECA.pdf?abstractid=5026045&mirid=1" target="_blank"><https://papers.ssrn.com/sol3/Delivery.cfm/5dd9d253-3762-429d-b02b-8aeb2979a4c9-MECA.pdf?abstractid=5026045&mirid=1></a>

[^69]: Blind image quality assessment via learnable attention-based pooling, <a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320319300925" target="_blank"><https://www.sciencedirect.com/science/article/abs/pii/S0031320319300925></a>

[^70]: Recycling Discriminator: Towards Opinion-Unaware Image Quality ..., <a href="https://www.semanticscholar.org/paper/086c57f6a682a9a0e71f89a9a708b7ea0852c620" target="_blank"><https://www.semanticscholar.org/paper/086c57f6a682a9a0e71f89a9a708b7ea0852c620></a>

[^71]: AIM 2024 Challenge on UHD Blind Photo Quality Assessment, <a href="https://dl.acm.org/doi/10.1007/978-3-031-91856-8_16" target="_blank"><https://dl.acm.org/doi/10.1007/978-3-031-91856-8_16></a>

[^72]: (PDF) AIM 2024 Challenge on UHD Blind Photo Quality Assessment, <a href="https://www.researchgate.net/publication/384295457_AIM_2024_Challenge_on_UHD_Blind_Photo_Quality_Assessment" target="_blank"><https://www.researchgate.net/publication/384295457_AIM_2024_Challenge_on_UHD_Blind_Photo_Quality_Assessment></a>

[^73]: \[PDF] VQualA 2025 Document Image Quality Assessment Challenge, <a href="https://openaccess.thecvf.com/content/ICCV2025W/VQualA/papers/Huang_VQualA_2025_Document_Image_Quality_Assessment_Challenge_ICCVW_2025_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/ICCV2025W/VQualA/papers/Huang_VQualA_2025_Document_Image_Quality_Assessment_Challenge_ICCVW_2025_paper.pdf></a>

[^74]: \[PDF] Blind Image Quality Assessment: A Brief Survey - arXiv, <a href="https://arxiv.org/pdf/2312.16551" target="_blank"><https://arxiv.org/pdf/2312.16551></a>

[^75]: Lightweight High-Performance Blind Image Quality Assessment, <a href="https://www.nowpublishers.com/article/Details/SIP-2023-0079" target="_blank"><https://www.nowpublishers.com/article/Details/SIP-2023-0079></a>

[^76]: \[PDF] A Survey of DNN Methods for Blind Image Quality Assessment ..., <a href="https://www.semanticscholar.org/paper/08778a2bdbf4c08b426e8a3767e1f0801388b51b" target="_blank"><https://www.semanticscholar.org/paper/08778a2bdbf4c08b426e8a3767e1f0801388b51b></a>

[^77]: Pairwise Comparisons Are All You Need, <a href="https://arxiv.org/html/2403.09746v2" target="_blank"><https://arxiv.org/html/2403.09746v2></a>

[^78]: Comparative Evaluation of Multimodal Large Language ..., <a href="https://www.mdpi.com/2504-2289/9/5/132" target="_blank"><https://www.mdpi.com/2504-2289/9/5/132></a>

[^79]: No-reference image quality assessment based on global ..., <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11457998/" target="_blank"><https://pmc.ncbi.nlm.nih.gov/articles/PMC11457998/></a>

[^80]: Image quality assessment based on multi-scale ..., <a href="https://ieeexplore.ieee.org/iel8/6287639/6514899/10845785.pdf" target="_blank"><https://ieeexplore.ieee.org/iel8/6287639/6514899/10845785.pdf></a>

[^81]: Teacher-Guided Learning for Blind Image Quality ..., <a href="https://openaccess.thecvf.com/content/ACCV2022/papers/Chen_Teacher-Guided_Learning_for_Blind_Image_Quality_Assessment_ACCV_2022_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/ACCV2022/papers/Chen_Teacher-Guided_Learning_for_Blind_Image_Quality_Assessment_ACCV_2022_paper.pdf></a>

[^82]: No-Reference Image Quality Assessment via Transformers, ..., <a href="https://www.researchgate.net/publication/358635020_No-Reference_Image_Quality_Assessment_via_Transformers_Relative_Ranking_and_Self-Consistency" target="_blank"><https://www.researchgate.net/publication/358635020_No-Reference_Image_Quality_Assessment_via_Transformers_Relative_Ranking_and_Self-Consistency></a>

[^83]: A Survey Deep Learning Based Image Quality Assessment, <a href="https://www.sciencedirect.com/science/article/pii/S1877050923008384/pdf?md5=7da01bf8470082391455c7d509db98cf&pid=1-s2.0-S1877050923008384-main.pdf" target="_blank"><https://www.sciencedirect.com/science/article/pii/S1877050923008384/pdf?md5=7da01bf8470082391455c7d509db98cf&pid=1-s2.0-S1877050923008384-main.pdf></a>

[^84]: zwx8981/LIQE - A Multitask Learning Perspective, <a href="https://github.com/zwx8981/LIQE" target="_blank"><https://github.com/zwx8981/LIQE></a>

[^85]: arXiv:2303.14968v1 \[cs.CV] 27 Mar 2023, <a href="https://arxiv.org/pdf/2303.14968" target="_blank"><https://arxiv.org/pdf/2303.14968></a>

[^86]: pyiqa.archs.liqe\_arch - pyiqa 0.1.13 documentation, <a href="https://iqa-pytorch.readthedocs.io/en/stable/autoapi/pyiqa/archs/liqe_arch/index.html" target="_blank"><https://iqa-pytorch.readthedocs.io/en/stable/autoapi/pyiqa/archs/liqe_arch/index.html></a>

[^87]: Exploring CLIP for Image Look & Feel, <a href="https://www.emergentmind.com/articles/2207.12396" target="_blank"><https://www.emergentmind.com/articles/2207.12396></a>

[^88]: Interpretable Image Quality Assessment via CLIP with ..., <a href="https://arxiv.org/pdf/2308.13094" target="_blank"><https://arxiv.org/pdf/2308.13094></a>

[^89]: Quality-Aware Image-Text Alignment for Real-World ..., <a href="https://www.alphaxiv.org/ko/overview/2403.11176v1" target="_blank"><https://www.alphaxiv.org/ko/overview/2403.11176v1></a>

[^90]: (PDF) ZEN-IQA: Zero-Shot Explainable and No-Reference ..., <a href="https://www.researchgate.net/publication/380746671_ZEN-IQA_Zero-Shot_Explainable_and_No-Reference_Image_Quality_Assessment_with_Vision_Language_Model" target="_blank"><https://www.researchgate.net/publication/380746671_ZEN-IQA_Zero-Shot_Explainable_and_No-Reference_Image_Quality_Assessment_with_Vision_Language_Model></a>

[^91]: \[PDF] Meta Learning for Blind Image Quality Assessment via Adaptive ..., <a href="https://papers.ssrn.com/sol3/Delivery.cfm/46de60bb-d0d2-4f55-bf4d-ce23e653f292-MECA.pdf?abstractid=4085273&mirid=1" target="_blank"><https://papers.ssrn.com/sol3/Delivery.cfm/46de60bb-d0d2-4f55-bf4d-ce23e653f292-MECA.pdf?abstractid=4085273&mirid=1></a>

[^92]: \[PDF] Blind Image Quality Assessment via Vision-Language ..., <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Blind_Image_Quality_Assessment_via_Vision-Language_Correspondence_A_Multitask_Learning_CVPR_2023_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Blind_Image_Quality_Assessment_via_Vision-Language_Correspondence_A_Multitask_Learning_CVPR_2023_paper.pdf></a>

[^93]: \[PDF] BIQABSF: Blind Image Quality Assessment Based on Statistical ..., <a href="https://www.researchsquare.com/article/rs-7860040/latest.pdf" target="_blank"><https://www.researchsquare.com/article/rs-7860040/latest.pdf></a>

[^94]: End-to-End Blind Image Quality Assessment Using Deep Neural ..., <a href="https://www.researchgate.net/publication/321090934_End-to-End_Blind_Image_Quality_Assessment_Using_Deep_Neural_Networks" target="_blank"><https://www.researchgate.net/publication/321090934_End-to-End_Blind_Image_Quality_Assessment_Using_Deep_Neural_Networks></a>

[^95]: An illustration of model agnostic explainability methods applied to ..., <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10187774/" target="_blank"><https://pmc.ncbi.nlm.nih.gov/articles/PMC10187774/></a>

[^96]: Quality Assessment Models Must Integrate Context, Reasoning, and ..., <a href="https://arxiv.org/html/2505.19696v1" target="_blank"><https://arxiv.org/html/2505.19696v1></a>

[^97]: Proceedings of the Thirty-Ninth AAAI Conference on Artificial ..., <a href="https://dl.acm.org/doi/proceedings/10.5555/3750840?tocHeading=heading6" target="_blank"><https://dl.acm.org/doi/proceedings/10.5555/3750840?tocHeading=heading6></a>

[^98]: \[PDF] RAPIQUE: Rapid and Accurate Video Quality Prediction of User ..., <a href="https://live.ece.utexas.edu/publications/2021/RAPIQUE.pdf" target="_blank"><https://live.ece.utexas.edu/publications/2021/RAPIQUE.pdf></a>

[^99]: Blind quality assessment of authentically distorted images, <a href="https://opg.optica.org/josaa/abstract.cfm?uri=josaa-39-6-B1" target="_blank"><https://opg.optica.org/josaa/abstract.cfm?uri=josaa-39-6-B1></a>

[^100]: \[PDF] Blind Image Quality Assessment Using Local Consistency Aware ..., <a href="https://kedema.org/paper/18_TCSVT_Wu.pdf" target="_blank"><https://kedema.org/paper/18_TCSVT_Wu.pdf></a>

[^101]: TuningIQA: Fine-Grained Blind Image Quality Assessment for ... - arXiv, <a href="https://arxiv.org/html/2508.17965v1" target="_blank"><https://arxiv.org/html/2508.17965v1></a>

[^102]: MCN: A Mixture Capsule Network for Authentic Blind Image Quality ..., <a href="https://www.sciencedirect.com/science/article/abs/pii/S0950705125018787" target="_blank"><https://www.sciencedirect.com/science/article/abs/pii/S0950705125018787></a>

[^103]: Blind Image Quality Assessment with Deep Learning: A Replicability ..., <a href="https://www.mdpi.com/2076-3417/13/1/59" target="_blank"><https://www.mdpi.com/2076-3417/13/1/59></a>

[^104]: \[PDF] Blind Image Quality Assessment Using Joint Statistics of Gradient ..., <a href="https://www4.comp.polyu.edu.hk/~cslzhang/paper/BIQA_GM-LOG_final-dc.pdf" target="_blank"><https://www4.comp.polyu.edu.hk/~cslzhang/paper/BIQA_GM-LOG_final-dc.pdf></a>

[^105]: Blind Image Quality Assessment Using Center-Surround Mechanism, <a href="https://dl.acm.org/doi/10.1145/3177404.3177425" target="_blank"><https://dl.acm.org/doi/10.1145/3177404.3177425></a>

[^106]: Blind Image Quality Assessment for Authentic ... - -ORCA, <a href="https://orca.cardiff.ac.uk/150369/1/Blind_IQA_print.pdf" target="_blank"><https://orca.cardiff.ac.uk/150369/1/Blind_IQA_print.pdf></a>

[^107]: A Blind Image Quality Index for Synthetic and Authentic ..., <a href="https://www.mdpi.com/2076-3417/13/6/3591" target="_blank"><https://www.mdpi.com/2076-3417/13/6/3591></a>

[^108]: Semi-Supervised Blind Quality Assessment with ..., <a href="https://openreview.net/pdf/82798844eaa974730eed78579767954e356d8423.pdf" target="_blank"><https://openreview.net/pdf/82798844eaa974730eed78579767954e356d8423.pdf></a>

[^109]: Blind Image Quality Assessment With Active Inference, <a href="https://ieeexplore.ieee.org/document/9376644/" target="_blank"><https://ieeexplore.ieee.org/document/9376644/></a>

[^110]: SGIQA: Semantic-Guided No-Reference Image Quality Assessment, <a href="https://ieeexplore.ieee.org/document/10679236/" target="_blank"><https://ieeexplore.ieee.org/document/10679236/></a>

[^111]: How Does Image Content Affect the Added Value of Visual Attention in Objective Image Quality Assessment?, <a href="http://ieeexplore.ieee.org/document/6423792/" target="_blank"><http://ieeexplore.ieee.org/document/6423792/></a>

[^112]: No-reference image quality assessment based on global and local content perception, <a href="http://ieeexplore.ieee.org/document/7805544/" target="_blank"><http://ieeexplore.ieee.org/document/7805544/></a>

[^113]: Deep Neural Networks for Blind Image Quality Assessment: Addressing the Data Challenge, <a href="https://arxiv.org/pdf/2109.12161" target="_blank"><https://arxiv.org/pdf/2109.12161></a>

[^114]: Collaborative Auto-encoding for Blind Image Quality Assessment, <a href="https://arxiv.org/pdf/2305.14684" target="_blank"><https://arxiv.org/pdf/2305.14684></a>

[^115]: A Lightweight Parallel Framework for Blind Image Quality Assessment, <a href="https://arxiv.org/pdf/2402.12043" target="_blank"><https://arxiv.org/pdf/2402.12043></a>

[^116]: \[PDF] Blind Image Quality Assessment Using A Deep Bilinear ..., <a href="https://ece.uwaterloo.ca/~z70wang/publications/TCSVT_BIQA.pdf" target="_blank"><https://ece.uwaterloo.ca/~z70wang/publications/TCSVT_BIQA.pdf></a>

[^117]: Blind Image Quality Assessment via Multiperspective ..., <a href="https://onlinelibrary.wiley.com/doi/10.1155/2023/4631995" target="_blank"><https://onlinelibrary.wiley.com/doi/10.1155/2023/4631995></a>

[^118]: Scale Guided Hypernetwork for Blind Super-Resolution ..., <a href="https://arxiv.org/abs/2306.02398" target="_blank"><https://arxiv.org/abs/2306.02398></a>

[^119]: (PDF) Scale Guided Hypernetwork for Blind Super- ..., <a href="https://www.researchgate.net/publication/371286763_Scale_Guided_Hypernetwork_for_Blind_Super-Resolution_Image_Quality_Assessment" target="_blank"><https://www.researchgate.net/publication/371286763_Scale_Guided_Hypernetwork_for_Blind_Super-Resolution_Image_Quality_Assessment></a>

[^120]: Content adaptive screen image scaling, <a href="http://ieeexplore.ieee.org/document/7025792/" target="_blank"><http://ieeexplore.ieee.org/document/7025792/></a>

[^121]: Content adaptive screen image scaling, <a href="https://arxiv.org/abs/1510.06093" target="_blank"><https://arxiv.org/abs/1510.06093></a>

[^122]: Blind Image Quality Assessment for MRI with A Deep Three-dimensional content-adaptive Hyper-Network, <a href="https://arxiv.org/abs/2107.06888" target="_blank"><https://arxiv.org/abs/2107.06888></a>

[^123]: A new psychovisual paradigm for image quality assessment: from differentiating distortion types to discriminating quality conditions, <a href="http://link.springer.com/10.1007/s11760-013-0445-2" target="_blank"><http://link.springer.com/10.1007/s11760-013-0445-2></a>

[^124]: Content-Distortion High-Order Interaction for Blind Image Quality Assessment, <a href="https://arxiv.org/pdf/2504.05076" target="_blank"><https://arxiv.org/pdf/2504.05076></a>

[^125]: DEFNet: Multitasks-based Deep Evidential Fusion Network for Blind Image Quality Assessment, <a href="https://arxiv.org/pdf/2507.19418" target="_blank"><https://arxiv.org/pdf/2507.19418></a>

[^126]: Embedding-Driven Data Distillation for 360-Degree IQA With Residual-Aware Refinement, <a href="https://arxiv.org/pdf/2412.12667" target="_blank"><https://arxiv.org/pdf/2412.12667></a>

[^127]: CVPR 2020 Open Access Repository, <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.html" target="_blank"><https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.html></a>

[^128]: \[PDF] Blind Quality Assessment for in-the-Wild Images via Hierarchical ..., <a href="https://duanhuiyu.github.io/files/2022/2022_BMSB_sun.pdf" target="_blank"><https://duanhuiyu.github.io/files/2022/2022_BMSB_sun.pdf></a>

[^129]: Attention integrated hierarchical networks for no-reference image ..., <a href="https://www.sciencedirect.com/science/article/pii/S1047320321002674" target="_blank"><https://www.sciencedirect.com/science/article/pii/S1047320321002674></a>

[^130]: Content-Distortion High-Order Interaction for Blind Image Quality ..., <a href="https://arxiv.org/html/2504.05076v1" target="_blank"><https://arxiv.org/html/2504.05076v1></a>

[^131]: CSPP-IQA: a multi-scale spatial pyramid pooling-based ... - NIH, <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9573815/" target="_blank"><https://pmc.ncbi.nlm.nih.gov/articles/PMC9573815/></a>

[^132]: DACNN: Blind Image Quality Assessment via a Distortion-Aware ..., <a href="https://www.researchgate.net/publication/361860571_DACNN_Blind_Image_Quality_Assessment_via_A_Distortion-Aware_Convolutional_Neural_Network" target="_blank"><https://www.researchgate.net/publication/361860571_DACNN_Blind_Image_Quality_Assessment_via_A_Distortion-Aware_Convolutional_Neural_Network></a>

[^133]: \[PDF] Towards Distortion-Debiased Blind Image Quality Assessment, <a href="https://openreview.net/pdf?id=6ko0tOQllI" target="_blank"><https://openreview.net/pdf?id=6ko0tOQllI></a>

[^134]: Deep Superpixel-Based Network For Blind Image Quality Assessment, <a href="https://www.researchsquare.com/article/rs-970679/v1" target="_blank"><https://www.researchsquare.com/article/rs-970679/v1></a>

[^135]: Deep Superpixel-based Network for Blind Image Quality Assessment, <a href="https://arxiv.org/abs/2110.06564" target="_blank"><https://arxiv.org/abs/2110.06564></a>

[^136]: Deep Blind Image Quality Assessment Using Dynamic Neural Model with Dual-order Statistics, <a href="https://ieeexplore.ieee.org/document/10416247/" target="_blank"><https://ieeexplore.ieee.org/document/10416247/></a>

[^137]: Deep Neural Network for Blind Visual Quality Assessment of 4K Content, <a href="https://arxiv.org/abs/2206.04363" target="_blank"><https://arxiv.org/abs/2206.04363></a>

[^138]: A Survey on Vision Transformer, <a href="https://ieeexplore.ieee.org/document/9716741/" target="_blank"><https://ieeexplore.ieee.org/document/9716741/></a>

[^139]: A Survey on Vision Transformer, <a href="https://arxiv.org/abs/2012.12556" target="_blank"><https://arxiv.org/abs/2012.12556></a>

[^140]: \[PDF] MANIQA: Multi-Dimension Attention Network for No-Reference ..., <a href="https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.pdf></a>

[^141]: Multi-dimension Attention Network for No-Reference Image Quality ..., <a href="https://arxiv.org/abs/2204.08958" target="_blank"><https://arxiv.org/abs/2204.08958></a>

[^142]: \[CVPRW 2022] MANIQA: Multi-dimension Attention Network for No ..., <a href="https://github.com/IIGROUP/MANIQA" target="_blank"><https://github.com/IIGROUP/MANIQA></a>

[^143]: (PDF) MANIQA: Multi-dimension Attention Network for No-Reference ..., <a href="https://www.researchgate.net/publication/360062619_MANIQA_Multi-dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment" target="_blank"><https://www.researchgate.net/publication/360062619_MANIQA_Multi-dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment></a>

[^144]: Blind Image Quality Assessment via Adaptive Graph Attention, <a href="https://dl.acm.org/doi/abs/10.1109/TCSVT.2024.3405789" target="_blank"><https://dl.acm.org/doi/abs/10.1109/TCSVT.2024.3405789></a>

[^145]: \[PDF] Perceptual Image Quality Assessment With Transformers, <a href="https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Cheon_Perceptual_Image_Quality_Assessment_With_Transformers_CVPRW_2021_paper.pdf" target="_blank"><https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Cheon_Perceptual_Image_Quality_Assessment_With_Transformers_CVPRW_2021_paper.pdf></a>

[^146]: Data-efficient image quality assessment with attention-panel decoder, <a href="https://dl.acm.org/doi/10.1609/aaai.v37i2.25302" target="_blank"><https://dl.acm.org/doi/10.1609/aaai.v37i2.25302></a>

[^147]: ASCAM-Former: Blind image quality assessment based on adaptive ..., <a href="https://www.sciencedirect.com/science/article/pii/S0957417422022862" target="_blank"><https://www.sciencedirect.com/science/article/pii/S0957417422022862></a>

[^148]: ch-andrei/VTAMIQ: Full-Reference Vision Transformer (ViT) - GitHub, <a href="https://github.com/ch-andrei/VTAMIQ" target="_blank"><https://github.com/ch-andrei/VTAMIQ></a>

[^149]: Attention Down-Sampling Transformer, Relative Ranking and Self-Consistency for Blind Image Quality Assessment, <a href="https://arxiv.org/pdf/2409.07115" target="_blank"><https://arxiv.org/pdf/2409.07115></a>

[^150]: Local Distortion Aware Efficient Transformer Adaptation for Image Quality Assessment, <a href="https://arxiv.org/pdf/2308.12001" target="_blank"><https://arxiv.org/pdf/2308.12001></a>

[^151]: Perceptual Image Quality Assessment with Transformers, <a href="https://arxiv.org/pdf/2104.14730" target="_blank"><https://arxiv.org/pdf/2104.14730></a>

[^152]: Ascam-Former: Blind Image Quality Assessment Based on Adaptive Spatial & Channel Attention, <a href="https://www.ssrn.com/abstract=4215685" target="_blank"><https://www.ssrn.com/abstract=4215685></a>

[^153]: ASCAM-Former: Blind image quality assessment based on adaptive spatial & channel attention merging transformer and image to patch weights sharing, <a href="https://www.sciencedirect.com/science/article/pii/S0957417422022862?dgcid=rss_sd_all&" target="_blank"><https://www.sciencedirect.com/science/article/pii/S0957417422022862?dgcid=rss_sd_all&></a>

[^154]: Blind Image Quality Assessment Via Cross-View Consistency, <a href="https://ieeexplore.ieee.org/document/9961939/" target="_blank"><https://ieeexplore.ieee.org/document/9961939/></a>

[^155]: Data-Efficient Image Quality Assessment with Attention-Panel Decoder, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/25302" target="_blank"><https://ojs.aaai.org/index.php/AAAI/article/view/25302></a>

[^156]: Data-Efficient Image Quality Assessment with Attention-Panel Decoder, <a href="https://arxiv.org/abs/2304.04952" target="_blank"><https://arxiv.org/abs/2304.04952></a>

[^157]: Blind CT Image Quality Assessment Using DDPM-derived Content and Transformer-based Evaluator, <a href="https://arxiv.org/abs/2310.03118" target="_blank"><https://arxiv.org/abs/2310.03118></a>

[^158]: Feature Denoising Diffusion Model for Blind Image Quality Assessment, <a href="https://arxiv.org/abs/2401.11949" target="_blank"><https://arxiv.org/abs/2401.11949></a>

[^159]: Multi-Modal Prompt Learning on Blind Image Quality Assessment, <a href="https://arxiv.org/html/2404.14949v2" target="_blank"><https://arxiv.org/html/2404.14949v2></a>

[^160]: Quality-Aware CLIP for Blind Image Quality Assessment, <a href="https://dl.acm.org/doi/10.1007/978-981-99-8537-1_32" target="_blank"><https://dl.acm.org/doi/10.1007/978-981-99-8537-1_32></a>

[^161]: Blind Image Quality Assessment via Vision-Language ... - Liner, <a href="https://liner.com/review/blind-image-quality-assessment-via-visionlanguage-correspondence-multitask-learning-perspective" target="_blank"><https://liner.com/review/blind-image-quality-assessment-via-visionlanguage-correspondence-multitask-learning-perspective></a>

[^162]: Revisiting Vision Language Foundations for No-Reference Image Quality Assessment, <a href="https://arxiv.org/pdf/2509.17374" target="_blank"><https://arxiv.org/pdf/2509.17374></a>

[^163]: Building Reasonable Inference for Vision-Language Models in Blind Image Quality Assessment, <a href="https://arxiv.org/pdf/2512.09555" target="_blank"><https://arxiv.org/pdf/2512.09555></a>

[^164]: Vision-Language Consistency Guided Multi-modal Prompt Learning for Blind AI Generated Image Quality Assessment, <a href="https://arxiv.org/abs/2406.16641" target="_blank"><https://arxiv.org/abs/2406.16641></a>

[^165]: Vision-Language Consistency Guided Multi-Modal Prompt Learning for Blind AI Generated Image Quality Assessment, <a href="https://ieeexplore.ieee.org/document/10574854?" target="_blank"><https://ieeexplore.ieee.org/document/10574854>?</a>

[^166]: Hybrid No-Reference Quality Assessment for Surveillance Images, <a href="https://www.mdpi.com/2078-2489/13/12/588" target="_blank"><https://www.mdpi.com/2078-2489/13/12/588></a>

[^167]: QMamba: On First Exploration of Vision Mamba for Image ..., <a href="https://arxiv.org/html/2406.09546v2" target="_blank"><https://arxiv.org/html/2406.09546v2></a>

[^168]: UNDERSTANDING THE GENERALIZATION OF BLIND IM, <a href="https://openreview.net/pdf?id=hzxvMqYYMA" target="_blank"><https://openreview.net/pdf?id=hzxvMqYYMA></a>

[^169]: Deep Learning for No-Reference Image Quality Assessment, <a href="https://www.scitepress.org/Papers/2025/135977/135977.pdf" target="_blank"><https://www.scitepress.org/Papers/2025/135977/135977.pdf></a>

[^170]: MCTINet: Blind Image Quality Assessment Using Meta ..., <a href="https://www.researchgate.net/publication/398802926_MCTINet_Blind_Image_Quality_Assessment_Using_Meta-learning_Convolution_Transformer_Integration_Network" target="_blank"><https://www.researchgate.net/publication/398802926_MCTINet_Blind_Image_Quality_Assessment_Using_Meta-learning_Convolution_Transformer_Integration_Network></a>

[^171]: Blind Image Quality Assessment via Transformer Predicted Error Map and Perceptual Quality Token, <a href="https://arxiv.org/pdf/2305.09353" target="_blank"><https://arxiv.org/pdf/2305.09353></a>

[^172]: VTAMIQ: Transformers for Attention Modulated Image Quality Assessment, <a href="https://arxiv.org/pdf/2110.01655" target="_blank"><https://arxiv.org/pdf/2110.01655></a>

