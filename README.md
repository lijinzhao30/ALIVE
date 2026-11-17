<p align="center">
  <img src="docs/assets/logo.png" alt="ALIVE" width="210">
</p>

<h1 align="center">ALIVE: Making Streaming Video Models Interactive</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2611.21366"><img src="docs/assets/buttons/paper.png" alt="Paper — arXiv:2611.21366" height="42"></a>
  <a name="model-coming-soon"><img src="docs/assets/buttons/model.png" alt="Hugging Face Model — coming soon" height="42"></a>
  <a name="dataset-coming-soon"><img src="docs/assets/buttons/dataset.png" alt="Hugging Face Dataset — coming soon" height="42"></a>
  <a href="https://lijinzhao30.github.io/ALIVE/"><img src="docs/assets/buttons/website.png" alt="Project page" height="42"></a>
  <a href="https://lijinzhao30.github.io/ALIVE/#demo"><img src="docs/assets/buttons/demo.png" alt="Watch the demo" height="42"></a>
</p>

<p align="center"><b>Continuously perceive. Coordinate evolving tasks. Seek the feedback needed to proceed.</b></p>

<p align="center">
  Jinzhao Li<sup>1,3,*</sup> · Yuan Xu<sup>2,3,*</sup> · Jinxuan Wu<sup>4</sup> · Yukun Wang<sup>1</sup> · Yijia Lei<sup>1</sup> · Hanyu Ji<sup>2</sup> · Yinuo Chen<sup>1</sup> · Zining Huang<sup>1</sup><br>
  Chi Zhang<sup>1</sup> · Hang Ye<sup>2</sup> · Yiwu Zhong<sup>2</sup> · Honglei Yan<sup>3</sup> · Panwang Pan<sup>3</sup> · Yizhou Wang<sup>2</sup> · Yin Li<sup>5</sup> · Miao Liu<sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup> Tsinghua University &nbsp; <sup>2</sup> Peking University &nbsp; <sup>3</sup> ByteDance<br>
  <sup>4</sup> Beihang University &nbsp; <sup>5</sup> University of Wisconsin–Madison<br>
  <sub>* Equal contribution &nbsp; † Corresponding author</sub>
</p>

## Demo

Watch ALIVE demonstrate **Streaming Task Coordination** and **Feedback-Seeking Interaction** in English and Chinese. Click the preview to watch the full narrated demo.

<a href="https://lijinzhao30.github.io/ALIVE/#demo"><img src="docs/assets/demo-preview.gif" alt="ALIVE demo preview: continuous video and a live conversation. Click to play the complete 4:38 demo with sound." width="100%"></a>

<p align="center"><a href="https://lijinzhao30.github.io/ALIVE/#demo"><b>▶ Watch the full demo · 4:38 · English + 中文</b></a> &nbsp; · &nbsp; <a href="https://raw.githubusercontent.com/lijinzhao30/ALIVE/main/docs/assets/alive-demo.mp4">Download MP4</a></p>

## Toward interactive streaming

Streaming video models can continuously observe the world, yet often handle user requests in isolation. **ALIVE** moves toward sustained interaction: it responds at the right moment, coordinates evolving tasks, and actively seeks missing information or visual evidence.

| 🔵 Proactive Monitoring | 🟣 Streaming Task Coordination | 🟠 Feedback-Seeking Interaction |
| --- | --- | --- |
| Monitor unfolding events and respond when the requested condition is met. | Maintain multiple requests and update active tasks as instructions change. | Ask for clarification, request a concrete user action, or propose continued monitoring when more evidence is needed. |

<a href="docs/assets/figures/teaser.pdf"><img src="docs/assets/figures/teaser.png" alt="ALIVE paper overview: proactive monitoring, evolving task coordination, and feedback-seeking interaction along a continuous stream." width="100%"></a>

ALIVE brings together a new interaction setting, large-scale training data, and a two-stage learning framework. It achieves the **highest overall scores among the models compared in the paper on all five benchmarks**, covering both general streaming video understanding and streaming interaction.

## Benchmark results

The comparisons below use the paper's reported overall scores for **ALIVE, JoyAI-VL-Interaction, AURA, and VideoChat3**. Each chart starts at zero; higher is better. FSI-Bench reports the **Explicit Decision Protocol (EDP)**.

<a href="docs/assets/benchmarks-understanding.png"><img src="docs/assets/benchmarks-understanding.gif" alt="General streaming video understanding: ALIVE scores 68.9 on OVO-Bench, 76.0 on StreamingBench, and 51.6 on ProactiveVideoQA." width="100%"></a>

<a href="docs/assets/benchmarks-interaction.png"><img src="docs/assets/benchmarks-interaction.gif" alt="Streaming interaction: ALIVE scores 40.3 on IPIBench and 52.8 on FSI-Bench EDP." width="100%"></a>

OVO-Bench, StreamingBench, and IPIBench report overall accuracy (%). ProactiveVideoQA reports PAUC at ω = 0.5. FSI-Bench reports four-task macro accuracy (%) under EDP.

## Learning to interact

ALIVE is trained in two stages. **Contrastive supervised fine-tuning** teaches when and how to respond. **Reinforcement learning with Behavior-Aligned Credit Assignment** combines trajectory-level and local advantages to refine strategy selection, task responses, and critical silence.

<a href="docs/assets/figures/method.pdf"><img src="docs/assets/figures/method.png" alt="The ALIVE training framework: contrastive supervised fine-tuning followed by reinforcement learning with Behavior-Aligned Credit Assignment." width="100%"></a>
