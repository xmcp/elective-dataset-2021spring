# elective-dataset-2021spring

某学校2021春季选课系统GIF验证码数据集（29338张） + 准确率98.4%的Baseline模型 + 上下游相关工具。

数据集采用 <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">知识共享署名-非商业性使用 4.0 国际许可协议</a> 进行许可。

Baseline模型和上下游相关工具采用 [MIT License](https://mit-license.org/) 进行许可。



## 数据集

`dataset/` 目录包含了收集到的所有带标签验证码数据，共29338张。

- `dataset/manual`：
  人工标注的带标签验证码GIF数据集，标签经过了elective验证因此都是正确的。共5471张。
- `dataset/auto-corr`、`dataset/auto-fail-tagged`：
  模型自动标注的带标签验证码GIF数据集，其中 `auto-corr` 是识别正确（通过了elective验证）的部分，`auto-fail-tagged` 是识别错误然后手工重新标注的部分（此部分不保证正确性）。共22931（正确）+936（错误）张。

使用时请注意，由于 [GitHub 的限制](https://docs.github.com/en/github/managing-large-files/what-is-my-disk-quota)：

- `auto-fail-tagged` 在仓库中存储为7-zip压缩包；
- `manual` 在仓库中存储为7个不超过48MB的7-zip分卷；
- `auto-corr` 没有存储在仓库中，而是压缩为14个不超过95MB的7-zip分卷放在了 [Release页面](https://github.com/xmcp/elective-dataset-2021spring/releases)。



## Baseline 模型

`baseline/` 目录包含一个简易的验证码识别模型。

此模型进行了提取关键帧、基于OpenCV的图像增强以及基于CNN的分类器等一系列工作以完成识别。

将训练集和测试集图片分别放入 `set-train` 和 `set-test` 后运行 `train.py` 进行训练，用一块TITAN RTX训练需要几分钟的时间。

用大约一万张图片训练好的 `checkpoints/model_29.pth` 能达到 98.4% 的整体精确度。

`predict_bootstrap.py` 在elective系统上测试当前模型，将检验正确的带标签图片放入 `bootstrap_img_succ` 目录，错误的图片放入 `bootstrap_img_fail` 目录。



## 上下游相关工具

- `crawl/`：
  验证码众包标注平台，可以从elective爬取验证码、辅助多名用户同时标注、检验正确性后将正确的数据放入 `img_correct` 目录。检验错误的验证码将被抛弃，这是初期的一个设计失误，这样将使得数据集的分布与真实分布有偏差。
- `retag/`：
  手工标注模型识别错误数据的工具。从 `bootstrap_img_fail` 读取标注错误图片，人工输入正确标注后移动到 `bootstrap_img_fail_tagged`。
- `serve/`：
  提供在线验证码识别服务的 HTTP RPC 服务器。`POST /fire` 并传入base64编码的验证码GIF来进行识别。



## 数据处理过程

首先，我们设立了众包标注平台，多名志愿者累计标注了超过五千张验证码。

有了这些数据后，我们利用OpenCV进行了简单的图片增强、二值化、分字、裁切，然后随手糊了一个简单的CNN网络来识别。在随意调参之后，模型的整体（四个字）准确率接近95%。

然后，我们利用此模型来对数据集进行自举：爬取验证码后调用模型识别然后检验正确性，其中识别错误的部分手工标注。这样我们可以轻易地扩大数据集的规模，从而提升模型效果。

经过了更多的随意调参，模型的整体准确率可以达到98.4%。因为继续提升准确率意义不大，就没有继续优化。考虑到 PyTorch 安装比较麻烦，模型不易于部署到用户的设备上，我们实现了一个 HTTP API 可以用于云端识别。



by *Elector Quartet* (按字典序 *@gzz, @Rabbit, @SpiritedAwayCN, **@xmcp***)

