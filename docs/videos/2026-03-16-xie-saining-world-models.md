# Xie Saining: World Models, Escape from Silicon Valley, AMI Labs, Twice Rejecting Ilya, LeCun, Fei-Fei Li, and 42

**Source**: https://youtu.be/iiBY0fqpThI
**Channel**: Zhang Xiaojun Podcast (张小珺商业访谈录)
**Published**: 2026-03-16
**Duration**: 06:45:29
**Watched on**: 2026-05-14
**Sectioning**: chapters (15 YouTube-supplied chapters; chapter 1 was `<Untitled Chapter 1>`, renamed to "开场白与本期定位" in the section JSON before chunking)
**Detected video language**: `zh` (no `info.json.language`; both `subtitles` and `automatic_captions` keys empty — no YouTube subs)
**Transcript source**: faster-whisper large-v3 (CPU int8) — produced by `docs/videos/transcribe_batch.py` from local audio. ASR consistently rendered the host as "小骏/小俊" — corrected from channel metadata to **张小珺 (Zhang Xiaojun)**. Guest name rendered "谢赛明" in places — corrected to **谢赛宁 (Xie Saining)**. Yann LeCun appears as "杨立坤/杨立昆/杨乐空/样"; Ilya Sutskever as "伊莱亚/伊丽娅/Elia". JEPA appears as "japa/了japa".
**Speakers**: Zhang Xiaojun (张小珺, host), Xie Saining (谢赛宁, guest — NYU CS associate professor, co-founder & Chief Science Officer of AMI Labs, vision / world-model researcher)

## TL;DR

- A 7-hour marathon interview structured as a research-life arc: SJTU ACM class → NUS undergrad RA → UCSD PhD with Zhuowen Tu (DSN + HED) → FAIR with Kaiming He (ResNeXt → MoCo → MAE) → NYU under LeCun's recruiting → 2024 Cambrian / Vi-STaR / REPA / RAE → co-founding **AMI Labs** with **Yann LeCun** in 2025. Self-frames as "**the normal one**" (Klopp's line, not Mourinho's) all the way through.
- **Twice rejected Ilya Sutskever**: 2018 OpenAI offer (Ilya phone-called angrily — "你为什么不讨论一下就把这个 offer 拒了") and 2024 SSI invite (the breakpoint was Ilya saying computer vision is "已经解决的很不错了"). Said *yes* to LeCun in mid-2025 — the explanation is "气场是否相投," not pure technical alignment.
- **AMI Labs day-one shape**: ~$1B target raise (Xie says it ranks among the largest seed rounds in history), pre-money ~$1B, ~25 people, six co-founders, four offices on day one (**Paris HQ, New York, Montreal, Singapore**); LeCun = Executive Chairman, Xie = CSO, separate CEO from Meta's Southern-Europe VP. Explicitly *not* in Silicon Valley because "**硅谷被催眠了**" by the LLM scaling consensus.
- **Technical thesis: LM 终将凋零** ("LMs will not die, they will just fade away"). Representation, not language, is the foundation of a world model. Cake metaphor (LeCun): SSL = body, supervised = icing, RL = cherry on top. Language is "拐杖 / 鸦片" — a crutch / opiate that prevents the leg muscles (visual representation) from training. The roadmap: pre-train a **multimodal world-model base**, then have LM / video-diffusion / action / planning / robotics as downstream "hammers" on top.
- **Two macro reframings** that close the episode: (1) Era 2 of AI data is "**download humans**" — a 4-month-old has seen more video than 30T LLM training tokens; YouTube + always-on AI glasses + robotics are the natural channels. (2) **AGI is a 伪命题** — human intelligence is a tiny specialized slice of the ~2^(2M) functions a retina with 2M nerve fibers could encode; **a squirrel that survives in the real world with hunger, emotion, and social behavior is a harder problem than IMO/IOI gold or going to Mars**. The closer: "**42**."

## Why it matters

The single longest and most personal podcast Saining has ever done — a documentary-grade walk through how 2010s CV came to its current crossroads from someone who was a load-bearing author on **ResNeXt, MoCo, MAE, PointContrast, ConvNeXt, DiT, SiT, Cambrian, Vi-STaR, REPA**, and **RAE**. It is also the most explicit on-mic statement of the **anti-LLM-consensus world-model thesis** the AMI Labs founding team is betting ~$1B on, with concrete operating details: a "reverse OpenAI" data flywheel via a grassroots alliance (factories, hospitals, agriculture — airplane engines with 1000 sensors), a 60-70% new-lab / 20-30% free-research split, and Paris as the deliberate non-Valley HQ. Episodes #133 in Zhang Xiaojun's series; pairs naturally with the Yao Shunyu and Su Yu episodes as the third panel of an academic-to-startup triptych.

## Section summaries

### §1 — 开场白与本期定位  [[00:00 → 01:19](https://youtu.be/iiBY0fqpThI?t=0)]

- Host 张小珺 opens from snowy New York during Chinese New Year, framing the episode as a marathon interview recorded in the coldest winter in recent years.
- Guest introduced as 华人青年科学家 **谢赛宁 (Saining Xie)**, who has just embarked on entrepreneurship with Turing Award winner **杨立昆 (Yann LeCun)**; their new lab "**Neolab AMI**" (AMI Labs) just closed "超大规模" first round of financing; team currently 25 people.
- Sets the humility motif Saining repeated to her in pre-show: "他不是那个**天选之子**,他是**普通的那一个**" — `the normal one`, not the chosen one. Foreshadows §15's Klopp / "normal one" recapitulation.
- Two cold-open teasers: (a) the **Ilya Sutskever** angry phone call after Saining rejected the OpenAI offer ("他们发给我一个 offer,然后说我不去抱歉"), and (b) the duality framing "**有爱的同时一定就有恨,他是一体两面**" — love and hate are two sides of one thing.

### §2 — The normal one  [[01:19 → 35:40](https://youtu.be/iiBY0fqpThI?t=79)]

- Has never done a podcast before; only now has "**不被人喜欢的勇气**" (courage to not be liked) to do one. ~13 years in the US — "我这个后训练现在有点崩" (my Chinese-English post-training is breaking down).
- Family — father a psychology major turned media/TV worker, "纯粹的私宅" who filled the house with books; mother ran a business and took young Xie traveling across China. No STEM in the family — credits this loose humanities-leaning upbringing for shaping his "world model."
- **Self-positioning as "B-class"**: contrasted with friends on the canonical A-class lane (top high school → 竞赛金牌 → top undergrad → four-major PhD → professorship). Says his path was full of randomness and 玄学; "智商不够" for the A-class lane.
- Recurring motif: "**这个世界总不让我做我想做的事情,但是我偏偏要做我想做的事情**." Refused gaokao when SJTU offered early admission (teachers said "you should aim for Tsinghua/PKU"); refused the default MSRA internship pipeline.
- **Hero — 侯小迪 (Hou Xiaodi)**, SJTU senior, first mainland undergrad to publish a CVPR paper (a **7-line-of-code** algorithm), author of *交大学生生存手册* — preaches that the purpose of research is exploring the unknown, not 灌水. Quote: "如果一个人把政策评分作为自己的至高追求,那么他就是这个政策的牺牲品."
- Cosmic-coincidence anecdote: in the ACM-class interview, Prof. **沈仁超** quizzed him on the author of *What Is Mathematics* — **Richard Courant** — "remember this name, one of the 20th century's greatest mathematicians." Years later Saining ended up at **NYU's Courant Institute**.
- **Why vision specifically** — childhood thought experiment on which sense he could lose; visual cortex occupies ~30% of cortex but activates ~70% of the brain on visual input; the eye is "**the only part of the brain exposed to the real world**." Cites the Cambrian explosion arms-race theory (~530M years ago).
- ACM class had ~30-40 students; he ranked ~10th-something — never tried to top it; **于勇** (ACM-class founder) designed signature courses like 学子讲坛 where students give 45-60min talks on anything *except* coursework.
- First real paper: **BMVC face-recognition / manifold clustering** under mentor **冯佳时 (Feng Jiashi)** at NUS, 2012 — the same year as **AlexNet**.
- People: [侯小迪](https://youtu.be/iiBY0fqpThI?t=980) [[16:20](https://youtu.be/iiBY0fqpThI?t=980)], [于勇](https://youtu.be/iiBY0fqpThI?t=1269), [沈仁超](https://youtu.be/iiBY0fqpThI?t=837), [Richard Courant](https://youtu.be/iiBY0fqpThI?t=883), [颜水成](https://youtu.be/iiBY0fqpThI?t=1383), [冯佳时](https://youtu.be/iiBY0fqpThI?t=1825), [马毅 / 孙剑 / 何凯明](https://youtu.be/iiBY0fqpThI?t=1306) (MSRA at the time).
- Numbers: 13 years in US; first PC at age 9; Hou's CVPR paper = **7 lines of code**; ACM class ~30-40 students, Xie ~10th; visual cortex ≈ 30%, brain activation ≈ 70%; pre-Cambrian "530 million years ago" no eyes; first paper 2012.

### §3 — 世界总不让我做 Vision  [[35:40 → 52:06](https://youtu.be/iiBY0fqpThI?t=2140)]

- PhD application season disastrous — no CV advisor offers, almost switched to recommender systems / general ML. After the **April 15 deadline** he flood-sent 套词 emails; **涂哲文 (Zhuowen Tu)** replied at the last minute.
- Took a 3 a.m. call from his Shanghai dorm with Tu, pitched himself, got rescued in the final days of admissions. Original offer was UCLA; a week before enrollment Tu told him he was jumping ship to an unknown destination. Saining replied: "**我马上说,我就选择你了 [...] 我觉得我不在意学校 [...] 重要的事情是我跟谁在做什么事情**." Destination turned out to be **UCSD**, then ranked much lower in CS/AI than today.
- The other CV professor he'd hoped to work with — **Serge Belongie** (ASR: "Search Blondie") — was simultaneously leaving UCSD. Saining's principle: bet on people and upside potential, not rankings.
- Pushes back on a Xiaohongshu comment ("谢赛宁在国内表现平平无奇,到了美国一鸣惊人"). Saining's preferred framing: "**不是一瞬间的荷尔蒙或者肾上腺素的爆发,这件事情可能是一个终其一生的一个建设,一种很宁静的一个过程**."
- First PhD paper (2013-14): **Deeply Supervised Nets (DSN)** — auxiliary supervision at multiple branches to mitigate vanishing gradients; thematically a precursor to residual-style designs. Last year DSN won **AISTATS Test of Time**, despite originally being rejected from NeurIPS at a review score of ~886/887 because a Program-Chair-caught squared-term typo killed it after rebuttal.
- Second paper **HED (Holistically-Nested Edge Detection)** — DSN applied to pixel-level edge detection at ICCV; **Marr Prize honorable mention**. "**很不幸,这是我最后一次拿 best paper [...] 每个人都这么祝我了,再也没有得到**."
- Praises Tu's generation — handwrote ~50,000 lines of C++ for image segmentation before PyTorch/GPUs/OSS ML libraries existed; credits Tu's advisor **朱松纯 (Song-Chun Zhu)** and **李飞飞** for "蹚出" the path.
- People: [Zhuowen Tu](https://youtu.be/iiBY0fqpThI?t=2178), [Serge Belongie](https://youtu.be/iiBY0fqpThI?t=2357), [朱松纯](https://youtu.be/iiBY0fqpThI?t=2599), [李飞飞](https://youtu.be/iiBY0fqpThI?t=2602).
- Numbers: April 15 deadline; ~50,000 lines of hand-coded C++; ~5-6 top-venue papers in PhD; NeurIPS rejection score ~886/887; AISTATS Test of Time 10 years later.

### §4 — 学术流浪  [[52:06 → 57:43](https://youtu.be/iiBY0fqpThI?t=3126)]

- **Five PhD internships** across NEC Labs → Adobe → Meta → Google Research → DeepMind. Advisor 陀老师 (Tu) was "exceptionally open-minded — even today this is hard to imagine." Saining drove "南家到北家" 8 hours each summer with two suitcases, subletting his dorm — "**居无定所,这种流浪式的研究员的生活,我还蛮开心的**."
- Two motivations: (1) "走出去看看" curiosity; (2) hedge against "**What if I'm wrong?**" — what if AI/CV isn't the most interesting thing in the world?
- NEC Labs Cupertino was the best memory (Chinese-majority group, lunches together, one CVPR paper); **于凯 (Yu Kai)** had also been there back when NEC was a deep-learning hub.
- **Adobe SF was a flop** — too "artistic"; he worked on mechanical-turk crowdsourced feedback for segmentation. "我就没做好" and still feels guilty toward his mentor. The slump extended into the first two months of his Meta internship — until "突然一个转机发生了" (teeing up §5: Kaiming He arrives).
- Lesson he passes to his students: half of his five internships produced nothing — **"其实也不是 the end of the world."**
- Numbers: 5 internships; 3-6 months each; ~8-hour SoCal↔Bay drive each summer; ~half produced nothing.

### §5 — 与何恺明的友谊  [[57:43 → 1:21:05](https://youtu.be/iiBY0fqpThI?t=3463)]

- **Kaiming He** joined FAIR ~halfway through Saining's internship; Saining's manager handed him off to Kaiming because Saining wasn't producing results. Saining taught Kaiming US logistics and Linux/clusters (Kaiming had only used Windows at MSRA).
- Compares Kaiming's effect on collaborators to **Steve Jobs's "reality distortion field"** — "**凯明他的魔力在于,他能把所有很普通的东西变成一个金字般之前的 idea**."
- In the last month of the internship they entered the **ImageNet Challenge** together — the work became **ResNeXt**. Placed 2nd; Saining argues they were effectively 1st because the winning entry was an ensemble of prior models while ResNeXt was a new framework. Retroactively a precursor to MoE — parallel groups inside each block, sparser + wider at same FLOPs, with a "scaling behavior" observable on ImageNet.
- Kaiming named it **ResNeXt** partly as a pun on **"Xie's ResNet"** — "she's restnet [...] 既是 next 既是下一代的 restnet 也是给了我的一些 credit." Kaiming names most of their papers.
- On Kaiming's advice Saining did **NOT** stay at FAIR after the internship — Kaiming told his interns (including **王小龙 / Wang Xiaolong**) to intern elsewhere. Saining went to Google, then DeepMind in London.
- **DeepMind (London, cold rain, walking home at 10-11pm)** — RL / embodied-agent work; he decided he didn't enjoy RL or robotics, but was deeply struck by DeepMind's org model: **bottom-up exploration that hardens into top-down execution once an idea matures**, plus PMs coordinating research teams. **AlphaFold's team was forming during his internship**.
- **Demis Hassabis** told interns at a Q&A that DeepMind's mission was to become a company that wins **multiple Nobel Prizes** — at the time everyone thought it absurd ("天方夜谭"); "现在我们看到他们已经至少实现了一步."
- **PhD thesis: "Deep Representation Learning with Induced Structural Priors"** — frames representation learning as the tree's root, with classification / segmentation / edge detection / video / action / embodied as branches. Better to dig the root deeper than chase branch-level best papers. Cautionary tale: **NAS** "wasted about two years" of the field.
- 2018 graduation — didn't apply for faculty jobs ("不配"). Interviewed at OpenAI — locked in a room **5-6 hours** doing a single hand-written A4 page of pencil problems by **张舒曼 (Shuman Zhang)**; got the offer, chose FAIR specifically to work with the "**three horsemen of computer vision**": Kaiming He, **Piotr Dollár**, **Ross Girshick**.
- Key quote on careers: "**不要在乎一个 point estimate [...] 因为所有的评价它到最后都会是一个积分,你需要时间的积累**."
- People: [Kaiming He](https://youtu.be/iiBY0fqpThI?t=3465), [Demis Hassabis](https://youtu.be/iiBY0fqpThI?t=4080), [Wang Xiaolong](https://youtu.be/iiBY0fqpThI?t=3921), [张舒曼](https://youtu.be/iiBY0fqpThI?t=4785), [Piotr Dollár](https://youtu.be/iiBY0fqpThI?t=4839), [Ross Girshick](https://youtu.be/iiBY0fqpThI?t=4839).
- Numbers: ImageNet placed 2nd; NeurIPS DSN score ~886/887; OpenAI interview ~5-6 hours; year 2018; FAIR stint = 4 years.

### §6 — 两次拒绝了 Ilya  [[1:21:05 → 1:37:50](https://youtu.be/iiBY0fqpThI?t=4865)]

- **First refusal (2018)**: Saining turned down an OpenAI offer without negotiation; Ilya called him personally, "**非常生气**, 然后他问我说,你为什么不讨论一下就把这个 offer 拒了,是我们给的钱不够吗?" Top-PhD pay at the time was ~**$400-500K** ("now at least 3× that"); money wasn't the issue.
- At that moment FAIR was the "**圣殿**" for top PhDs — more open, more academic, more certain than OpenAI; Saining thinks Ilya was being refused often back then by elite candidates choosing FAIR. Estimates "他 reach out 了一千个人,一万个人吧."
- **FAIR job talk was rocky** — allocated 1 hour, he spoke only 30 minutes because nobody told him the academic job-talk convention; researchers filled time with Q&A to save him. He was among the first fresh-PhD FAIR hires (likely #2 after **陈新雷 / Xinlei Chen**) — FAIR previously only hired established researchers like Kaiming.
- **Second refusal (July 2024)**: right after Ilya founded **SSI**, he emailed inviting collaboration. Saining had just started at NYU. Their walking-the-streets-of-New-York conversation was **philosophical** — main topic: "**怎么样给未来的人工智能给予爱的能力**" — *not* salary.
- **The decisive technical divergence**: Saining asked Ilya about computer vision / general perception; Ilya replied that he considers that problem "**已经解决的很不错了**." Saining reads this as SSI committing to a language-based route that isn't the one he wants to design — framed as "**兄弟爬山各自努力**," not zero-sum.
- **Research philosophy** — invoking **Hannah Arendt**: averse to "impact" as a goal — too aggressive, too male-coded. "**做这些事情的目的不是创造 impact 而是为了理解本身**." The point of publishing is to raise "**地球上智能总量**," not to brand oneself.
- **Anti-fame stance** — calls himself "**某种虚假的 fame 的一个受害者**" because 小红书 reposts label work as "谢赛宁团队"; asks 小编 to stop using his name/photo; tells his students never to self-promote on 小红书 or 知乎.
- Numbers: 2018 top-PhD pay ~$400-500K (Saining misspoke "2008"); two phone calls with Ilya total; FAIR 4 years; Ilya reached out to "1,000 or 10,000 people."

### §7 — 杨立昆和李飞飞往事  [[1:37:50 → 1:58:30](https://youtu.be/iiBY0fqpThI?t=5870)]

- **LeCun recruited Saining three times** — first at FAIR (LeCun was director, no direct co-work), second at NYU, "**第三次我们可以之后再聊**" (foreshadowing AMI).
- LeCun's visionary instinct shows in NYU's **Center for Data Science (CDS)** — set up over a decade ago as an independent unit (not under CS or Math); glass-walled, warm-orange, students-and-robots mixed in the sofa area; the early prototype of every "interdisciplinary AI center" US universities chase today.
- **李飞飞 (Fei-Fei Li)** — got to know her over a New York dinner; she now visits often for **World Labs**, with regular research meetings. Saining recommends her autobiography. His take: her greatest achievement is **NOT** building ImageNet the dataset; it's "**定义问题**" — in 2011/2012 image classification wasn't yet a clearly defined problem; setting the agenda mattered far more than the dataset itself.
- Two Saining ↔ Fei-Fei collaborations on the mic: **Thinking in Space** (multimodal LLM spatial intelligence) and **Cambrian-S** (defining which video questions matter).
- Rejects the host's "**你怎么走进 AI 的核心**" framing — calls it 吸引力法则 + everyone's research roots aligning (LeCun's early digit-recognition was also vision).
- Transition into representation learning: architecture / data / objective triad. **Kaiming told Saining in 2018-2019: "我们要做 scalable 的 model,把模型做大大大"** — the first person to explicitly say *scaling* to him.
- **LeCun's cake metaphor** (the load-bearing intuition for the rest of the episode): "**底层是你的蛋糕的 body,这一部分必须是 self-supervised learning [...] 在上面 supervised learning [...] 是 icing on the cake [...] 然后再往上 reinforcement learning,他只是 cherry on top**." No SSL base, RL cherry → no intelligence.
- 2015-2016 vision community designed dozens of **pretext tasks** for SSL (rotation prediction 0/90/180/270, grayscale colorization, context encoder hole-filling) — all 15-20 percentage points worse than ImageNet supervised pretraining; "**没有一个能打的**."
- Real motivation for SSL is **not** annotation cost (a common misread) — supervised learning forces "**infinite mappings**" into a single label (e.g. "chair" → avocado chair, designer chair, ...), pushing the model to memorize or learn spurious correlations; no path to **common sense**.
- People: [Yann LeCun](https://youtu.be/iiBY0fqpThI?t=5870), [Fei-Fei Li](https://youtu.be/iiBY0fqpThI?t=6108), [Kaiming He](https://youtu.be/iiBY0fqpThI?t=6651).
- Numbers: pretext-task SSL was 15-20 percentage points worse than ImageNet pretraining; ImageNet 1000 classes include ~200 dog breeds.

### §8 — 草蛇灰线:"表征的世界"  [[1:58:30 → 2:43:55](https://youtu.be/iiBY0fqpThI?t=7110)]

- **MoCo (2019)** — first paper to make contrastive learning "really work" for visual representation; key insight = learn a space where same-object views are close, different-object views far. Prior art: **CPC**, **Memory Bank**, LeCun's early metric-learning. Not "横空出世" — a step on a continuum.
- **Kaiming's research style** distilled: extreme focus ("某种极致的专注力"), strong taste, contrarian conviction, engineering ability, and **抽丝剥茧** — extracting the few load-bearing points others miss or refuse to articulate.
- **Kaiming's anti-armchair thesis**: "**这个梯度本身,这件事情,才是你真正的 idea 的来源 [...] 一开始你想的这个 idea,不是你的 idea,这个东西不属于你. 探索中的 idea,才是属于你的 idea.**" An idea you "想出来" sitting in your room is almost never your idea — either you're luckier than 10,000 others (unlikely), racing them on execution speed, or it's a bad idea others already failed on.
- Concrete **6-month research cycle** Saining now teaches NYU students: 1-2mo exploration (hack/play/reproduce baselines) → 2-3mo scaling + experiments → 1-2mo writing. Pivoting mid-cycle is mandatory — "**最差的 research 是什么样的 research? 就是一开始你定义好的一个问题 [...] 最后你发了一篇论文,这个论文的 idea 跟你一开始想的 idea 完全一致.**"
- **Research as SGD** — the gradient itself, not the A→B path, is the real source of ideas. A failing experiment that drops 10 points is *more* informative than a flat one; worst signal = "不好也不差." Kaiming taught Saining to **predict every experiment's result before running it**, so wrong predictions are usable gradients.
- **Bill Freeman's non-linear-game plot** (x = paper quality, y = career impact): flat for bad/mediocre work, vertical jump for breakthroughs. "**Researcher 更像一个发明家. 你这辈子真的只需要成功一次就够了**." Researchers are inventors, not chess players.
- Academia has become a finite game tail-wagged by OpenAI/Google/Meta launches. Saining's Google part-time gig had a counter-intuitive purpose: "**我之所以去 Google 做这个工作,原因是我想看看 Google 大家在做什么 [...] 这样我就知道我在学术界不做什么**" — don't compete on resource-rich axes.
- **Baseline obsession as Kaiming's hidden weapon**: "**你的 research 的上限其实取决于 baseline 的好坏.**" Most students *want* weak baselines for bigger gains; Kaiming wants baselines "高到不能再高" so any further gain is a real breakthrough. Kaiming single-handedly built FAIR's TPU infra (~**5000 TPU cores** rented from Google Cloud, originally for LMs) after the team gave up — **MoCo / MAE / DiT** all ran on it. FAIR-intern first lesson is famously Excel — a hand-designed experiment-tracking spreadsheet whose rows and columns are themselves a research-taste artifact.
- **Contrastive → MAE arc, the humbling**: MoCo v1/v2/v3 + ViT beat ImageNet supervised, "无比光明的未来" — but didn't scale. Pivot to **MAE (masked autoencoder)** — also produced beautiful representations, also didn't scale to LM-level impact. **PointContrast** extended SSL to 3D; methodology was universal across domains but none gave vision its "GPT moment."
- Saining counts ~**20-25 papers** that have "深远地影响了整个深度学习跟 AI 的进程" (LeNet, AlexNet, ImageNet, ResNet, R-CNN, Transformer, GPT-3, BERT, CLIP, ViT, GAN, DDPM, NeRF, Gaussian Splatting...). His own count: **zero**. DiT counts as **0.25** because it pushed the tangent of the research frontier but "wasn't fully his."
- **LM 终将凋零 motif (foreshadowing world models)**: "**LM 永不会死,但终将凋零 — 老兵不死,终将凋零. They won't die, they will just fade away.**" LM is a great daily tool; cannot be the foundation ("地基") of a universal intelligence system.
- Personal coda on Kaiming: not "无所不能的研究机器" — plays WoW and Hearthstone (天梯 higher than Saining), prefers IC over management, lectures PhDs that they hold a "Doctor of Philosophy" yet "**为什么你们培养出来的人一点哲学都不懂呢**." Outside interests: evolutionary biology, physics, quantum, philosophy.
- People: [Kaiming](https://youtu.be/iiBY0fqpThI?t=7118), [Bill Freeman](https://youtu.be/iiBY0fqpThI?t=8023), [Ross Girshick](https://youtu.be/iiBY0fqpThI?t=9302), [Yuxin Wu](https://youtu.be/iiBY0fqpThI?t=9305).
- Numbers: 6-month research cycle; **5000 TPU cores**; ~20-25 era-defining papers, Saining's count 0, DiT = 0.25; FAIR stint 4 years total; MoCo/MAE/PointContrast finished in his first 1-2 years at FAIR.

### §9 — Research taste 与《金刚经》  [[2:43:55 → 4:11:07](https://youtu.be/iiBY0fqpThI?t=9835)] — *the longest section*

- **Day-one gift from Kaiming at FAIR — the Diamond Sutra (金刚经)**, not a research book. Encoded lesson: "**凡所有相皆是虚妄,若见诸相非相,即见如来**." Research taste = breaking through the illusory `相` of paper acceptance, fame, and fast wins to chase what's actually true. Saining parallels this with Kant's 物自体 and Schopenhauer's "world as will and representation."
- **Kaiming's deadline discipline**: finish a publishable paper **one month before** the deadline, then polish for a month. OCD rule passed to Saining's students: "**你的一行论文不能有一行有小于百分之六十的文字**." "**paper 不是给你自己看,是给别人看**" — the communication interface must be 赏心悦目.
- **Research = filmmaking / storytelling**. Cites **Robert McKee's *Story*** — what matters isn't background but the *choices* under conflict. Cites **Scorsese**: "the most creative things are the most personal." Cites his SJTU A3 班 advisor 于老师: "**不是因为看见所以相信,因为相信所以看见**."
- **ConvNeXt (with 刘壮, now Princeton faculty)** — questioned the assumption that soft attention is what makes ViT work; controlled ablations showed architecture/design choices matter more. Kaiming ambitiously named it "**ConvNet for the 2020s**."
- **DiT (with intern Bill Peebles)** — Saining's last FAIR project. Started as a representation-learning probe (diffusion features vs SSL); pivoted in the last month when they realized the architecture was more efficient/scalable than U-Net. **Rejected at CVPR for "lack of novelty"** (Yann LeCun tweeted about it), accepted at the next venue unchanged — "**重或不重一点都不重要**." FAIR refused to attach their authors after Saining left ("不要借我们 FAIR 的名声"), so DiT was published under NYU/Berkeley affiliation. Bill went to OpenAI to drive DiT into **Sora** because nobody else "bought it."
- **Anti-fragility (Taleb)**: research must be an anti-fragile system — black-swan shocks (rejections) should produce more upside than damage.
- **NYU-era resource story**: used **Google TRC (TPU Research Cloud)** because GPUs were too expensive; explicitly framed academic PI life as analogous to startup fundraising. **NSF ≈ $500K total over 5 years per PI (~$100K/yr)**; industry grants ~$10-15K with **~100 schools competing**. Hiked Google trails with a collaborator to "pitch" the partnership. Funded the **Cambrian / Vi-STaR / REPA / RAE** series on this stack.
- **Vision as a perspective, not a task** — what CV must solve: (1) continuous, high-dim, noisy signals; (2) hierarchical/abstract representation = generalization; (3) massively parallel processing (cortex fires many places at once, intuitive physics); (4) cross-view feature sharing (a child's drawing of a dog, a cartoon dog, a real dog all collapse to "dog"). **LLMs live in `y`-space (label/communication); vision lives in `x`-space.**
- **Cambrian-S 5-stage roadmap toward a Predictive World Model** (cribbed from autonomous-driving level taxonomy):
  - **L0** = pure LLM (Plato's cave — only language)
  - **L1** = current MLLMs ("show and tell")
  - **L2** = streaming event cognition (continuous visual stream)
  - **L3** = spatial cognition — "**人活在这个世界上就是长镜头,我们的眼睛就是我们的相机**" (inspired by **贾樟柯** / **毕赣** long takes)
  - **L4** = predictive world model
- **Vi-STaR** was the first multimodal **System-2** (test-time scaling on visual reasoning) — done *months* before "test-time scaling" became a buzzword; Saining showed the benchmark to OpenAI's **Alex Kirillov** and **Bowen / 博文**, who then drove "**Think with Images**" inside OpenAI a year later. Bittersweet — industry labs increasingly don't cite or list authors, breaking the credit-assignment loop.
- **LLM is NOT self-supervised — it's strongly supervised.** The "label" is the entire civilization-scale tokenized knowledge humans uploaded to the internet — free but not unlabeled. Language is a *communication* tool, not a *thinking* tool — "a cup fell and broke" omits the dynamics and physics; that omission is intrinsic.
- **The crutch / opiate metaphor** (Saining extending LeCun): "**语言其实是一个毒药,或者语言其实是一个鸦片,你加多语言,你总是会觉得更幸福的 [...] 如果一个人一直吸鸦片,你就废了**." LM = crutch — you can walk but can't run; if you keep using crutches you never train the leg muscles.
- **REPA + RAE bet**: **representation is the only thing that matters** — once you have a good enough one, every other task becomes easy. Cites **马毅** at HKU: "**你们一定不能害怕高维度**." Future foundation = a **cognitive architecture** of multiple connected modules, where LLMs degrade to a communication interface decoding the foundation into language / pixels / actions (VLA-like).
- Scaling-law cameo: **C = 6 N D** (compute ≈ 6 × tokens × parameters).
- People: [Kaiming](https://youtu.be/iiBY0fqpThI?t=9837), [刘壮](https://youtu.be/iiBY0fqpThI?t=10666), [Bill Peebles](https://youtu.be/iiBY0fqpThI?t=10956), [LeCun](https://youtu.be/iiBY0fqpThI?t=11178), [Robert McKee](https://youtu.be/iiBY0fqpThI?t=10394), [Scorsese](https://youtu.be/iiBY0fqpThI?t=10491), [贾樟柯 / 毕赣](https://youtu.be/iiBY0fqpThI?t=12464), [李飞飞](https://youtu.be/iiBY0fqpThI?t=12804), [Alex Kirillov](https://youtu.be/iiBY0fqpThI?t=14294), [Bowen](https://youtu.be/iiBY0fqpThI?t=14298), [马毅](https://youtu.be/iiBY0fqpThI?t=14589), [Aravind (Perplexity)](https://youtu.be/iiBY0fqpThI?t=11499), [Taleb](https://youtu.be/iiBY0fqpThI?t=11657).
- Numbers: paper line-fill rule ≥ 60-70%; pre-Cambrian → present compressed to one day, behavioral modernity is the last 8-10 seconds; NSF $50K/yr × 5 years per PI; industry grants $10-15K with ~100 schools competing → "可买半个 H100 / 300 cluster / 3-4 cards"; scaling law C = 6ND.

### §10 — 世界模型是什么?  [[4:11:07 → 4:29:47](https://youtu.be/iiBY0fqpThI?t=15067)]

- **Strict definition**: a world model takes the current state s_t and an action/intervention a_t and learns a **predictive (transition) function f** that outputs the next state s_{t+1}. Not strictly temporal, but that's the canonical case.
- **Historical lineage**: physiologist **Kenneth Craik (1943)** → control theory (1960s-70s **Model Predictive Control** for moon landers — roll out an action sequence, score with a cost function, execute first step, repeat) → **Rich Sutton's Dyna** in model-based RL.
- **Sutton's Dyna ↔ Kahneman's System 1 / 2**: pure RL is "**primitive, model-free**" — reactive, like muscle memory after learning to drive; a strong world model enables planning, which is the same concept as the reasoning now hot in LLMs.
- "**State** should be a minimal-information description of the system sufficient for the agent's task" — don't model every air molecule (totally stupid). Connects directly to hierarchical / latent representation learning, and is why LLMs **violate the bitter lesson** — language is itself an extremely clever human-designed structure, not a learned representation.
- **LLMs are a "fundamentally flawed" world model**: controllability/safety today come only from post-training/fine-tuning/alignment; a real world model would let an agent predict the consequence of its action at inference time and add external constraints — a knife-holding robot should know rotating the blade toward the human is dangerous without ever having seen that scenario in data.
- **World model is a goal, not a technique**: "**世界模型不好定义,原因他其实是因为他不是一个技术路线 [...] 我们所有人都在通往世界模型的道路上.**" Different camps target different artifacts:
  - **Video-diffusion world simulators** — **Sora, ByteDance, Genie, Runway, Luma** — optimized for long, consistent, controllable video
  - **World Labs (Fei-Fei Li)** — explicit 3D-asset interface; **Autodesk just led a ~$200M round**; representation is explicit 3D, "100% won't err" in that space
  - **AMI's target — a "predictive brain"** — world-model representation is necessarily **latent** (not language-bound), inspectable from the side; LLMs remain a critical module but not the whole thing
- People: [Kenneth Craik](https://youtu.be/iiBY0fqpThI?t=15153), [Rich Sutton](https://youtu.be/iiBY0fqpThI?t=15323), [Fei-Fei Li](https://youtu.be/iiBY0fqpThI?t=15950).
- Numbers: Craik 1943; **Autodesk ~$200M led round into World Labs**.

### §11 — 从下载互联网,到下载人类  [[4:29:47 → 4:58:17](https://youtu.be/iiBY0fqpThI?t=16187)]

- Treating a video as a flat token sequence (e.g. **256 tokens × 128 frames**) and feeding to an LLM is "completely doesn't make sense" — Transformers' uniform attention is the wrong inductive bias for continuous signals with an underlying global state.
- **A real world model must (LeCun's framing)**: (1) understand the physical world, (2) have large associative memory, (3) reason, plan, do counterfactual/causal inference, (4) be controllable and safe.
- World model is not a replacement for LLMs but a more bitter-lesson base — video generation already shifts from modeling **p(y)** (label/language) to **p(x|y)** (pixels themselves), forcing the model to learn why a four-legged cat is more likely than a three-legged one. Pixels are still an "**excuse for humans**"; the true endgame ditches the regular pixel grid for representations not meant to be looked at.
- **Understanding vs generation is a false dichotomy** — both sit on top of a true world model as the base layer. Saining self-identifies as the **"表征派"** (representation faction) building that base.
- LLM scaling laws are "**watered down**" because benchmarks reward factual retrieval (forcing ~1:1 params:tokens à la **Chinchilla**); a visual world model will have a different scaling curve, **likely far smaller** — no need for trillion-parameter fact-storage, just a strong filter from ~**1 Gbit/s of sensory input** to ~**10-100 bits/s of action**, on **~20 watts of power**.
- **Data progression**: era 1 = "download the internet" for LLMs; era 2 = "**download human**" — "**我觉得过去的时代是 dump,或者 download 吧,internet 的时代,现在的时代是 download human 的时代,我们要把人类下载下来.**"
- **A 4-month-old child has seen more video than 30T LLM training tokens** ≈ **30 minutes of YouTube uploads**. Step 1: collect YouTube (despite ToS/IP cat-and-mouse — **ByteDance has a structural advantage** here). Step 2: collect **ego-vision from always-on wearables**.
- **Two natural product outlets**:
  - **AI glasses / wearables** giving "**infinite tokens**" of personal context. Current heart-rate trackers are a degenerate vertical world model — they collect signal but lack the intelligence layer to say "you're under too much stress, take a day off."
  - **General-purpose robotics** — today suffers from "**brain not strong enough**"; Spring Festival Gala dancing robots are vertical entertainment, not the elderly-care helper.
- Saining dislikes the "world model" hype term but cites **Berkeley's Jitendra Malik**: the one redeeming thing is it forces you to remember you're modeling the **world**, not just *words*.
- People: [LeCun](https://youtu.be/iiBY0fqpThI?t=16353), [Jitendra Malik](https://youtu.be/iiBY0fqpThI?t=17865).
- Numbers: 256 tokens × 128 frames; human FPS ~100Hz; brain bandwidth ~10^9 (1 Gbit/s) → action ~10-100 bits/s on ~20W; 4-month-old ≈ 30T LLM tokens ≈ 30 min YouTube upload; Chinchilla ~1:1 params:tokens.

### §12 — 和杨立昆创立 AMI 始末  [[4:58:17 → 5:45:53](https://youtu.be/iiBY0fqpThI?t=17897)]

- **Trigger**: a "**中等 paper 陷阱**" (middle-paper trap, like middle-income trap) — at NYU he could keep producing decent papers but resource limits made breakthroughs impossible.
- A mentor suggested asking LeCun in mid-2025; on their next Monday 1:1, LeCun **unprompted** told Saining: "**三连你先不要告诉别人,但我已经决定了,这个我现在想要做的事情,我觉得应该在外面做,我想要去创业,开一个公司**." Their world-model thesis was "**完全一致**."
- **Two-sided thesis**: "**the world needs a world model**" (LLMs can't solve real physical-world problems for farms, hospitals, airplane engines with **1000 sensors**) AND "**the world model needs the world**" (data and problem definitions live outside Silicon Valley, outside YouTube — high-dim, noisy, non-visual industrial signals that won't be uploaded to the internet).
- **Why no big lab can do it**: every frontier lab is locked in an "**AGI / scaling law / benchmark → resource allocation → arms race**" value chain. Researchers know video-understanding / world-model work matters, but resource allocation forces them into video-captioning teams as the only adjacent slot. He saw this firsthand at Google — the **REE paper** took ~1 year (partly due to student **伯扬 / Boyang**'s health issues); Googlers messaged him saying their managers killed identical work because of product cycles.
- **AMI positioning**: 60-70% new-lab execution, 20-30% completely-free frontier research. For-profit, needs a business model.
- **The "reverse OpenAI" / Mastercard-style coalition**: instead of downloading the internet → train GPT → push to market, AMI partners with a **grassroots alliance (草根联盟)** of industries (factories, hospitals, agriculture) who supply problem definitions and proprietary data, get an initial world model that delivers value, and feed data back to retrain. LeCun's "**neutral face**" (American + French, not Valley-resident) is a recruiting and partnering asset.
- **Anti-superhero hiring**: "**一个人很难被闪电击中两次**" — a person is rarely struck by lightning twice. Won't try to recruit the authors of "the 20-25 AI-history-defining papers." Wants strong-but-not-yet-famous mission-driven young researchers and wants to **preserve their visibility** rather than turn them into replaceable cogs.
- **The JEPA conversion arc** Saining lived through at NYU: "**质疑 JEPA → 理解 JEPA → 成为 JEPA**" — three life stages. World model is multi-modal (not just vision), pre-trained as a "universal" base with downstream **"hammers"** (LM, video diffusion, action, planning, robotics) sitting on top.
- **AMI day-one shape**: Saining = **co-founder & Chief Science Officer**; LeCun = **Executive Chairman ("captain")**; separate CEO handles operations. Compares the LeCun-as-CSO arrangement to **Pika**.
- **Day-one numbers**: ~**25 people**, target raise ~**$1B**, **four offices on day one — Paris (HQ), New York, Montreal, Singapore**. Several joiners walked away from **$15-20M Meta superintelligence offers** and large vested Google/Meta stock packages "想都不想" because they believed AMI is "the only place we can do this thing."
- **On LeCun the person**: a public 斗士 on Twitter but privately warm and humble — takes selfies with everyone, lets juniors disagree to his face. Crucially "**does not oppose LLMs — only the narrative that LLMs lead to human-level intelligence**." Still runs a weekly NYU group meeting and "**永远要给你写公式的**." Management style: sailing — give people trust, but correct course "as early as possible" when something drifts. Quote LeCun (paraphrased): "**my integrity as a scientist cannot accept this**."
- **Why YES to LeCun after twice NO to Ilya**: "**气场是否相投**" (whether the auras match) — chemistry, not pure logic. Saining can neither be a "**功成名就**" senior professor nor an 18-year-old who beds down in a Shenzhen factory (name-checks **Eddie at build.ai** who actually does that), so AMI is the middle path.
- People: [LeCun](https://youtu.be/iiBY0fqpThI?t=18004), [Ilya](https://youtu.be/iiBY0fqpThI?t=19981), [侯小迪](https://youtu.be/iiBY0fqpThI?t=18758), [张涛 / Manus](https://youtu.be/iiBY0fqpThI?t=18760), [Eddie / build.ai](https://youtu.be/iiBY0fqpThI?t=19948), [Boyang](https://youtu.be/iiBY0fqpThI?t=18492).
- Numbers: target raise ~**$1B**; ~**25 people** day one; **four offices**; airplane engine = **1000 sensors**; 60-70% new-lab / 20-30% free research; Meta superintelligence offers = **$15-20M**; some joiners had "几千万" in unvested stock.

### §13 — "硅谷被催眠了"  [[5:45:53 → 6:07:17](https://youtu.be/iiBY0fqpThI?t=20753)]

- **AMI raised roughly a $1B pre-money seed** — Saining says this ranks among the **largest seed rounds in history**. Stresses VC capital is **far more precious** than Meta/Google "money printer" budgets — every dollar must be deployed with care.
- **Deliberately not Silicon Valley**: "**我觉得 Six Valley [...] 已经被 Luxury Journal 催眠了**" (ASR — Saining = Silicon Valley + Wall Street Journal; the LLM scaling consensus). Will open SF offices once the fog lifts; "**我们的 location 一定是哪里有人才,我们公司就在哪里**."
- **Underdog / 草根联盟** identity, not 含着金汤匙. Outside the Valley most people believe them; inside, most don't — and "**做 research 也是一样,你们越不相信我,我越 happy [...] 我现在已经 all in 了,你跟不跟?**"
- **On LeCun, by way of Kaiming**: "**16-year-old adolescent extended to age 65**" — a slash-青年 with four hobbies: model airplanes, astrophotography (the nebula on LeCun's Zoom background is his own backyard shot), electronic music, jazz (his personal jazz-club website mentions **Charlie Parker**). LeCun knew both the **1972 Tarkovsky** and **2002 Soderbergh** versions of *Solaris* when Saining pitched a paper of that name.
- **Upcoming AMI paper "Solaris"** — a video-generation model. Saining reads Lem's novel as a parable that LLMs may not understand humans but only reflect them — "**海洋只是人对自己的投射**" — and wants the captain of the "world-model ship" to be someone with that breadth of taste.
- **JEPA reframe**: "**JEPA 不是一个模型,JEPA 不是一个具体的算法,JEPA 是一个整整个一套的 cognitive architecture [...] JEPA 是一个非常非常广阔的海洋,在这个海洋里面可以有好多好多的船在上面开,并且 LM 也是其中的一部分.**"
- **Skiing → startup metaphor**: "**滑雪是一个讲求平衡的运动 [...] 你要无所畏惧的把自己的肩膀朝向山下 [...] 人类的赞歌就是勇气的赞歌**" (the JoJo line). Leaning back = falling.
- **AMI cofounder roster** (six total across four offices): CEO from Meta's Southern-Europe VP slot; **VP of World Model "Mike"** (former director of LeCun's JEPA team at FAIR); **CRIO Pascal Feng** (Chinese, ex-startup founder) bridging research and product. Plan: ship a 2C product eventually, but not under pressure to do so before the world-model breakthrough. Roadmap horizon = "**1 year max — 几年当然都很不现实**."
- People: [LeCun](https://youtu.be/iiBY0fqpThI?t=20863), [Kaiming](https://youtu.be/iiBY0fqpThI?t=20863), [Charlie Parker](https://youtu.be/iiBY0fqpThI?t=20921), [Stanisław Lem](https://youtu.be/iiBY0fqpThI?t=20970), [Tarkovsky](https://youtu.be/iiBY0fqpThI?t=21050), [Soderbergh](https://youtu.be/iiBY0fqpThI?t=21050), [Mike Rabbat](https://youtu.be/iiBY0fqpThI?t=21712), [Pascal Feng](https://youtu.be/iiBY0fqpThI?t=21689).
- Numbers: ~$1B pre-money seed; 6 co-founders; 4 offices; took "**a week**" to decide; **1-year max** roadmap.

### §14 — 自大的人类!  [[6:07:17 → 6:18:45](https://youtu.be/iiBY0fqpThI?t=22037)]

- **AGI is a 伪命题 (pseudo-problem)**, echoing LeCun's debate with **Demis Hassabis** about whether "general intelligence" even exists. Out of ~**2^(2 million)** possible visual functions a retina with **2 million nerve fibers** could encode, humans can only access a tiny specialized subset. "**所以人的智能是一个非常 specialized 的智能.**"
- Recommends **Frans de Waal's *Are We Smart Enough to Know How Smart Animals Are?*** and earlier **Chimpanzee Politics**. Intelligence is a continuum: animals pass mirror tests; chimps do strategic reasoning; **corvids cache and re-cache food when watched by rivals**; whales have their own language; dogs/bats have their own sensory intelligences.
- **Citing Rich Sutton's podcast appearance**: building a **squirrel** that survives in the real world with intrinsic rewards, hunger, emotion, and social behavior is a harder problem than writing code, IMO, or IOI gold. "**一旦你能够去 build 一个松鼠的智能 [...] 后面的写 code 上火星上月球,这件事情都是再容易不过的事情.**"
- **Cosmic-arrogance reframe**: from a 造物主's view, re-creating a squirrel is greater than what human civilization built "**在这 530 个 million years 最后的 8 秒**." AMI's world model still targets human-level intelligence because it benefits the world most, but he wants to drop the arrogance.
- **Robotics as the right exit**: "**在你谈论什么 AGI super intelligence 之前,能不能先有一个足够 reliable 足够 channel 的 robot,能够在我们家庭的环境里面帮我去解决一些家务**" — what a **4-12-year-old** child can do at home. Cites DeepMind's **谭杰 (Tan Jie)**: robot development is wildly uneven — **limbs already surpass humans, but no one is building the robot 大脑 (brain)**.
- Calls for a "**预训练的下半场**" (second half of pre-training) for the world model — invoking **姚顺宇 (Yao Shunyu)**'s "second half" framing and a recent **Jim Fan** post. Robotics startups including **π / π-VLA (Physical Intelligence)** and DeepMind itself (which "**收敛到 Gemini via VLA**") lack the bandwidth to do this pre-training; they're stuck on hardware scaling laws, imitation learning, and language-model foundations.
- People: [LeCun](https://youtu.be/iiBY0fqpThI?t=22063), [Demis Hassabis](https://youtu.be/iiBY0fqpThI?t=22063), [Frans de Waal](https://youtu.be/iiBY0fqpThI?t=22197), [Rich Sutton](https://youtu.be/iiBY0fqpThI?t=22355), [谭杰](https://youtu.be/iiBY0fqpThI?t=22572), [姚顺宇](https://youtu.be/iiBY0fqpThI?t=22617), [Jim Fan](https://youtu.be/iiBY0fqpThI?t=22625).
- Numbers: 2 million retinal nerve fibers → 2^(2M) function space; 12-year-old can do all household chores; 530M years × last 8 seconds compressed for civilization.

### §15 — "42"  [[6:18:45 → 6:45:29](https://youtu.be/iiBY0fqpThI?t=22725)]

- Identifies with Liverpool FC manager **Jürgen Klopp**'s "**I am the normal one**" framing (vs. **Mourinho**'s "special one"). Klopp's "battery" metaphor — a coach transfers passion/energy to empower the team; Saining wants to play that role for his students and AMI.
- **Research has a "悲凉的底色"** (melancholy ground tone) — only **5-10%** of the time you feel real joy (when something works). Cites Kaiming saying something similar. AI's current era is *less* lonely because 小红书 / 微博 / 知乎 chatter at least breaks the "幽闭空间" feeling.
- **LeCun is "非常乐观"** because he lived through the AI winter and was proven right. Saining has watched LeCun's **identical repeated talk 10-20 times** (with its ugly slide style) and keeps finding new meaning — LeCun has become "**an inspiration, not just knowledge**," whose framework maps cleanly onto multimodal / large-model work and can guide escape from local optima.
- **The Washington Square Park walk** to his NYU office takes **5-10 minutes** — pianists, dancers, moms with strollers, old men playing chess, idle young people, NYU students with laptops — his most decompressing time. "**我发现这个世界比我们想象的大得多. 不是所有人都关心什么叫做 AI.**" Informs his sense of researcher social responsibility — "**正是伊丽娅当初给我打电话的时候,她想要跟我聊的东西,但我那时候还没有这些感悟**."
- **TV/film as AI prophecies**: **Person of Interest** (competing superintelligences), **Pantheon** (animation by **Ken Liu / 刘宇坤**, also recommended by **Sam Altman**), **Park Chan-wook's *No Other Choice* (《别无选择》)**, AI short film **Total Pixel Space** (this year's Runway AI Film Festival winner).
- **Life books**: **GEB (《集异璧之大成》)** — undergrad group-read tradition; **Zen and the Art of Motorcycle Maintenance** — "**把我掏空了**" rather than filled him up. Current answer to what matters: "**人与人之间的真诚的交流是重要的,也许其他都不重要**."
- **Research-as-connection vignette**: a VC invested in his startup because **Robin Rombach** (Stability/Black Forest CEO) — whom Saining had met only once at a conference — told the VC "**你们一定要投赛宁**." Trust built on academic papers can exceed personal connection.
- **On Seedance** (ByteDance video gen): very impressive; rumored ~**200B MoE diffusion model** (nobody else has made MoE work in diffusion at scale); but **90-95% of generative-model quality is a data problem** (cleaning, captioning, distribution, diversity, prompt alignment) — architecture is secondary. Sora and Veo trying to surpass it "不一定那么简单."
- **Anti-cliché rant on two clichéd ML-paper epigraphs**:
  - **Misusing Wittgenstein's** "*the limits of my language are the limits of my world*" as LM/language-determinism endorsement — that line is from the *Tractatus* and refers only to **propositionally-expressible reality**; later Wittgenstein **overturned it himself** with the "**语言游戏 / language game**" view where meaning comes from real-world practice/action: "**语言本身没有意义,这些 symbol 本身没有任何意义,它之所以发生异议是因为,它跟真实世界的实践发生了关系. 然后这件事情就很世界模型了.**"
  - **Misusing Feynman's** "*What I cannot create, I do not understand*" to endorse unified generative models — Feynman meant create in a real-world action sense, not "**a diffusion model 反向传播 a loss**, 这件事情完全是离谱的." Echoes Kaiming's call for ML researchers to read more philosophy.
- **Closer**: doesn't feel "命运在推我"; the universe *is* a giant world model but predicting fate would require "**地球这么大的一个计算机,或者说你要有一个,有整个宇宙作为你的计算机**" — and the answer might just be **42**.
- People: [Klopp](https://youtu.be/iiBY0fqpThI?t=22774), [Mourinho](https://youtu.be/iiBY0fqpThI?t=22779), [LeCun](https://youtu.be/iiBY0fqpThI?t=22848), [Kaiming](https://youtu.be/iiBY0fqpThI?t=22899), [Ilya](https://youtu.be/iiBY0fqpThI?t=23295), [Ken Liu](https://youtu.be/iiBY0fqpThI?t=23427), [Sam Altman](https://youtu.be/iiBY0fqpThI?t=23447), [Park Chan-wook](https://youtu.be/iiBY0fqpThI?t=23509), [Robin Rombach](https://youtu.be/iiBY0fqpThI?t=23845), [Wittgenstein](https://youtu.be/iiBY0fqpThI?t=23991), [Feynman](https://youtu.be/iiBY0fqpThI?t=24132).
- Numbers: ~20+ years a Liverpool fan; joy ≈ 5-10% of research time; watched LeCun's talk 10-20 times; Washington Square Park walk 5-10 min; Seedance rumored ~200B MoE; 90-95% data-problem claim; "42" closer.

## Notable quotes

> "**LM 永不会死,但终将凋零. 就是老兵不死,终将凋零 [...] They won't die, they will just fade away.** 就是说这个东西一定会有它的价值,它是一个很好的工具,我现在会天天使用 LM. 但它不是我们构建一个 universal,一个通用智能系统,我们智能系统的基石. 它不是这个世界模型的,这种大厦的地基."
> — Xie Saining, §8 [[02:24:19](https://youtu.be/iiBY0fqpThI?t=8659)] — *the recurring "LM 终将凋零" thesis*

> "**因为金刚经里面说这些所有事情如梦幻如泡影 [...] 凡所有相皆是虚妄,若见诸相非相,即见如来 [...] research taste 的来源就在于,大家能不能真的抛开所有的这些虚无的像,然后去一直去通往通往这个真理的道路.**"
> — Xie Saining, §9 [[02:45:07](https://youtu.be/iiBY0fqpThI?t=9907)] — *Kaiming's day-one gift, the Diamond Sutra as research-taste framework*

> "**这个梯度本身,这件事情,才是你真正的 idea 的来源 [...] 一开始你想的这个 idea,不是你的 idea,这个东西不属于你. 探索中的 idea,才是属于你的 idea.**"
> — Xie Saining, §8 [[02:05:04](https://youtu.be/iiBY0fqpThI?t=7504)] — *Kaiming's anti-armchair thesis; exploration as SGD*

> "**语言其实是一个毒药,或者语言其实是一个鸦片,你加多语言,你总是会觉得更幸福的 [...] 它有用,但它是一个 short cut. 就如果你一个人一直吸鸦片,你就废了. 然后如果它是一个拐杖,拄着拐的话,你也没有办法训练你的大腿的肌肉.**"
> — Xie Saining, §9 [[04:09:44](https://youtu.be/iiBY0fqpThI?t=14984)] — *extending LeCun's "language is a crutch" — language as opiate that atrophies real representation*

> "**伊莱亚的说法是,他觉得这件事情(computer vision)已经解决的很不错了,所以我觉得可能 SSI 有自己的,基于(语言)的这样一个路线,然后这条路线至少在现在为止,不是我想要去设计的路线.**"
> — Xie Saining, §6 [[01:26:05](https://youtu.be/iiBY0fqpThI?t=5165)] — *the technical breakpoint with Ilya in 2024*

> "**我觉得过去的时代是 dump,或者 download 吧,internet 的时代,现在的时代是 download human 的时代,我们要把人类下载下来.**"
> — Xie Saining, §11 [[04:47:52](https://youtu.be/iiBY0fqpThI?t=17272)] — *era-shift framing for the next decade of AI data*

> "**我觉得能够打造出来一只松鼠的智能,这件事情才是难的问题 [...] 他有自己的 goal [...] 他有自己的 intrinsic reward,他知道饥饿,他有自己的 emotion [...] 后面的写 code 上火星上月球,这件事情都是再容易不过的事情.**"
> — Xie Saining, §14 [[06:13:12](https://youtu.be/iiBY0fqpThI?t=22392)] — *the squirrel-vs-AGI reframe*

> "**我觉得 Silicon Valley [...] 已经被 Wall Street Journal 催眠了 [...] 但这件事情我觉得不会持续很久,被催眠的人总有醒来的一刻 [...] 我们的 location 一定是哪里有人才,我们公司就在哪里.**"
> — Xie Saining, §13 [[05:46:52](https://youtu.be/iiBY0fqpThI?t=20812)] — *why AMI's HQ is Paris, not the Valley*

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| Zhang Xiaojun (张小珺) | Host | [[00:12](https://youtu.be/iiBY0fqpThI?t=12)] |
| Xie Saining (谢赛宁) | Guest — NYU CS associate prof, AMI Labs CSO | [[00:35](https://youtu.be/iiBY0fqpThI?t=35)] |
| Yann LeCun (杨立昆) | Turing Award; AMI Labs Executive Chairman; recruited Saining 3× | [[00:40](https://youtu.be/iiBY0fqpThI?t=40)] |
| Ilya Sutskever (伊莱亚 / 伊丽娅) | Twice rejected — 2018 OpenAI + 2024 SSI | [[01:03](https://youtu.be/iiBY0fqpThI?t=63)] |
| Hou Xiaodi (侯小迪) | SJTU senior; 7-line CVPR paper; *Survival Manual* author | §2 [[16:20](https://youtu.be/iiBY0fqpThI?t=980)] |
| Yu Yong (于勇) | SJTU ACM-class founder | §2 [[21:09](https://youtu.be/iiBY0fqpThI?t=1269)] |
| Shen Renchao (沈仁超) | ACM-class interviewer; Courant quiz | §2 [[13:57](https://youtu.be/iiBY0fqpThI?t=837)] |
| Richard Courant | NYU's Courant Institute namesake | §2 [[14:43](https://youtu.be/iiBY0fqpThI?t=883)] |
| Yan Shuicheng (颜水成) | NUS LV-Lab PI | §2 [[23:03](https://youtu.be/iiBY0fqpThI?t=1383)] |
| Feng Jiashi (冯佳时) | NUS PhD mentor on first BMVC paper | §2 [[30:25](https://youtu.be/iiBY0fqpThI?t=1825)] |
| Ma Yi / Sun Jian / Kaiming He (马毅 / 孙剑 / 何凯明) | MSRA vision researchers (Saining couldn't intern there) | §2 [[21:46](https://youtu.be/iiBY0fqpThI?t=1306)] |
| Zhuowen Tu (涂哲文) | UCSD PhD advisor; rescued Saining at the deadline | §3 [[36:18](https://youtu.be/iiBY0fqpThI?t=2178)] |
| Serge Belongie | CV professor leaving UCSD as Saining arrived | §3 [[39:17](https://youtu.be/iiBY0fqpThI?t=2357)] |
| Song-Chun Zhu (朱松纯) | Tu's PhD advisor — cited as a path-paver | §3 [[43:19](https://youtu.be/iiBY0fqpThI?t=2599)] |
| Fei-Fei Li (李飞飞) | Co-paver of CV path; World Labs; Cambrian-S advisor | §3 [[43:22](https://youtu.be/iiBY0fqpThI?t=2602)] |
| Yu Kai (于凯) | At NEC Labs during its deep-learning era | §4 [[55:50](https://youtu.be/iiBY0fqpThI?t=3350)] |
| Kaiming He (何恺明 / 凯明) | Saining's FAIR mentor; ResNeXt / MoCo / MAE / DiT | §5 [[57:45](https://youtu.be/iiBY0fqpThI?t=3465)] |
| Steve Jobs | "Reality distortion field" metaphor for Kaiming | §5 [[59:02](https://youtu.be/iiBY0fqpThI?t=3542)] |
| Wang Xiaolong (王小龙) | Co-intern told by Kaiming to intern elsewhere | §5 [[1:05:21](https://youtu.be/iiBY0fqpThI?t=3921)] |
| Demis Hassabis | DeepMind CEO; "multiple Nobel Prizes" interns Q&A | §5 [[1:08:00](https://youtu.be/iiBY0fqpThI?t=4080)] |
| Shuman Zhang (张舒曼) | OpenAI interviewer (A4 page of pencil problems) | §5 [[1:19:45](https://youtu.be/iiBY0fqpThI?t=4785)] |
| Piotr Dollár | FAIR "three horsemen of CV" | §5 [[1:20:39](https://youtu.be/iiBY0fqpThI?t=4839)] |
| Ross Girshick | FAIR "three horsemen of CV" | §5 [[1:20:39](https://youtu.be/iiBY0fqpThI?t=4839)] |
| Xinlei Chen (陈新雷) | Likely the first fresh-PhD FAIR hire (Saining = #2) | §6 [[1:24:34](https://youtu.be/iiBY0fqpThI?t=5074)] |
| Hannah Arendt | Reframes "impact" → understanding | §6 [[1:31:17](https://youtu.be/iiBY0fqpThI?t=5477)] |
| Zhuang Liu (刘壮) | ConvNeXt co-author; Princeton faculty | §9 [[2:57:46](https://youtu.be/iiBY0fqpThI?t=10666)] |
| Bill Peebles | DiT co-author intern; now Sora head at OpenAI | §9 [[3:02:36](https://youtu.be/iiBY0fqpThI?t=10956)] |
| Robert McKee | *Story*; research-taste primer Saining recs to students | §9 [[2:53:14](https://youtu.be/iiBY0fqpThI?t=10394)] |
| Martin Scorsese | "The most creative things are the most personal" | §9 [[2:54:51](https://youtu.be/iiBY0fqpThI?t=10491)] |
| Jia Zhangke (贾樟柯) | Director; long-take inspiration for Cambrian-S | §9 [[3:27:44](https://youtu.be/iiBY0fqpThI?t=12464)] |
| Bi Gan (毕赣) | *Kaili Blues* director; long-take inspiration | §9 [[3:27:48](https://youtu.be/iiBY0fqpThI?t=12468)] |
| Alex Kirillov | OpenAI; saw Vi-STaR → drove Think with Images | §9 [[3:58:14](https://youtu.be/iiBY0fqpThI?t=14294)] |
| Yi Ma (马毅) | HKU; "你们一定不能害怕高维度" | §9 [[4:03:09](https://youtu.be/iiBY0fqpThI?t=14589)] |
| Aravind (Perplexity) | Showed Saining Perplexity demo at Blue Bottle | §9 [[3:11:39](https://youtu.be/iiBY0fqpThI?t=11499)] |
| Nassim Taleb | *Antifragile* | §9 [[3:14:17](https://youtu.be/iiBY0fqpThI?t=11657)] |
| Kenneth Craik | 1943 origin of "internal world model" concept | §10 [[4:12:38](https://youtu.be/iiBY0fqpThI?t=15158)] |
| Rich Sutton | Dyna paper; squirrel-intelligence framing | §10 [[4:15:23](https://youtu.be/iiBY0fqpThI?t=15323)] |
| Jitendra Malik | Berkeley; "world model, not word model" | §11 [[4:57:45](https://youtu.be/iiBY0fqpThI?t=17865)] |
| Zhang Tao / Manus (张涛 / 涛哥) | Manus co-founder; "love life" mentor advice | §12 [[5:12:40](https://youtu.be/iiBY0fqpThI?t=18760)] |
| Eddie (build.ai) | Founder who moved his team into a Shenzhen factory | §12 [[5:32:28](https://youtu.be/iiBY0fqpThI?t=19948)] |
| Boyang (伯扬) | NYU student; co-author on Google REE paper | §12 [[5:08:12](https://youtu.be/iiBY0fqpThI?t=18492)] |
| Charlie Parker (查理帕克) | Jazz musician on LeCun's personal jazz site | §13 [[5:48:41](https://youtu.be/iiBY0fqpThI?t=20921)] |
| Stanisław Lem (莱姆) | *Solaris* author; namesake of upcoming AMI paper | §13 [[5:49:30](https://youtu.be/iiBY0fqpThI?t=20970)] |
| Andrei Tarkovsky | 1972 *Solaris* director | §13 [[5:50:50](https://youtu.be/iiBY0fqpThI?t=21050)] |
| Steven Soderbergh | 2002 *Solaris* director | §13 [[5:50:50](https://youtu.be/iiBY0fqpThI?t=21050)] |
| Mike Rabbat | AMI VP of World Model; ex-director JEPA team at FAIR | §13 [[6:01:52](https://youtu.be/iiBY0fqpThI?t=21712)] |
| Pascal Feng (Pascal 冯) | AMI CRIO; Chinese; bridges research↔product | §13 [[6:01:29](https://youtu.be/iiBY0fqpThI?t=21689)] |
| Frans de Waal (德瓦尔) | Primatologist; the two animal-cognition recs | §14 [[6:09:57](https://youtu.be/iiBY0fqpThI?t=22197)] |
| Tan Jie (谭杰) | DeepMind robotics; "limbs strong, brain absent" | §14 [[6:16:12](https://youtu.be/iiBY0fqpThI?t=22572)] |
| Yao Shunyu (姚顺宇) | "Second half" framing reused for pre-training | §14 [[6:16:57](https://youtu.be/iiBY0fqpThI?t=22617)] |
| Jim Fan | Recent post on world-model pre-training | §14 [[6:17:05](https://youtu.be/iiBY0fqpThI?t=22625)] |
| Jürgen Klopp | Liverpool manager; "I am the normal one" | §15 [[6:19:34](https://youtu.be/iiBY0fqpThI?t=22774)] |
| José Mourinho (穆里尼奥) | "I am the special one" — the foil | §15 [[6:19:39](https://youtu.be/iiBY0fqpThI?t=22779)] |
| Ken Liu (刘宇坤) | Author behind Pantheon; ex-lawyer, ex-programmer | §15 [[6:30:27](https://youtu.be/iiBY0fqpThI?t=23427)] |
| Sam Altman | Also recommended *Pantheon* | §15 [[6:30:47](https://youtu.be/iiBY0fqpThI?t=23447)] |
| Park Chan-wook (朴瓒玉) | Dir. *No Other Choice* (《别无选择》) | §15 [[6:31:49](https://youtu.be/iiBY0fqpThI?t=23509)] |
| Robin Rombach | Stability → Black Forest CEO; vouched for Saining to VC | §15 [[6:37:25](https://youtu.be/iiBY0fqpThI?t=23845)] |
| Ludwig Wittgenstein (维特根斯坦) | *Tractatus* misquote target | §15 [[6:39:51](https://youtu.be/iiBY0fqpThI?t=23991)] |
| Richard Feynman (费曼) | "What I cannot create I do not understand" misquote target | §15 [[6:42:12](https://youtu.be/iiBY0fqpThI?t=24132)] |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| AMI Labs (Neolab AMI) | The new lab Saining co-founded with LeCun | [[00:45](https://youtu.be/iiBY0fqpThI?t=45)] |
| *What Is Mathematics* (Richard Courant) | ACM-class interview question | [[14:24](https://youtu.be/iiBY0fqpThI?t=864)] |
| 交大学生生存手册 (SJTU Student Survival Manual) | Hou Xiaodi's manifesto | [[17:43](https://youtu.be/iiBY0fqpThI?t=1063)] |
| AlexNet (ImageNet 2012) | Saining's "origin point" of deep learning | [[31:33](https://youtu.be/iiBY0fqpThI?t=1893)] |
| First BMVC face-recognition / manifold-clustering paper | Saining's NUS undergrad work | [[30:40](https://youtu.be/iiBY0fqpThI?t=1840)] |
| Deeply Supervised Nets (DSN) | First PhD paper; AISTATS Test of Time 2025 | §3 [[45:34](https://youtu.be/iiBY0fqpThI?t=2734)] |
| Holistically-Nested Edge Detection (HED) | Marr Prize honorable mention | §3 [[48:24](https://youtu.be/iiBY0fqpThI?t=2904)] |
| ResNet | Conceptual antecedent of ResNeXt | §3 [[47:50](https://youtu.be/iiBY0fqpThI?t=2870)] |
| ResNeXt (Aggregated Residual Transformations) | 2nd-place ImageNet 2016; MoE precursor | §5 [[1:00:40](https://youtu.be/iiBY0fqpThI?t=3640)] |
| AlphaFold | Forming during Saining's DeepMind internship | §5 [[1:08:42](https://youtu.be/iiBY0fqpThI?t=4122)] |
| PhD thesis: *Deep Representation Learning with Induced Structural Priors* | Saining's research identity | §5 [[1:09:54](https://youtu.be/iiBY0fqpThI?t=4194)] |
| NeurIPS workshop: Representation Learning with Structural Priors | Saining's organized workshop | §5 [[1:10:24](https://youtu.be/iiBY0fqpThI?t=4224)] |
| *Thinking in Space* | Saining ↔ Fei-Fei collab on MLLM spatial intelligence | §7 [[1:44:17](https://youtu.be/iiBY0fqpThI?t=6257)] |
| Cambrian-S | Saining ↔ Fei-Fei collab on video question definition | §7 [[1:44:31](https://youtu.be/iiBY0fqpThI?t=6271)] |
| ImageNet | Fei-Fei's signature; example of "defining the problem" | §7 [[1:43:28](https://youtu.be/iiBY0fqpThI?t=6208)] |
| *The Worlds I See* (Fei-Fei Li autobiography) | Saining recommends | §7 [[1:41:53](https://youtu.be/iiBY0fqpThI?t=6113)] |
| MoCo / Momentum Contrast (v1, v2, v3) | First contrastive SSL to really work | §8 [[1:58:38](https://youtu.be/iiBY0fqpThI?t=7118)] |
| CPC (Contrastive Predictive Coding) | Prior art to MoCo | §8 [[2:00:32](https://youtu.be/iiBY0fqpThI?t=7232)] |
| Memory Bank | Prior art to MoCo | §8 [[2:00:34](https://youtu.be/iiBY0fqpThI?t=7234)] |
| MAE (Masked Autoencoder) | Beautiful representations; didn't scale to LM-level | §8 [[2:27:48](https://youtu.be/iiBY0fqpThI?t=8868)] |
| PointContrast | SSL extension to 3D | §8 [[2:30:58](https://youtu.be/iiBY0fqpThI?t=9058)] |
| DiT (Diffusion Transformer) | Saining's "0.25"; rejected at CVPR | §8 [[2:19:39](https://youtu.be/iiBY0fqpThI?t=8379)] |
| DDPM | Cited in the 20-25 era-defining papers list | §8 [[2:21:14](https://youtu.be/iiBY0fqpThI?t=8474)] |
| Research is an Infinite Game (Saining CVPR talk) | Bill Freeman plot-style framing | §8 [[2:15:39](https://youtu.be/iiBY0fqpThI?t=8139)] |
| 金刚经 (Diamond Sutra) | Kaiming's day-one FAIR gift; research-taste source | §9 [[2:44:08](https://youtu.be/iiBY0fqpThI?t=9848)] |
| ConvNeXt — A ConvNet for the 2020s | With 刘壮; named by Kaiming | §9 [[2:57:54](https://youtu.be/iiBY0fqpThI?t=10674)] |
| SiT (flow matching in transformer) | DiT follow-up | §9 [[3:13:48](https://youtu.be/iiBY0fqpThI?t=11628)] |
| LDM / Stable Diffusion | Robin Rombach's line | §9 [[3:07:36](https://youtu.be/iiBY0fqpThI?t=11256)] |
| Cambrian-1 | First Cambrian paper | §9 [[3:15:50](https://youtu.be/iiBY0fqpThI?t=11750)] |
| Vi-STaR (visual System-2) | Months before "test-time scaling" became a buzzword | §9 [[3:56:38](https://youtu.be/iiBY0fqpThI?t=14198)] |
| REPA (Representation Alignment) | Representation-first bet | §9 [[4:01:14](https://youtu.be/iiBY0fqpThI?t=14474)] |
| RAE (Representation Autoencoder) | High-dim representation operating | §9 [[4:02:53](https://youtu.be/iiBY0fqpThI?t=14573)] |
| Sora (OpenAI; cites DiT) | Bill Peebles drove it from inside | §9 [[3:08:38](https://youtu.be/iiBY0fqpThI?t=11318)] |
| Think with Images (OpenAI) | Kirillov / Bowen launched after Vi-STaR | §9 [[3:58:42](https://youtu.be/iiBY0fqpThI?t=14322)] |
| *Story* (Robert McKee) | Research-taste primer | §9 [[2:53:14](https://youtu.be/iiBY0fqpThI?t=10394)] |
| *Antifragile* (Taleb) | Research as anti-fragile system | §9 [[3:14:17](https://youtu.be/iiBY0fqpThI?t=11657)] |
| Kenneth Craik 1943 (origin of mental world model) | Lineage of the term | §10 [[4:12:33](https://youtu.be/iiBY0fqpThI?t=15153)] |
| Model Predictive Control (MPC) | 1960s-70s control-theory antecedent | §10 [[4:13:42](https://youtu.be/iiBY0fqpThI?t=15222)] |
| Dyna (Rich Sutton) | Model-based RL paper | §10 [[4:15:23](https://youtu.be/iiBY0fqpThI?t=15323)] |
| ByteDance video model | Competing video-diffusion world simulator | §10 [[4:26:57](https://youtu.be/iiBY0fqpThI?t=16017)] |
| Genie (DeepMind) | Generative interactive environment | §10 [[4:27:00](https://youtu.be/iiBY0fqpThI?t=16020)] |
| Runway / Luma | Video-diffusion world-simulator camp | §10 [[4:27:03](https://youtu.be/iiBY0fqpThI?t=16023)] |
| World Labs (Fei-Fei Li) | Explicit-3D-asset interface; Autodesk-led ~$200M round | §10 [[4:27:50](https://youtu.be/iiBY0fqpThI?t=16070)] |
| Chinchilla scaling law | ~1:1 params:tokens balance | §11 [[4:44:43](https://youtu.be/iiBY0fqpThI?t=17083)] |
| Libet experiment (贝利特实验) | Decision-readiness potential | §11 [[4:34:33](https://youtu.be/iiBY0fqpThI?t=16473)] |
| REE (Google ↔ NYU collab paper) | The "killed-by-product-cycle" case study | §12 [[5:08:06](https://youtu.be/iiBY0fqpThI?t=18486)] |
| JEPA (Joint Embedding Predictive Architecture) | LeCun's framework Saining converted to | §12 [[5:10:40](https://youtu.be/iiBY0fqpThI?t=18640)] |
| *A Path Towards Autonomous Machine Intelligence* (LeCun 2022) | Foundational JEPA position paper | §12 [[5:37:13](https://youtu.be/iiBY0fqpThI?t=20233)] |
| Solaris (upcoming AMI video-gen paper) | Lem-titled; LeCun knew both film versions | §13 [[5:49:23](https://youtu.be/iiBY0fqpThI?t=20963)] |
| *Solaris* (Lem 1961 / Tarkovsky 1972 / Soderbergh 2002) | Source canon | §13 [[5:49:29](https://youtu.be/iiBY0fqpThI?t=20969)] |
| I-JEPA / LeJEPA | Representation theory follow-up | §13 [[6:04:05](https://youtu.be/iiBY0fqpThI?t=21845)] |
| *JoJo's Bizarre Adventure* | "人类的赞歌就是勇气的赞歌" source | §13 [[5:55:12](https://youtu.be/iiBY0fqpThI?t=21312)] |
| *Are We Smart Enough to Know How Smart Animals Are?* (de Waal) | Animal-cognition rec | §14 [[6:08:52](https://youtu.be/iiBY0fqpThI?t=22132)] |
| *Chimpanzee Politics* (de Waal) | Earlier de Waal | §14 [[6:09:57](https://youtu.be/iiBY0fqpThI?t=22197)] |
| π / π-VLA (Physical Intelligence) | Robotics startup stuck on VLA | §14 [[6:17:52](https://youtu.be/iiBY0fqpThI?t=22672)] |
| *Person of Interest* | TV: competing superintelligences | §15 [[6:29:33](https://youtu.be/iiBY0fqpThI?t=23373)] |
| *Pantheon* (Ken Liu / 万神殿) | AI-prophecy animation | §15 [[6:30:22](https://youtu.be/iiBY0fqpThI?t=23422)] |
| *No Other Choice / 别无选择* (Park Chan-wook) | Film on AI's alienation | §15 [[6:31:44](https://youtu.be/iiBY0fqpThI?t=23504)] |
| *Total Pixel Space* (Runway AI Film Festival winner) | AI short film | §15 [[6:32:39](https://youtu.be/iiBY0fqpThI?t=23559)] |
| *GEB* (Gödel, Escher, Bach) | SJTU undergrad group-read tradition | §15 [[6:33:37](https://youtu.be/iiBY0fqpThI?t=23617)] |
| *Zen and the Art of Motorcycle Maintenance* | "把我掏空了" | §15 [[6:35:08](https://youtu.be/iiBY0fqpThI?t=23708)] |
| *Tractatus Logico-Philosophicus* (Wittgenstein) | Misused epigraph rant target | §15 [[6:40:34](https://youtu.be/iiBY0fqpThI?t=24034)] |
| Seedance (ByteDance video gen) | Rumored ~200B MoE diffusion | §15 [[6:38:07](https://youtu.be/iiBY0fqpThI?t=23887)] |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| AMI Labs day-one team size: ~25 people | [Host] | [[00:51](https://youtu.be/iiBY0fqpThI?t=51)] |
| Saining ~13 years in the US ("后训练有点崩") | [Guest] | [[02:56](https://youtu.be/iiBY0fqpThI?t=176)] |
| First personal computer at age 9 | [Guest] | [[06:18](https://youtu.be/iiBY0fqpThI?t=378)] |
| Hou Xiaodi's CVPR paper: only 7 lines of code | [Guest] | [[16:59](https://youtu.be/iiBY0fqpThI?t=1019)] |
| Visual cortex ~30%, brain activation ~70% on visual input | [Guest] | [[25:54](https://youtu.be/iiBY0fqpThI?t=1554)] |
| Pre-Cambrian (~530M years ago): life had no eyes | [Guest] | [[26:28](https://youtu.be/iiBY0fqpThI?t=1588)] |
| ACM class ~30-40 students; Saining ~10th | [Guest] | [[34:50](https://youtu.be/iiBY0fqpThI?t=2090)] |
| First-paper year: 2012-13 (AlexNet moment) | [Guest] | [[31:30](https://youtu.be/iiBY0fqpThI?t=1890)] |
| PhD application: replied after April 15 deadline | [Guest] | [[36:26](https://youtu.be/iiBY0fqpThI?t=2186)] |
| Tu hand-coded ~50,000 lines of C++ for image segmentation | [Guest] | [[42:20](https://youtu.be/iiBY0fqpThI?t=2540)] |
| ~5-6 top-venue papers in PhD | [Guest] | [[45:11](https://youtu.be/iiBY0fqpThI?t=2711)] |
| DSN NeurIPS review score ~886/887; rejected post-rebuttal | [Guest] | [[1:15:23](https://youtu.be/iiBY0fqpThI?t=4523)] |
| DSN won AISTATS Test of Time 10 years later | [Guest] | [[1:16:36](https://youtu.be/iiBY0fqpThI?t=4596)] |
| Saining did 5 PhD internships | [Guest] | [[52:44](https://youtu.be/iiBY0fqpThI?t=3164)] |
| Each internship 3-6 months; ~8-hour SoCal↔Bay drive | [Guest] | [[54:55](https://youtu.be/iiBY0fqpThI?t=3295)] |
| Half of internships produced nothing | [Guest] | [[54:46](https://youtu.be/iiBY0fqpThI?t=3286)] |
| ResNeXt placed 2nd at ImageNet Challenge | [Guest] | [[1:00:50](https://youtu.be/iiBY0fqpThI?t=3650)] |
| OpenAI interview: locked in a room for 5-6 hours | [Guest] | [[1:19:33](https://youtu.be/iiBY0fqpThI?t=4773)] |
| Year of graduation / OpenAI offer: 2018 | [Guest] | [[1:20:19](https://youtu.be/iiBY0fqpThI?t=4819)] |
| Top-PhD pay ~$400-500K (Saining misspoke "2008", meant 2018) | [Guest] | [[1:21:27](https://youtu.be/iiBY0fqpThI?t=4887)] |
| Saining stayed at FAIR for 4 years | [Guest] | [[1:36:54](https://youtu.be/iiBY0fqpThI?t=5814)] |
| Ilya may have reached out to 1,000-10,000 people | [Guest] | [[1:28:20](https://youtu.be/iiBY0fqpThI?t=5300)] |
| Two phone calls with Ilya total (2018 + July 2024) | [Guest] | [[1:25:11](https://youtu.be/iiBY0fqpThI?t=5111)] |
| 2015-16 pretext-task SSL: 15-20 pp worse than ImageNet supervised | [Guest] | [[1:58:05](https://youtu.be/iiBY0fqpThI?t=7085)] |
| ImageNet 1000 classes contain ~200 dog breeds | [Guest] | [[1:53:34](https://youtu.be/iiBY0fqpThI?t=6814)] |
| Kaiming told Saining "we need to make models big" in 2018-19 | [Guest] | [[1:51:09](https://youtu.be/iiBY0fqpThI?t=6669)] |
| ~6-month research cycle: 1-2mo explore / 2-3mo experiments / 1-2mo writing | [Guest] | [[2:05:24](https://youtu.be/iiBY0fqpThI?t=7524)] |
| Kaiming bought ~5000 TPU cores for FAIR (originally for LM, repurposed) | [Guest] | [[2:33:00](https://youtu.be/iiBY0fqpThI?t=9180)] |
| ~20-25 era-defining DL/AI papers; Saining counts 0 (DiT = 0.25) | [Guest] | [[2:20:18](https://youtu.be/iiBY0fqpThI?t=8418)] |
| Kaiming finishes papers 1 month before deadline | [Guest] | [[2:47:55](https://youtu.be/iiBY0fqpThI?t=10075)] |
| Paper line-fill rule ≥ 60-70% | [Guest] | [[2:49:11](https://youtu.be/iiBY0fqpThI?t=10151)] |
| Cambrian-S: 538M years to today; behavioral modernity = last 8-10 sec | [Guest] | [[3:16:28](https://youtu.be/iiBY0fqpThI?t=11788)] |
| NSF: ~$500K total / 5 yr / PI (~$100K/yr) | [Guest] | [[3:23:44](https://youtu.be/iiBY0fqpThI?t=12224)] |
| Industry grants: $10-15K with ~100 schools competing | [Guest] | [[3:24:05](https://youtu.be/iiBY0fqpThI?t=12245)] |
| Scaling law cameo: C ≈ 6 N D | [Guest] | [[3:50:06](https://youtu.be/iiBY0fqpThI?t=13806)] |
| Autodesk led ~$200M round into World Labs | [Guest] | [[4:28:14](https://youtu.be/iiBY0fqpThI?t=16094)] |
| Video tokenization example: 256 tokens × 128 frames | [Guest] | [[4:30:26](https://youtu.be/iiBY0fqpThI?t=16226)] |
| Human FPS perception ~100 Hz | [Guest] | [[4:29:56](https://youtu.be/iiBY0fqpThI?t=16196)] |
| Brain bandwidth ~10^9 bits/s in, ~10-100 bits/s out, on ~20 W | [Guest] | [[4:46:32](https://youtu.be/iiBY0fqpThI?t=17192)] |
| 4-month-old has seen > 30T LLM tokens ≈ 30 min YouTube upload | [Guest] | [[4:48:28](https://youtu.be/iiBY0fqpThI?t=17308)] |
| Chinchilla: ~1:1 params:tokens | [Guest] | [[4:44:38](https://youtu.be/iiBY0fqpThI?t=17078)] |
| Airplane engine = ~1000 sensors | [Guest] | [[5:15:59](https://youtu.be/iiBY0fqpThI?t=18959)] |
| AMI: 60-70% new-lab / 20-30% free research | [Guest] | [[5:17:38](https://youtu.be/iiBY0fqpThI?t=19058)] |
| AMI target raise ~$1B; pre-money ~$1B | [Guest] | [[5:41:28](https://youtu.be/iiBY0fqpThI?t=20488)] |
| AMI day-one ~25 people; 4 offices (Paris HQ + NYC + Montreal + Singapore) | [Guest] | [[5:41:46](https://youtu.be/iiBY0fqpThI?t=20506)] |
| Meta superintelligence offers walked away from: $15-20M | [Guest] | [[5:42:32](https://youtu.be/iiBY0fqpThI?t=20552)] |
| LeCun still runs NYU group meeting one day a week | [Guest] | [[5:37:55](https://youtu.be/iiBY0fqpThI?t=20275)] |
| AMI co-founders: 6; took ~1 week to accept LeCun's offer | [Guest] | [[6:01:14](https://youtu.be/iiBY0fqpThI?t=21674)] |
| AMI roadmap horizon: max 1 year | [Guest] | [[6:02:30](https://youtu.be/iiBY0fqpThI?t=21750)] |
| 2 million retinal nerve fibers → 2^(2M) function space | [Guest] | [[6:08:06](https://youtu.be/iiBY0fqpThI?t=22086)] |
| 12-year-old can do all household chores | [Guest] | [[6:15:59](https://youtu.be/iiBY0fqpThI?t=22559)] |
| Civilization built in last 8 sec of 530M-year clock | [Guest] | [[6:14:49](https://youtu.be/iiBY0fqpThI?t=22489)] |
| Liverpool FC fan for 20+ years | [Guest] | [[6:19:24](https://youtu.be/iiBY0fqpThI?t=22764)] |
| Joy ≈ 5-10% of research time | [Guest] | [[6:21:36](https://youtu.be/iiBY0fqpThI?t=22896)] |
| Watched LeCun's repeated talk 10-20 times | [Guest] | [[6:24:43](https://youtu.be/iiBY0fqpThI?t=23083)] |
| Washington Square Park walk: 5-10 minutes | [Guest] | [[6:27:27](https://youtu.be/iiBY0fqpThI?t=23247)] |
| Seedance rumored ~200B MoE diffusion | [Guest] | [[6:38:29](https://youtu.be/iiBY0fqpThI?t=23909)] |
| Generative-model quality ~90-95% data problem | [Guest] | [[6:38:44](https://youtu.be/iiBY0fqpThI?t=23924)] |

## Open questions / gaps

- **AMI's data-flywheel claim ("world model needs the world")** is asserted without showing partner pipeline or revenue path; Saining repeatedly admits "I've never done a startup, never succeeded, never failed."
- **"A person is rarely struck by lightning twice"** is offered as the hiring principle but stated as taste — no evidence that famous-paper authors fail more often at second breakthroughs.
- **"Silicon Valley is hypnotized by the LLM narrative"** is asserted as obvious; no specific evidence; Saining even says "我不知道比例是怎么样."
- **Vision scaling law**: Saining says it will be "fundamentally different from LLMs" and require "far fewer parameters" but offers only intuition ("我现在的直觉是这样").
- **"A 4-month-old has seen more video than 30T LLM training tokens ≈ 30 min YouTube upload"** is presented as LeCun's recurring example without source.
- **AGI as 伪命题** is asserted with appeal to LeCun's authority but no independent definition of "general intelligence."
- **MoCo / MAE "don't scale" to LM-level impact** — mechanism (representation collapse? data efficiency? lack of generative loss?) is not unpacked.
- **Saining bet ("LLM will degrade to a communication interface; representation is the only thing that matters")** is asserted as a forward-looking bet without empirical demonstration in this episode.
- **"Computer vision faculty hiring is shrinking"** is asserted as a trend without numbers.
- **"FAIR's reorg / culture shift after ChatGPT made it anti-research"** is anecdote, not data.
- **DiT was "in fact" the real first-place ImageNet winner because the actual #1 was an ensemble** — value judgment, not what the leaderboard recorded; same caveat for the ResNeXt 2nd-place framing.
- **Year-misspeak**: Saining says top-PhD pay "in 2008 was $400-500K"; context makes clear he meant 2018.
- **Knife-holding robot "won't err without ever having seen the scenario"** — the generalization mechanism is asserted, not justified.
- **Seedance ~200B MoE diffusion** is acknowledged as 小道消息.
- **"Nobody else has made MoE work in diffusion at scale"** stated without citation.
- **"90-95% of generative-model quality is a data problem"** is a sweeping assertion offered without supporting evidence.
- **LeCun-recruited-3-times third time** is explicitly deferred ("之后再聊"); Saining did not confirm it was AMI on-record at that moment.
- **Fei-Fei's "specific low-point story"** Saining heard from her was deliberately omitted ("具体的事情可能就不方便讲了").

## Verification log

- **Sectioning**: chapters (15 author-supplied YouTube chapters); chapter #1 was `<Untitled Chapter 1>` and was renamed to "开场白与本期定位" in the section JSON before chunking. All other 14 titles preserved verbatim from `info.json.chapters`.
- **Transcript source**: faster-whisper large-v3 (CPU int8, local M4) — produced by `docs/videos/transcribe_batch.py` from local audio. YouTube provided no subtitles (manual or auto) for this video, so the standard yt-dlp path was unavailable.
- **Speaker name corrections**: guest's name was inconsistently rendered "谢赛明" / "谢塞宁" — corrected throughout to **谢赛宁 (Xie Saining)**. Host name corrected from "小骏" / "小俊" to **张小珺 (Zhang Xiaojun)** per channel metadata. LeCun appears as "杨立坤" / "杨立昆" / "杨乐空" / "样" — normalized to **Yann LeCun (杨立昆)**. Ilya appears as "伊莱亚" / "伊丽娅" / "Elia" — normalized to **Ilya Sutskever**. JEPA appears as "japa" / "了japa" — normalized.
- **Sections covered**: 15/15 ✅
- **Notable quotes traced verbatim**: 8/8 ✅ (each anchored by a distinctive 6-15-char substring in the local transcript)
- **Numbers traced**: 58/58 ✅
- **Sectioning method used**: chapters
- **Removed during verification**: none

## See Also

- `docs/videos/2026-05-01-su-yu-agent-tech-history.md` — Zhang Xiaojun's episode with Su Yu (the OSU NLP / Neocognition founder Saining is informally peer to); same host, same channel, complementary perspective from agent research rather than vision / world models.
- `docs/videos/2026-05-11-yao-shunyu-training-claude-gemini.md` — Zhang Xiaojun's episode with Yao Shunyu (the ReAct author whose "second half" framing Saining repurposes for world-model pre-training in §14); same host, same channel.
- `skills/research/youtube-wiki/SKILL.md` — the pipeline that produced this entry.
