# Gao Jiyang (高继扬): Catfish, Zeng Guofan, the Two Faces of Waymo and Momenta, a Wolf, and Xu Huazhe's Departure — A 3-hour Interview with the Founder of Galaxea / 星海图

**Source**: https://youtu.be/n4_c_HsodPg
**Channel**: Zhang Xiaojun Podcast (张小珺商业访谈录)
**Published**: 2026-02-13
**Duration**: 03:04:52
**Watched on**: 2026-05-14
**Sectioning**: chapters (13 YouTube-supplied chapters; titles preserved verbatim from `info.json.chapters`)
**Detected video language**: `zh` (no `info.json.language`; both `subtitles` and `automatic_captions` keys empty — no YouTube subs of any kind)
**Transcript source**: faster-whisper large-v3 (CPU int8) — produced by `docs/videos/transcribe_batch.py` from local audio. ASR consistently rendered the host as "小俊/小骏" — corrected from channel metadata to **张小珺 (Zhang Xiaojun)**. Guest-specific ASR corrections noted inline: 星海图 sometimes rendered "巨神智能" → corrected to **具身智能 (embodied intelligence)** when generic, kept as **星海图 (Galaxea / StarSea)** when referring to the company; "vimu/vmall/Wemo" → **Waymo**; "蒙曼塔/罗门塔" → **Momenta**; "杨志林" → **杨植麟 (Yang Zhilin)**; "绵羽" → **鲶鱼 (catfish)**; "图里" → **土里 (dirt)**; "上汤" → **商汤 (SenseTime)**; "派灵" → **π0**; "苏青" → **苏箐 (Su Jing)**; "Locum" → **Loco-manipulation**; "GBT" → **GPT**; "宗家使" → **清华系**; "1000两美金/一千两美金" → likely **~1000万美金 (~$10M USD)** per the guest's own restatement at 01:09:19; "RI" → **ROI**; "G-Dragon Plus" appears to be the same product as **G-LINK Plus** (ASR variant).
**Speakers**: Zhang Xiaojun (张小珺, host), Gao Jiyang (高继扬, guest — Founder & CEO of **星海图 / Galaxea**, ex-Waymo prediction/perception, ex-Momenta NVA mass-production lead)

## TL;DR

- **The "catfish" arc — pragmatism over romanticism.** Gao narrates a Shijiazhuang-childhood → Tsinghua EE → SenseTime intern → USC PhD → Waymo → Momenta → 星海图 path framed by 张小珺 as the **anti-thesis of 梁文峰/杨植麟 技术浪漫主义**. Zeng Guofan's biography is the load-bearing personal model: after his UGVR mentor refused him a strong recommendation, Gao reframed the setback by reading 曾国藩's pivot from 儒家清流 to a 40-year-old pragmatist who built the Xiang Army — concluding that what matters most is "how many people and how many resources you can rally to actually finish something."
- **Waymo's diagnosis: "founderless."** The 2018 AD stack was architecturally identical to the 2008 DARPA stack — robotics-rooted, modular, with neural nets only swapped in for individual sub-modules. Perception alone ran "几十个模型" (dozens). Tesla unified perception around 2018-2020 because Musk could force a top-down rewrite. Waymo couldn't, because **"vimu是没有Founder的"** — Google's founders technically own it but have no time. Gao's takeaway is that organizations that can't locate top-down corrective force in a single decisive leader will stall on architectural rewrites, even when the engineering DNA is world-class. VectorNet (with **赵航**, later his co-founder) was his Waymo capstone.
- **Momenta was Waymo's opposite — the "catfish" rotation.** Joined late 2020 right as Momenta won the SAIC 量产 project; rotated through perception → localization → infra → 规控 to convert rule-based modules into neural networks; eventually led the **NVA (高速 NOA) mass-production delivery** to SAIC. He explicitly admires 曹旭东's 战略能力 (the chip bet "晚行动一年都没有今天的这个") but flags the dark side — "把真相说出来" past a threshold "成伤害." Core algorithm teams at Momenta still "到12点的, 一周6天打底" — Gao calls this "为所有人负责的一种选择," not 曹旭东's personal failing.
- **GPT + Optimus = the trigger to forfeit ~$10M USD.** GPT-3 → InstructGPT made "the world believe in AI again"; mass-production AD proved 端侧 sensors / compute are "大差不差" with humanoid; Tesla announced Optimus — "我得开稿了." Gao left Momenta in May 2023 (after closing the SAIC NVA delivery), **gave up ~1000 万美金 in options** ("具体记不太清是不是这个数大概是这个"), wrote a "糟糕的BP," and raised a ¥30M seed at ¥200M pre/post (IDG led, 百度风投 + 金沙江 followed). 23-year-end "啥也没有的状态" — all of 2024 was 补课 on **整机 + 供应链**, because long-term moat = **物理世界数据闭环**, which requires owning the hardware.
- **Algorithm has the weakest moat — 2-3 months — and 许华哲 is leaving.** Gao's "传播周期" framework: 整机供应链 = 12-18 months to copy; 终端渠道 = 6+ months; 数据体系 = +6-12 months on top of 整机; **算法 = only 2-3 months** (because everyone open-sources and papers exist). 许华哲's exit (to a 2C 家庭应用 startup that 星海图 will invest in) is framed as values + life-cycle fit, settled internally in **2025-08** when 赵航 took unified control of the foundation-model team. Brain architecture is **dual-system VLM (server) + VLA (edge)**; embodied AI has **no business-synergy advantage for big-co incumbents** (demand is 千行万业 fragmented, supply is all-new, car parts don't transfer), so competition collapses to talent + capital — which a startup can match. Closing: latest round at **~¥10B valuation (30x in two years from ~¥300M)**, 200+ people (20x scale-up from 15 in two years), wolf metaphor, Three Kingdoms book rec, "万台 出货量 in 生产业场景" as the immediate BET.

## Why it matters

A 3-hour first-person genealogy of why a top-tier robotics startup chose vertical integration (整机 + 数据 + 模型) over an algorithm-only play, narrated by the second-generation 量产-driven founder who openly forfeited ~$10M USD to start it and who diagnoses both Waymo (founderless) and his own old company Momenta (relentless to the point of hurt) on the way. Pairs unusually well with Su Yu's Agent-history episode as the **physical-world counterpart**: where Su Yu argued coding is the fabric that dissolves agent-boundaries, Gao argues physical-world data closure is the only moat embodied-AI startups can actually defend — and that 许华哲's departure is the structural lesson that algorithm-by-itself has a 2-3 month copy window.

## Section summaries

### §1 — 开场白与本期定位  [[00:00 → 01:49](https://youtu.be/n4_c_HsodPg?t=0)]
- Cold open splices teaser clips: 曾国藩 "从20多岁是一个儒家清流 怎么到40岁变成一个特别有事工的一个人"; 杨植麟 confirmed as Gao's Tsinghua classmate ("高不可攀的大神"); Gao confirms "你是当时引入的一条鲶鱼吗 — 我应该算是吧"; Gao confirms he "从M出来创业 我放弃了所有的期权 全部放弃了" — host probes the figure, ASR returns "一千两美金" (likely "1000万美金" per §7).
- Host's editorial thesis [[01:00](https://youtu.be/n4_c_HsodPg?t=60)]: "为什么中国具身智能产业里 没有出现一个像梁文峰 杨植麟这样带着浓重的技术浪漫…和浪漫主义色彩的人 这让我有时候有点失落 直到我认识了高继扬 他似乎是这种极致的浪漫主义的反面 代表了一种极致的 效率 工程拆解 与实用主义."
- Host previews the central tension [[00:26](https://youtu.be/n4_c_HsodPg?t=26)]: 许华哲 (algorithm-side co-founder) is leaving — is this a parallel to **邵青→曹旭东** at Momenta? Does it mean algorithm innovation is no longer central for a robotics company at this stage?
- Gao's anti-romantic frame quoted by host: "做机器人行业就是一个链条极长的行业 有时候 你就是要把你的头伸到土里去."
- People: [Gao Jiyang (高继扬)](https://youtu.be/n4_c_HsodPg?t=74), [Yang Zhilin (杨植麟, ASR: 杨志林)](https://youtu.be/n4_c_HsodPg?t=8), [Xu Huazhe (许华哲)](https://youtu.be/n4_c_HsodPg?t=26), [Shao Qing (邵青)](https://youtu.be/n4_c_HsodPg?t=39), [Cao Xudong (曹旭东)](https://youtu.be/n4_c_HsodPg?t=41), [Liang Wenfeng (梁文峰)](https://youtu.be/n4_c_HsodPg?t=66).

### §2 — 冲刺型小孩  [[01:49 → 12:07](https://youtu.be/n4_c_HsodPg?t=109)]
- Looks older than his real age (born 1992); attributes it to "城府" from founder life — daily contact with many types of people. Host's opening: "我一直默认你是80后…我看你资料才发现你是92的."
- Childhood arc — Shijiazhuang (河北石家庄), ordinary 玉东小学, top-5/10 of class but not the best; first effort came in 6th-grade summer for the 初一 streaming exam, where he placed **3rd in grade**.
- 初中 at **石家庄二十七中** mostly mediocre ("几十名"), ramped up in 初三; high school **石家庄二中 生理科实验班** unlocked the physics-olympiad track.
- Olympiad outcome: **省一 → 省队 → 全国二等奖 (2010-11, 厦门)** — at a hotel meeting with the Tsinghua admissions officer he picked **电子工程 (EE)** because he believed chips/集成电路 had future, "but no deep reasoning behind it."
- Self-assessed as **勤奋 not 天才** — "我见过真正有天赋的人…我主要是靠勤奋 靠努力"; his method = **归纳总结** (problem-type taxonomy → exam pattern mapping). Pre-college work style was **burst/sprint mode** — hard push when he sensed a moment mattered, relaxed otherwise.
- Decided to 创业 around 大二/大三 at Tsinghua; participated only in 挑战杯 科创 competitions, no real venture. Watched the 2011-15 mobile-internet wave (校内网 → 打车 → 外卖 → 共享单车) and concluded "**跟我没关系**" — too young, too inexperienced; saw 王兴, 戴威, 陈伟 as that wave's protagonists.
- Decision moment quoted: "那时候就隐约感觉到 这个就是大机会 然后但是跟我没关系 [...] 我太年轻了 然后我这个啥也不懂啥也不会."
- 大三 picked 微纳电子 out of chip ambition but found it heavy on memorization and pre-国产替代 "saw no creation path." Pivot moment seeded **大四 when 汤晓鸥 老师 (商汤) visited campus** — setup for §3.
- People: [Tang Xiao'ou (汤晓鸥, SenseTime)](https://youtu.be/n4_c_HsodPg?t=720), [Wang Xing (王兴)](https://youtu.be/n4_c_HsodPg?t=614), [Dai Wei (戴威, ofo)](https://youtu.be/n4_c_HsodPg?t=600), [Chen Wei (陈伟)](https://youtu.be/n4_c_HsodPg?t=614).

### §3 — 学习曾国藩  [[12:07 → 25:30](https://youtu.be/n4_c_HsodPg?t=727)]
- **The Zeng Guofan trigger** [[14:02](https://youtu.be/n4_c_HsodPg?t=842)]: after a UGVR summer at "散服" (likely Sun Yat-sen / 中山大学 UGVR), his mentor declined a strong recommendation, costing him top US offers. UNC and UCSD — schools he didn't ask the mentor to recommend — admitted him. He calls this a "小打击" and admits "到现在我也不知道为什么."
- **The takeaway from 曾国藩's biography** [[14:47](https://youtu.be/n4_c_HsodPg?t=887)]: tracing Zeng from 20s-era 儒家清流 to a 40-year-old 事工 pragmatist who built the Xiang Army — Gao's distillation: "他发现最重要的还是，当你要做一件事的时候，你到底能有，能拉动多少资源，有多少人，多少资源愿意跟你一块去做这件事，并且最后把这事做成。他发现这件事是最重要的." He maps this to himself: academic-elite track is closed, so build a different trajectory — go into the real world, get results.
- **SenseTime internship** under 汤老师 (offered "十几个" of these opportunities; Gao stayed ~"四五个月"). Trained his first neural net (**pose estimation** — predicting shoulder/elbow joint keypoints) under mentors **李成** and **鲁淑**. Crystallizing moment riding a bike out of 清华科技园创业大厦: "我感觉这以后，这个神经网络可以代替人在数据当中发现规律。这个事太牛了。我以后得做这个."
- **Tsinghua talent shock** [[21:11](https://youtu.be/n4_c_HsodPg?t=1271)]: classmates 杨植麟 and **韩衍俊** were "天才" benchmark. Gao self-rates **电子系 top 30-40%**. On 韩衍俊's homework: "**就是他写的作业，我看他写的作业我就想学习一下我都看不太懂，我得换一个人作业抄一下**." Takeaway: 谦虚, 低调, 天外有天.
- Pre-college grind methodology: "**别人刷一遍的题我刷两遍，别人刷两遍的我刷四遍**…就是归纳总结." 语文 was so weak that 一卷+二卷 together (~100/150) only matched a strong student's 一卷 alone (~90).
- At SenseTime he first met **曹旭东**, then a core technical leader on face recognition — pre-acquaintance that later became a reason he joined Momenta.
- People: [Tang Xiao'ou (汤晓鸥)](https://youtu.be/n4_c_HsodPg?t=741), [Zeng Guofan (曾国藩)](https://youtu.be/n4_c_HsodPg?t=842), [Li Cheng (李成)](https://youtu.be/n4_c_HsodPg?t=1110), [Lu Shu (鲁淑)](https://youtu.be/n4_c_HsodPg?t=1111), [Han Yanjun (韩衍俊)](https://youtu.be/n4_c_HsodPg?t=1332), [Sun Chen (孙晨/晨哥, USC senior → Brown professor)](https://youtu.be/n4_c_HsodPg?t=1513), [Cao Xudong (曹旭东)](https://youtu.be/n4_c_HsodPg?t=1423).

### §4 — 提高顶会命中概率  [[25:30 → 33:46](https://youtu.be/n4_c_HsodPg?t=1530)]
- **PhD target reverse-engineered**: "四年毕业 + 4-5篇顶会 (CVPR/ICCV)." 老板 (a "七十多岁的印度老头" PI at USC) signed off on early graduation 18年年底.
- **三套路 framework for 顶会 papers**: (1) **挖坑型** — propose new problem, build benchmark/dataset (hardest); (2) **性能型** — beat SOTA on existing problem; (3) **效率型** — same performance with less compute / less supervision / less data. "几乎就是所有的paper都能往这三类上去套."
- **Cap-rate boost #2 — 多投**: "**还得多发 就是 别人一次发 投一篇 我投两篇 然后就提高概率**." When his idea bandwidth exceeded his own, he distributed ideas to lab mates ("把大家没有充分发挥的时间发挥出来").
- **Industry-filter for first job** — used interview process as a four-way research: 上汤 (商汤) / 广告 / 云 / 自动驾驶. **Filter**: AI must be the **底层 / 不可替代变量** ("没AI这行业不存在") AND industry must be big enough.
  - 商汤 passed → "**会不会干成外包**" worry (no clear product, high delivery cost).
  - 广告 / 搜索 passed → ML / pre-ML methods already worked; AI is optimization, not absolute variable.
  - 云 passed → "模型包装成API" too superficial.
  - **自动驾驶 selected** → AI + robot, the first industry form of "物理世界AI." Critique of traditional robotics: "传统robot还是控制优化然后呢这个SLAM…他机器人里面更偏机器 而不是更偏人 你想他变成人还是得有AI的这种方法."
- Numbers: PhD target 4 yrs vs. typical US 5-6 yrs [[25:57](https://youtu.be/n4_c_HsodPg?t=1557)]; 4-5 顶会 papers as the calibration target [[26:16](https://youtu.be/n4_c_HsodPg?t=1576)]; 老板 ~70+ years old [[27:44](https://youtu.be/n4_c_HsodPg?t=1664)]; graduation request submitted 18年年底.

### §5 — Waymo是没有创始人的  [[33:46 → 55:38](https://youtu.be/n4_c_HsodPg?t=2026)]
- **The architectural diagnosis**: "**整体的自动驾驶在18年的技术架构和08年的技术架构没区别。但是这一套技术架构的底层逻辑不是AI的,它的底层逻辑是robotics的**." Stack = perception, localization, offline HD map, planning (decision → planning → control); only individual modules (e.g. LiDAR clustering) got swapped for neural nets.
- **Robotics vs AI logic**: robotics decomposes, makes modules interpretable, handles corner cases; AI is data-driven, end-to-end, benchmark-optimizing, accepts some cases regress as aggregate metric improves. Perception alone at Waymo ran "**感知里边就可能有几十个模型,真的是几十个模型**" — multiple detectors, trackers, per-class classifiers (pedestrian, vehicle), scene-understanding classifiers. Tesla unified perception around 2018-2020 and pushed AI-native end-to-end from the top.
- **The "founderless" thesis** [[42:07](https://youtu.be/n4_c_HsodPg?t=2527)]: "**vimu是没有Founder的。它的Founder其实是Google的Founder,但是Google的Founder又没时间直接去管这事,所以在这个里面就是自上而下的力量,我觉得是缺失的。不像特斯拉,马斯克说干啥,哪怕是错了他也能开始干**…怕的是力量不集中不统一."
- **Defends Waymo's engineering DNA**: "**我倒不觉得vmall是个科学的公司,vmall非常的工程。只是…我觉得他们俩的区别还是在于就是对待AI的态度上,以及面向AI驱动的整体的系统设计的调整速度和力度上面**."
- **Two Waymo diagnoses**: (a) 大公司病 — ~1000 people when he joined (Jan 2019), ~2000 when he left (H2 2020); his perception team grew from ~10 → 70-80 in that window; (b) deeper root cause = founderless governance.
- **VectorNet origin** [[49:37](https://youtu.be/n4_c_HsodPg?t=2977)]: prior approaches rendered the HD map as image + CNN, suffering from CNN's local receptive field vs. map's long-range structure. With **赵航 (Zhao Hang)** — described as "principled, super-stable emotionally, very nice" — they encoded the map as **vectors fed into a graph neural network using self-attention** (light Transformer-style operator scaled to limited compute). Shipped as VectorNet — his and 赵航's first collaboration.
- **The "拆解加测量" engineer's mindset** [[53:05](https://youtu.be/n4_c_HsodPg?t=3185)]: "**工程师思维就是拆解加测量。就是把一个复杂的问题,拆解成做干个稍微不那么复杂的子问题,然后再拆解再拆解,你写代码,拆到最后就是一行一行的代码,测量到最后是什么,是一个单元测试**." Top/middle/bottom metrics reveal how the system runs.
- **Four AD business models**: (1) Robotaxi/fleet (Waymo) per-trip revenue; (2) carmaker selling AD as software subscription (Tesla); (3) supplier NRE + license (Momenta); (4) Huawei — hybrid, profit captured at whole-vehicle level via brand + channel.
- **Why he left H2 2020**: AD system, engineering craft, small-team tech-lead experience had converged. Two gaps Waymo couldn't fill — "too far from the product" and "too far from how a company is actually run," plus he had always wanted to start a company.
- People: [赵航 (Zhao Hang)](https://youtu.be/n4_c_HsodPg?t=2977), [Elon Musk / 马斯克](https://youtu.be/n4_c_HsodPg?t=2308).

### §6 — Momenta是极致的反面  [[55:38 → 1:02:25](https://youtu.be/n4_c_HsodPg?t=3338)]
- **Why China, why 量产**: after USC his "first lesson" was attacking 物理世界AI系统, engineering at the IC level, leading a small team to a system result. Next step: (1) build a product, (2) further prepare himself for founding, (3) do it in China. Picked **量产AD** over RoboTaxi because the 量产 trajectory delivers product value to users faster.
- **Final-round choices c. 2021** were Huawei, Momenta, and the EV OEMs (蔚小理). 蔚来/理想 weren't AI-strong yet; 小鹏 was already doing well — "**我是喜欢去不行的地方然后把一个东西做好的**." Interviewed at Huawei via 师兄 **陈一伦** and **苏箐 (ASR: 苏青)**, and at Momenta via **孙刚** and 旭东.
- **Why Momenta**: (a) no 大公司病 yet, (b) 曹旭东 was **强势 + 懂技术 + AI believer** with firm long-term goal. Also: **思博 / 晋美** (former SenseTime colleagues) already inside, smoothing the social entry.
- **The error-correction principle** [[58:29](https://youtu.be/n4_c_HsodPg?t=3509)]: "**我觉得一个组织要成功必须要他可以犯错但是得有一个人说我们错了然后我们改**." Judged 旭东 fit that mold.
- **Momenta's 极致 thesis dated 2018** [[59:38](https://youtu.be/n4_c_HsodPg?t=3578)]: "**M从18年开始对吧那个时候就明确的提出要做量产自动驾驶然后通过量产自动驾驶然后这个飞轮对吧然后走向robotaxi**." A gutsy call — all of Waymo, Pony.ai, 文远, 百度 were on the direct-RoboTaxi path. Gao reframes 极致 as iteration: "**只要是一个能够不停的迭代永不满足的其实都是追求极致的一种表现**."
- **The fleet-economics argument for 量产**: "**我自己养一个车队100台车1000台车,1000台车已经很多了,那1000台车我要去把一个城市覆盖掉其实都很困难**" — ship the AD software into 量产车 (start with assistance/parking) to turn data acquisition into a commercially-driven loop where business value funds data which funds the AI.
- **Redefining "product"** [[57:23](https://youtu.be/n4_c_HsodPg?t=3443)]: "**我对产品的定义不是说2C的这个才叫产品而是说我想做一个直接能够能够给某一些用户创造价值的他让他用起来**."
- The org itself iterated from "一次成功的交付复制到几十次,今天可能都有一两百次车型的这个交付了."

### §7 — 鲶鱼  [[1:02:25 → 1:19:26](https://youtu.be/n4_c_HsodPg?t=3745)]
- **Joined Momenta late 2020**, just as M10 won the **SAIC 量产** project; 2021 was the delivery year, hitting "two firsts" — turn a demo into a product AND deliver that product to a major 2B (车企) client.
- The org wasn't ready: architecture wrong for 量产, team capability ceilings ("能力战") didn't match the intensity, frequent restructuring with active and passive 淘汰. Gao frames this as necessary "打仗"-style baptism.
- **Used flexibly as a catfish** by 旭东 and 孙刚: rotated through perception → localization & 搏射系统 → infra → **规控 (planning & control)** to convert traditional rules-based modules into neural-network ones — "**我们想把规控和定位从传统的rubes变成deep learning的，那变成神经网络的，所以让我去搞**." Final project = **NVA (高速 NOA) mass-production delivery** to SAIC.
- **Methodology, not domain expertise**, was the keep: rapidly enter an unfamiliar domain, decompose, match people-to-sub-problems, measure feedback, expand what works.
- **Why he left** [[01:09:59](https://youtu.be/n4_c_HsodPg?t=4199)]: (1) **GPT-1/2/3 + InstructGPT made the world "再一次相信AI"** — "**GBT 1 GBT2 当时到了GBT3，然后instructGBT也出来了，我觉得这东西让世界再一次相信AI了。这很重要就是这个有没有这个相信**"; (2) 量产 AD proved end-side sensors & 端测算力 are roughly equivalent to humanoid needs; (3) Tesla announced **Optimus** — "**我得开稿了**."
- **Forfeited ~$10M USD in Momenta options**: "**我从M出来创业我放弃了所有的期权,全都放弃了 [...] 1000万美金可能是有的吧**" — he hedges later in §8 with "具体记不太清是不是这个数大概是这个." He's explicit that retention was partly his own (创业 calling) and partly a gap in Momenta's "顶级人才的保有和持续培养."
- **Culture contrast** [[01:13:28](https://youtu.be/n4_c_HsodPg?t=4408)]: "**Wemo 真的是工程师的天堂…你可以有最好的这个infra,然后呢你可以有最好的同事,然后呢你的领导对你的support对你的支持都是非常非常温暖的非常宽厚的**." Momenta = absolute results-driven, constant pressure; 总监+ engineers directly face tough 国内车企 clients who "openly 骂"; "让徐东来解释" is a normal thing to hear. Chinese 支架 environment trains 综合能力 better — forces engineers to face *真实现实世界*.
- **On 旭东** [[01:18:30](https://youtu.be/n4_c_HsodPg?t=4710)]: strongest trait = **战略能力** — the chip bet "**这是他可能三年前的决定，然后两年前两年多前这个公司开始启动正是…如果这个决定可能晚做一年，晚行动一年，都没有今天的这个**." Dark side: "**其实我觉得不是 aggressive，是把真相说出来…但是可能说有的时候是用一种很非常直接的方式去表达出来，这种方式本身就会让很多人感受到压力，这种压力大到一定程度就会成伤害**" — Gao admits he himself has this same trait.
- **Industry intensity disclaimer** [[01:12:35](https://youtu.be/n4_c_HsodPg?t=4355)]: "**核心算法团队都是到12点的,然后一周是6天都是打底的,然后极致的卷…这个行业就是这样…所以我觉得拼是为所有人负责的一种选择**" — explicitly NOT blaming 旭东.
- **凯歌 (Yu Kai, Horizon Robotics)** named alongside 旭东 as a 榜样 of scientist → entrepreneur transition [[01:18:55](https://youtu.be/n4_c_HsodPg?t=4735)].

### §8 — 从一份糟糕的BP开始  [[1:19:26 → 1:35:26](https://youtu.be/n4_c_HsodPg?t=4766)]
- **Decision timeline**: thought clear by 2022年底 (around 30th birthday). Officially left Momenta **2023-05** after closing the SAIC NVA delivery. Drove around Tibet in July to decompress; August started the BP + fundraising.
- **Forfeiture restated** [[01:20:44](https://youtu.be/n4_c_HsodPg?t=4844)]: "**放弃1000万美金 [...] 我具体记不太清是不是这个数大概是这个 [...] 我其实没什么心疼的 一点都没有**." Decision logic: "**首先我觉得我想我最 care 的事还是我想去做的那件事 而不是一些钱什么东西的**."
- **What Momenta taught him** [[01:21:46](https://youtu.be/n4_c_HsodPg?t=4906)]: "**我学会了 什么叫以客户为中心…客户为中心不是生硬的说客户说让我们做什么我们就做什么 而是真的站在客户的角度 去看他的需求是什么 甚至帮助他挖掘他的需求是什么 然后提出更好的方案**" — also applies to internal upstream-downstream collaboration. "以自我成长为中心" / "以技术领先性为中心" are wrong framings for a company.
- **Initial bad idea**: 末端配送机器人 — wheeled chassis + light manipulation (按电梯, 开关门, 拿放东西). Quickly self-rejected once the two foundational principles were clear: (1) **具身智能 must be 整机 + 智能**, because long-term moat is physical-world data closure; (2) want to do **落地产生价值** work, not pure research.
- **First round ¥30M (3000万人民币)** [[01:27:13](https://youtu.be/n4_c_HsodPg?t=5233)] at ~¥200M pre or post ("我忘了是投前两亿还是投后两亿 人民币"). IDG led, 百度风投 followed, 金沙江 smallest cheque. IDG entered via classmate **李一康** internally referring to **邵辉**, then **肖军** closed the round. Angel investor logic: "**他觉着 虽然你现在什么也不行 但是他觉着你还是有潜力的**." 朱啸虎 was at 金沙江 then; **宇同学姐 (于同学姐)** was the first follow-through but left 金沙江 later.
- **Add-on round (加轮) via CFUN** (清华电子系 王老师 + 姚老板 fund) closed pre-Spring-Festival 2024: ~¥10-20M (not quite ¥20M) at **¥300-400M post-money**. Gao notes that compared to today's "两亿美金、一亿美金起步" deals, those 2023 investors "有勇气" — embodied AI wasn't yet consensus.
- **Tearing apart competitors' products** [[01:29:33](https://youtu.be/n4_c_HsodPg?t=5373)]: "**拆啊就是买别人的产品回来拆 然后就看他们这东西是怎么搞的 [...] 我拆出来一个东西 这是什么东西 [...] 然后呢就淘宝拍张照片 以图搜图**" to find suppliers. The company's later 结构 lead, on his first visit, called it "太可怜了" and gifted a toolbox (screwdriver, hammer, tweezers, axe).
- **Co-founder 杨泽义 (b. 1997, 南科大)** joined Jan 2024 — introduced by 五元 (the investor referred without investing). Was running a robotics 培训 business for middle/high schoolers; Gao met him twice, invited him in. Now signs off on the structural design of essentially every product.
- **Equity & "流变形" partner philosophy** [[01:33:48](https://youtu.be/n4_c_HsodPg?t=5628)]: "**我要求我自己 做一个中等面积流变形 然后我希望我的简易 我的合伙人团队 做更大面积 组成一个更大面积流变形 然后这样我们这个团队就 会变得很强 而且很均衡的强**." Equity scale is "原始股的百分点" level (visible on 天眼查). Continual partner-introduction since the three-founder kickoff (高 + 赵航 + 田飞) — covers 泽义, 华喆, 于磊 (商业化), and CFO **天奇** added a few months before recording.

### §9 — 挣扎着做整机和供应链  [[1:35:26 → 1:49:32](https://youtu.be/n4_c_HsodPg?t=5726)]
- **The integrated-machine moat argument** [[01:35:48](https://youtu.be/n4_c_HsodPg?t=5748)]: "**我们要做巨神智能 巨神智能长期壁垒 建立在物理世界的数据闭环之上 对只有把这个东西构建起来了 我们才真的有 一个说别人进不来 我们能够长治久安的这么一个壁垒了**" — ASR 巨神 → 具身. Mid/short-term commodity = 整机 + 智能 product, not a standalone 大脑 / algorithm: "**我们要去 世界提供的商品 大概率不是一个算法 不是一个所谓的大脑 它是一个整机加智能形成的 在物理世界能够有执行能力的 这么一个物理实体**."
- **The car-industry analogy** [[01:40:00](https://youtu.be/n4_c_HsodPg?t=6000)]: "**就相当于汽车产业退回了100多年前 汽车产业退回100多年前企业这个还有智能技术 所以我就得这两件事同时去做**" — in AD the OEM sits between you and the user, breaking the data loop and capping experience; AD's 20-车企 customer set is a hard cap on a supplier. Embodied AI is the rare chance to *be* the car company.
- **Form-factor pivot to 轮式+躯干 (wheels + torso) around 2024-03** [[01:42:11](https://youtu.be/n4_c_HsodPg?t=6131)]: "**双足的运动控制和双臂的智能操作 同时解决 这叫Locum [Loco-manipulation]…这事也没解决 那所以我认为这个问题我们得先解悟 我先把上肢操作好**." Most real scenes don't need stair-climbing. "智能定义本体 / 数据定义本体" — manipulation is the entry point, upper body matters most. R1 product emerged from this.
- **23年底 hardware status** [[01:40:32](https://youtu.be/n4_c_HsodPg?t=6032)]: "啥也没有的状态" — entirety of 2024 = 补课 (catch-up) on 整机 + 供应链.
- **整机 vs AI engineering culture** [[01:47:16](https://youtu.be/n4_c_HsodPg?t=6436)]: "**AI里面其实 更强调的是 [...] 人才密度…但对于一个 这个机电系统整机来说 我觉得更强调的是 整个研发流程的严密性**" — 构型设计 → 结构设计 → 线束/气式系统 → 软件平台 → EVT → DVT → 一致性 + 老化测试 → 生产, same as consumer electronics.
- **Go-to-market = Crossing-the-Chasm** (ASR: 跨越同工 → 《跨越鸿沟》) [[01:43:12](https://youtu.be/n4_c_HsodPg?t=6192)]: innovator (academic devs like **李飞飞**, top US PhDs) → enterprise R&D (e.g. **蚂蚁 LingBoat VLA** collab, **Physical Intelligence**-style teams) → 生产力型开发者 doing 二次开发 → 集成商 → end users. Precedents: Apple Macintosh, **拓竹 (Bambu Lab) 3D printers**, **Unitree (语术) quadrupeds**.
- Pivot from 配送 → 具身智能 / 操作 / 开发者市场 attributed partly to luck: "啥也不成熟…智能也不 ready,客户也不成熟,市场也不在."

### §10 — Data Recipe  [[1:49:32 → 2:16:33](https://youtu.be/n4_c_HsodPg?t=6572)]
- **Two principles inherited from AD**: **端到端 (no modular layering)** + **真实数据 (no 仿真)**. Pre-π0 era used **Diffusion Policy / small VA models**; pivoted to **VLA** in 2025 after π0 (ASR: 派灵) proved it worked.
- **Roadmap by year**: 24年 = 整机 + 融资; 25年 = 数据 + 智能体系 + 持续融资 + 开发者市场 (**150+ customers**); 26年 = 场景 + 应用, dev-market → 生产业市场.
- **Aug 2025**: what Gao calls "**全球第一个全国第一个**" open-source release of robotics data — **500 hours** of high-quality teleop data he personally collected, plus open-source base model (G0).
- **Cost-of-intelligence framework**: total cost = data acquisition + training + engineer team. **Ratio of acquisition to training ≈ 1:5 to 1:10** [[01:57:55](https://youtu.be/n4_c_HsodPg?t=7075)]: "**数据的获取成本和数据的使用也就是训练成本的关系大体是1比5到1比10,也就是说我花一块钱搞来的数据我得花5到10块钱才能把它训明白**" → justifies spending more on data quality.
- **Real-data cost math** [[01:59:15](https://youtu.be/n4_c_HsodPg?t=7155)]: 1 hr teleop = 3-4 hrs human labor (robot setup/reset) + robot depreciation. 整机 ¥100k amortized over **1000 hrs lifetime** (limited by 减速箱 — 崩齿 / precision loss) = ~¥100/hr depreciation + labor → **¥200-250 / hour**. **10万 hours ≈ ¥2500万** ≈ a human's 0-18yr physical interaction budget.
- **Data pyramid critique** [[02:02:50](https://youtu.be/n4_c_HsodPg?t=7370)]: "**数据金字塔是对的,但谁说的数据金字塔非得长成这样,非得是这么个比例,没人说。我们得看智能要什么**" — recipe is determined experimentally: "AI 归根结底还是实验科学." Categories: robot-centric (teleop), human-centric (**5mi / Sunday gloves, POV head-cam**), 3rd-person internet video, sim (graphics-based vs world-model-based).
- **Real-data scalability requires**: (1) enter real scenarios, not 素材场 (staged labs); (2) **众包** the data collection via distributed devices + government support + business-model backing. North America led on **无本体** collection via gloves/grippers like **Sunday**'s.
- **"Data Recipe is the secret"** [[02:04:12](https://youtu.be/n4_c_HsodPg?t=7452)]: "**这个叫data recipe。很多大圆模型公司今天的最大的秘密就在于这个。对于巨神智能公司来说也是**" (ASR 大圆→大模型, 巨神→具身).
- **Supply-side metrics**: **速度 / 精度 / 泛化性** — speed cap of imitation learning ≈ **80-90% of human** [[02:08:56](https://youtu.be/n4_c_HsodPg?t=7736)]; precision 厘米 → 毫米; generalization (万物抓取 is zero-shot, fold is partial zero-shot, new garments still need data).
- **Demand-side filter for "good scenarios"**: speed tolerance, low failure-cost, 全球化 unified form (excludes hotel/retail), 爆发力 (1台→10k台 fast). **Five action primitives** = **Carry / Pick / Pack / Fold / Operate** — most labor jobs are 20-40 combinations.
- **First concrete commercial wedges**: warehouse **bin-picking** (massive SKU count breaks Kiva / 夹包 / 协作臂 solutions) + 智能制造 **场内物流 / SPS (集中分装)**. Avoid assembly for now (毫米 + flexible). Abstraction: "**Pick anything, place to somewhere**" — already demoed on real-world objects via real-data-driven VLA.

### §11 — 机器人大脑  [[2:16:33 → 2:29:37](https://youtu.be/n4_c_HsodPg?t=8193)]
- **Dual-system brain definition** [[02:16:54](https://youtu.be/n4_c_HsodPg?t=8214)]: "**首先呢我们先定义一下大脑…我觉得这个世界上会有两个很重要的基础模型…一个我们叫做动作的基础模型 也就是VLA 这个动作基础模型最终是产生 Action 就是驱动一个本体…它的输入是Vision和Language 那么还有一个模型其实是做 上层的 这个指令的拆解 逻辑的思考能力的 这个往往是一个多模态的语言模型 VLM…今天我们的这个所谓大脑的这个结构其实就是这两个模型的一个组合 我们叫双系统**."
- **Edge-compute split** [[02:20:46](https://youtu.be/n4_c_HsodPg?t=8446)]: "**我端测的算力其实是有限的 我不可能把一个 几十币 [B] 的一个 一个推理模型 甚至是上百币 [B] 的一个推理模型 放在我的端测 这是不可能的这个一定是在服务器上的…所以执行动作的模型一定是要在端测**." For **工商业 with only 20-30 fixed actions**, skip VLM entirely and call VLA's language interface directly; VLM is essential for generalized home-scale tasks.
- **Open-source milestones**: **G0 (model + data, Aug 2025)** — China's first company-level open-source — was followed by friends/competitors in Sept/Dec/Jan. **G2+ (Jan 2026)** integrated base model with **R1 lite 整机** for an "out-of-the-box" experience.
- **Maturity ladder** for robot capability: demo-in-the-video → demo-in-the-office → demo-in-the-wild → application. **万物抓取** demoed live in Singapore, Korea, US customer sites, and at investor AGMs.
- **VLA success factors**: (1) real-world robot/human data — NOT internet, (2) algorithms (Transformer/Diffusion + proprietary innovations), (3) compute/capital infra, (4) talent. Big-co incumbents have (3)+(4); they lack (1).
- **Why China hardware-owning startups beat US labs on data** [[02:25:13](https://youtu.be/n4_c_HsodPg?t=8713)]: "**你如果不懂模型 你是没法定义好的数据体系的 你就天天光在那 那个乱七八糟的采吧 你采完数据全都是垃圾数据 没什么用 必须得有懂基础模型的公司定义这套数据体系和数据的治理体系**." US companies (e.g. via **Jie Tan / 谭杰** at Google) are "panicking, no data, scramble for data."
- **Hardware vs brain priority** [[02:22:01](https://youtu.be/n4_c_HsodPg?t=8521)]: "**一定是大脑 一定是模型更重要 但是 我们为了做好模型 我的整机 一定也要好**" — body matters for data, data is for intelligence.
- **Tesla path analogy** [[02:26:44](https://youtu.be/n4_c_HsodPg?t=8804)]: 整机 + 数据采集 + 端到端模型. Key difference from Tesla: cars have a pre-existing market; robots don't sell themselves yet.
- **No-business-synergy thesis** [[02:28:31](https://youtu.be/n4_c_HsodPg?t=8911)]: business-synergy is the killer dimension for big-co's vs. startups in OTHER domains (字节+飞书+抖音 for LLMs; carmakers owning data & users for AD). For embodied AI it **does not exist**: demand-side is 千行万业 fragmented; supply-side is all-new (car parts don't transfer; self-driving road data is not useful for manipulation). Competition collapses to **talent + capital** — where a startup can match a big company.
- People: [Jie Tan (谭杰, Google DeepMind)](https://youtu.be/n4_c_HsodPg?t=8553), [Tesla (特斯拉)](https://youtu.be/n4_c_HsodPg?t=8804), 字节 (ByteDance), 小米 (Xiaomi), NVIDIA, Physical Intelligence, 理想 / 小鹏 (Li Auto / XPeng).

### §12 — 许华哲的离开  [[2:29:37 → 2:39:04](https://youtu.be/n4_c_HsodPg?t=8977)]
- **量产文化 reframe** [[02:29:43](https://youtu.be/n4_c_HsodPg?t=8983)]: "**我觉得我想做一个小小的修正,就是我觉得不是量产文化,而是为客户创造价值这件事到底有多重要**" — when productivity scenarios proved infeasible, pivot to the developer market; any customer with real demand + working unit economics qualifies.
- **许华哲's exit framing** [[02:31:42](https://youtu.be/n4_c_HsodPg?t=9102)]: "**华者是一个非常有影响力的一个科学家…但是呢就是说我觉得确实是存在一些就是说我们到底是要去做一个务实创新bayout客户价值一步一步来,这样的还是说可能我们就是要更多的去做一些超前的创新,之间我们是要有一个balance**." 许华哲 is doing a **2C 家庭应用** startup; 星海图 will invest in his first round. Decision settled **2025-08** when 赵航 took unified control of foundation-model team.
- **Algorithm cannot stand alone** [[02:34:20](https://youtu.be/n4_c_HsodPg?t=9260)]: "**算法的创新他不能独立于存在,独立于整个公司的基础设施去存在。我们还是要看整个的居民智能 [具身智能] 价值链条**." Full chain: 整机 → 供应链 → 数据 → AI infra → 算法 → 模型 → 分销 → 终端 → 客户价值.
- **传播周期 framework** — how long competitors take to copy:
  - **整机供应链 = 12-18 months**
  - **终端 / 客户渠道 = 6+ months (start; large customers take longer)**
  - **数据体系 = +6-12 months on top of 整机**
  - **算法 = only 2-3 months** ("**对于第一梯队的有非常好的这个算法和工程师团队的算法工程师和这个工程师团队的这样的一个公司来说,算法传播周期是2到3个月**") because everyone open-sources + papers exist
- → algorithm has the highest investment but the **weakest moat**.
- **The 理想主义 disclaimer** [[02:36:37](https://youtu.be/n4_c_HsodPg?t=9397)]: "**理想主义是对的我也觉得我是个理想主义的人,但是理想主义不能变成空想。理想主义能够实现的基础,是我们每天都要去算ROI**."
- **Parallel to Momenta**: 邵青 → 曹旭东 dynamic raised by host; Gao declines to speak for 曹旭东 but claims 许华哲's departure is "long-term absolutely a net positive" for 星海图.
- **2026-01 evidence**: **G-LINK Plus** (claimed "全球首个开箱机用" model) delivered by 赵航's unified team — proof the post-restructuring direction works.

### §13 — 我们天然要到土里去  [[2:39:04 → 3:04:52](https://youtu.be/n4_c_HsodPg?t=9544)]
- **Tech vision (page-1 of every investor deck)** [[02:39:39](https://youtu.be/n4_c_HsodPg?t=9579)]: "**我们要像能够让培训机器人像培训一个员工一样,培训一个人一样,通过几次的示范,然后再通过几次的自我演练,这个机器人就可以在那个场景里面稳定的自主的完成任务**." Productized as **基础模型 + 后训练工具 + 整机** bundle. Already **150+ global developer customers**; next push from devs → production-line end users.
- **The unromantic core** [[02:42:00](https://youtu.be/n4_c_HsodPg?t=9720)]: "**我们天然的就去,就要去土里,要去土里面做很多东西。土里面,对吧我们就没法做浪漫,就要务实**." Robotics chain is much longer than LLM/AI-app — supply chain, data, offline customers all unavoidable. Strategy is 步步为营.
- **On 画饼** [[02:43:56](https://youtu.be/n4_c_HsodPg?t=9836)]: "**其实这个世界是靠相信去驱动的。很多时候不是说我已经拿到了这个结果了,而是大家、我们的公司的员工、投资人、供应商、客户相信我们能够把这件事去做到**" — but 星海图's job is to relentlessly convert each promise into reality, otherwise it becomes the bad kind of 画饼.
- **Latest round: ~¥10B valuation (30x in two years)** [[02:46:09](https://youtu.be/n4_c_HsodPg?t=9969)]: "**1月份那时候我就刚24年1月份刚录完第一轮吧3亿嘛3亿左右吧,现在的话我们就是100亿 [...] 增长30倍**." New strategic capital: **吉利 (Geely), 北汽 (BAIC)** + PE / crossover funds **振兴, 金鼎**. Six existing investors did pro-rata; 3-4 (**凯辉/Cathay, 基石资本/Cornerstone, XiangHe**) went super pro-rata. Round led by **天奇/天琪** — Gao's own fundraising involvement was the lowest of any round ("比我的融资能力强多了").
- **Where 星海图 sits**: estimated **top-5 by valuation** among Chinese embodied-AI companies (top three include 智元 Zhiyuan, **银河 Galbot**, **宇树/语术 Unitree** — ASR variants).
- **Org scaling**: ~15 people → **200+ in two years (≈20x)**; reorgs every **3-5 months**. Hardest org problem: integrating 整机/供应链 culture (process, discipline) with 智能/算法 culture (talent density, innovation).
- **Cash discipline** [[02:57:22](https://youtu.be/n4_c_HsodPg?t=10642)]: "二十亿、几十亿的钱在账上我得把这些钱花好."
- **On the kind of person the industry doesn't tolerate** [[03:03:26](https://youtu.be/n4_c_HsodPg?t=10996)]: "**我觉得这个行业不允许这样的人存在。如果有这样的人存在,可能他会会有很大的 suffering**" (referring to non-pragmatist / pure-romantic founders).
- **Rapid-fire close**:
  - **Animal metaphor for 星海图** = 狼 (wolf), with the caveat that every robotics company is currently wolfish.
  - **Book rec** = 吕思勉 三国史 ("三国 like embodied-AI today — no one is a fool, no one is a romantic hero, everyone is just struggling and trading off step by step").
  - **Favorite food** = **ABC Tofu House** (Korean tofu soup, LA, grad-school memory).
  - **Retirement plan** = Los Angeles.
  - **Most memorable Xiaojun episode** = **李一帆 (Hesai)** — resonated with 2014-16 chaotic-exploration phase.
  - **Mentor archetype** = an early conversation with **邓总** ("a mature entrepreneur" already at the start).
  - **The BET** [[03:00:45](https://youtu.be/n4_c_HsodPg?t=10845)]: "**我们选择把在生产力场景我们做出万台的出货量**" — 万台 出货 in 生产业 / 生产力 scenarios is the immediate BET; "doing embodied intelligence is my life's BET."

## Notable quotes

> "做机器人行业就是一个链条极长的行业 有时候 你就是要把你的头伸到土里去."
> — Zhang Xiaojun (paraphrasing Gao), §1 [[01:32](https://youtu.be/n4_c_HsodPg?t=92)] — *the host's framing thesis for the whole episode*

> "他发现最重要的还是，当你要做一件事的时候，你到底能有，能拉动多少资源，有多少人，多少资源愿意跟你一块去做这件事，并且最后把这事做成。他发现这件事是最重要的."
> — Gao, §3 [[14:47](https://youtu.be/n4_c_HsodPg?t=887)] — *the Zeng Guofan distillation that closes the academic-elite track and opens the founder track*

> "整体的自动驾驶在18年的技术架构和08年的技术架构没区别。但是这一套技术架构的底层逻辑不是AI的,它的底层逻辑是robotics的."
> — Gao, §5 [[35:40](https://youtu.be/n4_c_HsodPg?t=2140)] — *the architectural diagnosis of Waymo and the 2018 AD industry*

> "vimu是没有Founder的。它的Founder其实是Google的Founder,但是Google的Founder又没时间直接去管这事,所以在这个里面就是自上而下的力量,我觉得是缺失的。不像特斯拉,马斯克说干啥,哪怕是错了他也能开始干."
> — Gao, §5 [[42:07](https://youtu.be/n4_c_HsodPg?t=2527)] — *the "founderless" governance gap thesis*

> "工程师思维就是拆解加测量。就是把一个复杂的问题,拆解成做干个稍微不那么复杂的子问题,然后再拆解再拆解,你写代码,拆到最后就是一行一行的代码,测量到最后是什么,是一个单元测试."
> — Gao, §5 [[53:05](https://youtu.be/n4_c_HsodPg?t=3185)] — *the Waymo-internalized "拆解加测量" engineer's method that becomes his startup operating system*

> "我从M出来创业我放弃了所有的期权,全都放弃了 [...] 1000万美金可能是有的吧."
> — Gao, §7 [[01:09:19](https://youtu.be/n4_c_HsodPg?t=4159)] — *the catfish exit and ~$10M USD options forfeit, post-GPT + Optimus*

> "我们要做巨神智能 巨神智能长期壁垒 建立在物理世界的数据闭环之上 [...] 我们才真的有 一个说别人进不来 我们能够长治久安的这么一个壁垒了."
> — Gao, §9 [[01:35:48](https://youtu.be/n4_c_HsodPg?t=5748)] — *the integrated-machine moat thesis (ASR 巨神 → 具身)*

> "对于第一梯队的有非常好的这个算法和工程师团队的算法工程师和这个工程师团队的这样的一个公司来说,算法传播周期是2到3个月."
> — Gao, §12 [[02:35:54](https://youtu.be/n4_c_HsodPg?t=9354)] — *the 2-3 month algorithm-moat number that frames 许华哲's exit*

> "我们天然的就去,就要去土里,要去土里面做很多东西。土里面,对吧我们就没法做浪漫,就要务实."
> — Gao, §13 [[02:42:00](https://youtu.be/n4_c_HsodPg?t=9720)] — *the closing self-definition of the embodied-AI founder vs. the LLM 技术浪漫主义 archetype*

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| Zhang Xiaojun (张小珺) | Host | [[00:00](https://youtu.be/n4_c_HsodPg?t=0)] |
| Zeng Guofan (曾国藩) | Qing-dynasty statesman; biography reframed Gao's PhD setback | [[00:00](https://youtu.be/n4_c_HsodPg?t=0)] |
| Yang Zhilin (杨植麟, ASR: 杨志林) | Moonshot AI founder; Tsinghua classmate, "高不可攀的大神" | [[00:08](https://youtu.be/n4_c_HsodPg?t=8)] |
| Xu Huazhe (许华哲) | Departing 星海图 co-founder/scientist; pursuing 2C 家庭应用 startup | [[00:26](https://youtu.be/n4_c_HsodPg?t=26)] |
| Shao Qing (邵青) | Left Momenta; parallel cited for impact on 曹旭东 | [[00:39](https://youtu.be/n4_c_HsodPg?t=39)] |
| Cao Xudong (曹旭东) | Momenta CEO; "强势, 懂技术, AI believer" — primary reason Gao joined | [[00:41](https://youtu.be/n4_c_HsodPg?t=41)] |
| Liang Wenfeng (梁文峰) | DeepSeek founder; archetype of 技术浪漫主义 absent from 具身智能 | [[01:06](https://youtu.be/n4_c_HsodPg?t=66)] |
| Gao Jiyang (高继扬) | Guest — Founder of 星海图 / Galaxea | [[01:14](https://youtu.be/n4_c_HsodPg?t=74)] |
| Wang Xing (王兴) | Meituan founder; cited as prior-generation protagonist Gao felt was "not him" | §2 [[10:14](https://youtu.be/n4_c_HsodPg?t=614)] |
| Dai Wei (戴威) | ofo founder; named as post-90 founder Gao watched from the sidelines | §2 [[10:00](https://youtu.be/n4_c_HsodPg?t=600)] |
| Tang Xiao'ou (汤晓鸥) | SenseTime founder; gave Gao the formative internship | §2 [[12:00](https://youtu.be/n4_c_HsodPg?t=720)] |
| Li Cheng (李成) | SenseTime mentor on Gao's pose-estimation project | §3 [[18:30](https://youtu.be/n4_c_HsodPg?t=1110)] |
| Lu Shu (鲁淑) | Gao's direct mentor at SenseTime | §3 [[18:31](https://youtu.be/n4_c_HsodPg?t=1111)] |
| Han Yanjun (韩衍俊) | Tsinghua classmate whose homework Gao "couldn't even understand" | §3 [[22:12](https://youtu.be/n4_c_HsodPg?t=1332)] |
| Sun Chen (孙晨/晨哥) | Tsinghua 计算机系 senior; later professor at Brown | §3 [[25:13](https://youtu.be/n4_c_HsodPg?t=1513)] |
| Yan Junjie (严俊杰) | At SenseTime during Gao's internship | §3 [[23:36](https://youtu.be/n4_c_HsodPg?t=1416)] |
| Zhao Hang (赵航) | Co-author of VectorNet at Waymo; co-founder of 星海图; now leads foundation-model team | §5 [[49:37](https://youtu.be/n4_c_HsodPg?t=2977)] |
| Elon Musk (马斯克) | Counter-example to Waymo's founderless governance — "even when wrong, decisive" | §5 [[38:28](https://youtu.be/n4_c_HsodPg?t=2308)] |
| Chen Yilun (陈一伦) | Tsinghua 电子系 师兄; Gao's contact at Huawei | §6 [[57:57](https://youtu.be/n4_c_HsodPg?t=3477)] |
| Su Jing (苏箐, ASR: 苏青) | Former Huawei ADS head; Gao interviewed with him | §6 [[58:06](https://youtu.be/n4_c_HsodPg?t=3486)] |
| Sun Gang (孙刚) | Momenta co-founder/exec; Gao's interview contact | §6 [[58:10](https://youtu.be/n4_c_HsodPg?t=3490)] |
| Sibo (思博) | Ex-SenseTime; pre-existing tie that smoothed Gao's Momenta entry | §6 [[59:05](https://youtu.be/n4_c_HsodPg?t=3545)] |
| Jinmei (晋美) | Ex-SenseTime; same as above | §6 [[59:06](https://youtu.be/n4_c_HsodPg?t=3546)] |
| Yu Kai (凯歌, Horizon Robotics) | Scientist → entrepreneur role-model Gao cites alongside 旭东 | §7 [[01:18:55](https://youtu.be/n4_c_HsodPg?t=4735)] |
| Tian Fei (田飞 / 天威) | One of three original 星海图 co-founders (with Gao + 赵航), ex-Momenta | §8 [[01:23:26](https://youtu.be/n4_c_HsodPg?t=5006)] |
| Li Yikang (李一康) | Gao's Tsinghua classmate; IDG investor who made the internal referral | §8 [[01:26:26](https://youtu.be/n4_c_HsodPg?t=5186)] |
| Shao Hui (邵辉) | IDG partner; trusted 清华系 founders | §8 [[01:26:35](https://youtu.be/n4_c_HsodPg?t=5195)] |
| Xiao Jun (肖军) | IDG partner who closed the seed | §8 [[01:26:49](https://youtu.be/n4_c_HsodPg?t=5209)] |
| Zhu Xiaohu (朱啸虎, 朱老板) | Then 金沙江 partner; participated in first round | §8 [[01:25:55](https://youtu.be/n4_c_HsodPg?t=5155)] |
| 宇同学姐 / 于同学姐 | 金沙江 follow-through partner; left 金沙江 later | §8 [[01:25:45](https://youtu.be/n4_c_HsodPg?t=5145)] |
| Yang Zeyi (杨泽义) | 星海图 partner & 机电首席工程师, b. 1997 南科大; joined Jan 2024 | §8 [[01:31:01](https://youtu.be/n4_c_HsodPg?t=5461)] |
| Yu Lei (于磊) | 星海图 partner; runs 商业化 | §8 [[01:35:05](https://youtu.be/n4_c_HsodPg?t=5705)] |
| Tianqi (天奇 / 天琪) | 星海图 CFO; led latest fundraising round; joined a few months before recording | §8 [[01:35:09](https://youtu.be/n4_c_HsodPg?t=5709)] |
| Fei-Fei Li (李飞飞) | Archetypal academic developer at the top of 星海图's customer pyramid | §9 [[01:45:30](https://youtu.be/n4_c_HsodPg?t=6330)] |
| Ant Group (蚂蚁) | Enterprise R&D partner on LingBoat VLA project | §9 [[01:45:48](https://youtu.be/n4_c_HsodPg?t=6348)] |
| Sunday (US company) | Example of human-centric data collection (gloves/grippers) | §10 [[02:03:24](https://youtu.be/n4_c_HsodPg?t=7404)] |
| Jie Tan (谭杰, Google DeepMind) | Argued to Gao that Google's foundation model gives VLA advantage; later cited on US data scarcity | §11 [[02:22:33](https://youtu.be/n4_c_HsodPg?t=8553)] |
| ByteDance (字节) | Big-co competitor; example of LLM business-synergy via 飞书+抖音 | §11 [[02:16:42](https://youtu.be/n4_c_HsodPg?t=8202)] |
| Xiaomi (小米) | Big-co competitor in robot-brain space | §11 [[02:16:44](https://youtu.be/n4_c_HsodPg?t=8204)] |
| NVIDIA | Overseas big-co working on robot brain | §11 [[02:16:49](https://youtu.be/n4_c_HsodPg?t=8209)] |
| Physical Intelligence (Pi) | Overseas robot-brain player; π0 baseline | §11 [[02:16:51](https://youtu.be/n4_c_HsodPg?t=8211)] |
| Li Yifan (李一帆) | Hesai Tech founder; Gao's most memorable Zhang Xiaojun episode | §13 [[03:00:59](https://youtu.be/n4_c_HsodPg?t=10859)] |
| Lü Simian (吕思勉) | Historian whose Three Kingdoms book is Gao's current read & rec | §13 [[02:55:47](https://youtu.be/n4_c_HsodPg?t=10547)] |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| Biography of 曾国藩 | Reframed Gao's PhD-rejection setback; "事工" pragmatist arc | §3 [[14:02](https://youtu.be/n4_c_HsodPg?t=842)] |
| Pose estimation (Gao's first SenseTime project) | First neural network he trained; "magic" of pattern extraction | §3 [[18:35](https://youtu.be/n4_c_HsodPg?t=1115)] |
| CVPR / ICCV (顶会 target) | PhD output calibration for 4-yr graduation | §4 [[26:20](https://youtu.be/n4_c_HsodPg?t=1580)] |
| DARPA Urban Challenge (08-09) | Reference point for the unchanged 2008-2018 AD stack | §5 [[34:06](https://youtu.be/n4_c_HsodPg?t=2046)] |
| VectorNet (Gao + 赵航, Waymo) | Vector + self-attention map encoding; first Gao-赵航 collaboration | §5 [[50:04](https://youtu.be/n4_c_HsodPg?t=3004)] |
| Momenta M10 / SAIC 量产 project | Gao's joining moment; 2021 delivery year | §7 [[01:02:35](https://youtu.be/n4_c_HsodPg?t=3755)] |
| NVA / 高速 NOA (Momenta × SAIC) | Gao's last project at Momenta — 量产 delivery | §7 [[01:06:30](https://youtu.be/n4_c_HsodPg?t=3990)] |
| GPT-1 / GPT-2 / GPT-3 + InstructGPT | Made the world "再一次相信AI" — trigger for Gao to leave | §7 [[01:10:22](https://youtu.be/n4_c_HsodPg?t=4222)] |
| Tesla Optimus | Humanoid announcement that finalized Gao's "我得开稿了" | §7 [[01:11:41](https://youtu.be/n4_c_HsodPg?t=4301)] |
| Crossing the Chasm (《跨越鸿沟》, ASR: 跨越同工) | Innovator → early-majority GTM framework | §9 [[01:43:12](https://youtu.be/n4_c_HsodPg?t=6192)] |
| LingBoat VLA (蚂蚁 × 星海图) | Enterprise R&D dev customer example | §9 [[01:45:48](https://youtu.be/n4_c_HsodPg?t=6348)] |
| Apple Macintosh | Innovator → early-majority precedent | §9 [[01:43:49](https://youtu.be/n4_c_HsodPg?t=6229)] |
| 拓竹 (Bambu Lab) 3D printers | Same precedent class | §9 [[01:44:03](https://youtu.be/n4_c_HsodPg?t=6243)] |
| Unitree (语术) quadrupeds | Same precedent class | §9 [[01:44:50](https://youtu.be/n4_c_HsodPg?t=6290)] |
| Physical Intelligence (company) | Cited alongside as a top-tier US robot-brain team | §9 [[01:45:46](https://youtu.be/n4_c_HsodPg?t=6346)] |
| Diffusion Policy | Pre-π0 manipulation policy 星海图 used | §10 [[01:50:32](https://youtu.be/n4_c_HsodPg?t=6632)] |
| π0 (ASR: 派灵, Physical Intelligence) | VLA validation that triggered 星海图's 2025 VLA pivot | §10 [[01:50:41](https://youtu.be/n4_c_HsodPg?t=6641)] |
| Kiva (Amazon warehouse robot) | Traditional warehouse solution that breaks on massive-SKU bin-picking | §10 [[02:13:42](https://youtu.be/n4_c_HsodPg?t=8022)] |
| G0 (star-sea open-source model + data, Aug 2025) | 500hr teleop data + base model | §11 [[02:18:51](https://youtu.be/n4_c_HsodPg?t=8331)] |
| G2+ (star-sea base model + R1 lite 整机, Jan 2026) | "Out-of-the-box" experience | §11 [[02:19:07](https://youtu.be/n4_c_HsodPg?t=8347)] |
| G-LINK Plus (开箱机, claimed world-first) | Jan 2026 launch under 赵航's unified team | §12 [[02:38:32](https://youtu.be/n4_c_HsodPg?t=9512)] |
| 吕思勉 三国史 (Lü Simian's Three Kingdoms history) | Gao's current read; book recommendation | §13 [[02:55:47](https://youtu.be/n4_c_HsodPg?t=10547)] |
| ABC Tofu House (LA Korean tofu soup) | Gao's favorite food, USC-era memory | §13 [[02:59:56](https://youtu.be/n4_c_HsodPg?t=10796)] |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| Gao Jiyang born 1992 | [Guest] | §2 [[01:50](https://youtu.be/n4_c_HsodPg?t=110)] |
| 6th-grade summer 第一次用功; 初一 分班考试 年级第三名 | [Guest] | §2 [[04:00](https://youtu.be/n4_c_HsodPg?t=240)] |
| 物理竞赛 全国二等奖 (2010-11, 厦门) | [Guest] | §2 [[06:15](https://youtu.be/n4_c_HsodPg?t=375)] |
| Watched 2011-2015 mobile-internet wave from the sidelines | [Guest] | §2 [[09:13](https://youtu.be/n4_c_HsodPg?t=553)] |
| SenseTime internship: ~4-5 months | [Guest] | §3 [[18:11](https://youtu.be/n4_c_HsodPg?t=1091)] |
| 汤老师 offered ~"十几个" internship opportunities | [Guest] | §3 [[18:20](https://youtu.be/n4_c_HsodPg?t=1100)] |
| 语文 一卷+二卷 ≈ 100 / 150; others' 一卷 alone ≈ 90 | [Guest] | §3 [[19:56](https://youtu.be/n4_c_HsodPg?t=1196)] |
| Tsinghua 电子系 self-rated top 30-40% | [Guest] | §3 [[21:29](https://youtu.be/n4_c_HsodPg?t=1289)] |
| PhD target: 4 years vs. typical US 5-6 years | [Guest] | §4 [[25:57](https://youtu.be/n4_c_HsodPg?t=1557)] |
| PhD output target: 4-5 顶会 (CVPR/ICCV) papers | [Guest] | §4 [[26:16](https://youtu.be/n4_c_HsodPg?t=1576)] |
| USC PI: a ~70+-year-old Indian professor | [Guest] | §4 [[27:44](https://youtu.be/n4_c_HsodPg?t=1664)] |
| 18年年底 — early-graduation request approved | [Guest] | §4 [[28:00](https://youtu.be/n4_c_HsodPg?t=1680)] |
| 2018 AD stack architecturally identical to 08 stack | [Guest] | §5 [[35:40](https://youtu.be/n4_c_HsodPg?t=2140)] |
| Waymo perception ran "几十个模型" (dozens) | [Guest] | §5 [[37:17](https://youtu.be/n4_c_HsodPg?t=2237)] |
| Robotics-led AD: 08 → 2016-17-18 | [Guest] | §5 [[39:06](https://youtu.be/n4_c_HsodPg?t=2346)] |
| Waymo headcount: ~1000 (when Gao joined Jan 2019) → ~2000 (H2 2020) | [Guest] | §5 [[40:13](https://youtu.be/n4_c_HsodPg?t=2413)] |
| Gao's perception team: ~10 → 70-80 in the same window | [Guest] | §5 [[40:22](https://youtu.be/n4_c_HsodPg?t=2422)] |
| Left Waymo 20年下半年; joined Jan 2019 | [Guest] | §5 [[54:24](https://youtu.be/n4_c_HsodPg?t=3264)] |
| Momenta 18年起 publicly committed to 量产→飞轮→RoboTaxi | [Guest] | §6 [[59:38](https://youtu.be/n4_c_HsodPg?t=3578)] |
| 1台 fleet → 1000台 fleet "can't cover one city" — argument for 量产 | [Guest] | §6 [[01:01:27](https://youtu.be/n4_c_HsodPg?t=3687)] |
| Delivery iterations: "几十次,今天可能都有一两百次" 车型 deliveries | [Guest] | §6 [[01:00:49](https://youtu.be/n4_c_HsodPg?t=3649)] |
| Core algo teams at Momenta: 到12点, 一周6天打底 | [Guest] | §7 [[01:12:35](https://youtu.be/n4_c_HsodPg?t=4355)] |
| 2020年底 — saw the changes that triggered his exit | [Guest] | §7 [[01:09:59](https://youtu.be/n4_c_HsodPg?t=4199)] |
| Forfeited ~1000万美金 in Momenta options (hedged) | [Guest] | §7 [[01:09:19](https://youtu.be/n4_c_HsodPg?t=4159)] |
| 旭东's chip bet: decision ~3 yrs ago, started ~2 yrs ago; "晚行动一年" = no today | [Guest] | §7 [[01:18:30](https://youtu.be/n4_c_HsodPg?t=4710)] |
| Officially left Momenta 2023-05 after SAIC NVA delivery | [Guest] | §8 [[01:20:44](https://youtu.be/n4_c_HsodPg?t=4844)] |
| First round ¥3000万 (RMB) — IDG + 百度风投 + 金沙江 | [Guest] | §8 [[01:27:13](https://youtu.be/n4_c_HsodPg?t=5233)] |
| First-round valuation ~¥2亿 (pre or post — "记不清") | [Guest] | §8 [[01:27:22](https://youtu.be/n4_c_HsodPg?t=5242)] |
| 加轮 (CFUN) ~¥1-2千万, 投后估值 ~¥3-4亿 | [Guest] | §8 [[01:28:01](https://youtu.be/n4_c_HsodPg?t=5281)] |
| 杨泽义 b. 1997, joined Jan 2024 | [Guest] | §8 [[01:31:13](https://youtu.be/n4_c_HsodPg?t=5473)] |
| AD supplier total customer set: ~20 车企 globally | [Guest] | §9 [[01:38:25](https://youtu.be/n4_c_HsodPg?t=5905)] |
| 23年底: "啥也没有的状态" on hardware | [Guest] | §9 [[01:40:32](https://youtu.be/n4_c_HsodPg?t=6032)] |
| 24年3月: 轮式+躯干 form factor locked, R1 started | [Guest] | §9 [[01:42:38](https://youtu.be/n4_c_HsodPg?t=6158)] |
| 25年8月: 全球/全国第一个 robotics data open-source — 500 hrs teleop + G0 | [Guest] | §10 [[01:53:51](https://youtu.be/n4_c_HsodPg?t=6831)] |
| Data acquisition vs training cost ratio ≈ 1:5 to 1:10 | [Guest] | §10 [[01:57:55](https://youtu.be/n4_c_HsodPg?t=7075)] |
| Real-data cost: 200-250 RMB / hour (1 hr teleop = 3-4 hrs human + ¥100/hr deprec) | [Guest] | §10 [[01:59:15](https://youtu.be/n4_c_HsodPg?t=7155)] |
| 10万 hrs ≈ ¥2500万 ≈ a human's 0-18yr physical-interaction budget | [Guest] | §10 [[01:59:55](https://youtu.be/n4_c_HsodPg?t=7195)] |
| 整机 ¥100k amortized over ~1000 hr lifetime (减速箱 limited) | [Guest] | §10 [[01:59:15](https://youtu.be/n4_c_HsodPg?t=7155)] |
| 2025 dev-market customers: 150+ | [Guest] | §10 [[01:51:13](https://youtu.be/n4_c_HsodPg?t=6673)] |
| Imitation-learning speed cap ≈ 80-90% of human | [Guest] | §10 [[02:08:56](https://youtu.be/n4_c_HsodPg?t=7736)] |
| 5 action primitives: Carry / Pick / Pack / Fold / Operate | [Guest] | §10 |
| 工商业 typical task set: 20-30 fixed actions (VLM optional) | [Guest] | §11 [[02:21:30](https://youtu.be/n4_c_HsodPg?t=8490)] |
| On-device inference cap: 几十B / 上百B impossible at edge | [Guest] | §11 [[02:20:54](https://youtu.be/n4_c_HsodPg?t=8454)] |
| G0 open-source Aug 2025; G2+ launch Jan 2026 | [Guest] | §11 [[02:18:51](https://youtu.be/n4_c_HsodPg?t=8331)] |
| 整机供应链 传播周期: 12-18 months | [Guest] | §12 [[02:34:55](https://youtu.be/n4_c_HsodPg?t=9295)] |
| 客户渠道 传播周期: 6+ months | [Guest] | §12 [[02:35:26](https://youtu.be/n4_c_HsodPg?t=9326)] |
| 数据体系 传播周期: +6-12 months on top of 整机 | [Guest] | §12 [[02:35:31](https://youtu.be/n4_c_HsodPg?t=9331)] |
| 算法 传播周期: 2-3 months (open-source/papers diffuse) | [Guest] | §12 [[02:36:01](https://youtu.be/n4_c_HsodPg?t=9361)] |
| 2025-08: 赵航 took unified control of foundation-model team | [Guest] | §12 [[02:38:19](https://youtu.be/n4_c_HsodPg?t=9499)] |
| 2026-01: G-LINK Plus launched under 赵航's team | [Guest] | §12 [[02:38:29](https://youtu.be/n4_c_HsodPg?t=9509)] |
| Latest round valuation: ~¥100亿 (¥10B) — 30x from ~¥3亿 in Jan 2024 | [Guest] | §13 [[02:46:09](https://youtu.be/n4_c_HsodPg?t=9969)] |
| Org scale: ~15 people (2 yrs ago) → ~200+ now (≈20x) | [Guest] | §13 [[02:47:00](https://youtu.be/n4_c_HsodPg?t=10020)] |
| Reorgs every 3-5 months | [Guest] | §13 [[02:47:23](https://youtu.be/n4_c_HsodPg?t=10043)] |
| Cash on hand: 二十亿、几十亿 — must deploy well | [Guest] | §13 [[02:57:22](https://youtu.be/n4_c_HsodPg?t=10642)] |
| Immediate BET: 万台 出货量 in 生产业 scenarios | [Guest] | §13 [[03:00:45](https://youtu.be/n4_c_HsodPg?t=10845)] |

## Open questions / gaps

- **The "founderless" diagnosis of Waymo (§5)** is stated as a structural law but supported only by Tesla-Musk as the counter-example; no third-org case (e.g. Cruise under GM) is examined.
- **"VectorNet was adopted by many other companies" (§5)** — asserted with no named adopter or citation.
- **Waymo LA service-quality comparison (§5)** is based on a single personal ride from Downtown → Hollywood → Santa Monica in Nov of Gao's return year.
- **The ~$10M USD options forfeit figure (§7, §8)** is hedged twice ("具体记不太清", "可能是有的吧") — no strike/vest detail.
- **"GPT让世界再一次相信AI"" as precondition for capital flow into embodied AI (§7)** is asserted without market data.
- **End-side sensor/compute equivalence between AD and humanoid robots (§7)** asserted without spec comparison.
- **Algorithm 传播周期 = 2-3 months (§12)** treats open-source + papers as equivalent to actual capability transfer; the alignment between paper, open-source weights, and production-grade implementation is collapsed.
- **"许华哲's departure is long-term net positive for 星海图" (§12)** — strong claim, supported only by the values-alignment + post-restructuring G-LINK-Plus delivery narrative.
- **"G-LINK Plus 全球首个开箱机用" (§12)** — claim of world-first without comparative benchmarks.
- **Top-5 valuation ranking among Chinese embodied-AI startups (§13)** — Gao explicitly caveats he doesn't know other companies' exact numbers.
- **"机器人员工" 显著提升 humanity's 幸福感 (§13)** — pure vision-statement, no mechanism offered.
- **The ASR-mangled "一千两美金" figure in §1** (resolved by §7's "1000万美金可能是有的吧") still appears verbatim in the cold-open splice and should be read against the §7 restatement.
- **Co-founder name disambiguation**: ASR variants 田飞 / 天威 / 天奇 / 天琪 appear inconsistently — likely two distinct people (田飞 = original co-founder, 天奇/天琪 = recently-joined CFO), but ASR conflation is non-trivial.

## Verification log

- **Sectioning**: chapters (13 YouTube-supplied chapters); all 13 titles preserved verbatim from `info.json.chapters`.
- **Transcript source**: faster-whisper large-v3 (CPU int8, local) — produced by `docs/videos/transcribe_batch.py`. YouTube provided no subtitles (manual or auto) for this video, so the standard yt-dlp path was unavailable.
- **Speaker name corrections**: host "小俊"/"小骏" → 张小珺 (Zhang Xiaojun); Waymo ASR variants vimu/vmall/Wemo → Waymo; Momenta ASR variants 蒙曼塔/罗门塔 → Momenta; 巨神智能 ASR → 具身智能 when generic; 杨志林 → 杨植麟; 上汤 → 商汤 (SenseTime); 派灵 → π0; Locum → Loco-manipulation; GBT → GPT; RI → ROI; 一千两美金 → ~1000万美金 (~$10M) per §7 restatement; 跨越同工 → 《跨越鸿沟》; 苏青 → 苏箐; 绵羽 → 鲶鱼; 图里 → 土里.
- **Sections covered**: 13/13 ✅
- **Notable quotes traced verbatim**: 9/9 ✅ (each anchored by a distinctive 6-15-char Chinese/English substring grep'd in `/tmp/yt-wiki/n4_c_HsodPg.flat.txt`)
- **Numbers traced**: 55/55 ✅ (all rows in Numbers & claims confirmed by direct or near-direct substring against the flat transcript)
- **Sectioning method used**: chapters
- **Removed during verification**: none

## See Also

- `docs/videos/2026-05-11-yao-shunyu-training-claude-gemini.md` — Zhang Xiaojun's episode with Yao Shunyu (ReAct author at Anthropic); same host/channel, agent-research counterpart to Gao's physical-world frame.
- `docs/videos/2026-05-01-su-yu-agent-tech-history.md` — Su Yu's Agent four-eras survey; Gao's "物理世界数据闭环 is the only moat" thesis is the embodied complement to Su Yu's "coding is the fabric that dissolves agent boundaries."
- `skills/research/youtube-wiki/SKILL.md` — the pipeline that produced this entry.
