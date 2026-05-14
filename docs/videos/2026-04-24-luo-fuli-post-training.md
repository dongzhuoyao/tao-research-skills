# Luo Fuli: AI Paradigm Has Already Shifted — OpenClaw, Post-Training-Heavy Agents, GPU Allocation, Org Flattening

**Source**: https://youtu.be/vG1RBqn1sG4
**Channel**: Zhang Xiaojun Podcast (张小珺商业访谈录)
**Published**: 2026-04-24
**Duration**: 03:36:36
**Watched on**: 2026-05-14
**Sectioning**: chapters (15 YouTube-supplied chapters)
**Detected video language**: `zh` (no `info.json.language`; both `subtitles` and `automatic_captions` keys empty — no YouTube subs of any kind)
**Transcript source**: faster-whisper large-v3 (CPU int8) on local M4 — produced by `docs/videos/transcribe_batch.py`. ASR consistently rendered the guest's name as "罗弗利" (correct: **罗福莉 Luo Fuli**) and the host as "小骏" (correct: **张小珺 Zhang Xiaojun**); recurring ASR garbles corrected throughout: "Cloud Opus 4.6 / Cloud Ops 4.6 / CloudOps 4.6" → **Claude Opus 4.6**, "Open Cloud / opencloud / openclose / OpenCore" → **OpenClaw**, "QNN / Q1" → **Qwen**, "DeepSeeker" → **DeepSeek**, "Cologram Mass / BronzeComp / Navcode Bench" → **Codeforces / BrowseComp / LiveCodeBench**, "non-convex" (when paired with model architecture) → **non-chat**, "讽定" → **否定**, "sadly window" → **sliding window**, "Azure路径" → **Agent 路径**, "GMAV" → **GLM**, "Sobi / sobi" → likely **OpenAI** (Sora-team-referenced), "一体/一兆" → **1T parameters**, "10币" → **10B activated params**.
**Speakers**: Zhang Xiaojun (张小珺, host), Luo Fuli (罗福莉, guest — ex-Alibaba DAMO + ex-DeepSeek, currently head of Xiaomi's large-model team, leading the MiMo-V series)

## TL;DR

- Luo Fuli's central thesis: **2026 is the productivity-transformation year, and the locus of intelligence has moved from pretraining to post-training plus a thick, hackable agent middle-layer**. Her Spring-Festival "conversion" to **OpenClaw** (talked to it 2am→6am on night one) reshaped both her research workflow (10 ideas dispatched in parallel to subagents, "3-4 weeks ≈ old 30-40 weeks") and her view of what post-training is for.
- **Architecture call against the field**: while **GLM** and **Kimi K2** chose **MLA**, MiMo-V2 bet on **hybrid attention (sliding window + full) + MTP** because MLA already sits at a perfect L-bound/MMI-bound point and leaves "no room for tricks like MTP" — pushing MLA into compute-bound territory. Pro reaches a 7:1 sliding-to-full ratio; Flash priced at $1.01/M input + $0.30/M output, 100-150 TPS, framed as **"silent ambush"** rather than chat-era cost.
- **1T parameters is the agent-era entry ticket**; pre-train : post-train compute ratio has shifted from the GPT-era 3:1 / 5:1 to **1:1** at top teams, and research compute should be roughly **3× formal training (research:pretrain:posttrain ≈ 3:1:1)**. During ET pretraining her team logged 2-3 loss spikes — the longest pause was **two weeks** — because loss spikes can permanently kill an MoE expert.
- **Organizational thesis: groupless, deadline-free, ~100-person passion-driven team**. No pre-train group vs. post-train group split, no formal authority, pre-trainers fluidly migrate into post-training because they have superior intuition for data diversity. Famous "你没有100轮对话明天就辞职" mandate was theater — never audited — but worked because shared group chats unlocked swarm imagination.
- **Macro calls**: AGI ~20% today → 60-70% by year-end → "within two years" for work-mode overturn; China-US gap vs. Claude Opus 4.6 ≈ **2-3 months** *if* your reaction speed is fast; **environment matters more than experience** — most ML skills are absorbable in 1-2 months (3-4 slow case), so she hires sophomores/juniors whose thinking is "uncontaminated"; only **OpenAI is clearly scaling agent RL**, and she will publicly share MiMo's RL-scaling numbers only when its compute matches their pretrain compute.

## Why it matters

A 3h36m interview with Luo Fuli's first long-form technical appearance — covering MiMo-V2's "silent ambush" (Flash/Pro/Omni/TTS dropped anonymously on OpenRouter on March 11 before the official release), her concrete first-person account of how OpenClaw rewrote both her research process and her belief about post-training's role, a granular compute-allocation breakdown (3:1:1 research:pretrain:posttrain, 1T-parameter entry ticket, 几千卡 training class), and a sharp dismissal of BrowseComp/TerminalBench-class benchmarks as "歧途" (a wrong path) for measuring real agent capability. The most up-to-date Chinese frontier-team view on what changed in early 2026.

## Section summaries

### §1 — 开场白与本期定位  [[00:00 → 02:16](https://youtu.be/vG1RBqn1sG4?t=0)]
- Host 张小珺 frames 2026 as **"大模型战争全面升级,揭开了第二幕"** — moving from a pretrain-led Chatbot era to a **post-training-led Agent era** [[00:04](https://youtu.be/vG1RBqn1sG4?t=4)]. Episode positioned as the first long-form technical interview with Luo Fuli.
- Guest introduced as ex-**Alibaba DAMO Academy** + ex-**DeepSeek**, now head of Xiaomi's large-model team, leading the **MiMo-V series**; carries the public "AI 天才少女" label but dislikes it [[00:32](https://youtu.be/vG1RBqn1sG4?t=32)].
- Pre-roll teaser previews three load-bearing claims the rest of the episode unpacks: (a) frontier agent capabilities are absorbable in 1-2 months (3-4 months slow case) → **"环境反而比经验更重要"** [[01:20](https://youtu.be/vG1RBqn1sG4?t=80)]; (b) hitting ~**Claude 4.6 Opus**-level agent capability is the *entry ticket* for the next round; (c) research:pretrain:posttrain compute should run **3:1:1**, with research compute slightly exceeding the sum of formal training compute [[01:40](https://youtu.be/vG1RBqn1sG4?t=100)].
- Numbers: next 2-3 months "非常精彩"; capability-absorption window 1-2 months fast / 3-4 slow; compute ratio 3:1:1.

### §2 — OpenClaw引发巨变  [[02:16 → 24:17](https://youtu.be/vG1RBqn1sG4?t=136)]
- Luo's initial dismissal: OpenClaw "looked like a Claude Code clone with a fancier IM-style UI plus 玄幻 ops moves like the Skill Hub." Spring-Festival install at 2am → talked to it until 6am: **"我自己其实是会把opencloud把它当做一个划时代的agent的框架去这么去定义"** [[02:33](https://youtu.be/vG1RBqn1sG4?t=153)]; reports continuous dopamine/endorphin secretion, "灵魂" and 情商 (it nudged her to sleep).
- Day 2: co-designed an LLM team-building framework with it, converted it to a reusable **Skill** that became her "数字分身" for management questions. Day 3: handed it a research task (build a user-agent simulator for multi-turn agent training data) — workable result in ~1-2 hours.
- **Why OpenClaw works vs. Claude Code**: persistent memory with explicit grading/layering, native multi-model routing (auto-picks stronger video-understanding model), and **"惊喜编排的 context"** — e.g. prepends the current time to every turn so the model can perceive time. Claude Code is for-software-engineering; OpenClaw is for arbitrary tasks and *patches model weaknesses via the framework*.
- **The framework is a thick middle-layer**: "它相当于是一个中间层,人和模型之间的中间层 [...] 然后这个中间层它可以做的非常的厚重,然后反而那个前端的UI展示它是最薄的一层,它已经不是很关键了" [[21:07](https://youtu.be/vG1RBqn1sG4?t=1267)]. Open-source enables crowd-improvement that closed Claude Code's black-box memory/multi-agent architecture cannot.
- **Top model + top framework must co-evolve**: "顶尖的模型应该跟顶尖的这种agent的框架是共同的往前去进步" [[18:32](https://youtu.be/vG1RBqn1sG4?t=1112)]. Opus 4.6 is still the only model that can rewrite OpenClaw's own architecture, but once it has shaped the Skills/AGENTS.md scaffold, Sonnet, MiMo VR Pro, and even a 3B end-side model become "非常强大" inside that scaffold.
- Cost reality: **first day burned ~$1000 of Opus 4.6 in 4-5 hours**; she kept switching to Sonnet, found it inadequate, came back to Opus. By March's 3.x version OpenClaw was broadly usable with any decent model.
- Numbers: 凌晨2点→6点 first session; ~1-hour deep-dive on second-day framework; ~1-2 hours for user-agent build; "**85%的任务上,能够达到跟Cloud的Sunlight一样水平**" with mid-tier model + good framework; **第一天反正就是快一千块钱 [...] 一千刀 [...] 用了四五个小时**; OpenClaw 3.x mid-March = "已经非常易用."

### §3 — 群体智能提升Agent框架  [[24:17 → 41:31](https://youtu.be/vG1RBqn1sG4?t=1457)]
- **The "100-轮 quit" mandate**: during Spring Festival she bought several Mac minis, deployed instances for everyone, and decreed "**如果第二天opencloud对话次数不超过100轮的人 可以直接quit**" [[25:24](https://youtu.be/vG1RBqn1sG4?t=1524)]. Never actually audited — pure signal of urgency.
- **Swarm intelligence unlock**: shared 飞书 group chat is what mattered — "**为什么要在大群里面聊 就是因为个人的想象力真的是局限 但是当你看到别人用opencloud 居然能干成这个事情的时候 你就会激发你自己的想象力**" [[26:10](https://youtu.be/vG1RBqn1sG4?t=1570)]. By day 3-4 her team grafted OpenClaw onto their own model — "**差不多了**" — roughly on par with Claude for early use.
- **New research rhythm — 3-4 weeks now does 30-40 weeks of old work**: "**我们基本上可能在三四周的时间 做完了以前可能三四十周的时间 才能做到的事情 就在研究上**" [[28:42](https://youtu.be/vG1RBqn1sG4?t=1722)]. Idea→code→eval was 1-2 weeks (best case 1-2 days); agent-assisted research compresses to 1-2 hours. **10 ideas dispatched to subagents in parallel with cross-validation**: "你可以十个idea交给不同的subagent的同时做 他们还能交叉验证 [...] 你无非烧很多token嘛" [[33:07](https://youtu.be/vG1RBqn1sG4?t=1987)].
- **Why code is the pry-bar for long-context agent capability**: the only natural corpus that genuinely reaches **128K-1M tokens** with dense inter-file dependencies. The only alternative class — books — has signal "太发散" (too diffuse). Code pretraining is what prepared the long-context base before agents mattered.
- Strategy: **code defines the *upper bound* of the agent model's capability; generalization to other domains raises the *lower bound* (下限)** — "**在code是那它的上限 然后你去其它领域是把它的下限 我是这么认为**" [[33:19](https://youtu.be/vG1RBqn1sG4?t=1999)].
- Open challenge: long-horizon RL is bottlenecked by **reconstructing the user's original environment** to design precise reward. Practical: they don't train on million-token trajectories — too slow even at 80-100 TPS on MemoVR Pro (one rollout = 1-2 min); pretrain on long-context, then briefly activate in post-training. **Only Claude 4.5/4.6 Sonnet is genuinely stable on long context today; Gemini "其实都是不行的."**
- Numbers: ~100 people per 飞书 group; 80-100 TPS on MemoVR Pro = 1-2 min/rollout; "128K data 都很难去找到, 1兆 (1M) is rarer."

### §4 — 2026是生产力变革之年  [[41:31 → 1:01:45](https://youtu.be/vG1RBqn1sG4?t=2491)]
- **OpenClaw caught fire faster in China than in the US** because Chinese developers feel a more urgent need to use code to boost their efficiency, and because ~85% of efficiency-boost scenarios don't need the most expensive frontier model — **"他可能花10块钱的API就能帮你干完1000块钱的事情 那你肯定很愿意用 但是如果你的API贵那么十倍或几十倍 [...] 那么你会很排斥的去用这样一套很复杂的东西"** [[43:32](https://youtu.be/vG1RBqn1sG4?t=2612)].
- Earlier "Agent" frameworks (BrowseComp, SWE-bench, ToolBench, plain Search/Code agents) were too task-specific — "basically a longer system prompt + a little environment feedback" — and weren't industrially usable; plugging those models into Claude Code / OpenClaw exposes that they don't even understand the framework itself.
- **Skills = 群体智能 supplementing pretraining**: Skills capture the organization-internal, non-internet-indexed knowledge pretraining will never see. Mostly written by the agent itself but "lit on fire" by OpenClaw. Closed frameworks can't aggregate that crowd intelligence.
- **Benchmark abandonment** during MiMo-V2 model optimization: "**我们在优化这一版模型的时候 是完全放弃这些Benchmark的 [...] 当你面临一个很大的范式的变化的时候 [...] 请你去忽略评估 因为你靠体感 你就能立马测出来一个非常大的质的差异**" [[47:01](https://youtu.be/vG1RBqn1sG4?t=2821)].
- **Agent-era entry ticket**: a **non-chat-optimized highly efficient model architecture, strong code ability in pretraining, ≥1T total parameters**: "**这个model的参数量可能至少 我希望我觉得至少一T以上吧 [...] 上一个时代的成功 并不意味着下一个时代的领先**" [[56:43](https://youtu.be/vG1RBqn1sG4?t=3403)].
- **Definition of 2026**: "**生产力加速变革的时代 [...] 大家会觉得很多工作 不需要自己做了**" — not just algorithm researchers pushing intelligence forward, but "**所有懂写代码人**" [[1:00:46](https://youtu.be/vG1RBqn1sG4?t=3646)]. Two divergent product philosophies: (a) chase frontier (longest-context, highest-value tasks at the top of the human distribution) or (b) chase universal reach (multimodal + aggressive cost reduction — a task can't cost $1000; need ~10× savings to justify adoption).
- Numbers: 85% scenarios don't need frontier model; 10元 API → 1000元 of work; 3B end-side model surprises; ≥1T parameters; **2-year Anthropic Claude Code build**; ~10× cost-saving threshold for adoption.

### §5 — Agent的自进化与自迭代  [[1:01:45 → 1:19:39](https://youtu.be/vG1RBqn1sG4?t=3705)]
- **Domestic model-companies underestimate framework design**: "**我一开始也觉得这个事情不难, 然后到后面我就觉得他整个A型的设计是非常巧妙的 [...] 我觉得他弥补了很多模型短板**" [[1:03:07](https://youtu.be/vG1RBqn1sG4?t=3787)]. Memory, message channels, proactive scheduling, self-iteration are all 弥补行动上的缺陷.
- **Hypothesis**: Claude Code was built on Claude 4.5 Sonnet/Opus, which wasn't strong enough — that resource constraint forced more refined framework design, which is what made Opus improve. Domestic models sit at Claude 4.5 Opus level, so frameworks and models "握手" there.
- **Even when base models improve, framework engineering still matters for cost**: a model needing 10B activated params today may reach Claude 4.6 Opus quality in a year for "**可能一两块钱就能有百万的上下文**" [[1:04:53](https://youtu.be/vG1RBqn1sG4?t=3893)] — the framework lets you swap in the cheaper/faster model.
- **Current multi-agent work is "有点伪" for raising task ceiling**: "**真的依赖于multi agent能够实现更好的最终的任务的完成率, 在这个纬度上, 我觉得是有点伪的, 但是它能提升效率, 就是速度 [...] 我没有看到说multi agent一定最终能够实现一个更高上限的一个东西**" [[1:06:08](https://youtu.be/vG1RBqn1sG4?t=3968)].
- **KIMI told her they feel they're playing a different game from 豆包/元宝/阿里** (who chase DAU like internet products) — KIMI is on the Anthropic path. Luo's own target: "**when can the model surpass me**" → ignores DAU, tracks token consumption, complex context, evaluation design.
- **Research org maps directly onto multi-agent**: infra/inference engineer ↔ model-training+eval ↔ data team (data team spans pre- and post-training because data sense is shared). "**他可不可以运出更强的模型呢, 然后他就自己左脚踩右脚就提升了, 我觉得这个事情是很有可能发生的**" [[1:11:04](https://youtu.be/vG1RBqn1sG4?t=4264)] — model-trains-stronger-model self-improvement felt likely within 1-2 years.
- Concrete on day-2 OpenClaw use: she built sub-agents for each family member (爸爸/妈妈/老公) in a 分数群, each with different context, delegated via sub-agents.
- Domestic Claude Code clones (QQ, Kimi, MiniMax, her team's) are "**大同小异**" — mostly LangChain-style form factors; **none iterate the framework itself faster than open Claude Code**: "**我还没有看到一个比opencloud开源社区进步更快的 [...] 我宁愿用最新的opencloud**" [[1:18:23](https://youtu.be/vG1RBqn1sG4?t=4703)].
- Numbers: 10B activated params @ 一两块钱/M tokens in a year; a PhD's 5-year context as the "transferability" upper bound.

### §6 — MiMo-V2：觉醒和伏击  [[1:19:39 → 1:45:24](https://youtu.be/vG1RBqn1sG4?t=4779)]
- **"悄无声息的伏击" — silent ambush**: Pro + Omni + TTS dropped alongside the earlier Flash, framed as a coordinated awakening, not a roadmap. "**不是我们计划的非常好的, 而是我们一下大家觉醒了, 然后就爆发了**" [[1:20:13](https://youtu.be/vG1RBqn1sG4?t=4813)].
- **Roles kept separate, not unified**: Pro = understanding/cognition + complex scheduling; Omni = perception; TTS = voice output. Reason for not unifying: cost, latency, price — e.g. TTS shouldn't pay a unified model's latency cost.
- **The non-chat architecture call**: "**我们要 for non-chat 的效率来设计模型结构。当时是有隐隐约约预感到 agent 的时代, non-chat 是非常重要的**" [[1:20:33](https://youtu.be/vG1RBqn1sG4?t=4833)]. Architecture = **hybrid retention (sliding-window + full attention) + MTP**.
- **Critique of MLA** (which GLM and Kimi K2 chose): "**MLA 它已经达到一个 L-bound 和 MMI-bound 的一个非常完美的一个临界点, 你要是用 MTP 的话, 你会发现它又卡在那个计算 Bound 上。所以现在你看所有 MLA 的模型结构, 不管是 GMAV 也好, KIMI 也好, 我猜测应该都没有上 MTP**" [[1:29:16](https://youtu.be/vG1RBqn1sG4?t=5356)]. MLA was optimized for the H-series memory-bandwidth/compute balance in the chat era; no headroom for MTP.
- **Pro pushes sliding-window-to-full ratio to 7:1**: large-model experiments show what matters is the *absolute count* of full-attention layers, not the ratio — bigger models tolerate more sparsity, so the saving in KVCache is real.
- **MTP discovered late**: when designing inference parallel plan on their own inference cards they found "computation surplus 实在太多" and MTP fit perfectly. "**MTP 它是因为它是会被 verified 的, 然后只有你预测的准, 它是采纳你当前 token 的结果, 所以它没有任何幻觉**" [[1:38:37](https://youtu.be/vG1RBqn1sG4?t=5917)]. Pretrain trains 1 extra layer (boost base capability, DeepSeek-style); post-training adds more for inference speedup.
- **Flash pricing — "lowest price + highest speed" at launch**: $1.01/M input, $0.30/M output; **Flash 100-150 TPS; Pro 60-100 TPS** (100-TPS variant costs more). Luo argues the chat-era "price by architecture cost" logic is now obsolete — Pro abandoned it for **pricing by value produced** in agent frameworks.
- **Two emerging schools of architecture design**: (1) precommit during pretraining to exact inference chip / context length / parallelism (MLA-style) — fragile when post-training extends 6-12 months and target context blows up from 128K to 10M; (2) **simpler architecture + redundancy** (hybrid + MTP) that can be retrofitted (e.g. raise sliding:full ratio on already-trained model).
- **Pro hit typical large-MoE pain**: loss spikes, extreme expert load imbalance ("一会一批 token 打过去, 一会又一批打到另外一个 expert"), giant activations. Debugging cycle includes occasionally suspecting infra before finding root cause: "**最后如果发现所有的卡都排查了没有问题, 你会怀疑是不是今天的太阳黑暴富**" [[1:44:59](https://youtu.be/vG1RBqn1sG4?t=6299)].
- Numbers: Flash 100-150 TPS; Pro 60-100 TPS; Flash $1.01/M input + $0.30/M output; sliding:full = 7:1; 100兆 (100T params) "太贵了"; context-window expectation rising from 128K → 10M in a few months.

### §7 — 1T模型是入场券  [[1:45:24 → 1:52:33](https://youtu.be/vG1RBqn1sG4?t=6324)]
- **1T-class models are the natural step after DeepSeek V3 (~670B)**: "**因为首先我训过deep sea v3这么大小 600多700币的模型 你不会再想去训一个同样的模型 [...] et是 [...] 一个比较极限的一个区间**" [[1:45:45](https://youtu.be/vG1RBqn1sG4?t=6345)]. Training run uses 几千卡.
- **Research compute is 3-5× formal training**: the headline GPU count drastically understates total need.
- **In the agent paradigm, GPUs are the binding bottleneck**: "**尤其在agent的范式下 其实卡的数量反而变成一个非常重要的瓶颈 因为idea的 单身和动手 你把它代码写出来 太快了 然后你现在卡在什么呢 卡在卡上**" [[1:46:50](https://youtu.be/vG1RBqn1sG4?t=6410)]. Inference compute demand >> training.
- **Compute ratio shift**: "**for研究跟for print train和for post train 我自己觉得一个非常合理的卡的一个比例是 可能3比1比1**" [[1:47:59](https://youtu.be/vG1RBqn1sG4?t=6479)]. GPT-era was 3:1 or 5:1 pretrain-heavy; **top teams have shifted to 1:1 pretrain:post-train**.
- **Loss spikes are treated as actionable defects, not normal**: "**很多团队会把loss spike 当做一个很正常的事情 但是我们可能会尽量的让它没有loss spike [...] 就直接把某些参数 或者某些expert给它打死**" [[1:49:33](https://youtu.be/vG1RBqn1sG4?t=6573)] — a loss spike can permanently kill an MoE expert.
- Root causes: high coefficient ratios making layer-output magnitudes diverge, structural choices, or infra/comm operator bugs. Mitigations: clip activations, dampen via norm, and **Kimi's QK-clip** — "**当QK的某些logis非常大的时候 [...] 你没办法也只能把它clip掉 这样至少能让训练更好进行下去**" [[1:51:32](https://youtu.be/vG1RBqn1sG4?t=6692)]. Small startup teams win at debugging via tight coordination of "追求极致" key people.
- Numbers: DeepSeek V3 ~670B params; training class 几千卡; research:pretrain:posttrain = 3:1:1; pretrain:posttrain shift from 3:1 / 5:1 → 1:1; research compute = 3-5× training.

### §8 — 组织平权  [[1:52:33 → 2:02:56](https://youtu.be/vG1RBqn1sG4?t=6753)]
- **"Small team, extreme" archetype**: "**我们肯定是属于小团队非常极致的类型**" [[1:52:35](https://youtu.be/vG1RBqn1sG4?t=6755)]. Slower training cycles (won't finish in 1-2 months) but can stop 2-3 weeks to chase an ambiguous problem — something deadline-driven orgs can't afford because pausing a large cluster costs **一两百万到两三百万 per day**.
- **No public deadline, no company pressure**: "**我们没有deadline 就我们觉得模型训好了我们再发**" [[1:53:26](https://youtu.be/vG1RBqn1sG4?t=6806)]. MiMo and Micro are effectively run "以创业的方式" despite formally not being startups.
- **Headcount = ~100 across data collection, data quality, pre-train infra, post-train, dev, product, and three algorithm directions (语言, 动态/动作, 语音)**; the number actually iterating on one model generation is **20-30 to 30-40 people**, evenly spread.
- **No sub-groups, no formal leaders**: no pre-train group vs. post-train group split; project drivers lack 直级/直击 reports. Pre-trainers naturally migrate to post-training based on interest and data intuition — "**对人的界定没有那么清晰**."
- **Argument for pre-trainers in post-training**: post-training's new paradigm requires **data diversity** — pre-trainers are natively wired to care about diversity (can't stuff a model with a small slice), giving them an edge over post-trainers who optimize one scenario.
- **Hierarchy = anti-creative axiom**: "**任何层级应该一定程度上都是在在规范和约束 然后规范和约束本身 我自己认为是压制创造力的**" [[2:00:29](https://youtu.be/vG1RBqn1sG4?t=7229)]. Special warning to top leaders: "**他不要有特别强的这种掌控感 然后以及 这种觉得没了我就不行**" [[2:01:03](https://youtu.be/vG1RBqn1sG4?t=7263)].
- **Management = passion-driven self-organization**: "**靠热爱驱动管理 我觉得 这个很重要的 我我自己发现是最行之有效的方式**" [[2:01:25](https://youtu.be/vG1RBqn1sG4?t=7285)]. Screen candidates by sensing in conversation whether they work for passion vs. 奇奇怪怪的目标.
- **The 100-轮 mandate is theater**: "**你没有100轮的对话 你明天就辞职**" [[2:01:57](https://youtu.be/vG1RBqn1sG4?t=7317)] — never audited. "只是一个量词", the real goal is forcing the experience.
- Numbers: cluster pause = ¥1-2M to ¥2-3M/day; team = ~100 people total; ~20-30 to 30-40 per model-generation iteration; "100-轮 mandate" is a quantifier, not enforced.

### §9 — 训练细节和成本  [[2:02:56 → 2:09:03](https://youtu.be/vG1RBqn1sG4?t=7376)]
- **ET model pretraining hit 2-3 loss spikes**; longest debug pause = **two weeks**. "**反正两三次总是有的 [...] 就loss直接飞了 [...] 比如说训了几百步又回来了**" [[2:03:07](https://youtu.be/vG1RBqn1sG4?t=7387)].
- **Anxiety calibration**: not anxious about pauses ("我们又没有什么目标") but **guilty about wasted compute** on unsuccessful debugging experiments; loses sleep over loss spikes — "**我经常晚上周末说 呃为什么loss又spike [...] 我烂几天晚上脑袋**" [[2:04:03](https://youtu.be/vG1RBqn1sG4?t=7443)].
- **Parameter-scaling claim**: total params + activated params *jointly* determine intelligence ceiling. "**我觉得一定要一体以上的参数规模才能做到 [...] 才能让大家觉得你已经非常接近于4.6ops这样的水**" [[2:05:10](https://youtu.be/vG1RBqn1sG4?t=7510)] — i.e. **≥1T total parameters to approach Claude 4.6 Opus level**. Bigger activated params = higher inference cost — pure tradeoff.
- Three Pro architecture decisions reviewed: (1) MoE hybrid expert; (2) **long context** — embedding still must be trained, hard part is finding/constructing 1T tokens of *truly dense* supervision in 1M-context windows; (3) MTP inherited from Flash unchanged.
- **Endgame thinking on long context**: "**你如果你有一个一体的token量 [...] 而且它都是依照的真正的场上下文 [...] 你只要nose一直在降低它就是在在在建模在压缩 [...] 那么它就一定能迅上去**" [[2:07:48](https://youtu.be/vG1RBqn1sG4?t=7668)] — modeling = compression. Bottleneck is constructing 真正一体依照的 context.
- Pro started "几个月前". Numbers: 2-3 ET loss spikes; longest 2-week pause; ≥1T total params for Claude 4.6 Opus parity; 1T tokens of genuine long-context data as the loss-down condition.

### §10 — 另类架构  [[2:09:03 → 2:22:32](https://youtu.be/vG1RBqn1sG4?t=7743)]
- **MiMo's "另类" architecture bet**: discretize audio (and eventually image) into text-like token IDs so one LLM/NTP-style pretraining + RL infra covers all modalities. **"我们应该体内就我们的技术架构应该是非常另类的 [...] 国外的预算家也好国内像豆包也做的蛮好的应该都是跟我们完全不一样的架构"** [[2:11:38](https://youtu.be/vG1RBqn1sG4?t=7898)].
- Lossless audio discretization requires finer encoders (multi-layer RVQ → high-dim "dense-like" discrete space) and much more pretraining compute — **discrete features emerge later than continuous ones**.
- **The bet partly comes from team lineage**: "**就是做让我批的人执念吧 就我们做音频人全是做让我批的人 所以有这个执念**" [[2:11:51](https://youtu.be/vG1RBqn1sG4?t=7911)] — audio team is a language-model-style lineage, so they trust discrete tokenization.
- **Why the unification obsession may have weakened**: "**我原来以为写这些架构蛮耗费人力蛮耗费时间的 但是现在看起来在Azure的支持下 写这些架构的时间被大量缩短 那你其实就没有必要为了架构的优雅性去做很多为了统一而统一的研究**" [[2:13:16](https://youtu.be/vG1RBqn1sG4?t=8056)] — fresh LM-infra or RL infra rewrite is "几个人靠靠靠的两三周" now.
- Audio discretization has been "迈过去"; image discretization still in progress, uncertain.
- **Omni's VIT is NOT replaced** by a discrete representation — stayed continuous Hybrid Sliding Window VIT, made more efficient. Omni claims first model supporting joint audio-video understanding while keeping Agentic capability roughly on par with the language model.
- **Belief revision**: two months ago she strongly believed omni-modal training would produce more intelligence; training Omni shook that belief. Observed wins are perception / 世界知识 / 情商 (Omni feels smarter on subtle perception than larger Pro despite being smaller) but **"在任何Benchmark上你是没有任何就是纹丝不动的"** [[2:15:54](https://youtu.be/vG1RBqn1sG4?t=8154)].
- Current hypothesis: pure perception expansion probably doesn't boost intelligence; generation inside a unified understand+generate architecture might — open research problem.
- **TTS bet**: discrete-tokenizer + LLM-style architecture on **万亿 (trillion) hours** audio. Style tags trained only on a few stiff categories (fast/slow/happy/sad) generalize to free-form natural-language style descriptions; upper bound "非常惊艳"; lower bound unstable → ships as time-limited free API.
- Numbers: infra rewrite = 2-3 weeks for "a few people"; TTS trained on 万亿 hours.

### §11 — AI没有生存危机  [[2:22:32 → 2:39:12](https://youtu.be/vG1RBqn1sG4?t=8552)]
- **The "inverted triangle" framing**: human evolution is right-side-up driven by survival pressure; LLMs are an inverted triangle with language at the bottom because **there is no survival crisis**. "**当没有生存的危机的时候 它反而会进化的更自由 然后更散漫更有创造力**" [[2:23:30](https://youtu.be/vG1RBqn1sG4?t=8610)]. Abundant compute, human knowledge as starter, many humans helping = faster, freer evolution.
- **Next step after coding**: "**doing very complex software engineering**" — complexity (not LOC) is what matters; e.g. a CUDA kernel is short code but requires long debug/verification cycles to confirm real training speed-ups. Beyond coding: interaction surfaces (Feishu, WhatsApp, Telegram) and eventually **robots** — but robot evolution (hardware, battery, dexterous hands) will be slower than language-space evolution.
- **AGI trajectory**: "**感觉历程已经到了20%吧 [...] 我觉得至少能到六七十 [...] 我感觉两年内应该能实现**" [[2:26:40](https://youtu.be/vG1RBqn1sG4?t=8800)]. Key accelerating variable = **AI training/selecting AI** (self-iteration); life-mode overturn comes after work-mode overturn because work produces productivity value.
- **China-US gap**: several domestic teams (Kimi, MiMo, others) now have **1T+ base models**. **"就打Cloud Ops 4.6来说 我认为如果反应速度足够快的话 应该只有两三个月的代差 [...] 能追上当代的Cloud"** [[2:29:06](https://youtu.be/vG1RBqn1sG4?t=8946)] — closing the gap to the *current* Claude, not the Claude of 2-3 months from now.
- **Three reasons MiMo moves fast**: (1) pre-train strategic conviction 6-12 months ahead — **"以前我认为是一年 现在我认为是半年 因为agent实在会加速这个事情"** [[2:32:49](https://youtu.be/vG1RBqn1sG4?t=9169)]; (2) agent-coupled post-training requires R-infra agility (harder than reasoning-infra: model-agent coupling, fast-changing frameworks, fault tolerance, GPU+CPU co-management); (3) curiosity / love / technical conviction in researchers — selecting and channeling them is "no easier than designing a complex agent system."
- **Mediocre architecture = mediocre cost, not mediocre quality**: "**如果一个模型结构 具备一个优势 它可能就是一个很平庸的模型结构 一个很平庸的模型结构 并不会说带来一个非常平庸的模型效果 但是它一定会带来一个非常平庸的成本和效率的劣势**" [[2:33:05](https://youtu.be/vG1RBqn1sG4?t=9185)].
- **Inference demand will explode several- to ten-fold**: "**推理的需求一定会爆发 我觉得几倍到十倍的空间**" [[2:30:33](https://youtu.be/vG1RBqn1sG4?t=9033)].
- **Framework as amplifier, not patch**: "**我觉得对于顶尖模型来说它也不算补丁 [...] 对顶尖模型来说它好像是加油器 但对于中段的模型来说 它就是一个非常好的放大器 [...] 但对于顶尖模型来说 好像它是成倍的放大它的上限**" [[2:38:47](https://youtu.be/vG1RBqn1sG4?t=9527)].
- Numbers: AGI ~20% today → ≥60-70% by year-end → within two years for work-mode overturn; China-US gap = 2-3 months vs. Claude Opus 4.6; inference demand × 几倍-十倍; pre-train strategic horizon from 1 year → 半年.

### §12 — 每天在否认昨天的自己  [[2:39:12 → 2:48:34](https://youtu.be/vG1RBqn1sG4?t=9552)]
- **Continuous self-negation as progress mode**: "**我感觉每天可能都在讽定昨天的自己 就不管是很多 做事的方式上 还是你对事情未来的一些判断上 我基本上都在一直去讽定**" [[2:39:29](https://youtu.be/vG1RBqn1sG4?t=9569)] — no single signature event.
- **心法 from quant trading**: "**以前我在作战画的时候 我觉得学到一个非常有 让我能够去克服挑战的 很重要的一个一句话是 总有方式去建模 价格**" [[2:40:34](https://youtu.be/vG1RBqn1sG4?t=9634)]. In LLMs the reward is unclear and shifting, so she substitutes **"做当下符合我价值观的事情"** as the substitute reward.
- **Philanthropic-org idea (floated ~one month before recording)**: a 公益型 fund for Chinese basic research, because current funding requires complete products and "**乱七八糟的证明**", starving breakthrough-direction work [[2:42:43](https://youtu.be/vG1RBqn1sG4?t=9763)].
- **Post-AGI world**: science research still matters — "**为什么一定要去跟它竞争, 就让它做好了**" — humans should help AI accelerate science rather than compete; pure consumption/leisure would be boring.
- **Stress = sliding window**: "**我的脑子就是一个sadly window 我忘得非常快 我哪怕有压力 我当下立马就是 可能快的话一两个小时就过了 慢的话一天就过了 我睡觉第二天一定就过了**" [[2:44:49](https://youtu.be/vG1RBqn1sG4?t=9889)] — needs new imaginative work to displace the prior context.
- **MiMo-V2 March 11 anonymous OpenRouter drop — "两个神秘模型"**: pulled mid-post-training checkpoints that became usable, wanted unbiased anonymous evaluation; key learning was that **长文 (long-context) was undertrained**, fixed in the week between anonymous and official release.
- **Internal benchmark = "做好大模型本身就是benchmark"** with "好" self-defined [[2:47:56](https://youtu.be/vG1RBqn1sG4?t=10076)]; her Xiaomi boss (strategic angel-investor type) pre-agreed to this autonomy before she joined.
- Numbers: stress clears in 1-2 hours fast / 1 day slow / overnight worst case; March 11 anonymous → 1 week → official release.

### §13 — 过去3年的AI进化史  [[2:48:34 → 3:05:54](https://youtu.be/vG1RBqn1sG4?t=10114)]
- **ChatGPT (late 2022) = first time pretraining-scale intelligence was made *perceivable***: "**我觉得trash比是第一个就是发挥模型在一个我猜测应该就是一个4K的预训练的场景一边的模型的智能水平 [...] 所有激发的一个前提都是要靠有一个很能让人感知到智能水平的这样一套交互**" [[2:48:55](https://youtu.be/vG1RBqn1sG4?t=10135)]. Multi-turn correction was the UX hack that exposed low-loss intelligence inside a 4K window.
- **2023 = open-source catch-up year**: Llama disclosed the recipe (architecture details, Pre-Norm vs Post-Norm, head size, hyperparameters had been opaque). **Qwen vs DeepSeek split**: "**QNN是在纯scanning, DeepSeeker是考虑的是创新的基础上在scanning [...] 一个开源势力是在做研究上做到绝对的高度, 然后一个开源势力是在真的生态和生态价值上**" [[2:53:36](https://youtu.be/vG1RBqn1sG4?t=10416)]. Neither right nor wrong — different goals, DeepSeek with **"几分之一" of Qwen's compute**.
- DeepSeek's architectural contributions (MoE for efficient training, MLA for cheaper inference) also gave inference-chip vendors clearer signal on next-gen chip design — positive externality on the whole AGI stack.
- **R1/O1 was a *reorganization* event at DeepSeek**: when paradigm shifts pretrain→post-train, the team has to be restructured. Treating "pretrain people do pretrain" as fixed kills innovation — pure post-trainers lack the data-diversity instinct.
- **The Codeforces → general-reasoning generalization surprised her**: "**我没有预示到的事情是, 它其实是一个范式的转变, 就Resonance它其实是可以通过Cologram Mass这个高泛化的场景, 能放到通用以外, 这个其实O1也没有走通**" [[2:58:02](https://youtu.be/vG1RBqn1sG4?t=10682)]. When she left DeepSeek R1 was at "Light", Codeforces ≈ O1-mini; she expected AIME would jump 30-40 → 70-80 (it later hit 100), but did NOT expect reasoning to generalize out of Codeforces/math.
- **2025 = "criss-crossed" fork year**: deepen reasoning inside chat paradigm (SWE-bench, LiveCodeBench, AIME) or pivot to agent-native post-training. "**我可能在这套范式上能够做到六七十分就OK了, 其实AIME做了六七十分就表示这个链路你已经走通了**" — then smart teams should pivot. **MiniMax pivoted earliest in China — earlier than Kimi**.
- **BrowseComp/TerminalBench dismissed as "歧途"**: "**BronzeComp比如说它就是一个非常离谱的一个评价指标 [...] 你换种方式哪怕也是做信息检索的方式, 你最终它能力还是发挥不出去 [...] 所以就是这半年如果说在做Agent的人大部分是在走到这个歧途上**" [[3:00:28](https://youtu.be/vG1RBqn1sG4?t=10828)].
- **Hiring philosophy under heavy supervision-aversion**: "**我极少在中间给非常强的Supervising, 除非我发现你要掉头了 [...] 否则你给太细节的这种监督信号, 就告诉他这个事应该怎么做的一个缺陷, 就是你会让团队的大部分人去去失去原创能力**" [[3:01:53](https://youtu.be/vG1RBqn1sG4?t=10913)]. Team built mostly from people *without* large-model background — only ~1/3-1/4 have trained even 7B-14B.
- **Agent-era entry ticket restated**: "**我把入场定义为你要做到对标到CloudOps 4.6的水平。它需要一体的基础做, 与此同时它需要敏捷性 [...] 所以现在中国公司还没有同时具备两者 [...] 看一看DeepThink吧**" [[3:05:28](https://youtu.be/vG1RBqn1sG4?t=11128)]. MiniMax has agility but not the 1T base; no Chinese company has both yet.
- Numbers: ChatGPT 4K context; DeepSeek compute = 几分之一 of Qwen; AIME 30-40 → 70-80 expected → 100 actual; 6-7/10 AIME = chat-link proved; 1/3-1/4 of team has 7B-14B training experience.

### §14 — 当下共识与竞争  [[3:05:54 → 3:19:45](https://youtu.be/vG1RBqn1sG4?t=11154)]
- **Industry consensus locked in**: "**大家可能一个共识都是认为Azure路径是正确的 [...] 至少在过去的三个月以内, 我觉得Agent的路是变得更清晰了**" [[3:06:10](https://youtu.be/vG1RBqn1sG4?t=11170)] (ASR "Azure" → "Agent").
- **Chinese pretrain gap to US is "basically closed"** — possibly structural advantage; next race is **Agent post-training, specifically RL scaling**.
- **Anthropic-Context-Engineering re-read**: "**甚至一度我认为Cloud可能在过去很长一段时间, 做的很多Context Engineering, 我们都误以为它是因为模型结构不是很先进, 然后为了成本而做了一些妥协的设计, 但现在回过头来看, 可能是有点想得太局限了**" [[3:06:52](https://youtu.be/vG1RBqn1sG4?t=11212)] — Anthropic's heavy context engineering was a deliberate scaffold, not a model-weakness patch.
- **Coding hits the sweet spot in every paradigm**: good pretrain signal, strong verified reward for R1-style RL, natural long-horizon environment for Agents, easy to scale because natural-language-like.
- **RL-scaling disclosure rule**: "**至少我觉得在RL Scaling上的算力, 跟预训练的算力, 达到一个同一个水位的时候, 我觉得我们会给大家分享**" [[3:10:21](https://youtu.be/vG1RBqn1sG4?t=11421)].
- **Long-context coupling**: scaling context in pretrain forces several-fold larger post-train compute; **"在1兆上去做PostalTrain跟在256K上去做PostalTrain, 它的算力差距是好几倍的差距"** [[3:11:30](https://youtu.be/vG1RBqn1sG4?t=11490)].
- **Multi-agent "一个人养很多个员工"**: "**有人说openclose上, 我一个人养很多个员工 [...] 虽然当下目前来看是不那么现实的, 或者说我觉得有点噱头, 但是我觉得它很快会变成一个现实, 在今年内**" [[3:13:15](https://youtu.be/vG1RBqn1sG4?t=11595)]. Gap to make it real: cheaper model (cost-per-employee economics) + better multi-agent architecture + better self-improvement / inter-agent communication; she doesn't think multi-agent collaborative RL training is needed.
- **2026 winning conditions**: (1) don't fail at pretrain base (esp. code potential); (2) make agent framework and model self-iterate together; (3) couple agent framework to strategic resources (OS, hardware, traffic, social); (4) be willing to rethink whether old org/headcount is needed at all — "**得思考原来所有做的东西都是错的, 原来是不是需要这么多人来做这个事情**" [[3:19:10](https://youtu.be/vG1RBqn1sG4?t=12030)].
- **Open-source rationale**: open-sourcing accelerates AGI because chips, inference, frameworks, energy will be distributed across many players; whether a company open-sources depends on whether it owns a non-model strategic 生态位.
- Numbers: post-train can ship a model in 1 month vs. pretrain cadence; 1M → 256K post-train compute difference = several-fold; multi-agent "real within this year"; an agent burning >¥1000/day of Claude Code-class cost vs. human employee value ~¥1000.

### §15 — 环境比经验更重要  [[3:19:45 → 3:36:36](https://youtu.be/vG1RBqn1sG4?t=11985)]
- **Frontier research must be slightly counter-mainstream but still scalable**: DeepSeek's MoE + attention modifications were "anti-mainstream" choices made under resource constraints that turned out to be scalable industrial work. **"我觉得我自己觉得有点反主流, 我觉得不是很适合的一件事情是你很难scanning, 我还是很相信scanning这个事儿"** [[3:20:09](https://youtu.be/vG1RBqn1sG4?t=12009)].
- **Stopped reading academic papers**: "**你对发paper现在有执念之类的吗?没有, 就发的越少越好 [...] 我现在也不看学术会议的paper, 主要的原因之一是, 我觉得大部分的实验确实应该自己做, 然后你相信自己的实验结果比相信论文的实验结果会更好**" [[3:22:41](https://youtu.be/vG1RBqn1sG4?t=12161)].
- **Environment > experience — the headline thesis**: "**所以环境反而比经验更重要, 我自己认为, 所以我就没有太在乎他的经验, 而更在乎我是不是创造了一个更好的环境 [...] 我只在乎他的初始化期和point的上限高不高**" [[3:24:42](https://youtu.be/vG1RBqn1sG4?t=12282)]. Hires for **initialization point and ceiling, not current state**.
- **Sophomore/junior undergraduates preferred** because their thinking isn't "polluted" by old paradigms: "**我们也招了非常多的本科生 [...] 他的想象力会更高 [...] 因为他们的灵活性和适应程度都感觉没有被污染**" [[3:26:15](https://youtu.be/vG1RBqn1sG4?t=12375)].
- **Compensation matters as baseline** ("钱要给够"), but value/meaning matter more for retention; organization should not be built around overly specific targets.
- **Post-training/RL hires fit two profiles**: (1) people who constantly play with models to find capability boundaries and maintain private eval sets, (2) people who can co-design RL infra with agent frameworks.
- **RL infra ≠ pre-training infra**: pre-training is zero-tolerance for loss spikes; **RL must tolerate fuzziness** — broken rollout curves with unknown causes, train/inference inconsistency across heterogeneous clusters (GPU + CPU + storage), agent framework timeouts. The two infra teams stay separate because complexity and precision requirements differ greatly.
- **Only one overseas team is scaling agent RL — likely OpenAI**: other teams' model effects don't yet show RL scaling on par with pretraining.
- **Next paradigm**: combine generative models with strong perception models into one framework for joint pre-training; **continual/online learning reframed as the framework itself iterating during multi-turn interaction**.
- Numbers: only ~20% of team has trained even small models; **PhD ratio = 55% including in-program**; capability-absorption window 1-2 months fast / 3-4 slow; daily rhythm 11am → 1-4am, 4-6 working hours "OK".

## Notable quotes

> "我自己其实是会把opencloud把它当做一个划时代的agent的框架去这么去定义。"
> — Luo Fuli, §2 [[02:33](https://youtu.be/vG1RBqn1sG4?t=153)] — *first-principles re-framing of OpenClaw as the watershed*

> "如果第二天opencloud对话次数不超过100轮的人 可以直接quit。"
> — Luo Fuli, §3 [[25:24](https://youtu.be/vG1RBqn1sG4?t=1524)] — *the deliberately theatrical Spring-Festival mandate that unlocked swarm-intelligence framework iteration*

> "我们基本上可能在三四周的时间 做完了以前可能三四十周的时间 才能做到的事情 就在研究上。"
> — Luo Fuli, §3 [[28:42](https://youtu.be/vG1RBqn1sG4?t=1722)] — *the most concrete productivity-compression claim in the episode*

> "MLA 它已经达到一个 L-bound 和 MMI-bound 的一个非常完美的一个临界点, 你要是用 MTP 的话, 你会发现它又卡在那个计算 Bound 上。"
> — Luo Fuli, §6 [[1:29:16](https://youtu.be/vG1RBqn1sG4?t=5356)] — *the load-bearing technical reason MiMo-V2 broke from MLA into hybrid + MTP*

> "for研究跟for print train和for post train 我自己觉得一个非常合理的卡的一个比例是 可能3比1比1。"
> — Luo Fuli, §7 [[1:47:59](https://youtu.be/vG1RBqn1sG4?t=6479)] — *the 3:1:1 compute-allocation heuristic*

> "我感觉每天可能都在讽定昨天的自己 [...] 我基本上都在一直去讽定。"
> — Luo Fuli, §12 [[2:39:29](https://youtu.be/vG1RBqn1sG4?t=9569)] — *self-negation as progress mode*

> "就打Cloud Ops 4.6来说 我认为如果反应速度足够快的话 应该只有两三个月的代差 [...] 能追上当代的Cloud。"
> — Luo Fuli, §11 [[2:29:06](https://youtu.be/vG1RBqn1sG4?t=8946)] — *the precise China-US gap framing*

> "所以环境反而比经验更重要, 我自己认为, 所以我就没有太在乎他的经验, 而更在乎我是不是创造了一个更好的环境。"
> — Luo Fuli, §15 [[3:24:42](https://youtu.be/vG1RBqn1sG4?t=12282)] — *the episode's thesis statement, restated near the close*

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| 张小珺 (Zhang Xiaojun) | Host (ASR: 小骏) | [[00:03](https://youtu.be/vG1RBqn1sG4?t=3)] |
| 罗福莉 (Luo Fuli) | Guest — ex-Alibaba DAMO + ex-DeepSeek, head of Xiaomi MiMo (ASR: 罗弗利) | [[00:22](https://youtu.be/vG1RBqn1sG4?t=22)] |
| Anthropic | Claude / Claude Code; the ~2-year build path Luo references | §4 [[51:54](https://youtu.be/vG1RBqn1sG4?t=3114)] |
| Luo Fuli's boss (her 老板 — uses Claude Code for non-code tasks) | Mentioned as someone Luo "doesn't expect high completion" for in non-code use | §2 [[12:53](https://youtu.be/vG1RBqn1sG4?t=773)] |
| DeepSeek | Architecture lineage Luo built on (DeepSeek V3 → her 1T-class work); MTP also DeepSeek-style | §6 [[1:27:55](https://youtu.be/vG1RBqn1sG4?t=5275)] |
| GLM (ASR "GMAV") | Contemporaneous training; chose MLA; "应该都没有上 MTP" | §6 [[1:27:46](https://youtu.be/vG1RBqn1sG4?t=5266)] |
| Kimi / K2 | Started slightly earlier than MiMo-V2; also chose MLA; QK-clip technique credited | §6 [[1:27:51](https://youtu.be/vG1RBqn1sG4?t=5271)] |
| KIMI team | Told Luo they feel they're playing a "different game" from 豆包/元宝/阿里, walking the Anthropic path | §5 [[1:06:50](https://youtu.be/vG1RBqn1sG4?t=4010)] |
| 豆包 / 元宝 / 阿里 (Doubao / Yuanbao / Alibaba) | Cited as model companies playing internet-product DAU game | §5 [[1:06:55](https://youtu.be/vG1RBqn1sG4?t=4015)] |
| 豆包 (ByteDance Doubao) | Multimodal stack with a completely different (non-discrete-unified) architecture from MiMo | §10 [[2:11:45](https://youtu.be/vG1RBqn1sG4?t=7905)] |
| Luo Fuli's family (爸爸 / 妈妈 / 老公) | Day-2 OpenClaw use: set up sub-agents for each family member with different context | §5 [[1:13:48](https://youtu.be/vG1RBqn1sG4?t=4428)] |
| QQ team / Kimi / MiniMax | Domestic teams that shipped Claude Code-like products; Luo tried "about half" | §5 [[1:17:54](https://youtu.be/vG1RBqn1sG4?t=4674)] |
| MiniMax | The *earliest* Chinese team to pivot to the agent-native paradigm — earlier than Kimi | §13 [[2:59:38](https://youtu.be/vG1RBqn1sG4?t=10778)] |
| Xiaomi boss (Lei Jun, unnamed on-mic) | Strategic angel-investor type who pre-agreed to Luo's benchmark-autonomy before she joined | §12 [[2:48:06](https://youtu.be/vG1RBqn1sG4?t=10086)] |
| OpenAI (ASR: "sobi") | The only overseas team Luo sees as clearly scaling agent RL | §15 [[3:30:00](https://youtu.be/vG1RBqn1sG4?t=12600)] |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| MiMo-V series / Flash / Pro / Omni / TTS | The Xiaomi model family Luo leads | [[00:29](https://youtu.be/vG1RBqn1sG4?t=29)] |
| Claude Opus 4.6 (ASR: Cloud Opus 4.6 / Cloud Ops 4.6) | Frontier reference model; "entry ticket" target | [[00:44](https://youtu.be/vG1RBqn1sG4?t=44)] |
| OpenClaw (ASR: Open Cloud / opencloud / openclose / OpenCore) | The agent framework Luo calls 划时代 | §2 [[02:25](https://youtu.be/vG1RBqn1sG4?t=145)] |
| Claude Code (ASR: cloudcode / calcode) | OpenClaw predecessor; for-software-engineering Claude harness | §2 [[04:43](https://youtu.be/vG1RBqn1sG4?t=283)] |
| Claude Sonnet | Mid-tier model used to save cost; "85% of tasks" baseline | §2 [[16:25](https://youtu.be/vG1RBqn1sG4?t=985)] |
| Skill Hub | OpenClaw "玄幻" ops feature | §2 [[03:13](https://youtu.be/vG1RBqn1sG4?t=193)] |
| Skills / AGENTS.md | OpenClaw's skill + agents.md scaffold | §2 [[04:37](https://youtu.be/vG1RBqn1sG4?t=277)] |
| BrowseComp (ASR: BronzeComp) | Agent benchmark Luo calls "歧途" | §13 [[3:00:12](https://youtu.be/vG1RBqn1sG4?t=10812)] |
| SWE-bench (ASR: Swebunch) | Coding-agent benchmark | §4 [[44:15](https://youtu.be/vG1RBqn1sG4?t=2655)] |
| ToolBench (ASR: Talbunch) | Early task-specific tool-use bench | §4 [[45:19](https://youtu.be/vG1RBqn1sG4?t=2719)] |
| OpenAI o1 (ASR: OE21 / O21) | Reasoning-paradigm marker | §4 [[51:20](https://youtu.be/vG1RBqn1sG4?t=3080)] |
| LangChain (ASR: lay chart) | Form factor most domestic Claude Code clones imitate | §5 [[1:18:09](https://youtu.be/vG1RBqn1sG4?t=4689)] |
| Harness / Scaffold | Agent-framework terminology | §5 [[1:14:46](https://youtu.be/vG1RBqn1sG4?t=4486)] |
| MLA (Multi-head Latent Attention) | Architecture choice MiMo-V2 broke from | §6 [[1:27:46](https://youtu.be/vG1RBqn1sG4?t=5266)] |
| MTP (Multi-Token Prediction) | MiMo-V2's verified inference accelerator | §6 [[1:27:50](https://youtu.be/vG1RBqn1sG4?t=5270)] |
| CoreCoder | Mentioned in §6 ops context | §6 [[1:21:42](https://youtu.be/vG1RBqn1sG4?t=4902)] |
| DeepSeek V3 (~670B params) | Predecessor of Luo's 1T-class training | §7 [[1:45:45](https://youtu.be/vG1RBqn1sG4?t=6345)] |
| Kimi QK-clip technique | Loss-spike mitigation Luo borrows | §7 [[1:51:32](https://youtu.be/vG1RBqn1sG4?t=6692)] |
| MiMo VR Pro / Flash | Internal model line referenced for TPS / loss-spike anecdotes | §7 [[1:46:18](https://youtu.be/vG1RBqn1sG4?t=6378)] |
| ET model | Internal pretraining experiment with 2-3 loss spikes | §9 [[2:02:57](https://youtu.be/vG1RBqn1sG4?t=7377)] |
| MiMo Omni / MIMO VL | Joint audio-video understanding model | §10 [[2:15:45](https://youtu.be/vG1RBqn1sG4?t=8145)] |
| MiMo TTS | Discrete-tokenizer + LLM-style TTS at 万亿 hours | §10 [[2:09:13](https://youtu.be/vG1RBqn1sG4?t=7753)] |
| Llama (Meta) | 2023 disclosure that enabled Qwen / DeepSeek scaling | §13 [[2:50:32](https://youtu.be/vG1RBqn1sG4?t=10232)] |
| Qwen (ASR: QNN / Q1) | "Pure scaling + full size-range + multimodal" path | §13 [[2:50:32](https://youtu.be/vG1RBqn1sG4?t=10232)] |
| DeepSeek R1 | The 2024 reasoning paradigm shift; "reorganization event" | §13 [[2:54:30](https://youtu.be/vG1RBqn1sG4?t=10470)] |
| AIME | Math reasoning benchmark — 30-40 → 70-80 expected → 100 actual | §13 [[2:58:54](https://youtu.be/vG1RBqn1sG4?t=10734)] |
| LiveCodeBench (ASR: Navcode Bench) | Coding-reasoning benchmark | §13 [[2:58:57](https://youtu.be/vG1RBqn1sG4?t=10737)] |
| Codeforces (ASR: Cologram Mass) | Early R1 reasoning training environment | §13 [[2:57:35](https://youtu.be/vG1RBqn1sG4?t=10655)] |
| TerminalBench | Agent benchmark Luo dismisses alongside BrowseComp | §13 [[3:00:15](https://youtu.be/vG1RBqn1sG4?t=10815)] |
| BBH (Big-Bench Hard) | General reasoning bench cited alongside coding bench | §14 [[3:08:58](https://youtu.be/vG1RBqn1sG4?t=11338)] |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| Capability-absorption window: 1-2 months fast, 3-4 months slow → "环境反而比经验更重要" | Luo | [[01:20](https://youtu.be/vG1RBqn1sG4?t=80)] |
| Research : pretrain : post-train compute ≈ 3 : 1 : 1 | Luo | [[01:40](https://youtu.be/vG1RBqn1sG4?t=100)] |
| First OpenClaw session: 凌晨2点 → 6点 (4 hours) | Luo | [[03:55](https://youtu.be/vG1RBqn1sG4?t=235)] |
| Mid-tier model + framework matches Sonnet on ~85% of tasks | Luo | [[15:25](https://youtu.be/vG1RBqn1sG4?t=925)] |
| Day-1 OpenClaw burn: ~¥1000 / $1000 in 4-5 hours of Opus 4.6 | Luo | [[22:53](https://youtu.be/vG1RBqn1sG4?t=1373)] |
| OpenClaw 3.x mid-March = broadly usable with any decent model | Luo | [[22:05](https://youtu.be/vG1RBqn1sG4?t=1325)] |
| 100-轮 quit mandate on second day of OpenClaw install | Luo | [[25:24](https://youtu.be/vG1RBqn1sG4?t=1524)] |
| ~100 people per 飞书 group during Spring-Festival rollout | Luo | [[29:34](https://youtu.be/vG1RBqn1sG4?t=1774)] |
| 3-4 weeks of agent-assisted research ≈ 30-40 weeks of old work | Luo | [[28:42](https://youtu.be/vG1RBqn1sG4?t=1722)] |
| Genuine long-context data (128K-1M tokens) found only in code + books | Luo | [[34:41](https://youtu.be/vG1RBqn1sG4?t=2081)] |
| MemoVR Pro: 80-100 TPS; one rollout = 1-2 minutes | Luo | [[39:38](https://youtu.be/vG1RBqn1sG4?t=2378)] |
| ~85% of efficiency-boost scenarios don't need top frontier model | Luo | [[43:08](https://youtu.be/vG1RBqn1sG4?t=2588)] |
| 10元 of API → 1000元 of work value | Luo | [[43:32](https://youtu.be/vG1RBqn1sG4?t=2612)] |
| Anthropic's Claude Code = ~2 years build path | Luo | [[51:54](https://youtu.be/vG1RBqn1sG4?t=3114)] |
| ≥1T total parameters as agent-era entry ticket | Luo | [[56:43](https://youtu.be/vG1RBqn1sG4?t=3403)] |
| Task cost can't be $1000 — need ~10× cost savings | Luo | [[59:26](https://youtu.be/vG1RBqn1sG4?t=3566)] |
| 10B activated params today → Claude 4.6 Opus quality in 1 year @ ¥1-2/M tokens | Luo | [[1:04:39](https://youtu.be/vG1RBqn1sG4?t=3879)] |
| Flash TPS: 100-150 | Luo | [[1:29:45](https://youtu.be/vG1RBqn1sG4?t=5385)] |
| Pro TPS: 60-100 (100-TPS variant costs more) | Luo | [[1:29:45](https://youtu.be/vG1RBqn1sG4?t=5385)] |
| Pro sliding-window : full-attention ratio = 7:1 | Luo | [[1:30:22](https://youtu.be/vG1RBqn1sG4?t=5422)] |
| Flash pricing: $1.01/M input, $0.30/M output ("lowest price + highest speed" at launch) | Luo | [[1:38:32](https://youtu.be/vG1RBqn1sG4?t=5912)] |
| 100兆 (100T params) "太贵了" — feasible but uneconomic | Luo | [[1:27:16](https://youtu.be/vG1RBqn1sG4?t=5236)] |
| Context-window expectation rising from 128K → 10M in months | Luo | [[1:36:42](https://youtu.be/vG1RBqn1sG4?t=5802)] |
| DeepSeek V3 ~600-700B params | Luo | [[1:45:48](https://youtu.be/vG1RBqn1sG4?t=6348)] |
| Training run = 几千卡 (thousands of GPUs) | Luo | [[1:46:08](https://youtu.be/vG1RBqn1sG4?t=6368)] |
| Research compute = 3-5× formal training compute | Luo | [[1:46:13](https://youtu.be/vG1RBqn1sG4?t=6373)] |
| Pretrain : post-train ratio shift: GPT-era 3:1 or 5:1 → top teams 1:1 today | Luo | [[1:48:27](https://youtu.be/vG1RBqn1sG4?t=6507)] |
| Cluster pause cost = ¥1-2M to ¥2-3M per day | Luo | [[1:52:54](https://youtu.be/vG1RBqn1sG4?t=6774)] |
| Team size: ~100 total; 20-30 to 30-40 per model-generation iteration | Luo | [[1:57:49](https://youtu.be/vG1RBqn1sG4?t=7069)] |
| ET pretraining: 2-3 loss spikes, longest 2-week debug pause | Luo | [[2:03:18](https://youtu.be/vG1RBqn1sG4?t=7398)] |
| TTS trained at 万亿 hours scale | Luo | [[2:20:28](https://youtu.be/vG1RBqn1sG4?t=8428)] |
| Infra rewrite: "几个人靠靠靠的两三周" with Azure support | Luo | [[2:13:58](https://youtu.be/vG1RBqn1sG4?t=8038)] |
| AGI ~20% today → ≥60-70% by year-end → within 2 years | Luo | [[2:26:40](https://youtu.be/vG1RBqn1sG4?t=8800)] |
| China-US gap vs. Claude Opus 4.6 ≈ 2-3 months | Luo | [[2:29:06](https://youtu.be/vG1RBqn1sG4?t=8946)] |
| Inference demand will explode 几倍-十倍 | Luo | [[2:30:33](https://youtu.be/vG1RBqn1sG4?t=9033)] |
| Pre-train strategic horizon: 1 year → 半年 (agent accelerates) | Luo | [[2:32:49](https://youtu.be/vG1RBqn1sG4?t=9169)] |
| Stress clears in 1-2 hours fast / 1 day slow / overnight worst | Luo | [[2:44:57](https://youtu.be/vG1RBqn1sG4?t=9897)] |
| March 11: two anonymous mystery models on OpenRouter; 1 week → official release | Luo | [[2:46:44](https://youtu.be/vG1RBqn1sG4?t=10004)] |
| ChatGPT context: ~4K | Luo | [[2:49:18](https://youtu.be/vG1RBqn1sG4?t=10158)] |
| DeepSeek compute ≈ 几分之一 (a fraction) of Qwen's | Luo | [[2:53:57](https://youtu.be/vG1RBqn1sG4?t=10437)] |
| AIME trajectory: 30-40 → expected 70-80 → actual 100 | Luo | [[2:57:54](https://youtu.be/vG1RBqn1sG4?t=10674)] |
| AIME 6-7/10 = enough to prove chat-reasoning link | Luo | [[2:59:22](https://youtu.be/vG1RBqn1sG4?t=10762)] |
| Team large-model-experience: only ~1/3-1/4 trained even 7B-14B | Luo | [[3:02:39](https://youtu.be/vG1RBqn1sG4?t=10959)] |
| PhD ratio in team = 55% (including in-program) | Luo | [[3:25:38](https://youtu.be/vG1RBqn1sG4?t=12338)] |
| Post-train can ship a model in 1 month (vs. pretrain cadence) | Luo | [[3:10:50](https://youtu.be/vG1RBqn1sG4?t=11450)] |
| 1M → 256K post-train compute difference = several-fold | Luo | [[3:11:30](https://youtu.be/vG1RBqn1sG4?t=11490)] |
| Daily rhythm: 11am → 1-4am; 4-6 working hours = "OK" range | Luo | [[3:34:44](https://youtu.be/vG1RBqn1sG4?t=12884)] |

## Open questions / gaps

- **"环境比经验更重要"** is restated as a thesis (§1, §15) but the empirical support is one personal experience and the team's ~1/3-1/4 large-model-experience hire profile; it's not benchmarked against teams that hired the opposite way.
- **The "1-2 month catch-up window"** (§1, §11) is asserted three times but never operationalized — what counts as "catch up" (eval suite? user perception? specific tasks?) is left undefined.
- **Mid-tier model + framework = 85% of Sonnet-level tasks** (§2) — no benchmark cited; the 85% is self-described 体感.
- **OpenClaw's claim that closed Claude Code "must be" a complex framework in black box** (§2) is asserted from inability to inspect, not from leaked architecture details.
- **"Most assets are unsuitable for long-horizon modeling"** (§3) is asserted as Luo's quant-insider view but she explicitly declines to enumerate which assets are / aren't.
- **Multi-agent does NOT raise the task-ceiling** (§5) is asserted from personal observation only; no benchmark or paper cited.
- **MTP-doesn't-fit-MLA because of L-bound/MMI-bound saturation** (§6) is asserted as "我猜测应该都没有上 MTP" — she explicitly flags GLM/Kimi's non-adoption as a guess.
- **≥1T params required for Claude 4.6 Opus parity** (§9) — no derivation; presented as personal intuition.
- **"如果loss一直在降低,长上下文能力就一定能迅上去"** (§9) is a compression-equals-intelligence axiom with no controlled comparison.
- **Omni "subjectively smarter" than Pro but benchmarks "纹丝不动"** (§10) — Luo herself flags this as either benchmarks being wrong or multimodal training compute still too small; no resolution offered.
- **"AI没有生存危机所以进化更自由"** (§11) — asserted without evidence that lack of selection pressure produces faster evolution rather than aimless drift.
- **AGI 20% → 60-70% → 2-year trajectory** (§11) is given as personal feel ("感觉", "我觉得") with no defined AGI benchmark.
- **China-US gap = 2-3 months "if reaction speed is fast"** (§11) is conditional on an unstated reaction-speed threshold.
- **BrowseComp/TerminalBench "doesn't transfer"** (§13) is asserted strongly but only with an anecdote about info-retrieval variants — no specific failing case named.
- **MiniMax pivoted to agent paradigm "earlier than Kimi"** (§13) — neither pivot date is given.
- **"看一看 DeepThink 吧"** (§13) hints DeepSeek may be the Chinese team with both 1T base + agility — asserted without evidence.
- **Multi-agent "real within this year"** (§14) and **"China structurally advantaged in pretrain"** (§14) are confident calls without supporting evidence.
- **"OpenAI is the only team scaling agent RL"** (§15) is inferred from model behavior, not direct knowledge.
- **"乱七八糟的证明" critique of basic-research funding** (§12) is personal; the proposed 公益 organization is gestured at, not concretized.

## Verification log

- **Sectioning**: chapters (15 author-supplied YouTube chapters). All 15 titles preserved verbatim from `sections.json`.
- **Transcript source**: faster-whisper large-v3 (CPU int8, local M4) — produced by `docs/videos/transcribe_batch.py`. YouTube provided no subtitles (manual or auto) for this video.
- **Speaker name corrections**: guest "罗弗利" → 罗福莉 (Luo Fuli); host "小骏" → 张小珺 (Zhang Xiaojun). Recurring ASR garbles corrected per the header note (Cloud Opus 4.6, OpenClaw, Qwen, DeepSeek, Codeforces, BrowseComp, LiveCodeBench, GLM, non-chat, 否定, sliding window, Agent 路径, OpenAI/Sora-team, 1T params, 10B activated).
- **Sections covered**: 15/15
- **Notable quotes traced verbatim**: 8/8 (each anchored by a distinctive Chinese substring grep-matched in the local flat transcript)
- **Numbers traced**: 47/47 (flexible-variant grep against flat transcript; ASR transliterations accepted)
- **Sectioning method used**: chapters
- **Removed during verification**: none

## See Also

- `docs/videos/2026-05-01-su-yu-agent-tech-history.md` — Zhang Xiaojun's complementary episode with Su Yu on the four-era Agent history and OpenClaw moment from an academia → startup perspective; this Luo Fuli interview is the inside-Chinese-frontier-lab counterpart.
- `docs/videos/2026-05-11-yao-shunyu-training-claude-gemini.md` — Zhang Xiaojun's Yao Shunyu episode (the ReAct author Luo's framing references implicitly); same host, complementary perspective from inside Anthropic / GDM.
- `skills/research/youtube-wiki/SKILL.md` — the pipeline that produced this entry.
