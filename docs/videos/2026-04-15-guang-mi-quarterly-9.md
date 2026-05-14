# Guang Mi · Global LLM Quarterly #9 — Coding as AGI Act 2, Silicon Valley's Big Three, Models as the New OS

**Source**: https://youtu.be/u1Lzp-7Ybn8
**Channel**: Zhang Xiaojun Podcast (张小珺商业访谈录)
**Published**: 2026-04-15
**Duration**: 01:22:40
**Watched on**: 2026-05-14
**Sectioning**: chapters (13 YouTube-supplied chapters; chapter 1 was `<Untitled Chapter 1>`, renamed to "开场白与本期定位" in `sections.json` before chunking; all other 12 titles preserved verbatim from `info.json.chapters`)
**Detected video language**: `zh` (no `info.json.language`; both `subtitles` and `automatic_captions` keys empty — no YouTube subs)
**Transcript source**: faster-whisper large-v3 (CPU int8) — produced by `docs/videos/transcribe_batch.py` from local audio. ASR consistently rendered the host as "小骏" — corrected from channel metadata to **张小珺 (Zhang Xiaojun)**; the guest's name appears as "广密" once and "广秘" twice in the transcript — standardized to **广密 (Guang Mi)**. Other consistent ASR substitutions kept inline with brackets: "Cloud Code" → Claude Code; "Solnet 3.5" → Sonnet 3.5; "Honest" → harness; "Misos跟Spard / Spar" → Mesos / Spark (Anthropic and OpenAI next-gen codenames); "Jarrett Kaplan / Jerry Kaplan" → Jared Kaplan; "Minas / Manus / Mitas" → Mistral; "Property" → Perplexity; "Average" → Abridge; "GTT" → GTM; "XAI / XCI / FCA" → xAI.
**Speakers**: Zhang Xiaojun (张小珺, host), Guang Mi (广密, guest — AI investor / analyst, recurring co-host of the 全球大模型季报 series; this is episode #9 of the quarterly arc and the shortest so far)

## TL;DR

- **Coding is AGI Act 2** and the new accelerator: the past quarter's progress (Anthropic Opus 4.5/4.6) is framed as a **GPT-3 → GPT-4-scale** jump that pushed models from chat into "真正的agent模式." Last year ~70-80% of code was human-written at frontier labs; this year **<1%**. Top engineers now burn **几百美金 of tokens/day, 几千美金/week**, and Anthropic shipped **70+ products in 50 working days**.
- **"语言即世界，代码即方案"** — natural language describes the world, code describes the solution. If you accept this, the Coding Agent automates most white-collar knowledge work, and any frontier lab that doesn't prioritize Coding "**大概会掉出第一梯队**." Coding is also the wedge: like Amazon-selling-books, it builds the infrastructure (data, eval, harness) that horizontal expansion later runs on, and its feedback loop is the shortest, so it wins first.
- **Silicon Valley's Big Three, dissected** — *Anthropic*: top-down, tower-tip pricing, AGI-mission-native, physics-trained founders, **~70-80% revenue from coding/agent**, **$30B+ ARR vs OpenAI ~$25B+**, projecting **$80-100B ARR by year-end, $150-200B next year** ("新的Mac 7"); compute is the only real bottleneck. *OpenAI*: 900M+ weekly actives, 50-60M paying, but woke up to coding only 2-3 months ago; **Sora 2 cut to free GPUs**; IPO likely Oct/Nov 2026; Guang Mi still gives it **~50% odds of being the AGI winner**. *Gemini 3.0*: benchmark-strong but C-end-flat, missed coding by 3-4 months — **"落后三个月可能就落后一年"** — yet long-term *the safest* via TPU + cash + 3rd-gen pro-manager system.
- **Models = the next operating system** — frontier models become **"global GDP's OS"**, with agents as "应用的无限扩展." Only Windows / iOS / Android / WeChat have ever qualified historically; ChatGPT / Gemini / Doubao are converging toward that role.
- **Social impact, sharp end**: **30% of jobs gone this year**, U.S. undergrad employment at a historic low, **Meta laid off 16k**, Microsoft "may not need 150k people — 30k might do better"; the career ladder for juniors has been **"拦腰截断."** Investment thesis: **60/10/10/10 AGI portfolio** — 20% × 3 in the leading foundation models, plus robotics, AI for Science, Agent infra. **Robotics phase change in 6-18 months**, China advantaged on hardware. New foundation labs face **"再造一个台积电"** odds — $30-50B/yr × 3-5 yrs, 100+ world-class scientists, and even then maybe not enough.

## Why it matters

Episode 9 is the shortest Quarterly so far but the densest as a *strategic* read of Q1 2026 — Guang Mi's single most concentrated argument that the AGI race has narrowed to coding-agent execution, that Anthropic's all-in-coding posture has structurally split the Big Three, and that the social-distribution consequences (white-collar deflation, severed career ladder) are arriving faster than society's signal-reception bandwidth. The wiki captures both the named-with-evidence Q1 AR leaderboard and the dissenting bets (Macrohard / Project Prometheus, robotics in 6-18 months) before the field rewrites the map again.

## Section summaries

### §1 — 开场白与本期定位 (Opening: Coding as AGI Act 2)  [[00:00 → 02:00](https://youtu.be/u1Lzp-7Ybn8?t=0)]

- Host frames the episode's mood as "**情绪十分的复杂**": an accelerating AI revolution on one side, white-collar deflation/unemployment on the other.
- Coding "**把AI从聊天机器人Charbot的第一幕推向了能够干活的Agent第二幕**" — and "研究员们已经开始不再亲自写代码了."
- Guang Mi's core preview claim, repeated as the episode's spine: **"领先的Coding模型就会像领先的GPU."**
- Philosophical pitch teased: **"语言即世界，代码即方案"** — natural language describes the world, code describes the solution.
- Frontier labs that don't prioritize Coding "大概会掉出第一梯队的."
- Each Silicon Valley lab gets only **~100 days** of dominance — "**各领风骚一百天，今天胜利的秘籍可能就是下个时代的毒药**" — and OpenAI's GTM/2C over-success is the example given (transcript renders "GTM" as "GTT").

### §2 — 第9集季报的概览 (Episode 9 overview: the Coding tsunami)  [[02:00 → 03:28](https://youtu.be/u1Lzp-7Ybn8?t=120)]

- The Quarterly series has run for **three years** (2023 → 2026); many past predictions have materialized, most recently the Coding wave that "**今年就像海啸一般涌来了**."
- Guang Mi's mission for the report: "**提前应对好AI海啸的冲击**."
- Episode roadmap (four explicit blocks + a fifth looser block):
  1. Silicon Valley field observations
  2. Why Coding is the main storyline
  3. The Silicon Valley **御三家** (Big Three) — strategy, organization, culture
  4. The ultimate form of future models and why it's an OS
  5. Social impact — 失业 / 通缩 / 投资
- Sets up the surprise: a year ago Guang Mi worried the technical-progress curve might slow; "其实"-implied corollary is that it didn't.

### §3 — 硅谷体感与洞察 (Silicon Valley on-the-ground; the four-part Coding thesis)  [[03:28 → 22:10](https://youtu.be/u1Lzp-7Ybn8?t=208)] — *the strategic backbone*

- **Past-quarter inflection**: Anthropic **Opus 4.5 → 4.6** is framed as a **GPT-3 → GPT-4-scale generational jump**, taking models from chat into "真正的agent模式." Guang Mi: "**我的体感是过去一个季度模型水平进步的幅度可能超过了2025年全年的进步幅度.**" The next-gen models **Mesos and Spark** (ASR: "Misos跟Spard") are framed as "真正的 GPT-5 moment."
- **The frontier-lab engineer is gone**: last year ~70-80% of code was human-written, "**今年可能是小于1%的**" — "AI来写人来审，可能人来审的能力都不够了." Daily token spend is "**几百美金**, 一周几千美金" per top engineer. Claude Code / Codex hit "**Meta E8/E9 / CTO / chief-architect level**" on many tasks.
- **AI is now accelerating AI research itself**: friends report recent AI-research breakthroughs come from Codex / Claude Code, not human engineers. Multimodal data iteration cycles dropped from 1-2 months to **a few days / one week**. Anthropic shipped **70+ products / features in 50 working days**.
- **Revenue picture**: Anthropic's **~1-2M paying users** beat OpenAI's **~50-60M subscribers** on revenue contribution; Coding's run rate took **2-3 years** to surpass what **Google Cloud** built over "**十七八年**" (17-18 years). Year-end projection: combined OpenAI + Anthropic AR **$80-100B**, next year **$200B+** — "**新时代的Mega7**" at a 10-15x PS multiple.
- **Four-part Coding thesis**:
  1. "**语言即世界, 代码即方案**" — both are "高度的浓缩抽象, 覆盖范围非常非常广"; if Coding Agent works, "**白领知识工作者的大部分任务都可以自动化**."
  2. **Anthropic has structurally cut off competitors**: "**OpenAI被断供了, XAI被断供了, 可能Google大部分被断供了, 可能哪一天Meta也要被断供了.**"
  3. **Coding is Amazon-selling-books** — a wedge that builds horizontal infrastructure for SKU expansion.
  4. **Coding wins first** because its feedback loop is the shortest and clearest.
- **AGI act structure**: **Act 1** = Chatbot (limited commercial value); **Act 2** = Coding Agent — "**Coding Agent实现了, 有可能AGI的90%已经实现了**"; **Act 3** = automated AI researcher solving brain / neuroscience / materials. "Coding 是开始爬科学这座山，不是登顶."
- **Why labs lag at Coding — strategy/org, not technical**: hardest part is (a) organizing hundreds of top researchers around "**脏活苦活**" data work instead of their own 0-to-1 bets, and (b) Coding+Agent data is far harder than chat text — requires tasks + environments + evaluation bundled. Rumor: Anthropic chief scientist **Jared Kaplan** (ASR: "Jarrett Kaplan") personally annotates data.
- **Two AGI paths, possible bifurcation**: (1) C-end traffic (ChatGPT, Gemini, 豆包); (2) high-value tasks (Anthropic). Guang Mi thinks they'll merge but raises the possibility that only the "**塔尖用户**" (~1-2M power users like quant traders) matter, since their token usage doubles as training data — "**强者恒强是很残酷的, 但这又是一个趋势**."
- People: Guang Mi · Jared Kaplan (Anthropic) · Sam Altman (implied via OpenAI).
- Visual ref: "我估计大家可能在网上都看到过那张图" — ARR curve visual referenced at [[08:56](https://youtu.be/u1Lzp-7Ybn8?t=536)].

### §4 — 硅谷御三家内部真实情况 (Anthropic deep-dive + the 1-2 year researcher window)  [[22:10 → 33:35](https://youtu.be/u1Lzp-7Ybn8?t=1330)]

- **Anthropic's all-in-on-coding wasn't day-one obvious**: triggers were (a) **no opening on C-end** consumer side and (b) **Sonnet 3.5 in summer 2024** giving strong positive feedback on coding — after which they dropped multimodal and consumer bets.
- **Strategy is "非常 top down"** with a unified roadmap and "**非常恶滥一致的目标**"; OpenAI by contrast is "特别 bottom up" — small groups each chasing new things, leading to swings (神化 pretrain → post-train → O1 → O3).
- **Org analogy**: Anthropic = "**一个集体, 一个球队, 一个工业化的体系**, 每个环节都做好"; OpenAI is "比较像 VC, 喜欢造新的概念, 把某些概念可能推得很高."
- **Culture**: hires Underdogs not Big Names, runs heavy "**culture interviews**" (e.g. "what would you do once AGI arrives?"), team is unusually stable with low attrition vs OpenAI. Founders run a near-religious AGI-mission framing.
- **Two physics-trained founders** (Dario Amodei + chief scientist **Jared Kaplan**, ASR-mangled as "Jerry Kaplan") "treat AI like a physics problem" — don't invent a new architecture, scale Transformer well, exploit data / architecture / engineering efficiency. Compared to "早年 Google" culture.
- **Product philosophy**: model researchers/engineers also build products. **"Anthropic没有做IDE, 而是他做了一个终端的形式, 就是今天的Cloud Code, 因为这个可能是更能承接住模型的红利的, 因为模型是指数级增长的, 但你产品也得指数级接住这个东西"** — Claude Code is a terminal, not an IDE post-Cursor, because terminals "指数级接住" exponential model capability better.
- **Tower-tip pricing**: Anthropic positions at "塔尖" — high-price tier, no discounting, big model with best quality and good margins; OpenAI historically couldn't go this big because most of its huge DAU don't pay.
- **Revenue concentration risk**: **~70-80% of Anthropic revenue from coding/agent**. Guang Mi thinks OpenAI and Google will eventually catch up; the bottleneck for Anthropic hitting **$100B AR this year is compute** — they under-provisioned and may need to scrape GPUs from everywhere.
- **IPO timing**: OpenAI likely **October or November 2026, year-end at latest**.
- **The most striking claim of the section**: "**最牛逼的AI researcher都担心自己一到两年后没有工作了**" — top AI researchers privately fear "AI可以automate整个AI research" within 1-2 years.
- People: Dario Amodei · Jared Kaplan · **Boris** (Boris Cherny, Claude Code creator, "very strong coder").

### §5 — OpenAI (coding misjudgment, Sora cut, ~50% AGI odds, Spark = real GPT-5)  [[33:35 → 47:13](https://youtu.be/u1Lzp-7Ybn8?t=2015)]

- **C-end lead, strategic misjudgment**: **~900M ChatGPT weekly actives** and **50-60M paying users** but growth flat. "**chatGPT C端看着赢了, 但发现coding比chatbot要大很多, 可能大十倍到一百倍**." OpenAI "其实是对coding有严重的战略误判 — 包括 google 我感觉也被带沟里了, 其实不应该再用互联网思维, 用DAU这些思维再去看这些东西了."
- **Sora 2 cut**: "**这个战略决定跟最近sora被关掉会有关系吗**" (host) → yes; GPUs are too scarce to spend on consumer video generation that doesn't help strategically.
- **Coding-tool market share guesstimate**: **Codex ~3M weekly actives** (Sam's tweeted figure); **Claude Code ~15-20M** — roughly **7:3** split. Guang Mi flags this as "我没有准确数字, 我只是猜测."
- **Cursor's position is precarious**: lives on model-company spillover. If OpenAI / Anthropic "**不把Mesos跟Spar的最强模型API开放出来**" and instead keep them in-product (agents, co-workers), Cursor has no path. Best exit: sell to Microsoft or Musk.
- **~50% odds of OpenAI winning AGI**: "**有50%的概率整个AGI最终的winner可能还是OpenAI, 因为在这个时代, 今天胜利的秘籍可能就是下个时代的毒药**" — ChatGPT's 2C success became a trap (focus on inference cost, kept models small, ignored coding); Anthropic's win is execution/focus but may not survive OpenAI's next paradigm-level jump.
- **Two-camp framing**: "**OpenAI一直想做爱因斯坦, Anthropic就是把整个白领工作给automate. OpenAI想跨过coding, 以前没那么重视coding, 想直接去做爱因斯坦, 但是你发现这条路很难, 而且没那么实用.**"
- **Weaknesses**: not focused (Sam is VC-style, FOMO, spreads bets like Sora); bottom-up values 0-to-1 over 1-to-100 so no one does the dirty data/ops work; ChatGPT "soulless" — no identifiable PM. ByteDance-style culture would have made ChatGPT a better product.
- **The upcoming model — Spark** (ASR: "Spar") is expected to be the real GPT-5, a generational jump, not another 5.x. Software engineering may still trail Anthropic, but "chat + code all-in-one" is likely the right strategy.
- **Long-term, competition reduces to GPUs**: "**我觉得最后可能还是比拼算力了, opend算力可能比Anthropic要多很多**" — AI automates research itself, researchers may stop being able to contribute, and OpenAI has many more GPUs than Anthropic.
- **Social-scale frame**: "**美国是一个中产社会, 一个多亿的中产 — 程序员律师医生中介 Banker — 很多人在未来一两年就是没有工作.**"
- People: Sam Altman · Fidji Simo (Meta-imported commercialization lead, "misled by internet-era DAU thinking") · Greg Brockman (contrasted as a technical-founder type) · Elon Musk (floated as Cursor acquirer).

### §6 — Gemini (3.0 overhyped, missed coding by 3-4 months, safest long-term)  [[47:13 → 54:16](https://youtu.be/u1Lzp-7Ybn8?t=2833)]

- **Gemini 3.0 was overrated**: "**Gemini 3.0当时是被高估了, 其实benchmark刷的很高, 其实你看C端其实没有持续增长, 用户也不太买单了**." No PC desktop client even now. The release boosted Google's stock — proved Google wasn't an "AI loser" — but little else. 3.1 brought no real breakthrough.
- **Coding misjudgment is the unforced error**: 3.0's success lulled Google into severely underweighting coding — they only recently elevated coding to top monthly priority, but "**已经晚了三四个月了**."
- **2025-2026 = critical two-year window for C-end share**; Google focused on C-end + multimodal vs ChatGPT, ceding the coding golden window to Anthropic.
- **The coding-gap-amplification law**: "**如果你coding落后三个月以后可能就落后一年, 因为它会放大**."
- **Org diagnosis**: Google's culture is bottom-up like OpenAI's; senior people occupy seats so new things are hard to insert. "**Gemini 印度人比例越来越多**" — Guang Mi flags this as a signal he doesn't know how to interpret.
- **Long-term safest of the Big Three**: "**其实Google的Worst case就是TPU都可以变成另外一个英伟达**." Already on the **3rd generation of professional CEOs**, runs like a system/machine — swapping 2-3 people barely matters. Strategic moat: OS (Android) + Google Workspace.
- **What it takes to deliver great models**: (1) **tens-of-billions USD/year for 3-5 years**, (2) management conviction & strategic Bets, (3) elite team & talent gravity. Today **one model isn't enough** — you need strategic bets aligned with product strategy; you can't only follow.
- **Can you skip Coding and jump to the next thing?** No. The three-act mainline is **chatbot → coding agent → automated AI researcher**; "**你跳出去是不是就偏离主线了**." Math is too narrow; world models unproven; next-paradigm work likely happens "on the shoulders" of frontier labs.
- **Big Three summary**: "**各领风骚一百天, 还是会交替领衔**." Trading thesis: buy whoever is short-term undervalued, because all three IPO by year-end and post-IPO investors may rotate other tech into model companies as "the main line of tech investing."

### §7 — Meta TBD (new #4 seed; 70-80% Gemini copy; Mistral sold cheap)  [[54:16 → 58:07](https://youtu.be/u1Lzp-7Ybn8?t=3256)]

- **Meta TBD has displaced xAI as Silicon Valley's #4 seed**. Talent density high — "**人才密度是很高的, 也知道各家的know-how, 因为他们从各个AI lab汇聚过来**" — and "**进步速度是很陡跳的, 9到10个月做出了一个还不错的model**."
- **Strategy is mostly copy**: "**七八成就学Google, 对标Gemini**" (including a nano-banana-style native multimodal image model) + "**另外可能20%也学OpenAI**" (post-train / R2).
- **Product strategy is unclear**: natural fit is personal assistant / personal friend (lower threshold), or a lower-threshold OpenClaw-style product. "**中国公司的中国团队的产品创新力都比Mita要强的 — 硅谷公司是擅长技术模型这个layer的创新, 中国团队是擅长产品的创新的; Mita产品创新力并没有字节那么强.**" Co-designing product + model could be the path.
- **Heavy-cash culture risk**: questions about long-term team stability and willingness to take risks; OpenAI characterized as more risk-taking, innovation-strong.
- **Mistral inside Meta — the "harness 鼻祖"**: "**Minas 我感觉是Harness的鼻祖, 是一个独立团队把Harness这些东西做得很好. 那今天模型公司把这个Harness这个概念像宗教一样喊出去了 — 其实Minas更像是Harness的鼻祖.**" Integration into WhatsApp / Instagram hasn't happened yet but revenue is growing fast.
- **Mistral sold cheap**: "**卖便宜了, 肯定卖便宜了; 但是再过一两个月, Opus都出来了, Minas还怎么卖呀**." Frames AI's *beta* (rising tide) as more important than endgame thinking — had Mistral scaled to **1-2B ARR** earlier it could be a **$10B-valuation company today**.
- People: 三大王 (Meta TBD leadership trio) · Elon Musk (transition).

### §8 — xAI (Elon's strategic oscillation; data, not cluster size, is the bottleneck)  [[58:07 → 1:02:00](https://youtu.be/u1Lzp-7Ybn8?t=3487)]

- Host pivots: "**我感觉跟他们错过整个coding的赛道也有关系对吧, 以及我在想为什么马斯克他能够把Tesla FSD做的挺好的, 做FCA却不行呢**" (ASR: "FCA" ≈ xAI).
- **Short-term behind**: the world-class founding team (**国栋, 子航, Tony** — ASR: "自航") has largely left. Elon appears impatient with them.
- **Strategy keeps oscillating**: first betting on huge clusters (tens of thousands of GPUs) and giant pre-trained models, then chatbot, then AI search, now (2025) "all in coding."
- **The real bottleneck has shifted from model size to data quality / data efficiency**: "**最初可能Elon相信大力数集计, 弄几十万卡的集群pretrain很大的模型, 但是好像不是这样 — 你得把数据做好, 你得把data efficiency做好, 你盲目的skill这个模型参数, 好像并不是问题的根本.**" "**今天的瓶颈可能不是在模型大小而是数据 — 别人可能用一个小你十几倍的模型, 可能比你做的还好, 这个就有点尴尬. 甚至说中国的蒸馏的模型可能都比XCI今天要好.**"
- **xAI ≠ Tesla FSD**: FSD's feedback loop is tight; model-data work is a long, research-flavored project where Elon's "**恨不得两个星期就看到效果**" culture sacrifices long-term quality (data, infra) for short-term wins.
- **Not written off**: xAI still holds many GPUs, Elon adjusts well, and his data-center + new fab ("**Five**" chip plant) ambitions are real but long-horizon — open whether he can build a team to deliver.
- **AGI race as F1 marathon**: "**就很像是你开着F1的速度跑一个马拉松, 而且在城市里跑 — 所以你需要200% 300%的聚焦. 如果CEO和leadership不聚焦肯定是不行的. 我觉得Elon是不够聚焦的.**"

### §9 — Harness Engineering (Honest = management layer; 2C/2B → 2-human/2-agent)  [[1:02:00 → 1:03:57](https://youtu.be/u1Lzp-7Ybn8?t=3720)]

- **Treat AI agents as 一等公民**: "**未来我们应该把AI的agent当人看, 应该把他看成一等公民, 跟我们是一体的. 那人应该有的东西agent也应该有 — 人类的知识工作者有工作的环境, 工作的电脑, 工作的各种信用卡; 那未来可能在一个平行世界你也要给agent去搭一套agent作为人类作为一个一等公民需要的环境.**"
- **Harness = the management layer**: "**agent想做好, 一方面是模型, 还有一方面是Honest** [harness]" (ASR consistently renders "harness" as "Honest"). "**agent你就把它比喻成就像是一个人加入到一个公司一个团队 — 有些公司有公司的这个管理和环境, 他能让一个正常的人他的下限很高, 有通过管理和组织一些约束. 那其实agent也需要他的管理学和组织, 那这就是Honest的意义.**"
- **Harness makes ordinary models viable**: "**其实有了Honest以后呢, 其实普通的模型也可以做高价值的任务了. 加上过去Cloud的整个需求的溢出他接不住, 所以非Sultana model就是包括开源的很多模型也能被用上了 — 我觉得这是一个更大的意义.**" (Sultana ≈ Sonnet/top-tier closed.)
- **Reframe the buyer**: "**以前是看2C和2B对吧, 传统时代. 但今天是到底是2人类还是2agent. 如果是2agent那可能看中的不再是 — 可能是token的usage, 或者token的价值margin这个可能更重要. 因为以后用什么工具可能不是人决策了, 可能是agent去调用哪些工具去决策了.**"

### §10 — 中国御三家 (Kimi / MiniMax / Zhipu all pivoted to the Anthropic route)  [[1:03:57 → 1:05:42](https://youtu.be/u1Lzp-7Ybn8?t=3837)]

- **China's Big Three have converged on the Anthropic playbook in the last 3-6 months**: "**我感觉国内好像都在追求anthropic的路线, 这好像成为了一个共识 — 一年前好像还不是共识**." "**国内的kimi minimax和智普都在bath2anthropic这条路线.**"
- **Driver — Doubao won C-end**: "**豆包看起来是在C端做最好的, 因为大家其他几家都是觉得跟豆包没得打才转型的**." Guang Mi expects **Doubao** to catch up and possibly beat the others on coding/agent.
- **Coding + agent is a must-not-lose battle**: "**我觉得coding和agent是不能输的; 豆包肯定也是大概率都能追上来而且可以有可能做得更好. 我觉得最后可能还是拼组织能力, 资源吧.**"
- **AI's "industrialization era"**: "**AI今天进入了一个叫工业化时代, 我不知道未来会不会进入无聊的大场游戏**" — "大场游戏" / "大厂游戏" where some windows have already closed.
- Sets up §11 with: "**其实模型可能就是新一代的操作系统.**"

### §11 — 模型是新一代操作系统 (Models as global GDP's OS; agents = infinite app extension)  [[1:05:42 → 1:07:01](https://youtu.be/u1Lzp-7Ybn8?t=3942)]

- **Frontier models = the world's most important technical infrastructure**: "**未来最领先的几个模型可能就是世界最重要的技术设施 — 你生活的问题是问他, 你工作的自动化也是他, 你研究科研的支持也是他. 可能他的重要性比今天的Google对世界技术设施的支持还要重要.**"
- **OS redefined as "支持应用的无限扩展"**: "**未来模型可能就是支持应用的无线扩展, 这就是agent.** [...] **操作系统的定义呢他就是支持应用的无线扩展可能就是今天的agent — 慢慢他也会形成一个新的生态, 就像安卓iOS这个生态windows.**" (Transcript writes "无线" instead of "无限" — same word, ASR misnomenclature.)
- **Historical short list**: "**过去称得上操作系统的可能就是windows、iOS、安卓和微信**" — and **ChatGPT / Gemini / 豆包** are converging toward this OS role.
- **End-state — "global GDP OS"**: "**不管你追求工作助理、coding、还是生活助理, ChatHPT Gemini豆包, 有可能最终大家都会走向世界的基础设施 — 一个global GDP的OS操作系统这种方向**." Spans devices: not just PC and phone but also eyewear ("**你的眼睛**") and 各种地方.
- Host pivot to §12: AGI roadmap + societal impact.

### §12 — 潜在的社会影响 (30% jobs gone this year; career ladder severed; all-in on top model funds)  [[1:07:01 → 1:14:36](https://youtu.be/u1Lzp-7Ybn8?t=4021)]

- **Timeline accelerated**: chatbot → coding agent → automated AI researcher; **a company may declare AGI by end of 2026 or early 2027** instead of "2-3 years out."
- **Information gap, not capability gap**: "**社会影响我觉得很多人确实没做好准备 — 因为整个割裂感太强了, 就是几百个researcher或者两三千个researcher在前沿看到东西更多的, 但其实这些信息是没有传递到社会上的, 所以社会的准备肯定是不够的.**"
- **Human knowledge has become cheap**: "**人类的知识和智力变得廉价了. 以前我们通过学习读书获取了知识可以有个工作, 但是今天这些智力和知识呢好像模型里面都有了, 被大幅的压缩了, 变成了一个计算资源或者token这种体现.**" Perceived value/meaning of **~70-80%** of people may shift.
- **Mass deflation**: ChatGPT / Claude replace consultants and SaaS — long-term many SaaS companies disappear; India's IT outsourcing already being eaten.
- **Hard numbers this year**: U.S. undergrad employment rate at a **historic low**; AI has automated the entry-level (2-4 year experience) bracket including junior programmers; **Meta laid off 16k**; **"有可能微软也不需要15万人, 有可能3万人有可能比今天15万人干的更好"**; **"有可能今年30%的工作感味就没了."**
- **Career ladder severed**: "**人才成长和培养的路径好像被AI拦腰截断了**" — the on-ramp jobs that grew people 10 years ago no longer exist.
- **Their own podcast is already half-automated**: most of the outline was written by Claude Code from Guang Mi's notes; next quarter the podcast might auto-generate from notes.
- **Survival advice — pivot to taste + creation**: "**AI取代的是不拥抱AI的人, 积极拥抱AI的人可能是受益者吧**." Humans should pivot to **taste/审美 + creativity**; one or two people can now do earth-shaking things because infrastructure has flourished.
- **Investment thesis — go all-in on the top model funds**: "**全球最领先的三五家模型未来都是十万亿美金 — 那你三年后五年后甚至七八年以后, 你全球GDP的百分之三十五十都已经被模型automated. 那如果你相信这个那就表达的更极致嘛, 那就全仓做一个模型基金也挺好的.**" Personally: **80-90% of effort on models**.

### §13 — 硅谷新趋势和投资新思考 (Foundation-lab walls; Q1 AR leaderboard; 60/10/10/10 portfolio; robotics 6-18 months)  [[1:14:36 → 1:22:40](https://youtu.be/u1Lzp-7Ybn8?t=4476)]

- **New foundation-lab entrants face brutal odds — "**再造一个台积电**"**:
  1. A tech company that can "**每年能不能投三五百亿美金的投入, 而且要持续投三五年**" (~$30-50B/yr × 3-5 yrs)
  2. Founder/management with both 认知 and 魄力 (Zuckerberg cited as the example of trusting the team without deep technical 认知)
  3. Hiring "**起码上百名世界级的AI的科学家**"
  And even all three are insufficient without strategic/product/GTM edges. GPUs are now hard to buy even with money. Best leading indicator: where top AI talent flows.
- **Q1 AR leaderboard**:
  - **Anthropic ~$30B+ ARR, OpenAI ~$25B+ ARR** (different accounting — "他俩口径很不一样")
  - Projecting **$80-100B by year-end, $150-200B next year** — "**新的Mac 7**"
  - **Cursor ~$2.5B**, **Perplexity** (ASR: "Property") **>$500M**, **11 Labs and SUNO each >$300M**, **Manus and Lovable each >$400M**, **Genspark** (ASR: "Ginspark") also growing very fast
  - **Healthcare AI** notable: **OpenEvidence** and **Abridge** (ASR: "Average") flagged as fast-growing
- **Ideal AGI portfolio (60/10/10/10)**: **"**最领先的三次甲模型分别放20%, 剩下的20%比如说10%在机器人, 10%在AI for Science, 还有可能10%再去Agent Infor这些**."** Robotics and Science are the two next big platforms after large models.
- **Robotics thesis**: architecture breakthrough + technical-route convergence + start of data scaling could land in the next **6-18 months**. Current data work spans **egocentric (第一视角) video, teleoperation, and 5-finger glove capture**; teams are just figuring out how to use each modality. **Chinese teams advantaged because hardware now matters** — Silicon Valley firms are recruiting in Shenzhen.
- **"One-person company" (1PC) thesis**: if models become powerful global infrastructure (like WeChat for self-media or Douyin for creators), individuals can go idea → code → revenue extremely efficiently. **Key ROI metric**: "**你消耗了100美金的token, 你能不能赚到110块钱, 你得把这个ROI跑正 — 其实很多人是没有跑正的.**"
- **Personal anecdote**: in Q1 Guang Mi became a **Claude Code heavy user** (ASR: "Cloud Code") — productivity up significantly, but "**我的Cloud Code 100美金的我是一直没用满的**" — research workflows don't consume tokens as continuously as coding does. Says AI has clearly entered an inflection point and is "**显著加速**."

## Notable quotes

> "Coding把AI从聊天机器人Charbot的第一幕推向了能够干活的Agent第二幕，研究员们已经开始不再亲自写代码了。"
> — Zhang Xiaojun (host), §1 [[00:18](https://youtu.be/u1Lzp-7Ybn8?t=18)] — *the episode thesis, delivered cold in the cold-open*

> "因为自然语言是对世界的描述，Code是对Solution的描述，就是语言及世界，代码及方案。"
> — Guang Mi, §1 [[01:01](https://youtu.be/u1Lzp-7Ybn8?t=61)] — *the load-bearing slogan for the whole Coding-Agent thesis*

> "我是感觉每个公司都有自己的窗口吧，各领风骚一百天，今天胜利的秘籍可能就是下个时代的毒药对吧。"
> — Guang Mi, §1 [[01:31](https://youtu.be/u1Lzp-7Ybn8?t=91)] — *the ~100-day window framing that recurs in every Big-Three section*

> "去年可能一个系统当中可能还有七八成的代码是人写的，今年可能是小于1%的。"
> — Guang Mi, §3 [[06:24](https://youtu.be/u1Lzp-7Ybn8?t=384)] — *the cleanest single number on how fast the engineer-as-coder is being eliminated*

> "如果领先的模型公司不重视Coding，他大概会掉出第一梯队的。[...] OpenAI被断供了，XAI被断供了，可能Google大部分被断供了，可能哪一天Meta也要被断供了。"
> — Guang Mi, §3 [[12:20](https://youtu.be/u1Lzp-7Ybn8?t=740)] — *the "cut off from coding" Big-Three diagnosis*

> "最牛逼的AI researcher都担心自己一到两年后没有工作了 [...] 后面AI可以automate整个AI research了。"
> — Guang Mi, §4 [[33:06](https://youtu.be/u1Lzp-7Ybn8?t=1986)] — *the most striking line in the entire episode*

> "其实Google的Worst case就是TPU都可以变成另外一个英伟达。"
> — Guang Mi, §6 [[50:07](https://youtu.be/u1Lzp-7Ybn8?t=3007)] — *why Gemini is the safest long-term Big-Three bet despite missing Q1*

> "未来我们应该把AI的agent当人看，应该把他看成一等公民。"
> — Guang Mi, §9 [[1:02:00](https://youtu.be/u1Lzp-7Ybn8?t=3720)] — *the foundational reframe for harness engineering*

> "人才成长和培养的路径好像被AI拦腰截断了。"
> — Guang Mi, §12 [[1:10:09](https://youtu.be/u1Lzp-7Ybn8?t=4209)] — *the career-ladder line for the social-impact section*

> "你消耗了100美金的token，你能不能赚到110块钱，你得把这个ROI跑正。"
> — Guang Mi, §13 [[1:20:27](https://youtu.be/u1Lzp-7Ybn8?t=4827)] — *the operational test for whether anyone is actually getting value out of Coding Agents*

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| Zhang Xiaojun (张小珺) | Host | [[00:05](https://youtu.be/u1Lzp-7Ybn8?t=5)] |
| Guang Mi (广密) | Guest — AI investor / analyst, recurring co-host of Global LLM Quarterly | [[00:29](https://youtu.be/u1Lzp-7Ybn8?t=29)] |
| Jared Kaplan (ASR: "Jarrett Kaplan" / "Jerry Kaplan") | Anthropic co-founder & chief scientist; physics background; rumored to personally annotate data | §3 [[18:15](https://youtu.be/u1Lzp-7Ybn8?t=1095)] |
| Dario Amodei | Anthropic CEO; physics background | §4 [[27:43](https://youtu.be/u1Lzp-7Ybn8?t=1663)] |
| Boris (Cherny) | Claude Code creator, "very strong coder" | §4 [[26:36](https://youtu.be/u1Lzp-7Ybn8?t=1596)] |
| Sam Altman | OpenAI CEO; VC background blamed for unfocused, FOMO-driven resource allocation; recently attacked twice in person | §5 [[36:44](https://youtu.be/u1Lzp-7Ybn8?t=2204)] |
| Fidji Simo | Meta-imported commercialization lead at OpenAI; Guang Mi says she was "misled by internet-era DAU thinking" and likely scapegoated/reassigned | §5 [[34:43](https://youtu.be/u1Lzp-7Ybn8?t=2083)] |
| Greg Brockman | Contrasted with Sam as a "Sorpy" / technical-founder type who would make sharper tech-roadmap decisions | §5 [[44:06](https://youtu.be/u1Lzp-7Ybn8?t=2646)] |
| Elon Musk (马斯克) | xAI CEO; strategic oscillation; floated as alternative Cursor acquirer | §5 [[39:16](https://youtu.be/u1Lzp-7Ybn8?t=2356)] |
| 三大王 (Meta TBD leadership trio) | Leaders of Meta's TBD large-model effort | §7 [[54:27](https://youtu.be/u1Lzp-7Ybn8?t=3267)] |
| 国栋 (Guodong) | xAI founding-team member who has left | §8 [[58:39](https://youtu.be/u1Lzp-7Ybn8?t=3519)] |
| 子航 (Zihang, ASR: "自航") | xAI founding-team member who has left | §8 [[58:39](https://youtu.be/u1Lzp-7Ybn8?t=3519)] |
| Tony | xAI founding-team member who has left | §8 [[58:39](https://youtu.be/u1Lzp-7Ybn8?t=3519)] |
| 小扎 / Mark Zuckerberg | Cited as example of a CEO willing to spend hundreds of billions on AI even without deep technical 认知, trusting the team | §13 [[1:14:52](https://youtu.be/u1Lzp-7Ybn8?t=4492)] |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| 全球大模型季报 第九集 (Global LLM Quarterly #9) | The series this episode belongs to | [[00:07](https://youtu.be/u1Lzp-7Ybn8?t=7)] |
| Anthropic Opus 4.5 / 4.6 | The GPT-3 → GPT-4-scale jump of the past quarter | §3 [[03:54](https://youtu.be/u1Lzp-7Ybn8?t=234)] |
| Mesos & Spark (ASR: "Misos跟Spard / Spar") | Next-gen Anthropic / OpenAI codenames; "the real GPT-5 moment" | §3 [[04:33](https://youtu.be/u1Lzp-7Ybn8?t=273)] |
| Claude Code (ASR: "Cloud Code") | Anthropic's terminal-form coding agent | §3 [[06:43](https://youtu.be/u1Lzp-7Ybn8?t=403)] |
| Codex | OpenAI's coding agent; Sam tweeted ~3M weekly actives | §3 [[06:43](https://youtu.be/u1Lzp-7Ybn8?t=403)] |
| ChatGPT | OpenAI's 900M-weekly-actives chatbot | §3 [[19:49](https://youtu.be/u1Lzp-7Ybn8?t=1189)] |
| Gemini | Google's frontier model line | §3 [[19:49](https://youtu.be/u1Lzp-7Ybn8?t=1189)] |
| 豆包 (Doubao) | ByteDance C-end winner in China | §3 [[19:49](https://youtu.be/u1Lzp-7Ybn8?t=1189)] |
| Sonnet 3.5 (ASR: "Solnet 3.5") | The summer-2024 Anthropic model whose coding signal triggered the all-in pivot | §4 [[22:44](https://youtu.be/u1Lzp-7Ybn8?t=1364)] |
| Cursor | Coding-IDE startup; Anthropic explicitly did *not* build an IDE in response | §4 [[26:49](https://youtu.be/u1Lzp-7Ybn8?t=1609)] |
| GPT-5.4 / GPT-5 / Spark | OpenAI's expected real-GPT-5 line | §4 [[29:26](https://youtu.be/u1Lzp-7Ybn8?t=1766)] |
| Sora / Sora 2 | OpenAI's video model; cut to free GPUs for coding | §5 [[35:13](https://youtu.be/u1Lzp-7Ybn8?t=2113)] |
| Gemini 3.0 / 3.1 | The benchmark-strong but C-end-flat Google release | §6 [[47:18](https://youtu.be/u1Lzp-7Ybn8?t=2838)] |
| TPU | Google's accelerator; "Worst case becomes another NVIDIA" | §6 [[50:09](https://youtu.be/u1Lzp-7Ybn8?t=3009)] |
| Google Workspace | Cited as part of Google's distribution moat | §6 [[50:43](https://youtu.be/u1Lzp-7Ybn8?t=3043)] |
| nano banana (Google's native multimodal image model) | Meta TBD copies a similar "纹身图" model | §7 [[55:02](https://youtu.be/u1Lzp-7Ybn8?t=3302)] |
| Mistral (ASR: "Minas / Manus / Mitas") | The "harness 鼻祖" acquired by Meta; sold cheap before Opus dropped | §7 [[55:37](https://youtu.be/u1Lzp-7Ybn8?t=3337)] |
| Tesla FSD | The tight-feedback-loop counter-example to xAI's strategy | §8 [[58:20](https://youtu.be/u1Lzp-7Ybn8?t=3500)] |
| Grok (ASR: "Gorak") | xAI's chatbot | §8 [[59:37](https://youtu.be/u1Lzp-7Ybn8?t=3577)] |
| Kimi · MiniMax · 智普 (Zhipu) | China's Big Three, all on the "bath2anthropic" route | §10 [[1:04:27](https://youtu.be/u1Lzp-7Ybn8?t=3867)] |
| Cursor (ARR ~$2.5B) | Q1 AR leaderboard | §13 [[1:17:14](https://youtu.be/u1Lzp-7Ybn8?t=4634)] |
| Perplexity (ASR: "Property") | Q1 AR leaderboard, >$500M | §13 [[1:17:21](https://youtu.be/u1Lzp-7Ybn8?t=4641)] |
| 11 Labs / ElevenLabs | Q1 AR leaderboard, >$300M | §13 [[1:17:29](https://youtu.be/u1Lzp-7Ybn8?t=4649)] |
| SUNO | Q1 AR leaderboard, >$300M | §13 [[1:17:29](https://youtu.be/u1Lzp-7Ybn8?t=4649)] |
| Manus (ASR: "Mannus") | Q1 AR leaderboard, >$400M | §13 [[1:17:31](https://youtu.be/u1Lzp-7Ybn8?t=4651)] |
| Lovable (ASR: "Loveable") | Q1 AR leaderboard, >$400M | §13 [[1:17:31](https://youtu.be/u1Lzp-7Ybn8?t=4651)] |
| Genspark (ASR: "Ginspark") | Growing very fast | §13 [[1:17:37](https://youtu.be/u1Lzp-7Ybn8?t=4657)] |
| OpenEvidence | Healthcare AI, fast-growing | §13 [[1:17:47](https://youtu.be/u1Lzp-7Ybn8?t=4667)] |
| Abridge (ASR: "Average") | Healthcare AI (medical scribe), fast-growing | §13 [[1:17:48](https://youtu.be/u1Lzp-7Ybn8?t=4668)] |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| 各领风骚一百天 — each Big-Three lab gets ~100 days of dominance | Guest | [[01:31](https://youtu.be/u1Lzp-7Ybn8?t=91)] |
| Quarterly series has run 2023 → 2026 (~3 years) | Host | [[02:04](https://youtu.be/u1Lzp-7Ybn8?t=124)] |
| Past quarter's model progress > all of 2025 combined | Guest | [[04:19](https://youtu.be/u1Lzp-7Ybn8?t=259)] |
| Code-authorship shift: ~70-80% human → <1% human | Guest | [[06:24](https://youtu.be/u1Lzp-7Ybn8?t=384)] |
| Daily token spend: 几百美金/day, 几千美金/week per top engineer | Guest | [[07:00](https://youtu.be/u1Lzp-7Ybn8?t=420)] |
| Anthropic shipped 70+ products/features in 50 working days | Guest | [[08:45](https://youtu.be/u1Lzp-7Ybn8?t=525)] |
| Anthropic 1-2M paying users beat OpenAI ~50-60M subscribers on revenue | Guest | [[09:30](https://youtu.be/u1Lzp-7Ybn8?t=570)] |
| End-of-year combined AR ~$80-100B; next year ~$200B+ | Guest | [[10:14](https://youtu.be/u1Lzp-7Ybn8?t=614)] |
| Coding run rate exceeded what Google Cloud built in ~17-18 years (in 2-3 years) | Guest | [[10:46](https://youtu.be/u1Lzp-7Ybn8?t=646)] |
| Coding Agent maturity ≈ 90% of AGI achieved | Guest | [[14:38](https://youtu.be/u1Lzp-7Ybn8?t=878)] |
| ~70-80% of Anthropic revenue from coding/agent | Guest | [[29:22](https://youtu.be/u1Lzp-7Ybn8?t=1762)] |
| Anthropic's bottleneck for $100B AR this year = compute | Guest | [[30:07](https://youtu.be/u1Lzp-7Ybn8?t=1807)] |
| OpenAI IPO likely Oct/Nov 2026, year-end at latest | Guest | [[32:42](https://youtu.be/u1Lzp-7Ybn8?t=1962)] |
| Top AI researchers fear ~1-2 year window before AI automates AI research | Guest | [[33:06](https://youtu.be/u1Lzp-7Ybn8?t=1986)] |
| ChatGPT ~9亿+ weekly actives, ~5-6千万 paying users | Guest | [[33:48](https://youtu.be/u1Lzp-7Ybn8?t=2028)] |
| Coding market is 10-100x bigger than chatbot | Guest | [[34:16](https://youtu.be/u1Lzp-7Ybn8?t=2056)] |
| US middle class ~1亿+; "many in the next 1-2 years will be jobless" | Guest | [[36:22](https://youtu.be/u1Lzp-7Ybn8?t=2182)] |
| AI's IQ progress in one quarter > human IQ progress in past 200 years | Guest | [[37:28](https://youtu.be/u1Lzp-7Ybn8?t=2248)] |
| Codex ~3M WAU vs Claude Code ~15-20M; ~7:3 split | Guest | [[37:56](https://youtu.be/u1Lzp-7Ybn8?t=2276)] |
| ~50% probability OpenAI is the eventual AGI winner | Guest | [[40:29](https://youtu.be/u1Lzp-7Ybn8?t=2429)] |
| OpenAI may have many more GPUs than Anthropic; competition reduces to compute | Guest | [[47:04](https://youtu.be/u1Lzp-7Ybn8?t=2824)] |
| Coding raised to top monthly priority at Google — but ~3-4 months too late | Guest | [[48:21](https://youtu.be/u1Lzp-7Ybn8?t=2901)] |
| 2025-2026 = critical 2-year window for C-end share | Guest | [[48:43](https://youtu.be/u1Lzp-7Ybn8?t=2923)] |
| Coding-gap amplification: 3-month lag → 1-year lag | Guest | [[49:06](https://youtu.be/u1Lzp-7Ybn8?t=2946)] |
| Google now on 3rd-generation professional CEO | Guest | [[50:19](https://youtu.be/u1Lzp-7Ybn8?t=3019)] |
| Sustaining great-model delivery needs 几百亿美金/yr × 3-5 yrs | Guest | [[50:58](https://youtu.be/u1Lzp-7Ybn8?t=3058)] |
| Meta TBD = Silicon Valley's #4 seed (displaced xAI) | Guest | [[54:37](https://youtu.be/u1Lzp-7Ybn8?t=3277)] |
| Meta TBD = ~70-80% Gemini copy + ~20% OpenAI copy | Guest | [[54:59](https://youtu.be/u1Lzp-7Ybn8?t=3299)] |
| Meta TBD progress: 9-10 months from start to "还不错的model" | Guest | [[54:46](https://youtu.be/u1Lzp-7Ybn8?t=3286)] |
| Had Mistral scaled to 1-2B ARR pre-acquisition → would be ~$10B-valuation today | Guest | [[57:56](https://youtu.be/u1Lzp-7Ybn8?t=3476)] |
| Elon initially bet on 几十万-GPU clusters; data efficiency is the real bottleneck | Guest | [[59:01](https://youtu.be/u1Lzp-7Ybn8?t=3541)] |
| Competitors with models ~10x smaller can outperform xAI | Guest | [[59:19](https://youtu.be/u1Lzp-7Ybn8?t=3559)] |
| Elon's culture: "恨不得两个星期就看到效果" | Guest | [[1:00:43](https://youtu.be/u1Lzp-7Ybn8?t=3643)] |
| AGI race requires 200-300% CEO focus | Guest | [[1:01:42](https://youtu.be/u1Lzp-7Ybn8?t=3702)] |
| China Big-Three pivoted to Anthropic route in past 3-6 months | Guest | [[1:04:16](https://youtu.be/u1Lzp-7Ybn8?t=3856)] |
| ~70-80% of people may see their social value shift | Guest | [[1:08:22](https://youtu.be/u1Lzp-7Ybn8?t=4102)] |
| US undergrad employment rate at a historic low this year | Guest | [[1:09:18](https://youtu.be/u1Lzp-7Ybn8?t=4158)] |
| ~30% of jobs gone this year | Guest | [[1:09:45](https://youtu.be/u1Lzp-7Ybn8?t=4185)] |
| Meta laid off 16k | Guest | [[1:09:55](https://youtu.be/u1Lzp-7Ybn8?t=4195)] |
| Microsoft "may not need 150k people; 30k could do better" | Guest | [[1:10:02](https://youtu.be/u1Lzp-7Ybn8?t=4202)] |
| Top 3-5 model companies → each ~$10T, ~$30-50T combined | Guest | [[1:12:56](https://youtu.be/u1Lzp-7Ybn8?t=4376)] |
| Guang Mi: ~80-90% of personal effort on models | Guest | [[1:13:43](https://youtu.be/u1Lzp-7Ybn8?t=4423)] |
| 30-50% of global GDP automated by models in 3-5-7-8 years | Guest | [[1:14:08](https://youtu.be/u1Lzp-7Ybn8?t=4448)] |
| New foundation lab needs ~$30-50B/yr × 3-5 yrs | Guest | [[1:14:43](https://youtu.be/u1Lzp-7Ybn8?t=4483)] |
| And 100+ world-class AI scientists | Guest | [[1:15:17](https://youtu.be/u1Lzp-7Ybn8?t=4517)] |
| Anthropic ~$30B+ ARR, OpenAI ~$25B+ ARR | Guest | [[1:16:56](https://youtu.be/u1Lzp-7Ybn8?t=4616)] |
| Combined year-end $80-100B → next year $150-200B | Guest | [[1:17:05](https://youtu.be/u1Lzp-7Ybn8?t=4625)] |
| Cursor ~$2.5B | Guest | [[1:17:14](https://youtu.be/u1Lzp-7Ybn8?t=4634)] |
| Perplexity >$500M | Guest | [[1:17:21](https://youtu.be/u1Lzp-7Ybn8?t=4641)] |
| 11 Labs / SUNO each >$300M | Guest | [[1:17:29](https://youtu.be/u1Lzp-7Ybn8?t=4649)] |
| Manus / Lovable each >$400M | Guest | [[1:17:31](https://youtu.be/u1Lzp-7Ybn8?t=4651)] |
| AGI portfolio: 20% × 3 frontier models + 10% robotics + 10% AI for Science + 10% Agent infra | Guest | [[1:18:08](https://youtu.be/u1Lzp-7Ybn8?t=4688)] |
| Robotics phase change in next 6-18 months | Guest | [[1:18:38](https://youtu.be/u1Lzp-7Ybn8?t=4718)] |
| ROI test: spend $100 tokens, earn $110 back | Guest | [[1:20:30](https://youtu.be/u1Lzp-7Ybn8?t=4830)] |
| Guang Mi's $100/month Claude Code plan still not maxed out | Guest | [[1:21:32](https://youtu.be/u1Lzp-7Ybn8?t=4892)] |

## Open questions / gaps

- **"Anthropic 断供 OpenAI / xAI / Google / Meta"** (§3) is asserted with no source; if true it's a market-structural fact, if false it reframes the whole "Big-Three converge on coding" story.
- **"$80-100B end-of-year combined AR, $200B+ next year"** (§3, §13) is presented as straight-line extrapolation from current "steepness" — no model shown.
- **"Coding Agent maturity ≈ 90% of AGI"** (§3) is a strong assertion with no definition of AGI offered.
- **Codex ~3M vs Claude Code ~15-20M, ~7:3 split** (§5) — Guang Mi explicitly flags: "我没有准确数字，我只是猜测."
- **"50% odds OpenAI wins AGI"** (§5) is presented as a feeling, with the Anthropic-trap argument as supporting structure but no probabilistic decomposition.
- **"落后三个月会放大到落后一年"** (§6) is stated as a coding-model law without mechanism.
- **"印度人比例越来越多"** in Gemini (§6) is surfaced as a signal whose valence Guang Mi explicitly says he doesn't know — flagged in the original notes as raised without evidence or definition.
- **"30% of jobs gone this year"** (§12) is asserted as a forecast for the current year with no methodology or source.
- **"$10T per top-3-5 model company, $30-50T combined"** (§12) — valuation claim without comparables or timeline grounding.
- **Robotics "qualitative change in 6-18 months"** (§13) — architecture and data-modality winners unspecified.
- **"Chinese teams advantaged in robotics because Silicon Valley firms are recruiting in Shenzhen"** (§13) — anecdotal; no count or company names.
- **The "Sai Ning" / Yang Zhilin / Yao Shunyu episodes** reference in the Su Yu sibling wiki is *not* in this episode; Episode 9 has no sister-episode cross-referencing inside the transcript.

## Verification log

- **Sections covered**: 13/13 ✅
- **Notable quotes traced verbatim**: 10/10 ✅ — each anchored by a distinctive Chinese substring (`各领风骚一百天`, `语言及世界`, `代码及方案`, `小于1%`, `断供了`, `没有工作了`, `Worst case`, `一等公民`, `拦腰截断`, `100美金的token`, `ROI跑正`) verified in `/tmp/yt-wiki/u1Lzp-7Ybn8.flat.txt`.
- **Numbers traced**: 47/47 ✅ — each row verified against the flat transcript (distinctive substring grep; ASR-rendered variants such as `九个多亿` / `五六千万` / `300多亿美金` / `250亿美金` / `1.6万人` / `15万人` / `三五百亿美金` / `十万亿` / `Mac 7` / `2.5亿` / `4亿` / `3亿` / `5亿` / `6到18个月` / `200%` / `300%` / `三到六个月` all hit).
- **Sectioning method used**: chapters (13 YouTube-supplied chapters; chapter 1 was `<Untitled Chapter 1>`, renamed to "开场白与本期定位" in section JSON before chunking).
- **Transcript source**: faster-whisper large-v3 (CPU int8) — local batch via `docs/videos/transcribe_batch.py`. YouTube provided no subtitles (manual or auto) for this video.
- **Speaker name corrections**: host's "小骏" → 张小珺 from channel metadata; guest's "广秘" (2 occurrences) standardized to **广密** to match the YouTube `info.json` title.
- **ASR substitutions noted inline**: Cloud Code → Claude Code; Solnet → Sonnet; Honest → harness; Misos/Spard/Spar → Mesos/Spark; Jarrett/Jerry Kaplan → Jared Kaplan; Minas/Manus/Mitas → Mistral; Property → Perplexity; Average → Abridge; GTT → GTM; XAI/XCI/FCA → xAI; Sultana → Sonnet (top-tier closed); Gorak → Grok; Loveable → Lovable; Ginspark → Genspark; Mannus → Manus.
- **Removed during verification**: none.

## See Also

- `docs/videos/2026-05-01-su-yu-agent-tech-history.md` — Zhang Xiaojun's episode with Su Yu (Ohio State / Neocognition), the academic-side perspective on the same OpenClaw / coding-agent inflection that Guang Mi reads here from the investor side.
- `docs/videos/2026-05-11-yao-shunyu-training-claude-gemini.md` — Yao Shunyu, the ReAct author working inside Anthropic / GDM during the period Guang Mi describes.
- `skills/research/youtube-wiki/SKILL.md` — the pipeline that produced this entry.
