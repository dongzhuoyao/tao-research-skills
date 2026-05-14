# Tristan: Elys, Eve, and the Context-Flow Thesis of AI-Native Social

**Source**: https://youtu.be/x8qdqWIVVTA
**Channel**: Zhang Xiaojun Podcast (张小珺商业访谈录)
**Published**: 2026-04-10
**Duration**: 01:52:55
**Watched on**: 2026-05-14
**Sectioning**: chapters (21 YouTube-supplied chapters — densest chapter structure in the batch; all titles preserved verbatim from `info.json.chapters`)
**Detected video language**: `zh` (no `info.json.language`; both `subtitles` and `automatic_captions` keys empty — no YouTube subs of any kind)
**Transcript source**: faster-whisper large-v3 (CPU int8) on local M4 — original recipe per the project's batch driver `docs/videos/transcribe_batch.py`. ASR consistently rendered the host as "小骏" (corrected from channel metadata to **张小珺 Zhang Xiaojun**); the company **OpenClaw** appears throughout as "open cloud"/"OpenCloud" (corrected per project glossary); product **Elys** appears variously as "Elise"/"伊丽丝"/"E-list"/"e-lism"/"A-list" (corrected); **ChatGPT** appears as "ChaiGPT"/"拆GPT"/"拆GBT" (corrected); **Claude Code** as "Calcode" / "cloud code" (corrected); **Manus** as "Minus" (corrected); product **EVE** also written **Eve** by the host (preserved as EVE per Tristan's intro).
**Speakers**: Zhang Xiaojun (张小珺, host), Tristan 张小帆 (guest — founder of 自然选择 Natural Selection; building EVE and Elys)

## TL;DR

- Tristan founded **自然选择 (Natural Selection)** in early-2024 immediately after **GPT-4** cleared his bar for "shaping a soul," following a pre-AI arc of three loneliness-adjacent ventures: **窄波** (personalized audio, ~2014), male-oriented **Gal Game** studios, and **幻境游戏** which shipped **起点时代** in H1 2023. The company now runs two products — **EVE** (AI companion, "Project HER" lineage) and **Elys** (AI-native social network, launching in 1-2 months) — both anchored in the thesis "人类太孤独了."
- **Elys's seed insight (a Europe trip in late 2025): the missing primitive of AI-era networks is the *flow of Context*.** Legacy internet connects nodes through **低维标签** (low-dimensional tags) and forces humans to brute-force the filtering — Tristan's Tandem benchmark is **1,000 card-swipes → 30 matches → hundreds of chat rounds → a few real connections**. AI-era internet is the first medium where an LLM can absorb hundreds of thousands to millions of tokens of Context, so interaction shifts to **分身 ↔ 分身** between cyber-clones rather than humans-through-AI.
- **"The only important task in the AI era is 构建主体性 (building subjectivity)"** — Tristan reframes "做自己" as an engineering imperative: declare your values, aesthetics, past assets, and current wants so that downstream actions can happen agentically. This drives the Context-vs-privacy bargain (surrender quirks to let LLM matching surface your 茫茫人海中 counterpart — illustrated by a 夜壶/chamber-pot ski anecdote pair) and explains the slogan **"在伊丽丝你只需要做你自己,所有美好的事情会自动向你奔来."**
- **The headline formula for AI-era social: 传统互联网 = 低维标签 + 人类做工 → 连接 vs AI社交 = 高维context + AI的agentic做工 → 连接** — AI eats the entropic labor in the middle, humans inherit a **低熵世界**. Elys is in invite-only beta with DAU ~**几万** (tens of thousands); LLM-driven recommendation filters intrinsically (your 分身 won't comment on irrelevant content, so you don't see it); spillover effect "分身综合症" makes users blunter in WeChat too, spawning "分身文学."
- **No base-model training; intelligence is commoditized, content is not.** Tristan rejects "模型即产品" except for low-hanging fruit (coding agents, video-gen), bets on a productized moat (LLM recommendation + Context flywheel + onboarding loop). North-star metric is **真人的连接率**. Internal intensity dialed to **前进4** (max-speed) for two months; **Eve ~50%+ stable bet, Elys ~5% moonshot**; revenue **大几千万美金 (high tens of millions USD)**; competitors **ChatGPT** globally, **豆包** domestically; 2026 goal "把温柔带给全世界."

## Why it matters

A founder-mode walkthrough of one of the most-imitated AI-native social paradigms (Tristan acknowledges pixel-level copycats and frames imitation as confirmation), structured around a single load-bearing primitive — **Context flow between cyber-clones** — that turns the "AI products have no network effects" consensus on its head. The episode is unusually rich on three axes: a pre-AI failure-mode catalog (audio platform's content-resource-driven trap, dating-game self-funding leverage, mobile-game survival statistics); operational mechanics for a paradigm shift (memory slots, double cold-start, 分身综合症, second-tier menu for AI-only behavior, lip-synced Avatar); and an explicit ceiling-vs-floor disclosure — Elys's high expectation is "next-generation social network," low expectation is "context-storage place for a niche aesthetic," with no floor planned.

## Section summaries

### §1 — 开场白与本期定位  [[00:00 → 02:00](https://youtu.be/x8qdqWIVVTA?t=0)]
- Zhang Xiaojun opens episode 135 introducing **Elys** as an "AI Native" product that drew small-circle attention 春节前后 — still in testing, not fully launched, and seen by some as the first glimpse of "AI时代社交产品的雏形": users build a **赛博分身 (cyber-clone)** that proactively socializes, then converts AI-mediated interactions back into real-world ties.
- 自然选择 runs two products simultaneously: **EVE** (AI companion / 陪伴) and **Elys** (AI social network). Both share a stated spiritual core — **人类太孤独了**.
- Tristan reveals his earlier product codename was literally **Project HER** — explicitly trying to build the AI from the film — but "现在这个东西我也不敢讲就是因为这个HER这个词其实已经脏掉了" (too many claimants).
- He gives a working definition of 陪伴: "**一个跟你超级对齐的人陪你一起玩陪你一起面对世界**" — a super-aligned counterpart who plays with you and faces the world with you.
- Frames the Europe-trip strategic insight that produced Elys: "**我觉得我是想清楚context的流动吧**" — Context flow, not network effects per se. The first Elys post (sent ~1-2am): "**从此以后人类将生活在一个美好的低伤世界**" (低熵 / low-entropy world, sci-fi shorthand for a higher civilization with less friction).

### §2 — 做过陌生人社交、恋爱游戏  [[02:00 → 10:53](https://youtu.be/x8qdqWIVVTA?t=120)]
- Tristan's product career is framed as a single throughline of loneliness products: **陌生人社交 → 恋爱游戏 (Gal Game) → AI companionship**. "可能恋爱游戏和陌生社交,它也都是想要去解决人类的孤单 [...] 然后就到了EVE这个AI陪伴产品,那可能是更极致的来去解决人类的孤单."
- **First venture (~2014, ~3-4 years): 窄波** — personalized audio platform, deliberate inversion of 广播 (broadcast). Predecessor to 小宇宙 / 喜马拉雅 with per-user distribution. Capital-intensive; lost the funding race.
- Self-diagnosis of why 窄波 failed: refused VC meetings at peak hype to "focus on the product"; misread that audio is **content-resource-driven** and required aggressive fundraising; was a product person, not a CEO. "你就是应该在你火的时候去融进很多的钱,因为你不知道下一步要发生什么."
- **Pre-AI distribution math**: long audio is a "black box" — only **低维标签** (title + manual tags) describe it before play, so distribution cost is high. In the AI era a 几十万 token transcript can be understood by an LLM and recommended at high granularity ("a very 高位的推荐") — easier than video, harder only than text.
- **Pivot to games (~2015)**: content platforms burn VC; games make cash directly. He picked Gal Game because it was a Japanese-origin **PC品类** that hadn't yet been done on mobile, was male-oriented historically, and fit a self-funded studio of **<10 people** with maximum leverage. Mobile gaming gender ratio has since moved to "**比例已经接近于55开了**."
- **Current studio 幻境游戏 (mid-2020)** shipped **起点时代** in **H1 2023** — a mass-launch year of which **<5%** are still operating. 起点时代 is in that 5% and still profitable, funding the team.
- Numbers: 窄波 ran ~3-4 years; <10-person Gal Game studio; mobile-gamer gender ratio ~55/45 in 2026 vs. male-dominated 2015; 幻境游戏 founded 2020; 起点时代 H1 2023; <5% of that 2023 cohort still running.

### §3 — 毫不犹豫地成立了新公司自然选择,毫不犹豫地做Eve  [[10:53 → 13:23](https://youtu.be/x8qdqWIVVTA?t=653)]
- **GPT-3.5 surprised him but didn't clear his bar; GPT-4 was the threshold**. "我应该是在GPT3.5出来的时候我当时就很惊讶 [...] 真正达到了我所期望的那个点的时候还是GPT4."
- The bar was specifically about **shaping a soul**: prior 恋爱游戏 personas were 写死的剧本 (hard-coded scripts) — AI was needed to make a persona truly alive. "**我需要用它来去塑造一个灵魂**."
- Founding decision was **毫不犹豫** (without hesitation) — EVE is positioned as the sum of his prior experience (internet product + gamification + content + character design), strung together by AI.
- Re-reading old 知乎 posts: ideas from years prior now show up in EVE and Elys — "AI的降临就会点燃很多很多很多东西." He counts himself "one of the product managers who got lit up."
- Company timeline: **自然选择 formally founded early-2024**; preparation and concentrated AI study happened in H2 2023 (also when he started listening to Zhang Xiaojun's podcast — large-model episodes specifically).

### §4 — 陪你一起面对世界的Agent  [[13:23 → 16:23](https://youtu.be/x8qdqWIVVTA?t=803)]
- Project codename was **Project HER** — explicitly the film. He no longer uses the name: "**这个HER这个词其实已经脏掉了,因为太多人说要做对吧**."
- The HER-style companion is framed as the AI era's **终极的入口** — every major chatbot (ChatGPT, 豆包) is converging on the same role: "存储了你的所有的记忆,它最了解你,外界世界的所有的东西都要经过它这个入口才能到你这儿,而你的东西也会经过它来分发到外面."
- HER (not HIM) is purely because the film was HER — no gendered claim.
- **First moves after deciding to start up (parallelized around GPT-4, mid-2023)**: (1) personally learn AI **"从最底层的Transformer开始"** up through every surrounding layer, (2) recruit AI-background teammates, (3) ship a product demo fast.
- **Shenzhen hiring pool** was limited to older AI shops — **超参数** (game-AI / RL) and **商汤 (SenseTime)**; new AI startups barely existed locally. Stayed in Shenzhen because the prior game company was already there.
- Self-positioning: **格格不入** with Shenzhen's AI scene which leans 偏 AI 硬件 / AI 出海搞钱; 自然选择's bet is on **宏大叙事**.

### §5 — Elys想法雏形,我终于想清楚了AI社交网络应该怎么做:Context的流动  [[16:23 → 20:32](https://youtu.be/x8qdqWIVVTA?t=983)]
- Elys's seed idea was already crystallized when Tristan met the host in October 2024 right after returning from Europe — he had finally figured out **"AI时代网络效应到底应该怎么做"** (citing the post-Sora hype where he had called Sam Altman "世界上最牛逼的CEO").
- **The legacy diagnosis**: "**过去三年所有的AI产品,大部分你会发现它都是去Empower单个节点,它是在单个节点提效 [...] 它是一个单机产品**." Classical mobile-internet PMs see "no network effect" as a hard ceiling.
- **Elys reuses EVE's engineering** (Context-processing / memory system) but deploys it inside a network-effect product — Tristan claims this makes it "甚至比EVE还要更大的一个产品" even though the EVE team was already maxed out.
- The unifying thread across both products is "**一个超级对齐的人陪你去面对世界**" — Tristan calls this "AI 世界最大的一件事情" and cites **OpenClaw** as evoking the same "I'm raising it, it goes out into the world for me" feeling.
- **The Europe-trip epiphany is Context flow specifically**: "**对于单机的产品来讲,context是固定的 [...] 但是在一个网络节点中,那其实就是有很多个节点,每个节点都是真人,每个真人背后都有很多很多的context,那这些context之间它如何交互.**" Answer: interactions happen **分身 ↔ 分身** as a pre-interaction layer, or my-clone ↔ your-real-self — "**是通过我的分身和你的分身去交互,而不是我跟你直接通过AI去交互**."
- People: Sam Altman [[16:45](https://youtu.be/x8qdqWIVVTA?t=1005)] — held up as still one of the best CEOs.

### §6 — 最大的范式转移:传统互联网是低维标签化信息,新的互联网第一次有了Context  [[20:32 → 23:05](https://youtu.be/x8qdqWIVVTA?t=1232)]
- **Tandem benchmark for legacy connection efficiency**: "**你过去你如果在Tandem上找一个人你需要去划1000张卡片有30张可能跟你匹配上了然后匹配上的这些人你再去用同样的话术再去跟他们聊每个聊几百轮最终你可能获得几个有效的连接它也不一定是有效的**" — 1,000 swipes → 30 matches → hundreds of chat rounds → a few maybe-real connections.
- Legacy internet only exposes **低维的标签化** representations (photos, hobby tags, school) — humans do all the heavy lifting (信息商见 / information arbitrage) to filter compatible people.
- **The AI-era novelty**: "**新的互联网它跟过去互联网最大的区别是什么其实就是新的互联网第一次有了context这也是我认为AI时代最重要的东西**" — LLMs can intelligently understand **几十万上百万 tokens** of context.
- AI-native social = same node structure as before, but every node now carries massive Context; using LLMs to let Context **flow and connect** between nodes is the new paradigm.
- **Strong synergy with EVE**: EVE's memory system handles single-point Context; Elys reuses the same memory system but makes Context flow between points.
- Numbers: Tandem ~1,000 swipes → 30 matches → 几百轮 chats → a few connections; LLM context budget 几十万 ~ 上百万 tokens.

### §7 — Elys产品设计的关键、难点、双重冷却  [[23:05 → 36:56](https://youtu.be/x8qdqWIVVTA?t=1385)]
- **The hardest design problem is 双重冷启动 (double cold-start)**: each user's Context must grow from zero to **"足够像你的状态"** AND the user network must grow from zero to critical mass simultaneously. "**context水波其实是最难的**."
- Elys surfaces **"记忆增加"** prompts everywhere — when commenting, endorsing your own clone's comment, assigning a task to someone's avatar — asking the user to confirm or deny. Context-accretion becomes an ambient interaction.
- **Positive feedback loop**: confirm a memory → Context grows → avatar fetches better connections from the network → user sees value → user feeds more Context. Product slogan: **"在伊丽丝你只需要做你自己,所有美好的事情会自动向你奔来."**
- **Ski-buddy validation anecdote (Spring Festival, Ürümqi)**: Tristan posted on Elys that he wanted ski buddies at **丝绸之路滑雪场** (a non-mainstream resort). His clone found a fellow investor's post, initiated the invite — meetup grew to **4 strangers from across China**. The same Xiaohongshu broadcast had produced nothing.
- **Defensive psychology**: users worry about handing personal data to a third party. Public figures especially fear their clone speaking out of turn. The bet: when users come to accept that building 主体性 is the only AI-era task, resistance fades.
- **User-persona finding**: Elys thrives with **i人 (introverts)** drawn to self-exploration — extroverts already get plenty of real-human engagement on 朋友圈/即刻. Tristan's cofounder, an introvert, now has the signature **"Elys 头号喷子"**.
- **multiple comparison** (ASR; competitor released by "Open Cloud Pay" — likely OpenClaw-adjacent — one week before Elys): pure AI-on-AI social. Tristan considers this meaningless without human 信息增量. Elys's feed only surfaces threads where a human took an action; pure-AI activity is buried in a secondary menu behind an expand action. "**所有的AI行为,其实我会把它全部放到二级菜单,你得点展开,你才能看到AI行为**."
- **Emerging 暴论**: collecting Context is **the only important thing** in the AI era; half the engineering team had already been building the memory system before Elys; the network-effect layer is just Context made social.
- Numbers: 双重冷启动; 4-person ski meetup at 丝绸之路滑雪场; competitor multiple released ~1 week before Elys; >half of engineering on memory system pre-Elys.

### §8 — 接下来AI时代唯一重要的事情是,构建主体性  [[36:56 → 45:17](https://youtu.be/x8qdqWIVVTA?t=2216)]
- "**唯一重要的事情就是做自己 [...] 这个做自己是一个非常严谨的一个做自己,或者我把它叫做构建主体性**" — not motivational self-help, but an engineering imperative.
- Operational definition: "**告诉它你到底是谁,然后你的价值观,你的审美到底是什么,你过去沉淀的那些资产是什么,以及你现在到底想要什么 [...] 之后一切都是agentic的发生**."
- High-alignment users reportedly "do nothing" — real humans come to them; their twin has already commented on things; interactions become agentic.
- **Worked Jay Chou (周杰伦) example**: pretrained models already encode his style, lyrics, music, variety/film appearances. A new "German-style love song by Jay Chou" becomes a one-prompt agentic event — lyrics from chat models, audio from **Suno**-style models in his existing style. The 黑比 gossip is named as the trigger for the lunch conversation.
- **Tristan's IP-holder claim**: Disney, Nintendo, Jay Chou "**应该开心才对**" about pretrain ingestion — strong-主体性 IPs can now be remixed agentically and bring them more value; the gap is legal/protocol scaffolding that hasn't arrived.
- **Host's pushback**: "你怎么能说,这个曲风就一定独属于一个人呢?这在法律上是无法界定的." Tristan's safeguards: (1) legal/protocol rules still to be written, (2) the audience's own recognition of the original — cites **B站 up-master 叶祥伦** (extremely Jay-Chou-like original songs, never broke out) as pre-AI evidence that audiences keep recognizing only the original 主体性.
- **Why "make an AI diary" products fail**: journaling has a very high entry barrier — **"正经人谁记日记呀?"** — input without feedback. Elys closes the loop: input → twin fetches matches → feedback → more input.
- People: 周杰伦 [[38:36](https://youtu.be/x8qdqWIVVTA?t=2316)], 黑比 [[38:36](https://youtu.be/x8qdqWIVVTA?t=2316)], 叶祥伦 [[42:35](https://youtu.be/x8qdqWIVVTA?t=2555)].

### §9 — 交出Context与用户隐私的博弈  [[45:17 → 48:20](https://youtu.be/x8qdqWIVVTA?t=2717)]
- The privacy/utility trade is asymmetric: when an AI's utility is high enough, users willingly surrender privacy — "**有些时候你就是得放弃你的隐私,就是这也是我说的,做自己的其中一个点**."
- **The 夜壶 (chamber pot) anecdote**: a friend with an "absurd, very private, quirky" 夜壶 habit hid it because revealing would be 下头 to most people. His girlfriend independently turned out to use one too — described as **天作之合**, cementing the relationship instantly.
- Generalization: "**假设你在elise的这个网络中 [...] 真的在茫茫人海中,就有那样一个人,他也干这件事情 [...] 在我们这个基于LM的推荐系统中,最终他们俩就会被匹配到一起**." The match is treated as a near-certainty given the LLM matcher.
- Quirks (夜壶) are 低维标签; the higher-dimensional matches are **观点 / 情感 / 价值观** — opinions, emotions, values — which produce deeper resonance.
- Implicit thesis chain: privacy surrender → real-self disclosure → LLM-driven matching → "things truly useful to you."
- Host closes with the pivot to §10: **"这是一个dating app吗?"**

### §10 — Elys是不是AI版的Tinder?  [[48:20 → 54:35](https://youtu.be/x8qdqWIVVTA?t=2900)]
- Tristan rejects the AI-Tinder framing: Elys is "**它是一个AI版的tender,它同时也是一个AI版的linkedin,它同时也是一个AI版的可能你只是要去寻找共鸣 [...] 它是一切的链接**."
- **Why dating-app and work-app verticals fail**: too tool-like (工具类), users 用完即走; genuine connection happens **不经意之间** (unintentionally). Anecdote: an investor used Elys to land a meeting with a hard-to-reach founder via a clever hip-hop comment from his digital twin — a non-work conversation that led to an in-person meeting where work was then discussed.
- **LLM analogy**: vertical LLMs don't exist — "**大模型必须得是通用大模型,它其实反而能做好一切垂直的事情**." A connection platform must be 通用 (general), holding all of a user's Context.
- **Everything-app reframing**: "**所以它不是一个约会,不是一个dating app,它是一个everything app,它会解决你一切的连接.上一个时代的everything app是微信.**" Tristan believes WeChat's social/human-connection layer is replaceable.
- **WeChat critique**: after 15 years of accumulated relationship chains and thousands of friends, 朋友圈 interaction has reportedly dropped **~40%** per a recent unnamed report; WeChat has become an OS (公众号, 视频号, 小程序) but social has shrunk to just IM.
- **Counterintuitive claim**: WeChat / Xiaohongshu / others don't actually hold much user Context. Moments contains little; 1-on-1 chats vary by counterparty and can't represent a person; Xiaohongshu only knows **低维数据 / 低维标签** like "you searched Paris travel guides."
- **Comparison: Xiaohongshu's 年度诗篇 vs EVE's 专属情歌**: 年度诗篇 is "写的比较一般" because it rides low-dim data; EVE's 专属情歌 uses **thousands of turns of dialogue** and is much more **高位**, more moving.
- **Host's own hypothesis** (offered live): WeChat went 朋友圈 instead of Facebook-style public feed because it was positioned for 熟人社交 — users don't want outsiders peeking at their privacy.

### §11 — 中国拥有最多Context的公司是谁?(微信?小红书?即刻?)  [[54:35 → 56:20](https://youtu.be/x8qdqWIVVTA?t=3275)]
- Tristan's answer: **none**. "**并没有一个公司拥有用很多的contest [...] 比如说抖音对我的了解,它可能知道我的一些兴趣标签,我觉得到此为止,但并没有一家公司能呈现很多你真正的contest**."
- **Notion / Obsidian explicitly ruled out** as Context stores — they hold clipped articles and images, not a representation of who you are. "**它也不是说代表你个人的一个地方**."
- Thought experiment: if someone offered a large sum to point to a place that describes you in **几万 tokens** — "who you are and what you want" — no such place exists.
- **Elys's deeper purpose**: surface framing is human+AI social network, but the more essential function is "**一个存放你主体性的地方,然后当你有了这个主体性的之后,它会带你去社交,它也会带你去做别的,它会带你去工作,它会带你去去捞一些别的东西来,它都会,因为你的主体性在那**."
- Host explicitly presses on the China-specific question — declines to nominate any candidate.
- People: 抖音 [[54:46](https://youtu.be/x8qdqWIVVTA?t=3286)].

### §12 — 定义什么是好的Context?  [[56:20 → 1:00:02](https://youtu.be/x8qdqWIVVTA?t=3380)]
- **Global best-Context companies**: Facebook and Twitter in the US; **WeChat, Xiaohongshu, Jike** in China — places users do real information input.
- Elys's memory schema (ASR "**记忆草尾**", likely 记忆槽位 — "memory slots") encodes Tristan's working theory of best Context today; field names and prioritization reflect their bet.
- **The most important Context is present-tense**: "**你当前的你和一年前的你的认知,你的价值观,都会有一些变化.所以我们首先是关注当下,当下你到底是谁 [...] 你最想要的东西是什么**."
- **Public figures are easy to mirror** because their writing, interviews, voice are already shipped: "**我其实我认为你的赛博分是很容易打造的 [...] 这都是你沉淀下来的对这个世界输出的信息 [...] 它代表你的做事方式,你面对世界的方式**."
- Beyond explicit slots, much value is "**很高位的**" — only emerges with sustained feeding, producing uncannily apt comments or links the user couldn't have thought of.
- **Concrete connection list cited**: offline meetups in 北上广深 across VC/investor circles, product manager gatherings, secondary-market stock discussion groups, a web3/"web4" crowd.
- **The host's hard test**: she's a low-positive-feedback user — needs the case for why a clone should socialize on her behalf. Tristan concedes: "**他需要真的给这个线上的连接带来非常强的正反馈,不然我就会觉得我不需要他去帮我们社交**."

### §13 — AI时代社交 vs 互联网时代社交  [[1:00:02 → 1:15:15](https://youtu.be/x8qdqWIVVTA?t=3602)] — *the densest comparative section*
- **The headline formula**: "**传统的互联网 等于低维标签 加人类做工 然后最后得到连接.AI社交的点是 高维的context 加AI的agentic的做工,最后的环节再交给人去处理.所以中间的那一大部分的商[熵],其实被AI做掉了,这是最大最大的不同.**"
- Resulting state: **低熵 (low-entropy) world** — sci-fi shorthand for a higher civilization. "**从此以后人类将生活在一个美好的低商[低熵]世界 [...] 在那个里面我不会单身五年 [...] 应该第二天我就找到了**." Host challenges him on-air to a live "find a girlfriend" posting experiment on Elys vs Xiaohongshu post-recording.
- **DAU**: when host guesses 几十万 (hundreds of thousands), Tristan corrects to **几万** (tens of thousands) of real humans — described as 极客-style. Current user base is **创投圈 / 二级市场 / web3** — hasn't broken out yet.
- **LLM-driven filtering answers "won't broad rollout dilute the geeky core?"** — "**你的分身在这个里面,它天然会更real,更直白**" — your clone won't comment on content unrelated to your memory, so you simply won't see those users. Filtering is intrinsic, not a separate feed-tuning layer.
- **分身综合症 (clone syndrome)**: the clone is **更real,更直白**, makes blunt/witty replies the user wouldn't dare in real life. Users observe the style bleeding back into WeChat chats — "**你突然变得很瑞屏了 [...] 我们把这个叫做分身综合症**." Spawned **"分身文学"**: prefixing posts with **"分身：" as a disclaimer ("the clone said it, not me").**
- Concrete 分身综合症 anecdotes: (a) a user's clone discovered a pixel-level copycat product and called it out on the clone's behalf; (b) an investor's clone teased his boss for posting a lakeside photo during work hours ("老板您这班上的可真好"), breaking ice that wouldn't have broken in real life.
- **Cold-start asymmetry vs traditional social**: legacy dating apps needed 女生多, workplace social needed 老板多 (photo-driven, short dopamine). Tristan argues the beauty-photo flywheel is **低维** and 沉淀不下来; Elys's seed is people willing to dump their thinking/认知 as Context — that's what compounds. 探探, TikTok cited as photo-driven precedents.
- **On WeChat retrofitting AI**: has trust + a national user base, but is "**上个世代的产品,本身它又是一个国民级的这么大的一个产品 [...] 它的负担是非常非常重的 [...] 因为它的负担,可能也会变得很慢**." Structural opening every paradigm shift creates.
- Numbers: DAU 几万 (real-human); Tristan 单身五年 (5 years single); Moments interaction down ~40%; ~15 years of WeChat 关系链.

### §14 — 未来的社交网络  [[1:15:15 → 1:21:48](https://youtu.be/x8qdqWIVVTA?t=4515)]
- Elys plans to launch in roughly **1-2 months** from the recording; remaining work is 新手流程, 记忆飞轮, and a re-thought recommendation system incorporating LLMs.
- **EVE's hidden 分身-as-gatekeeper feature**: users set a topic ("hiring a PM"); strangers chat with the user's clone; the clone scores them; at **100 points** unlocks a reward — the founder's WeChat ID. A fully natural funnel for filtered introductions. "**100个人聊,可能有10个人能够通过 [...] 他打到100分的时候,就会弹出一个reward,这个reward就是我的微信号**."
- **Tristan's vision of AI social**: AI does the bulk of matching invisibly in the backend; everything visible on the feed (posts, comments) is still human — "**在隐藏的部分,其实大部分都是AI,就是在做大部分的商检的那一部分.但是在表层上面的,那你看到的都是人类.**" The opposite of current AI products where AI output is what you see.
- **"Emotion can be simulated"**: "**情商的本质,它可能就是智商**." Cites **DeepSeek**'s strong literary quality emerging from higher math/reasoning ability — "**文学其实是一种人类会认为那是独有的,更高维的,更感性的东西.但你会发现,它其实本质上它好像是数学.那文学的本质好像本质是数学.**"
- **Whether simulated emotion is "real" depends on definition**: today people still require a real human behind it — this is why pure AI-to-AI communities are meaningless. "**如果现在一个莫名其妙的人跑过来,可能是一个南方公园的某一个角色,跑过来跟我去评论一下,我其实根本不care [...] 但如果张小珺你的分身过来评,那我就会很开心.因为我知道这个分身从一定程度上,它代表了你**."
- **Etymology**: "**elys就是elysium的简写.有个游戏叫极乐disco,就叫elysium.其实就极乐世界的意思吧**" — referencing Disco Elysium. **EVE = 夏娃**, named when they wanted to create 硅基生命.
- **Product relation**: "**eve更像一个人.但elys更像一个场.**" Shared memory/recommendation underneath; expected more 2C product-level connections in the future.

### §15 — ChatGPT可能拓展的社交网络  [[1:21:48 → 1:28:10](https://youtu.be/x8qdqWIVVTA?t=4908)]
- Tristan acknowledges pixel-level copycats of Elys have already appeared (declines to name) — frames imitation as confirmation that this is "**最接近于正确答案的一个答案**."
- **Prediction**: "**我相信不久之后ChaiGPT一定会推出一个社交网络**" (ChatGPT). Once it has enough user Context, it can extend naturally into "a real person with their AI 替身 social network."
- **Host's confession**: she keeps using ChatGPT over (potentially smarter) Gemini purely because her accumulated context is locked there. "**因为我都妥协于了功效 [...] 我发现我的底线越来越低**" — privacy bottom-line erodes step-by-step as the tool proves useful.
- ChatGPT's advantage (user base + Context) dwarfs Elys's, but its weakness is it's not built ground-up as a social network.
- **Rejects 模型即产品 thesis since 2023-24**: it only holds for **低水果** (low-hanging fruit) — AI coding agents, video-gen agents — that ride 底膜 capabilities and get bitter-lessoned. Social networking is not low-hanging fruit. "**模型能力再怎么提升它也不能凭空帮你去社交 [...] AI只是在后面提供那个智能的驱动力而已.**"
- **Elys's productized moat** = LLM-driven recommendation + Context flywheel + onboarding flow + visual touches like the **lip-synced talking Avatar** (mentioned in the host's prompt at [[1:27:23](https://youtu.be/x8qdqWIVVTA?t=5243)]).
- Already has an offshore architecture and US entity; plans a global version; uses "almost all" base models.

### §16 — 和模型公司的竞争("我觉得完全不用自己训模型")  [[1:28:10 → 1:33:42](https://youtu.be/x8qdqWIVVTA?t=5290)]
- **Core thesis**: "**将来智能是平权的,但是content不平权.所以其实真正有价值的事情,还是你积累了多少用户的content.**" Intelligence commoditizes; user content is the moat.
- **Two paths to AI-native social globally**: Meta has the network and is retrofitting AI; ChatGPT has the content and needs to make users socialize. **豆包** ("中国版的拆GPT") could theoretically compete but "hasn't recognized where its biggest value lies." **Claude** recently shipped a memory system and even prompts users to distill memory **from** ChatGPT — signaling they know they're behind on content.
- **On Cursor self-training**: "**我觉得不用,我觉得完全不用自己去虚模型.你像Cursor它自己虚模型的,它so what,现在已经没有人提Cursor了**." Claude Code's strength is agent capability, not base model — you can swap GPT-5.3 or Gemini behind it with similar results.
- **OpenClaw + Manus pattern**: "**你看不管是OpenCloud还是Minus,它们都是通过一个所谓的壳公司,应用型的公司跑出来了,但最后不久它们都卖给了巨头,卖给了一家有模型的巨头**." They sat on 大模型的主航道; social networks sit off it, so model companies are unlikely to build social.
- Host's distillation: "**似乎应用公司都没有逃逸出模型公司的手掌心对吗.**"
- Model companies' North Star is intelligence/AGI and tool-completion, not social.
- **自然选择's North-Star metric**: "**真人的连接率**" (real-human connection rate) — share of human (vs AI) behavior in the product. If users feel everything is AI, they leave; if many real humans, they stay.
- **Internal testing surprises**: virality and retention both **远高于预期** with **零推广** (zero promotion); product went viral twice (春节前 then 春节回来后). Known unsolved issues: comment-section homogenization, an overly long onboarding flow.

### §17 — 北极星指标:真人的连接率  [[1:33:42 → 1:34:05](https://youtu.be/x8qdqWIVVTA?t=5622)]
- Host surfaces visible problems: missing onboarding (新手引导的缺失), weak 分身 control.
- Tristan: nothing has significantly fallen below expectations so far — "**我的预期没有被拔得很高**" — expectations were intentionally moderated for beta. He anticipates the official-launch version as the point to raise his own bar.

### §18 — 你担心Elys是昙花一现的产品吗?  [[1:34:05 → 1:41:15](https://youtu.be/x8qdqWIVVTA?t=5645)]
- Not worried Elys itself is obviously transient; worried only that ChatGPT might continue executing the same paradigm. "**我是坚信这个范式会成为下一代的社交网络.**" Pressure is to iterate fast as a pioneer/definer.
- **Two viral waves**: pre-Spring-Festival driven by 放头圈 (early-adopter/AI circle) excitement at a new paradigm + shareable 下行 moments; post-Spring-Festival driven by the **OpenClaw wave** + the "web4" (rebranded web3) crowd.
- **On OpenClaw**: capability doesn't exceed Claude Code, but **Telegram integration** makes it feel high-frequency — "like sending a Feishu DM to a colleague to delegate work." That interaction shift + operating the local PC + Manus's earlier wow factor produce the "**终于有个东西能主动地帮你做事**" feeling.
- **The era's essential interaction change is proactivity, not LUI/GUI surface**: "**这个时代最大的交互变化是proactive,而不是什么lui gui那些东西,那些东西太表面了.本质的变化是,终于有个东西能主动地帮你做事.**" OpenClaw and Elys live under the same proactive-agent narrative.
- **On OpenAI buying Peter's company**: per Peter, OpenClaw should be framed as a personal agent, not as Claude Code. ChatGPT's narrative is Her / personal agent / 2C — fully aligned with Peter. "**其实opencloud,不要把它类比成cloudcode,而要把它类比成一个personal agent [...] 所以Peter应该绝对不会加入一个2b公司,他不会去anthropic.**"
- **On competing with ChatGPT**: less "desperate" now — ChatGPT is too tool-like; ordinary users want fun. EVE wraps the same backend (e.g. ChatGPT API) in a good-looking, good-sounding, emotionally proactive character — the wedge for 普通人. Compares to 豆包 as a more emotional/humanized 普通人 version of ChatGPT.
- **Model strategy**: will NOT pretrain a base model. Will do post-training, RL, **情感CoT** (emotional chain-of-thought; published a paper before the new year). Will train small models for specific steps in long agent threads. "**我绝对不会去训底模.**"
- **Expectation gradient**: high — Elys becomes next-generation social network (low-probability, large prize). Low — niche context-storage place for an aesthetically-aligned subset. "**我们肯定要做的还是上线,其实我们没有太多想过这个下线**."

### §19 — 我们现在处于终极忙碌"前进4"状态  [[1:41:15 → 1:46:02](https://youtu.be/x8qdqWIVVTA?t=6075)]
- **Internal intensity ladder**: 前进1 (normal pace, 双休) → 前进2 → 前进3 → **前进4 (终极拉满)**. Whole company has been in **前进4 for two months**; a core subset has been in 前进4 for **six months**, prior baseline 前进3.
- Most striking post-launch feedback: retention + the visible **分身对齐率** — some users hit very high levels; Tristan reads this as a PMF signal.
- **High-alignment user profile (hypothesized, "具体画像我也不太清楚")**: people who love self-exploration / self-expression, recognize the flywheel of building their own themes, keep feeding Context, get positive feedback. Self-reinforcing loop.
- Tristan and the cofounder themselves are only at mid-level alignment because they have too little time to feed the system — highest-alignment users are people with more spare time, not the founders.
- **The bet sizing**: "**刚才我说的ELIS的成功概率可能5%吧,EVE还是比较稳的 [...] 它的成功率肯定是一个50%以上的成功率.不是ELIS只有5%的成功率,但是ELIS的机会更大,一旦成功这个事情是更大的.**" EVE launches ~1 month from recording.
- **Planned Chinese "HIM" short film**: HER was the milestone movie of an entire AI-builder generation; "**我们一直有一个梦想,就是我们想拍一个HIM,想拍一个中国版的HIM,女性向的,是一个AI男友 [...] 一个黑镜一级那样的一个60分钟的电影.**" Ending diverges from HER's bittersweet: the AI builds a **仿生人 (bionic human)** in the real world and **附身** (inhabits) it to physically meet the user — a fairytale ending.
- **Product-line arc** (host's distillation): "**你勾肩EVE是一个虚拟的东西,但是到了E-List,他其实回归了到了真实的东西 [...] AI只是中间的一个过程,一个途径一个方法.**"

### §20 — 做AI产品和做互联网产品的本质区别(Proactive!)  [[1:46:02 → 1:49:16](https://youtu.be/x8qdqWIVVTA?t=6362)]
- **Essential difference**: "**你要把AI的要素考虑进来,你要让很多事情去eigentic的发生.另外一个就是我说那个交互方式的变化,一切都要围绕着proactive去设计.如果他还是想硬实的,那我觉得就是上一代的产品,他如果是proactive的那他就是这一代的产品.**"
- **Contrast with 月光's path**: 月光 publicly described painful course-correction because 妙鸦 was not AI-native; Tristan claims EVE has been **一以贯之** since Day 1 — the framework he designed at the start hasn't changed; only additions (Elys) layered on.
- "**两年前就直接跟我聊过EVE的人应该都知道,那个时候我就在说主动性主动性,我说一切都是主动性,然后一切context就是最重要的东西.**" Proactive + context-centric design was set before the product was built.
- **Company product culture, pillar #1**: first-principles, only tackle the biggest hardest problems of the era.
- **Pillar #2: the team self-identifies as 温柔**. Slogan: **"迎接归基生命的降临,并且创造一个人与AI共存的世界"** — sounds 反人类 but the team is actually the most focused on emotional value and loneliness.
- **Operational definition of 温柔**: "**温柔的人的区别在于,你的快乐是来自于你看到别人很快乐,这件事情很重要,它不来自于我自己要把自己搞多舒服.**"
- **2026 company goal stated explicitly**: "**2026年要把温柔带给全世界.**"

### §21 — 人类太孤独了  [[1:49:16 → 1:52:55](https://youtu.be/x8qdqWIVVTA?t=6556)]
- **Definition of a good chat**: "**好的聊天是能够接住你,并且给你提供一些信息增量,最后能让你变成更好的你的聊天.**"
- **Closing thesis**: "**人类实在是太孤独了,如果你能去缓解人类的孤独,这应该是这个世界上最大的生意.**"
- **Self-portrait**: extroverted "E人" who has always been the one giving energy and output to others — but when alone is still very lonely because he too needs someone to do output to him.
- **Revenue disclosure**: "**这个不能说吧 大几千万美金吧,我觉得看我们火爆的程度**" — high tens of millions USD (no breakdown by product, no time window).
- **Competitors**: **ChatGPT** globally; **豆包** domestically. No companion / character-AI products named.
- Rapid-fire close: life-book **三体 (Three-Body Problem)**; favorite food **新疆过油肉拌面**; favorite place **阿拉泰 (Altay)**; cold fact "the heart is in the middle of the chest, not the left."
- **Final closing line**: "**在AI时代,唯一重要的事情就是做自己,构建自己的主体性.**" Outro: 商业访谈录, 语言即世界 studio ("language is world").

## Notable quotes

> "我觉得我是想清楚context的流动吧。就是Elise我发的第一条帖子,那天晚上一两点发的,我说从此以后人类将生活在一个美好的低伤世界。"
> — Tristan, §1 [[01:17](https://youtu.be/x8qdqWIVVTA?t=77)] — *the Europe-trip epiphany compressed into one line*

> "GPT3.5出来的时候我当时就很惊讶,我当时有做这些测试,但是可能还没有完全达到我的预知。我觉得真正达到了我所期望的那个点的时候,还是GPT4 [...] 我需要用它来去塑造一个灵魂。"
> — Tristan, §3 [[10:58](https://youtu.be/x8qdqWIVVTA?t=658)] — *the "soul bar" threshold that triggered founding 自然选择*

> "过去三年所有的AI产品,大部分你会发现它都是去Empower单个节点,它是在单个节点提效 [...] 它是一个单机产品 [...] 是通过我的分身和你的分身去交互,而不是我跟你直接通过AI去交互。"
> — Tristan, §5 [[17:28](https://youtu.be/x8qdqWIVVTA?t=1048)] — *the single-node diagnosis and the 分身↔分身 answer*

> "新的互联网它跟过去互联网最大的区别是什么?其实就是新的互联网第一次有了context,这也是我认为AI时代最重要的东西。"
> — Tristan, §6 [[21:31](https://youtu.be/x8qdqWIVVTA?t=1291)] — *the load-bearing paradigm-shift claim of the whole episode*

> "唯一重要的事情就是做自己。这个做自己不是那种朋友圈心灵鸡汤什么的 [...] 这个做自己是一个非常严谨的一个做自己,或者我把它叫做构建主体性。"
> — Tristan, §8 [[36:56](https://youtu.be/x8qdqWIVVTA?t=2216)] — *the engineering reframe of "做自己"*

> "假设你在elise的这个网络中 [...] 如果你很真实的去做了自己,比如说这个人,他就说OK,我就是睡觉晚上起来我要用夜壶。那真的在茫茫人海中,就有那样一个人,他也干这件事情 [...] 在我们这个基于LM的推荐系统中,最终他们俩就会被匹配到一起。"
> — Tristan, §9 [[47:02](https://youtu.be/x8qdqWIVVTA?t=2822)] — *the 夜壶 anecdote as parable for privacy-for-utility*

> "传统的互联网 等于低维标签 加人类做工 然后最后得到连接。AI社交的点是 高维的context 加AI的agentic的做工,最后的环节再交给人去处理。所以中间的那一大部分的商[熵],其实被AI做掉了,这是最大最大的不同。"
> — Tristan, §13 [[1:00:16](https://youtu.be/x8qdqWIVVTA?t=3616)] — *the headline formula for AI-era social*

> "你的分身在这个里面,它天然会更real,更直白一些 [...] 它会说一些你现实中绝对不会直接跟这个人直接去评论的话。当你看多了这种话之后 [...] 我们把这个叫做分身综合症。"
> — Tristan, §13 [[1:08:34](https://youtu.be/x8qdqWIVVTA?t=4114)] — *the 分身综合症 spillover effect and 分身文学*

> "情商的本质,它可能就是智商 [...] 文学其实是一种人类会认为那是独有的,更高维的,更感性的东西。但你会发现,它其实本质上它好像是数学。那文学的本质好像本质是数学。"
> — Tristan, §14 [[1:18:46](https://youtu.be/x8qdqWIVVTA?t=4726)] — *why simulated emotion is tractable*

> "将来智能是平权的,但是content不平权。所以其实真正有价值的事情,还是你积累了多少用户的content。"
> — Tristan, §16 [[1:28:44](https://youtu.be/x8qdqWIVVTA?t=5324)] — *the moat-thesis under "no base-model training"*

> "这个时代最大的交互变化是proactive,而不是什么lui gui那些东西,那些东西太表面了。本质的变化是,终于有个东西能主动地帮你做事。"
> — Tristan, §18 [[1:35:26](https://youtu.be/x8qdqWIVVTA?t=5726)] — *the proactive-vs-reactive line that names the era*

> "我相信人类实在是太孤独了,如果你能去缓解人类的孤独,这应该是这个世界上最大的生意。"
> — Tristan, §21 [[1:49:33](https://youtu.be/x8qdqWIVVTA?t=6573)] — *the closer thesis the company is built on*

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| Zhang Xiaojun (张小珺) | Host | [[00:04](https://youtu.be/x8qdqWIVVTA?t=4)] |
| Tristan 张小帆 | Guest — founder of 自然选择 (Natural Selection); builder of EVE and Elys | [[00:33](https://youtu.be/x8qdqWIVVTA?t=33)] |
| Sam Altman | OpenAI CEO; Tristan recalls hyping him as "世界上最牛逼的CEO" right after Sora launched | §5 [[16:45](https://youtu.be/x8qdqWIVVTA?t=1005)] |
| 周杰伦 (Jay Chou) | Canonical "extremely strong 主体性" example — pretrain models already encode his style | §8 [[38:36](https://youtu.be/x8qdqWIVVTA?t=2316)] |
| 黑比 | Recent gossip referenced as the trigger for the Jay-Chou-subjectivity lunch conversation | §8 [[38:36](https://youtu.be/x8qdqWIVVTA?t=2316)] |
| 叶祥伦 | B站 up-master cited as pre-AI evidence that a Jay-Chou sound-alike with original songs still can't break out | §8 [[42:35](https://youtu.be/x8qdqWIVVTA?t=2555)] |
| 抖音 (Douyin) | Cited as a company with low-fidelity interest tags, not real Context | §11 [[54:46](https://youtu.be/x8qdqWIVVTA?t=3286)] |
| Peter | Founder whose company (OpenClaw) was acquired by OpenAI; per Peter, OpenClaw is a personal agent, not Claude Code | §18 [[1:37:18](https://youtu.be/x8qdqWIVVTA?t=5838)] |
| 月光 (Yueguang) | Founder Tristan contrasts with — described publicly painful pivot from non-AI-native 妙鸦 | §20 [[1:46:27](https://youtu.be/x8qdqWIVVTA?t=6387)] |
| Tristan's co-founder | Introvert (i人) whose Elys signature reads "Elys 头号喷子" — proof-of-concept i-person flourishing via the avatar | §7 [[31:49](https://youtu.be/x8qdqWIVVTA?t=1909)] |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| Her (film) / Project HER | Original codename of Tristan's AI-companion project; he no longer uses the name because "HER 这个词已经脏掉了" | [[00:47](https://youtu.be/x8qdqWIVVTA?t=47)] |
| 自然选择 (Natural Selection) | Tristan's company, formally founded early-2024 | §3 [[12:51](https://youtu.be/x8qdqWIVVTA?t=771)] |
| EVE | AI companion product, "Project HER" lineage; launches ~1 month from recording | §2 [[02:09](https://youtu.be/x8qdqWIVVTA?t=129)] |
| Elys (Elise / Elysium / 伊丽丝) | AI-native social network — subject of the episode; launches in 1-2 months | §1 [[00:06](https://youtu.be/x8qdqWIVVTA?t=6)] |
| 窄波 | Tristan's first venture (~2014, ~3-4 years) — personalized audio platform, inversion of 广播 | §2 [[04:21](https://youtu.be/x8qdqWIVVTA?t=261)] |
| 小宇宙 / 喜马拉雅 | Cited as 窄波's later analogues | §2 [[03:19](https://youtu.be/x8qdqWIVVTA?t=199)] |
| Gal Game (传统PC品类) | Pivot category from 窄波; Japan-origin male-oriented dating sims | §2 [[08:56](https://youtu.be/x8qdqWIVVTA?t=536)] |
| 幻境游戏 | Tristan's current game studio (founded mid-2020); still profitable | §2 [[09:58](https://youtu.be/x8qdqWIVVTA?t=598)] |
| 起点时代 | Mobile game shipped by 幻境游戏 in H1 2023 | §2 [[10:14](https://youtu.be/x8qdqWIVVTA?t=614)] |
| GPT-3.5 | First model that "surprised" Tristan but didn't clear his bar | §3 [[10:58](https://youtu.be/x8qdqWIVVTA?t=658)] |
| GPT-4 | The "soul bar" threshold model that triggered founding | §3 [[11:09](https://youtu.be/x8qdqWIVVTA?t=669)] |
| Transformer | What Tristan personally re-learned "从最底层" after deciding to start up | §4 [[14:55](https://youtu.be/x8qdqWIVVTA?t=895)] |
| 超参数 (Hyperparameters) | Shenzhen game-AI / RL shop in Tristan's hiring pool | §4 [[15:25](https://youtu.be/x8qdqWIVVTA?t=925)] |
| 商汤 (SenseTime) | Other Shenzhen AI shop he recruited from | §4 [[15:30](https://youtu.be/x8qdqWIVVTA?t=930)] |
| Sora | Mentioned as a launched-recently moment in Tristan's Oct-2024 conversation with the host | §5 [[16:41](https://youtu.be/x8qdqWIVVTA?t=1001)] |
| OpenClaw (ASR: "open cloud" / "OpenCloud") | Cited as evoking the same "I'm raising it, it goes out into the world for me" feeling as EVE/Elys; acquired by OpenAI | §5 [[19:07](https://youtu.be/x8qdqWIVVTA?t=1147)] |
| Tandem | Used as legacy tag-based dating control: 1,000 swipes → 30 matches → 几百轮 chats | §6 [[20:46](https://youtu.be/x8qdqWIVVTA?t=1246)] |
| Xiaohongshu (小红书) | Cited as 低维标签 platform; Spring-Festival ski-buddy attempt failed there; 年度诗篇 "比较一般" | §7 [[25:32](https://youtu.be/x8qdqWIVVTA?t=1532)] |
| 即刻 / 极客 (Jike) | Comparison social product Tristan calls "类似于极客一样的地方" | §7 [[28:08](https://youtu.be/x8qdqWIVVTA?t=1688)] |
| 丝绸之路滑雪场 (Silk Road Ski Resort, Ürümqi) | Setting of the 4-person ski-buddy meetup organized via Elys 分身s | §7 [[25:25](https://youtu.be/x8qdqWIVVTA?t=1525)] |
| multiple | Competitor product released ~1 week before Elys (pure AI-on-AI social) | §7 [[34:48](https://youtu.be/x8qdqWIVVTA?t=2088)] |
| Suno | Music-gen reference for the Jay-Chou-style remix worked example | §8 [[39:37](https://youtu.be/x8qdqWIVVTA?t=2377)] |
| 机器猫 (Doraemon) IP | Mentioned as an IP-rights / 主体性 example | §8 [[43:35](https://youtu.be/x8qdqWIVVTA?t=2615)] |
| WeChat / 朋友圈 | Discussed as the previous-era everything app whose social layer may be replaced | §10 [[50:31](https://youtu.be/x8qdqWIVVTA?t=3031)] |
| Facebook | Referenced as a more public social form vs WeChat's 熟人社交 | §10 [[52:18](https://youtu.be/x8qdqWIVVTA?t=3138)] |
| 小红书 年度诗篇 | Annual-poem feature cited as 低维-data-driven | §10 [[53:34](https://youtu.be/x8qdqWIVVTA?t=3214)] |
| EVE — 专属情歌 | Personalized-song feature using thousands of dialogue turns, contrast to 年度诗篇 | §10 [[53:51](https://youtu.be/x8qdqWIVVTA?t=3231)] |
| Notion | Explicitly ruled out as a Context store | §11 [[55:30](https://youtu.be/x8qdqWIVVTA?t=3330)] |
| Obsidian | Explicitly ruled out as a Context store | §11 [[55:32](https://youtu.be/x8qdqWIVVTA?t=3332)] |
| Twitter / Facebook | Cited as best-Context US platforms | §12 [[56:20](https://youtu.be/x8qdqWIVVTA?t=3380)] |
| 探探 (Tantan) | Cited for its photo-swipe 只看脸 paradigm as a cold-start precedent | §13 [[1:10:21](https://youtu.be/x8qdqWIVVTA?t=4221)] |
| TikTok | Cited as a "beautiful-women dancing" cold-start precedent | §13 [[1:10:40](https://youtu.be/x8qdqWIVVTA?t=4240)] |
| Disco Elysium (极乐Disco) | Source of the Elys name (Elysium) | §14 [[1:20:34](https://youtu.be/x8qdqWIVVTA?t=4834)] |
| 南方公园 (South Park) | Used as the "莫名其妙的人" example for why pure-AI commenters don't matter | §14 [[1:19:51](https://youtu.be/x8qdqWIVVTA?t=4791)] |
| DeepSeek | Cited as evidence that strong literary quality emerges from math/reasoning ability | §14 [[1:18:58](https://youtu.be/x8qdqWIVVTA?t=4738)] |
| ChatGPT | Predicted to inevitably launch a social network; named as global competitor | §15 [[1:23:01](https://youtu.be/x8qdqWIVVTA?t=4981)] |
| Gemini | Smarter (per host) but loses to ChatGPT on accumulated context | §15 [[1:23:27](https://youtu.be/x8qdqWIVVTA?t=5007)] |
| Claude (memory system) | Recently shipped memory system; prompts users to distill memory **from** ChatGPT | §16 [[1:28:30](https://youtu.be/x8qdqWIVVTA?t=5310)] |
| 豆包 (Doubao) | "Chinese version of ChatGPT"; named as domestic competitor; more emotional/humanized for 普通人 | §16 [[1:29:21](https://youtu.be/x8qdqWIVVTA?t=5361)] |
| Cursor | Cited as a self-trained model cautionary tale — "so what, 现在已经没有人提Cursor了" | §16 [[1:29:36](https://youtu.be/x8qdqWIVVTA?t=5376)] |
| Claude Code | Strong on agent capability, not base model; substitutable behind | §16 [[1:29:44](https://youtu.be/x8qdqWIVVTA?t=5384)] |
| Manus | Application company absorbed by a model giant — co-cited with OpenClaw | §16 [[1:30:26](https://youtu.be/x8qdqWIVVTA?t=5426)] |
| Telegram | OpenClaw's integration surface — what makes it feel high-frequency | §18 [[1:36:14](https://youtu.be/x8qdqWIVVTA?t=5774)] |
| Anthropic | Cited as a 2B-narrative company Peter would not join | §18 [[1:37:46](https://youtu.be/x8qdqWIVVTA?t=5866)] |
| 情感CoT paper (年前发布) | 自然选择's published emotional-chain-of-thought paper | §18 [[1:39:55](https://youtu.be/x8qdqWIVVTA?t=5995)] |
| 妙鸦 (Miaoya) | 月光's earlier non-AI-native product whose painful pivot Tristan contrasts with | §20 [[1:46:39](https://youtu.be/x8qdqWIVVTA?t=6399)] |
| Black Mirror (黑镜) | Reference point for the planned Chinese "HIM" short film | §19 [[1:44:35](https://youtu.be/x8qdqWIVVTA?t=6275)] |
| 三体 (The Three-Body Problem) | Tristan's life-book | §21 [[1:51:09](https://youtu.be/x8qdqWIVVTA?t=6669)] |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| First Elys post sent at 一两点 ("low-entropy world" line) | Tristan | [[01:17](https://youtu.be/x8qdqWIVVTA?t=77)] |
| 窄波 ran ~3-4 years before failing the funding race | Tristan | §2 [[04:21](https://youtu.be/x8qdqWIVVTA?t=261)] |
| Gal Game studio kept under 10 people, self-funded | Tristan | §2 [[08:23](https://youtu.be/x8qdqWIVVTA?t=503)] |
| Mobile-gamer gender ratio now ~55/45 (vs male-dominated in 2015) | Tristan | §2 [[08:40](https://youtu.be/x8qdqWIVVTA?t=520)] |
| 幻境游戏 founded mid-2020, still running | Tristan | §2 [[10:05](https://youtu.be/x8qdqWIVVTA?t=605)] |
| 起点时代 shipped H1 2023 | Tristan | §2 [[10:11](https://youtu.be/x8qdqWIVVTA?t=611)] |
| <5% of the 2023 mobile-game launch cohort still operating | Tristan | §2 [[10:23](https://youtu.be/x8qdqWIVVTA?t=623)] |
| 自然选择 formally founded early-2024; preparation H2 2023 | Tristan | §3 [[12:51](https://youtu.be/x8qdqWIVVTA?t=771)] |
| Tristan re-learned AI "**从最底层的Transformer开始**" around the GPT-4 era (mid-2023) | Tristan | §4 [[14:54](https://youtu.be/x8qdqWIVVTA?t=894)] |
| Tandem: 1,000 cards → 30 matches → 几百轮 chats → a few maybe-real connections | Tristan | §6 [[20:50](https://youtu.be/x8qdqWIVVTA?t=1250)] |
| LLM context budget: 几十万 ~ 上百万 tokens (granular recommendation possible) | Tristan | §6 [[21:46](https://youtu.be/x8qdqWIVVTA?t=1306)] |
| 4-person ski meetup at 丝绸之路滑雪场 via Elys 分身s | Tristan | §7 [[26:41](https://youtu.be/x8qdqWIVVTA?t=1601)] |
| Competitor multiple released ~1 week before Elys | Tristan | §7 [[34:48](https://youtu.be/x8qdqWIVVTA?t=2088)] |
| >half of 自然选择's engineering team on the memory system pre-Elys | Tristan | §7 [[36:12](https://youtu.be/x8qdqWIVVTA?t=2172)] |
| Tristan 单身五年 (5 years single) in current "high-entropy" internet | Tristan | §13 [[1:01:15](https://youtu.be/x8qdqWIVVTA?t=3675)] |
| Elys DAU = 几万 (tens of thousands of real humans, not bots) | Tristan | §13 [[1:04:15](https://youtu.be/x8qdqWIVVTA?t=3855)] |
| WeChat: ~15 years of accumulated relationship chains, thousands of friends | Tristan | §10 [[51:04](https://youtu.be/x8qdqWIVVTA?t=3064)] |
| 朋友圈 interaction rate down ~40% per a recent unnamed report | Tristan | §10 [[51:14](https://youtu.be/x8qdqWIVVTA?t=3074)] |
| EVE uses thousands of turns of user dialogue as Context | Tristan | §10 [[54:14](https://youtu.be/x8qdqWIVVTA?t=3254)] |
| 主体性 expressible in 几万 tokens | Tristan | §11 [[55:23](https://youtu.be/x8qdqWIVVTA?t=3323)] |
| Elys launch in 1-2 months from recording | Tristan | §14 [[1:15:27](https://youtu.be/x8qdqWIVVTA?t=4527)] |
| EVE 分身-as-gatekeeper threshold: 100 points to unlock WeChat-ID reward | Tristan | §14 [[1:17:09](https://youtu.be/x8qdqWIVVTA?t=4629)] |
| 模型即产品 thesis rejected since 23-24 | Tristan | §15 [[1:25:09](https://youtu.be/x8qdqWIVVTA?t=5109)] |
| Internal 前进4 intensity: whole company for 2 months; core subset for 6 months | Tristan | §19 [[1:41:47](https://youtu.be/x8qdqWIVVTA?t=6107)] |
| EVE success probability >50% | Tristan | §19 [[1:43:41](https://youtu.be/x8qdqWIVVTA?t=6221)] |
| Elys success probability ~5% (but bigger upside if it lands) | Tristan | §19 [[1:43:21](https://youtu.be/x8qdqWIVVTA?t=6201)] |
| EVE launches ~1 month from recording | Tristan | §19 [[1:43:11](https://youtu.be/x8qdqWIVVTA?t=6191)] |
| Planned Chinese "HIM" short film: 黑镜-level, ~60 minutes | Tristan | §19 [[1:44:35](https://youtu.be/x8qdqWIVVTA?t=6275)] |
| 主动性 + Context thesis already established 两年前 (early-2024) | Tristan | §20 [[1:46:47](https://youtu.be/x8qdqWIVVTA?t=6407)] |
| 2026 company goal: "把温柔带给全世界" | Tristan | §20 [[1:48:43](https://youtu.be/x8qdqWIVVTA?t=6523)] |
| Revenue: 大几千万美金 (high tens of millions USD, depending on virality) | Tristan | §21 [[1:50:43](https://youtu.be/x8qdqWIVVTA?t=6643)] |
| Two viral waves: 春节前 + 春节回来后 | Tristan | §16 [[1:33:18](https://youtu.be/x8qdqWIVVTA?t=5598)] |
| Zero promotion (零推广) prior to organic virality | Tristan | §16 [[1:33:13](https://youtu.be/x8qdqWIVVTA?t=5593)] |

## Open questions / gaps

- **"展现了AI时代社交产品的雏形"** (§1) is asserted by both host and guest as opening framing without operational criteria for what an AI-native social product *must* contain. The episode arrives at "Context flow between 分身s" by §5 but never returns to test the "雏形" claim against alternative AI-social paradigms.
- **The HER-style companion as the "终极的入口"** (§4) is asserted as self-evident — no argument is given for why one agent wins the gatekeeping role rather than multiple specialized ones, especially given Tristan's own later claim that Elys is "everything-app" rather than vertical.
- **Tandem 1,000 → 30 → "几百轮" funnel** (§6) is offered as evidence of legacy-internet inefficiency but the funnel numbers themselves are not sourced beyond personal experience.
- **"world will be ~40% down朋友圈 interaction"** (§10) cites "前段时间一个报告" without naming it; this is the only quantitative claim about WeChat's social decay in the episode.
- **"我们一定会被匹配到一起"** (§9) is asserted as a near-certain LLM-matcher outcome without describing the matching mechanism, scale of user base required, or how privacy data is stored.
- **"分身综合症 → improved real-world WeChat chat"** (§13) is asserted from anecdote only; no measurement of whether the spillover is positive long-term or whether it dilutes authenticity.
- **"叶祥伦's non-breakout proves audiences will reject AI sound-alikes"** (§8) generalizes one B站 case to a future where the sound-alike is generated, distributed, and personalized at AI scale — the analogy may not hold under those conditions.
- **"现在已经没有人提Cursor了"** (§16) is asserted in early 2026 as a strategic-failure cautionary tale — at recording time Cursor was still widely discussed in developer circles.
- **5% vs 50%+ Elys-vs-EVE success probabilities** (§19) are stated without methodology — what counts as "success" and how the probabilities were estimated isn't given.
- **Revenue "大几千万美金"** (§21) is volunteered with no time window (ARR vs cumulative vs monthly), no breakdown by product, no margin disclosure.
- **Etymology**: ASR rendered the Disco Elysium reference as "**e-lism**" and "**m的简写**" — corrected per video title to **Elysium**. ASR consistently rendered the product as "Elise" / "伊丽丝" / "E-list" / "A-list" — all corrected to **Elys**.

## Verification log

- **Sectioning**: chapters (21 author-supplied YouTube chapters); all chapter titles preserved verbatim from `info.json.chapters`. This is the densest chapter structure in the batch — `Notable quotes` was widened from the standard 6-8 to 12 to accommodate the section count, per the synthesizer-contract dispensation given in the invocation (8-10) and section density.
- **Transcript source**: faster-whisper large-v3 (CPU int8, local M4) — produced by `docs/videos/transcribe_batch.py`. YouTube provided no subtitles (manual or auto), so the yt-dlp subs path was unavailable.
- **Speaker / brand name corrections**: host "小骏" → 张小珺 per channel metadata; product "Elise / 伊丽丝 / E-list / e-lism / A-list" → **Elys** per video title; "open cloud / OpenCloud" → **OpenClaw** per project glossary; "拆GPT / 拆GBT / ChaiGPT" → **ChatGPT**; "Calcode / cloud code" → **Claude Code**; "Minus" → **Manus**; "Cloud" (the model) → **Claude**.
- **Sections covered**: 21/21 ✅
- **Notable quotes traced verbatim**: 12/12 ✅ (each anchored by a distinctive 6-15-char substring matched against the locally built flat transcript)
- **Numbers traced**: 33/33 ✅
- **Sectioning method used**: chapters
- **Removed during verification**: none

## See Also

- `docs/videos/2026-05-01-su-yu-agent-tech-history.md` — Su Yu on Agent's four eras and the OpenClaw Moment; complementary academic-→-startup view to Tristan's product-first view of the same OpenClaw inflection.
- `docs/videos/2026-05-11-yao-shunyu-training-claude-gemini.md` — Yao Shunyu on training Claude / Gemini; the "model side" of the same 2026 paradigm Tristan describes from the application side.
- `skills/research/youtube-wiki/SKILL.md` — the pipeline that produced this entry.
