# Su Yu: An Agent Survey — Four Eras, OpenClaw, and the Dissolution of Boundaries

**Source**: https://youtu.be/Xxz5uh0L1mE
**Channel**: Zhang Xiaojun Podcast (张小珺商业访谈录)
**Published**: 2026-05-01
**Duration**: 02:17:49
**Watched on**: 2026-05-14
**Sectioning**: chapters (15 YouTube-supplied chapters)
**Detected video language**: `zh` (no `info.json.language`; `subtitles` and `automatic_captions` keys both empty — no YouTube subs of any kind)
**Transcript source**: faster-whisper large-v3 (CPU int8) on local M4 — original recipe per the project's batch driver `docs/videos/transcribe_batch.py`. ASR consistently rendered the guest's name as "苏玉" (should be **苏煜 Su Yu**) and the host as "小俊/小骏" (should be **张小珺 Zhang Xiaojun**) — both corrected from the YouTube `info.json` title.
**Speakers**: Zhang Xiaojun (张小珺, host), Su Yu (苏煜, guest — Ohio State University CS professor, founder of Neocognition, 2025 Sloan Research Fellow, Language Agent researcher)

## TL;DR

- Su Yu maps Agent research into four eras — **Logical Agent (1950s-90s) → Neural Agent (post-2000 Deep RL on Atari/AlphaGo) → Semantic Parsing → Language Agent (post-2022)** — and frames every era through a **Memory + Autonomy** dichotomy. Language is the inflection variable in both human evolution and AI evolution; LLMs play the role language did for civilization.
- The **OpenClaw Moment** is structurally identical to the **ChatGPT Moment**: underlying tech was already ready, the watershed was an interaction-form shift + a "YOLO" open-permission release. Anthropic's coding bet pulled the whole industry toward productivity-agent convergence in 2026.
- All Agent categories — browser-use/desktop-use/mobile-use, GUI/CLI/API, coding vs. tool-use — are dissolving into one **Universal Digital Agent**. Coding is "the most fundamental fabric" doing the dissolving: GUI/CLI/API can all be re-expressed as code.
- **Neocognition** closed a **$40M seed** in March 2026 (~6 months post-founding). Thesis: **specialization > general intelligence** via **continual learning of broad world models** (where "world model" includes the symbolic, social, workplace stuff — not just next-frame video prediction). Goal: democratize expert-level agents so individuals, not just frontier labs, can build them.
- For 2026 the main frontier will be **continual learning / self-learning**; **reliability + speed + cost-effectiveness** are the real bottlenecks. **GUI will not be replaced by CLI** — GUI piggybacks on humanity's accumulated digital knowledge, and a CLI-only future would fail for the same reasons Tim Berners-Lee's Semantic Web failed after 20+ years.

## Why it matters

A 2h17m technical survey from one of the earliest scholars to pivot from Semantic Parsing to Language Agent (his group built Mind2Web, SeeAct, LLM-Planner, UGround, MMMU). Combines a 4-era historical framing, an empirical 3-year timeline of Language Agent breakthroughs (CoT → ReAct → Mind2Web → AutoGPT → SeeAct → UGround → Claude Code → OpenClaw), startup-thesis transparency on Neocognition's $40M seed and why "world model" should be defined far more broadly than Yann-LeCun-style vision JEPA, and a clear technology-democratization stance on what AI researchers' real responsibility is post-OpenClaw.

## Section summaries

### §1 — Opening: framing 2026 as Act Two of AI  [[00:00 → 02:00](https://youtu.be/Xxz5uh0L1mE?t=0)]
- Host Zhang Xiaojun positions this episode as the continuation of an arc — prior episodes with 福利 and 广密 traced AI from "Act One: Chat" to "Act Two: Agent". 2026 is "Agent's high-frequency year."
- Guest Su Yu introduced: Ohio State CS professor, founder of Neocognition, 2025 Sloan Research Fellow, Language Agent researcher — "one of the few scholars who witnessed the full Agent evolution."
- Su Yu opens with two paradigm-shift markers: ChatGPT Moment = LLM paradigm shift; **OpenClaw Moment** = Agent / Personal Agent paradigm shift. Adds the load-bearing thesis for the whole episode: all Boundaries are converging toward one Universal Digital Agent, and that convergence is tied to **Coding**.
- Papers: *Artificial Intelligence: A Modern Approach* — referenced as the canonical era summary [[01:17](https://youtu.be/Xxz5uh0L1mE?t=77)].

### §2 — Who is Su Yu  [[02:00 → 03:30](https://youtu.be/Xxz5uh0L1mE?t=120)]
- Hunan → Tsinghua CS undergrad → US PhD → Ohio State professor → founded OSU NLP Group → moved to Silicon Valley in 2025 and founded Neocognition.
- OSU NLP group's known works enumerated on-mic: **Mind2Web, SeeAct (ASR: "CACT"), LLM-Planner** in computer-use agents; **MMMU** as the most widely used multimodal LLM benchmark.
- Caveats listeners upfront: heavy Mandarin-English code-switching is unavoidable; he'll translate when he can.

### §3 — Agent's four-era tech history: Logical → Neural → Semantic Parsing → Language  [[03:30 → 27:21](https://youtu.be/Xxz5uh0L1mE?t=210)] — *the technical backbone*
- **Su Yu's definition of Agent** (three criteria): (1) an **Entity** with a **Boundary**; (2) embedded in some **Environment**; (3) performing **Goal-Directed Activities**. By this definition all animals — and especially humans — qualify as Agents. The question has been "贯穿 AI 的始终" since the 1940-60s founding era.
- **Why the field fragmented**: building a fully-general Agent was deemed counter-productive at the time, so AI broke into Vision, NLP, Logic/Reasoning sub-fields. Su Yu invokes 三国演义's "分久必合,合久必分" — and notes that under LLMs these subfields are now re-converging.
- **The Memory + Autonomy framework** (load-bearing for the whole episode): **Memory** = knowledge expression / acquisition / update / forgetting; spans Semantic Knowledge, Episodic Memory, and Procedural Memory (in humans all stored uniformly in neural synapses). **Autonomy** = Perception → Reasoning → Decision Making → Action. Memory is the substrate of Autonomy — "一体两面."
- **Logical Agent era (1950s-90s)**: expert systems built on first-order logic (一阶谓词逻辑) + Inference Engine. Memory limited to finite Logical Statements; expressiveness bounded by the logic language. Higher-order / fuzzy / probabilistic logic only patched small parts. Collapsed under the **Knowledge Acquisition Bottleneck** (engineers hand-encoding domain experts' knowledge) — directly triggered the 80s-90s AI Winter.
- **Russell & Norvig anecdote**: Stuart Russell personally told Su Yu that *Artificial Intelligence: A Modern Approach* (1st ed. ~1995) is "本质上是一本关于 agent 的书" — Chapter 1 defines intelligent agent — but that framing has been forgotten.
- **Neural Agent era (post-2000, peaks post-2010)**: Atari Deep RL → AlphaGo → Dota → StarCraft. Under Memory/Autonomy lens still very limited — networks at 10M-100M params (≈1亿 量级), single-game scope, input=pixels / output=action set. **Reasoning is implicit** — one forward pass of fixed compute regardless of problem complexity. Generality extends only to "same architecture, different game"; sample efficiency catastrophic (millions of plays per game). Su Yu credits Demis Hassabis's personal preference for games + practical fit (high repeatability) for why games became the canonical Deep RL environment (and flags this as his guess).
- **Semantic Parsing (parallel NLP track, post-2000)**: convert natural language into formal meaning representation that a machine (KB / DB / website) can execute. Widens the action space. Su Yu's own PhD area; alumni now lead the LLM-Agent generation — **Percy Liang** (Stanford), **Luke Zettlemoyer** (UW; ELMo at AI2, RoBERTa at Meta — "GPT 在很大程度上是受到这些工作的启发的"), **Tao Yu** (HKU), **Huan Sun** (OSU).
- **Language Agent era (post-2022)**: Su Yu co-authored the **2024 Language Agent tutorial** with Yang Di, Yao Shunyu, and Tao Yu. Defining feature: **language as scaffold (脚手架)** for every Agent function — perception (language understanding), reasoning (CoT enables adaptive computing — more tokens = more compute for harder problems), action (formal/machine languages let agents act in any digital world).
- **LLM-as-Memory**: training itself is a memory-shaping process — compression of language surface forms into a meaning representation / world model. Counters the "stochastic parrot 随机鹦鹉" view.
- **Compressed-timeline analogy**: AI now advances per-year/month what previously took a decade. Mirrors evolution — eukaryotes ~1B years, mammals 200-400M, Homo ~2M+, symbolic expression ~100k, written language ~5-6k. Language is the inflection variable in both biological and AI evolution.
- People: [Stuart Russell](https://youtu.be/Xxz5uh0L1mE?t=732) [[12:12](https://youtu.be/Xxz5uh0L1mE?t=732)], [Peter Norvig](https://youtu.be/Xxz5uh0L1mE?t=722) [[12:02](https://youtu.be/Xxz5uh0L1mE?t=722)], [Demis Hassabis](https://youtu.be/Xxz5uh0L1mE?t=1062) [[17:42](https://youtu.be/Xxz5uh0L1mE?t=1062)], [Percy Liang](https://youtu.be/Xxz5uh0L1mE?t=1239) [[20:39](https://youtu.be/Xxz5uh0L1mE?t=1239)], [Luke Zettlemoyer](https://youtu.be/Xxz5uh0L1mE?t=1241) [[20:41](https://youtu.be/Xxz5uh0L1mE?t=1241)], [Yao Shunyu (姚顺宇)](https://youtu.be/Xxz5uh0L1mE?t=1310) [[21:50](https://youtu.be/Xxz5uh0L1mE?t=1310)].
- Numbers: Logical Agent era ~1950s-90s; AI Winter 80s-90s; backprop popularized ~1985; AIMA 1st ed ~1995; Neural Agent param scale 数十million-1亿; Atari "几百万盘" per game [[17:19](https://youtu.be/Xxz5uh0L1mE?t=1039)]; symbolic expression ~100k yrs [[26:39](https://youtu.be/Xxz5uh0L1mE?t=1599)]; written language ~5-6k yrs [[26:54](https://youtu.be/Xxz5uh0L1mE?t=1614)].

### §4 — Language emerged late but caused exponential civilizational growth  [[27:21 → 29:28](https://youtu.be/Xxz5uh0L1mE?t=1641)]
- Su Yu's evolutionary analogy (the topic of his **first blog post in 2023**): language appeared very late in human evolution but acted as an "爆炸式的加速剂" for civilization; LLMs are now playing the same role for AI/Agent evolution.
- **Semantic Parsing vs Language Agent — the essential distinction is foundation, not "using language."** NLP has always been about using language for AI. Before LLMs, you could only anchor to a specific database / knowledge graph / website. LLMs provide a **strong prior + an internally-built language-based world model**, so a language agent can drop into any environment and produce "at least reasonably" useful behavior.

### §5 — A 3-year recap of Language Agent's key works  [[29:28 → 40:56](https://youtu.be/Xxz5uh0L1mE?t=1768)]
- **CoT (Chain-of-Thought, early 2022)** — Su Yu's starting marker. Language's adaptive computing / adaptive reasoning is the essential difference from prior agents.
- **ReAct (Yao Shunyu, ~Oct 2022)** — extended CoT from math problems to agent settings with external environment: perceive → reason → action loop. Su Yu: many Agent milestones "look simple in retrospect, but having the insight at the right moment and shipping it is not easy."
- **LLM-Planner (OSU, late 2022, contemporary with ChatGPT)** — among the first to use LLMs for robot/embodied planning; Google's **SayCan** (ASR: "SECAN") is the canonical counterpart.
- **Mind2Web (OSU, 2022-10 start, early 2023 release)** — first LLM-based web / computer-use agent emphasizing the **generalist** angle (any website). Same era: **WebArena** from Graham Neubig (CMU, 2023-07) took the RL-environment path — full replicas of a handful of sites for reproducibility.
- **Toolformer (Meta, 2023-02, Luke Zettlemoyer as a lead)** — first LLM tool-use work; Satya Nadella circulated it company-wide at Microsoft.
- **AutoGPT (2023-03)** — at the time, GitHub's fastest-ever star growth. Hit 100k stars almost instantly, ~180k now (similar to OpenClaw's current count).
- **2023 H2 — GPT-4V triggers the multimodal turn**: OSU shipped **MMMU** (first multimodal LLM benchmark) and **SeeAct** (multimodal web agent on GPT-4V; the team hacked together their own API because GPT-4V had no API yet).
- **2024 — web → desktop/mobile expansion**: **OSWorld** (Yutao's group, 2024-03/04) for desktop; **SWE-bench** drives coding agents.
- **UGround (OSU, 2024 H2)** — the embodiment shift: agents should "use the computer like humans do" with **visual perception + pixel-level actions** (click, type), abandoning HTML-based representations. **Claude Computer Use (Oct 2024)**, OpenAI Operator, Claude in Chrome all later adopted this embodiment.
- **2025 — Anthropic's coding explosion**: Cursor's influence drove Anthropic to release Claude Code; in 2025 H2 base-model coding leveled up dramatically. Su Yu's lived experience: after **Opus 4.5**, "within one or two months Silicon Valley basically stopped writing code by hand." **OpenClaw** released November 2025; truly peaked around February 2026.

### §6 — At the end of the day: one Universal Digital Agent; coding dissolves the boundaries  [[40:56 → 45:18](https://youtu.be/Xxz5uh0L1mE?t=2456)]
- All taxonomies — browser-use vs desktop-use vs mobile-use; GUI vs CLI vs text; API vs coding+tool-use — were "比较临时性的." End state: **a universal digital agent that can do everything a human can do in the digital world, possibly better**. Surface modalities are "means to an end" and the boundaries are "正在快速消弥."
- **Coding is the fundamental fabric**. Su Yu credits Dario Amodei for nailing both points: coding is foundational, and boundaries are dissolving. "Coding 它是非常 fundamental … 它是这个最根本性的 fabric … 你所有东西都能用 code 来表达."
- Concrete mechanism for boundary dissolution: GUI / CLI / API can all be made equivalent via code — "因为 GUI 本身其实就是通过 code 的 render 而来的."
- "Language Agent" subsumes "Coding Agent" — programming language is language. Language is never just natural language; it's all symbolic forms (programming languages, diagrams, gestures), all derived from natural language. So the taxonomy doesn't need to change.
- Pushes back on the framing "natural language = scaffold for humans, code = scaffold for machines": all language ultimately serves the same purpose — describing and manipulating the world. From an agent's point of view, the natural/formal split is "并不是一个特别本质的区别."

### §7 — "I was one of the earliest to pivot from Semantic Parsing to Language Agent"  [[45:18 → 48:56](https://youtu.be/Xxz5uh0L1mE?t=2718)]
- Su Yu positions himself as one of the earliest research groups to pivot from Semantic Parsing to Agent research, "naturally on the path to Language Agent" once LLMs arrived.
- When NLP wasn't yet "显学" (CV and motion-learning were hotter), he picked the niche of Semantic Parsing despite hearing "some NLP profs telling their students: pick any topic, just don't pick Semantic Parsing" — small community, hard to publish, low citation.
- **The motivation was a problem that bugged him**: people are becoming "digital slaves" — software is so complex that ordinary users need months of Excel training, or years to become AWS experts; hundreds of features × thousands of workflows force humans to think like machines.
- **PhD defense manifesto** (which he calls "现在想想可能有点比较中二"): **"Let machines understand human thinking, don't let humans think like machines."** Same goal carried from Semantic Parsing through Language Agent — "只是现在用的技术、用的手段稍微不同."

### §8 — OpenClaw Moment ≅ ChatGPT Moment  [[48:56 → 55:10](https://youtu.be/Xxz5uh0L1mE?t=2936)]
- **Structural isomorphism — the tech was already ready; the watershed is interaction form.** Pre-ChatGPT, LMs evolved BERT (2018) → ELMo (2019) → GPT-1/2/3 over years. OpenAI's act was to fine-tune into a chatterbot and release it to the public. "底层的技术实际上是没有太大的变化的，更多是一个交互形式的变化." OpenClaw is the same — "做 agent 的人去看 OpenClaw 的 codebase 会有一种 nothing is new here 的感觉，但实际上它是一个交互形式的深刻变化."
- **OpenAI was itself surprised by ChatGPT's reception**; the success later exacerbated Ilya-led fundamental-research vs. applied tensions, "甚至可能导致了 OpenAI Sam Altman 的整个公变."
- **OpenClaw's two key changes**: (1) **form** — runs inside instant-messaging surfaces (WhatsApp etc.), has its own environment, always-on 24/7; (2) **permissions** — earlier academic + corporate agents were cautious about scope/permissions because "agents acting like humans is dangerous and produces harmful behavior"; OpenClaw "就不管这些 permission 这些 safety，反正所有东西都给我打开，agent 想干嘛就干嘛."
- **Being open-source enables the YOLO permission stance**: closed-source products doing the same "会出大问题."
- **Second-order impact already visible**: Anthropic's Claude Code is absorbing OpenClaw features; OpenAI is pulling experimental projects to focus on agent + productivity coding; Jensen Huang now says "every enterprise needs a Cloud Strategy"; Chinese big-tech is moving fast; recent layoff waves are tied to changing perceptions of agent capability.

### §9 — US-China diffusion patterns differ; China is more "全民化" at the application layer  [[55:10 → 1:02:05](https://youtu.be/Xxz5uh0L1mE?t=3310)]
- **Breadth contrast**: OpenClaw is hotter in China than in the US but in different ways. US adoption stays mostly within developer/tech circles ("an open-source project"); China is "全民化" — local governments push it, the narrative is "an era-defining industry opportunity" or "an individual upliftment tool" with a "you'll be left behind if you don't learn it" undertone.
- **Vivid concrete example**: Chinese seniors whose adult children are too busy show up at offline events with laptops asking for help installing OpenClaw — sometimes paying to install, then paying to uninstall when it didn't help. Su Yu sees the net effect as positive even with waste.
- **Application-layer speed is China's AI-era edge** — citing Eric Schmidt (ASR: "Eric Schmitz"): the US has always been slow at the application layer; China moves much faster on front-end tech adoption. Now that base-model intelligence has crossed a "good-enough" threshold, lots of things that previously weren't worth doing have become worth doing — and whoever finds and seizes that value wins.
- **Agent researchers' responsibility**: OpenClaw's barrier-to-entry is still high; most people can't extract value from it. The job of researchers is to make agents **truly usable and accessible** so every individual with a unique insight can convert it into value — "技术民主化," avoiding the alternative of "core tech gated by frontier labs." Su Yu is especially attuned to this watching layoff news: "如果 job displacement 的速度远超新工作机会产生的速度，社会会出问题."
- **General vs specialized**: A "single entry point to the entire digital world" agent indeed suits model companies. But that's not the only opportunity. "这个世界不是一个世界，它是由可能几百万个小世界组成的." Each micro-world needs **specialization** to produce real value. Big model companies' organizational structure and business model push them toward platform/unified products — "即使他们选择去做这个可能也做不好." Big opportunity for non-model-companies and individuals.

### §10 — Neocognition: $40M seed, "specialization over AGI"  [[1:02:05 → 1:20:30](https://youtu.be/Xxz5uh0L1mE?t=3725)]
- **Company name**: **NeoCognition** — "Neo" from **neocortex** (new cortex). Positioned as an **Agent research lab**, long-term scope covers all intelligence-related problems. Su Yu is on leave from his professorship.
- **Thesis: specializing intelligence > general intelligence.** General intelligence is already "good enough" — drop a digital-world problem on Claude Code / OpenClaw / Project Computer and you get ~60-70% success. The next stage of differentiation comes from specialization, not raw capability.
- **Worldview**: "这个世界实际上是由几百万个小世界组成的" — each profession / domain / company / software / website is a micro-world; their summed entropy is "几乎无限." No single agent / model captures it all — adaptation + specialization is mandatory.
- **Approach is horizontal, naturally enterprise-leaning** (B2B demands more depth from agents). Observes the **"SaaSpocalypse"**: SaaS valuation logic is breaking, margins compressed by agents, software companies pivoting from "tool" to "labor market" — delivering results (AI employees) on their own platforms rather than selling tools.
- **Funding: $40M seed (4000 万美金) — large for a seed.** Founded mid-2025 → seed closed March 2026 (~6 months). Su Yu calls it "比较顺利、比较幸运." Macro observation: **the US market is sharply bifurcated** — frontier neo-labs raise tens of millions to billions per round; OpenAI + Anthropic alone account for 30-50% of total market fundraising. Mid-tier VCs are struggling — either becoming megafunds (A16Z, Lightspeed) or boutique vertical firms.
- **What investors care about now**: track record at frontier labs (different valuation logic), thesis differentiation, team being "one of the best," commercial scale, and whether OpenAI/Anthropic can easily copy. Su Yu's response to "why can't the big labs do this?" — pick a track with **unbounded upside but high uncertainty** (analogous to robotics: VLA / world-model / hardware-led paths all live), which mathematically accommodates multiple players and tech directions.
- **Technical bet: world model — but defined far more broadly than vision JEPA.** Su Yu rejects narrowing world models to next-frame video prediction / 3D reconstruction / Yann-LeCun-style latent planning. His definition: **the world model an intern builds on their first day on the job** — surface org chart vs. real org chart, who actually decides, how to find the right approver, which software does what, internal workflows, theory-of-mind for colleagues. That's a **micro-world specialized world model**, partially symbolic, almost entirely non-video.
- **Target product**: not one expert agent (HR / Finance / Legal) but **a continual-learning method** that produces an expert agent for any given domain / role / environment. Critiques both current paradigms: (1) model-lab RL post-training in synthetic RL gyms is "天壤之别" from how humans actually continually learn; (2) OpenClaw / Claude Code-style **non-parametric learning** (SOUL.md / SKILL.md / harness / meta-harness / auto-harness) — Su Yu has long been bullish on this direction but believes its ceiling is limited. Neither paradigm has solved the problem yet.

### §11 — Continual Learning, World Models, GUI vs CLI  [[1:20:30 → 1:44:34](https://youtu.be/Xxz5uh0L1mE?t=4830)] — *the longest section*
- **"Continual Learning" is being used too loosely.** Catastrophic-forgetting setting, personalization, recursive self-improvement, OpenClaw-style adaptation, RL post-training — all called continual learning. Su Yu cares less about *how* you learn and more about *what* you learn: **the target should be a world model**. Continual learning and world model "本质上是一件事情."
- **Why "Neocognition" (Neo = neocortex)**: neocortex is ~70% of the human brain, evolved over a very short time (~200M years, only in mammals), but supports vision, language, hearing, logic, planning. Short evolution time + high functional load forced evolution to find a single **general enough learning machinery** that gets repeated many times.
- **Cortical column + Jeff Hawkins's "Thousand Brains" theory**: ~150k cortical columns of highly similar structure. Su Yu cites **Hawkins's *A Thousand Brains***: each cortical column learns a **world model**, and these world models are **not restricted to the physical world** — language, mathematics, democracy, rule-of-law are all world models. Each concept has multiple redundant world-model representations across columns.
- **Cognitive maps / conceptual framework — well-established in cognitive science, far from solved in AI**. Even setting aside Hawkins's specific theory, human continual learning produces cognitive maps. AI agents are nowhere near this — central problem his startup is targeting.
- **Language vs Vision in the neocortex**: vision occupies the largest area, but **language is what makes humans unique**. Aligned with Chris Manning: "我们的视力大概率是不如大猩猩的 … 视力不是最好的，听力不是最好的 … 但我们的语言是独一无二的，而这是导致我们这些文明和 intelligence 这么不同的根本原因."
- **Terence Deacon's *The Symbolic Species* — symbol-brain co-evolution**: Homo → Homo sapiens jump wasn't gradual; symbolic ability created new (cultural) selection pressure, "自己开了一局新的游戏." Language crosses both space (the present moment) and time (across generations), generating fast selection pressure.
- **MIT Fedorenko's fMRI work — language ≠ thought**: Wernicke / Broca don't light up during reasoning tasks, suggesting individual thought doesn't strictly need language. Su Yu's rebuttal: **language is the scaffolding** for learning — the hippocampus converts short-term to long-term memory during sleep, which is internalization. "Individual thought doesn't need language, but **civilization needs language**."
- **"Language Agent" the term will eventually become redundant.** Even if future base models are world models rather than LLMs, agents will always have language as a fundamental capability. Language Agent is a transitional name — eventually "我就叫它 agent 就好了."
- **GUI won't disappear — humans are visual animals.** HCI research: visualizing the same information lets the brain understand it ~0.X seconds faster. GUI also serves validation, trust-winning, and auditing. For agents, the question becomes "should agents use GUIs?" — Su Yu's answer leans yes.
- **GUI encodes accumulated knowledge / constraints / business logic.** 99% of the digital world already has a GUI; the GUI has domain knowledge baked into its design. Agents that use GUI well can **piggyback on all this accumulated knowledge** without re-implementing every CLI/API. That's the only way to "immediately reach all corners of human society" — especially the long tail.
- **Text is 1D; vision is 2D+** — higher-dimensional representations are more efficient for complex structures. Another fundamental advantage of GUI for agents.
- **MCP / CLI replacing GUI globally? Compare to Semantic Web.** Tim Berners-Lee's Semantic Web (first-order logic / Description Logic on the whole internet) was pushed for 20+ years and adoption remained near-zero. Cause wasn't technical — it's human nature and how society runs: you can't publish one standard (like MCP) and expect the whole world to rewrite their systems within a few years. Large banks still run decades-old COBOL. GUI's adoption succeeded because **"GUI suits humans."**
- **Global optimum ≠ local optimum** — even if CLI is globally optimal for agents, many local solutions are already "good enough" with no incentive to migrate. Conclusion: "我不觉得 CLI 会全面取代 GUI."

### §12 — Biggest Agent bottleneck right now? 2026 expectations?  [[1:44:34 → 1:47:09](https://youtu.be/Xxz5uh0L1mE?t=6274)]
- **The bottleneck isn't a single capability — it's a coupled problem set.** Memory, self-learning, continual learning, world model, specialization "其实所有这些东西都是同一件事情." Their joint failure shows up as deficient **reliability + speed + cost-effectiveness**.
- **Unifying three-layer framing**: continual-learning / self-learning are the *mechanism*; world model is the *content* being learned; specialization / becoming-expert-agent is the *outcome*. "条条大路通罗马."
- **2026 prediction**: **continual learning / self-learning will be 2026's main frontier theme.** World-model-based continual learning is "one of the best" paths but expect multiple parallel bets — "这也是有意思的地方."
- **Parallel second axis: societal diffusion speed.** How fast can the tech reach more corners of society? Directly tied to reliability/speed/cost.
- **Evidence the entry barrier is still too high**: OpenAI and Anthropic now run "partner tier" (ASR: "patent tier") models where they hire armies of **Forward Deployed Engineers** to embed at customer sites and build agents — itself a direct symptom of unresolved reliability/speed/cost problems.

### §13 — What are the big labs betting on?  [[1:47:09 → 1:52:47](https://youtu.be/Xxz5uh0L1mE?t=6429)]
- **2026 = year of convergence.** A year ago labs were betting on genuinely different things; Anthropic's "一家独大" "给大家打了个样" and pulled everyone toward similar productivity-focused agents. But "还是会有一些新的 bets 出来."
- **Anthropic / OpenAI**: productivity-everything; OpenAI "now also converging this direction."
- **Google**: strong base model + best ecosystem, but adoption / momentum somehow lagging — "可能里面有些更深层次的东西我没有看清楚."
- **xAI / Elon — "Macrohard"** (Su Yu's "Microsoft translation" line about Musk's new org). Bet: replace all software, do all knowledge work via computer-use agents. Technical path expected to mirror Tesla FSD — "一个比较偏小的模型，视觉这种 video 为主，去直接做 end-to-end modeling." "At least is a different bet, I don't know if it can succeed, good luck." xAI's internal turmoil is hurting execution.
- **Jeff Bezos — Project Prometheus (普罗米修斯)**, ~**$6-7B raised**, Bezos back as co-CEO. Large computer-use agent component but ultimately aimed at his home turf: manufacturing, logistics, infrastructure, factories.
- **"Computer-use agent" is no longer narrow** — whether agents drive GUI / CLI / tools, "computer-use agent 现在慢慢变成一个 general digital agent."
- **China — ByteDance and Zhipu by name.** ByteDance's **UI-TARS** series (ASR: "UiTorch") and 豆包手机; Zhipu (智谱) started computer-use early with **AutoGLM**. Su Yu has personal ties via Tang Jie (唐杰) at Tsinghua — after Mind2Web in summer 2023, he gave a talk that led to a joint **AgentBench** paper, "算是 agent 最早的 benchmark 之一."
- **Post-OpenClaw, everyone needs a "Cloud Strategy"** — directly echoing Jensen Huang's line.
- **Safety ≈ capability problem (segue into §14)**: agents are like interns — easy to make safety mistakes; "一个老师傅就不会." But genuine **security** (worst-case adversarial) is different and needs dedicated methods.

### §14 — Our generation lived the full Agent cycle; I love building conceptual frameworks  [[1:52:47 → 2:10:13](https://youtu.be/Xxz5uh0L1mE?t=6767)]
- **Why researchers are leaving academia for startups now**: In early Agent (proof-of-concept) days, a clever low-cost idea was enough — well-suited to a university lab. From 2025 on, interesting Agent ideas demand serious resources (money, GPUs, API spend, a fast-iterating team) — a profile fundamentally mismatched with academia's DNA. That, more than salary, is why so many professors are spinning out.
- **His own academia → startup path follows the same logic**: he originally left a Microsoft job paying 3-4× academic salary because he is "interest-rich" — at any moment ~10 ideas in parallel; only academia let him chase all of them. Today the bottleneck has shifted from ideas to deployment, so the optimal environment changed.
- **Intellectual style — conceptual-framework builder.** Explicitly not fast-thinking, not blessed with great memory. Edge is breadth + synthesis: absorb material across domains, string it together, see the connections. Startup is "a new environment full of new stimuli" that lets him keep expanding the framework.
- **Childhood origin of the habit**: atypical "good student" — would sneak out at 3am to play games at internet cafes, then attend school at 7am. The constant was voracious reading of anything paper in the house (history, politics, 故事会, 言情小说). **"读书它本身就是一个构造世界的过程"** — building worlds in your head from text, then building relations between those worlds.
- **Personality — "魂不灵" (hún bù líng), self-coined.** Two parts: (1) doesn't fixate on wanting things — sleeps and eats fine even when something is unresolved; (2) but when he decides to pursue something, "I put my mind to it, I put my effort to it" — high confidence he'll get it. Track record: Hunan small-county kid, top of his school, first/second cohort of Tsinghua 自主招生 (-30 points), top-10 in Hunan gaokao, full scholarship.
- **Why a startup was inevitable for him**: never agonized over the decision. At sufficient depth, Agent research and production become inseparable — the biggest learning signal for next-gen Agents comes from continual learning in deployed environments, and universities cannot deploy at scale. The only question was *when* and *what direction*, not *whether*. Waited until 2025 because the underlying tech (tool use, coding ability, multimodal) and his own understanding of where Agent bottlenecks actually live finally felt "ready."
- **Predictions for Agent's future**:
  - **Technical**: continual learning solved within a few years; agents penetrate every corner of society; production relations across industries restructured.
  - **Existential risk**: discounted near-term. The Singularity scenario isn't an intelligence problem — it requires **innate goals, intention, survival pressure**, and there's no known mechanism to inject those into current AI. All current AI objectives are externally given.
  - **Real concern — job displacement + concentration of returns**: agents take over knowledge work without (a) enough new jobs being created or (b) a redistribution mechanism; gains accrue to a few frontier labs and capital pools. Severe societal impact.
  - **His response — democratize access to frontier agent capabilities**: lower the bar so any individual with a good idea + unique insight can quickly build an agent and monetize it without frontier-lab-scale capital. Framed as every AI researcher's responsibility.

### §15 — Final rapid-fire close  [[2:10:13 → 2:17:49](https://youtu.be/Xxz5uh0L1mE?t=7813)]
- **Favorite food**: 火锅 (hotpot).
- **Must-read #1 — *A Brief History of Intelligence***: bought day-of (2023), recommended to every student. "AI + Evolution + Neuroscience 最好的、最通俗易懂的书." A chapter titled *Mice in the Imaginarium* directly inspired Su Yu's own paper *LRMs in the Imaginarium / learning tools through simulated trial and error*.
- **Must-read #2 — *A Thousand Brains*** (Jeff Hawkins): "a very bold theory of how the brain works"; evidence still thin but "based on my current reading, this theory makes a lot of sense."
- **Personal canon of AI papers** (the historian's tour): 1940s McCulloch-Pitts neuron paper → Turing's work → Hinton's backpropagation (popularizer if not first formulator) → **AlexNet (2012)** "the Renaissance of neural networks" → **word2vec (2013)** brought NN into NLP "when the whole NLP field looked down on neural nets" (also the year his PhD started) → **BERT (2018)** "the first foundation model for language that actually worked at scale" → **Transformer (2017)** → **attention (two 2014 papers — seq2seq + NYU's MT paper)** → **residual / shortcut** (from ResNet, ultimately highway networks) → **GPT / ChatGPT** "deep influence beyond just AI" → **Chain-of-Thought**. "These are all connected."
- **Current key bet**: "All the way continue learning, all the way world modeling."
- **On Xiaojun's podcast**: listened to many — more in the past, less since starting the company. Halfway through the **7-hour Sai Ning episode** ("listened twice, still haven't finished"); also caught the Yang Zhilin and Yao Shunyu episodes plus the entrepreneurship ones.
- **On the studio name 语言即世界 (Language Is World)**: "That's the truth. That's my belief."
- **Closing disclaimer** (asked Xiaojun whether to keep this in): "My personal experience is in no way meant to mislead anyone — especially young people. Skipping class to play games, or playing games at high intensity, is not something to be encouraged."

## Notable quotes

> "Agent 首先它应该是 Entity，就是它是一个实体，它有它的 Boundary [...] 然后它是需要在外界环境，在某一种环境当中去工作 [...] 第三个要素是它在这个环境工作，并不是在那随机的游荡 [...] 它是要去进行叫做 Goal Directed Activities，就是它是有目的性的。"
> — Su Yu, §3 [[04:07](https://youtu.be/Xxz5uh0L1mE?t=247)] — *the three-criteria Agent definition*

> "为了达成这样的目标，它需要至少两项广义上的能力，我叫做一项是 Memory，第二项是 Autonomy [...] 你的 Memory 是你的 Autonomy 的整个的基础。"
> — Su Yu, §3 [[08:18](https://youtu.be/Xxz5uh0L1mE?t=498)] — *the Memory + Autonomy framework that anchors the whole episode*

> "At the end of the day, 大家想要的就是一个 universal digital agent — 就是一个可以在 digital world 里面做人能做到所有事情, 甚至做得更好的这样的一个 agent."
> — Su Yu, §6 [[41:35](https://youtu.be/Xxz5uh0L1mE?t=2495)]

> "Let machines understand human thinking, don't let humans think like machines. 就是让机器去理解人的语言、人的想法,而不是让人去像机器一样思考。"
> — Su Yu, §7 [[47:57](https://youtu.be/Xxz5uh0L1mE?t=2877)] — *his PhD-defense manifesto, still the through-line from Semantic Parsing to Agent*

> "它底层的技术实际上是没有太大的变化的，更多是一个交互形式的变化。但这个交互形式的变化，反而是整个事情的导火索一样。"
> — Su Yu, §8 [[50:15](https://youtu.be/Xxz5uh0L1mE?t=3015)] — *on the ChatGPT ≅ OpenClaw structural isomorphism*

> "这个世界实际上是由几百万个小世界组成的 [...] 每一个职业、每一个 domain、每一个 profession、到每一个公司、甚至到每一个环境 [...] 它其实都是自己的一个小世界，而这些世界加起来的 entropy 是几乎无限的。"
> — Su Yu, §10 [[1:03:50](https://youtu.be/Xxz5uh0L1mE?t=3830)] — *Neocognition's specialization-thesis foundation*

> "Individual thought doesn't need language, 但是 civilization needs language."
> — Su Yu, §11 [[1:36:13](https://youtu.be/Xxz5uh0L1mE?t=5773)] — *answering the Fedorenko fMRI dissociation*

> "All the way continue learning, all the way world modeling."
> — Su Yu, §15 [[2:15:31](https://youtu.be/Xxz5uh0L1mE?t=8131)] — *his one-line current bet*

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| Zhang Xiaojun (张小珺) | Host | [[00:01](https://youtu.be/Xxz5uh0L1mE?t=1)] |
| Fu Li (福利) | Guest in earlier AI-evolution episode | [[00:06](https://youtu.be/Xxz5uh0L1mE?t=6)] |
| Guang Mi (广密) | Guest in earlier AI-evolution episode | [[00:06](https://youtu.be/Xxz5uh0L1mE?t=6)] |
| Su Yu (苏煜) | Guest — OSU NLP professor, Neocognition founder, 2025 Sloan | [[00:30](https://youtu.be/Xxz5uh0L1mE?t=30)] |
| Peter Norvig | Co-author *AIMA* (1st ed. ~1995) | [[12:02](https://youtu.be/Xxz5uh0L1mE?t=722)] |
| Stuart Russell | Co-author *AIMA*; told Su Yu the book is "fundamentally about agents" | [[12:12](https://youtu.be/Xxz5uh0L1mE?t=732)] |
| Demis Hassabis | DeepMind lead; cited as the reason games became the canonical Deep RL environment | [[17:42](https://youtu.be/Xxz5uh0L1mE?t=1062)] |
| Percy Liang | Stanford, semantic-parsing alum → LLM/Agent leader | [[20:39](https://youtu.be/Xxz5uh0L1mE?t=1239)] |
| Luke Zettlemoyer | UW; ELMo at AI2, RoBERTa at Meta; Toolformer co-lead | [[20:41](https://youtu.be/Xxz5uh0L1mE?t=1241)] |
| Tao Yu (于涛) | HKU, semantic-parsing alum; OSWorld | [[21:03](https://youtu.be/Xxz5uh0L1mE?t=1263)] |
| Huan Sun (孙欢) | OSU colleague, semantic-parsing alum | [[21:08](https://youtu.be/Xxz5uh0L1mE?t=1268)] |
| Yang Di (杨迪) | Co-author of 2024 Language Agent tutorial | [[21:50](https://youtu.be/Xxz5uh0L1mE?t=1310)] |
| Yao Shunyu (姚顺宇) | ReAct author; co-author of 2024 Language Agent tutorial | [[21:50](https://youtu.be/Xxz5uh0L1mE?t=1310)] |
| Satya Nadella | Microsoft CEO; circulated Toolformer company-wide | §5 [[32:30](https://youtu.be/Xxz5uh0L1mE?t=1950)] |
| Graham Neubig | CMU; WebArena team | §5 [[33:00](https://youtu.be/Xxz5uh0L1mE?t=1980)] |
| Dario Amodei | Anthropic CEO; credited for "coding-is-foundational" thesis | §6 [[42:20](https://youtu.be/Xxz5uh0L1mE?t=2540)] |
| Sam Altman | OpenAI; mentioned in ChatGPT-aftermath / 公变 context | §8 [[52:00](https://youtu.be/Xxz5uh0L1mE?t=3120)] |
| Ilya Sutskever | OpenAI; mentioned re fundamental-research vs applied tension | §8 [[51:40](https://youtu.be/Xxz5uh0L1mE?t=3100)] |
| Jensen Huang (黄仁勋) | NVIDIA; "every enterprise needs a Cloud Strategy" | §8 [[54:40](https://youtu.be/Xxz5uh0L1mE?t=3280)] |
| Eric Schmidt (口播: "Eric Schmitz") | Ex-Google CEO; cited on US application-layer slowness | §9 [[57:00](https://youtu.be/Xxz5uh0L1mE?t=3420)] |
| Yann LeCun | Meta / JEPA — vision-based world-model paradigm Su Yu pushes back on | §10 [[1:14:00](https://youtu.be/Xxz5uh0L1mE?t=4440)] |
| Jeff Hawkins | Author of *A Thousand Brains* | §11 [[1:25:00](https://youtu.be/Xxz5uh0L1mE?t=5100)] |
| Chris Manning | Stanford NLP; co-signed "language is what makes humans unique" | §11 [[1:29:00](https://youtu.be/Xxz5uh0L1mE?t=5340)] |
| Terence Deacon | Author of *The Symbolic Species* — symbol-brain co-evolution | §11 [[1:32:00](https://youtu.be/Xxz5uh0L1mE?t=5520)] |
| Fedorenko (MIT) | fMRI work on language ≠ thought dissociation | §11 [[1:35:00](https://youtu.be/Xxz5uh0L1mE?t=5700)] |
| Tim Berners-Lee | Semantic Web; cautionary analogue for "MCP / CLI replacing GUI" | §11 [[1:42:26](https://youtu.be/Xxz5uh0L1mE?t=6146)] |
| Elon Musk | xAI / "Macrohard"; computer-use agent as one of his biggest bets | §13 [[1:48:26](https://youtu.be/Xxz5uh0L1mE?t=6506)] |
| Jeff Bezos | Project Prometheus, ~$6-7B; back as co-CEO | §13 [[1:49:48](https://youtu.be/Xxz5uh0L1mE?t=6588)] |
| Tang Jie (唐杰) | Tsinghua; Zhipu / AutoGLM connection; AgentBench collaborator | §13 [[1:51:00](https://youtu.be/Xxz5uh0L1mE?t=6660)] |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| *Artificial Intelligence: A Modern Approach* (Russell & Norvig, 1995) | Canonical AI textbook; Russell told Su Yu it's "fundamentally about agents" | [[11:49](https://youtu.be/Xxz5uh0L1mE?t=709)] |
| Backpropagation (~1985 popularization) | Neural-net foundation | [[13:03](https://youtu.be/Xxz5uh0L1mE?t=783)] |
| AlphaGo (DeepMind) | Neural Agent benchmark | [[13:34](https://youtu.be/Xxz5uh0L1mE?t=814)] |
| Atari Deep RL | Neural Agent canonical task | [[13:43](https://youtu.be/Xxz5uh0L1mE?t=823)] |
| Dota / StarCraft AI | Neural Agent extensions | [[13:49](https://youtu.be/Xxz5uh0L1mE?t=829)] |
| ELMo (Zettlemoyer et al., AI2) | Pre-GPT LM that influenced GPT | [[20:52](https://youtu.be/Xxz5uh0L1mE?t=1252)] |
| RoBERTa (Zettlemoyer team, Meta) | Better-trained BERT | [[20:46](https://youtu.be/Xxz5uh0L1mE?t=1246)] |
| 2024 Language Agent tutorial (Su Yu, Yang Di, Yao Shunyu, Tao Yu) | Defines the term | [[21:50](https://youtu.be/Xxz5uh0L1mE?t=1310)] |
| Chain-of-Thought (CoT, early 2022) | Adaptive computing via language; Language Agent era starting marker | §5 [[30:00](https://youtu.be/Xxz5uh0L1mE?t=1800)] |
| ReAct (Yao Shunyu, ~Oct 2022) | Perceive-reason-action loop | §5 [[30:30](https://youtu.be/Xxz5uh0L1mE?t=1830)] |
| LLM-Planner (OSU, late 2022) | Among first to use LLMs for embodied planning | §5 [[31:30](https://youtu.be/Xxz5uh0L1mE?t=1890)] |
| Mind2Web (OSU, 2022-10 start, early 2023 release) | First LLM-based generalist web agent | §5 [[32:00](https://youtu.be/Xxz5uh0L1mE?t=1920)] |
| SayCan (Google, ASR: "SECAN") | LLM-for-robot-planning | §5 [[32:00](https://youtu.be/Xxz5uh0L1mE?t=1920)] |
| Toolformer (Meta, 2023-02) | First LLM tool-use; Satya Nadella circulated it | §5 [[33:00](https://youtu.be/Xxz5uh0L1mE?t=1980)] |
| AutoGPT (2023-03) | GitHub's fastest star growth ever | §5 [[34:02](https://youtu.be/Xxz5uh0L1mE?t=2042)] |
| MMMU (OSU) | First multimodal LLM benchmark | §5 [[36:00](https://youtu.be/Xxz5uh0L1mE?t=2160)] |
| SeeAct (OSU, ASR: "CACT") | GPT-4V-based multimodal web agent | §5 [[36:00](https://youtu.be/Xxz5uh0L1mE?t=2160)] |
| WebArena (CMU, 2023-07) | RL-environment style web-agent reproducibility | §5 [[33:30](https://youtu.be/Xxz5uh0L1mE?t=2010)] |
| OSWorld (Yutao group, 2024-03/04) | Desktop agent | §5 [[37:00](https://youtu.be/Xxz5uh0L1mE?t=2220)] |
| SWE-bench (~2023 H2) | Drove coding agents | §5 [[37:00](https://youtu.be/Xxz5uh0L1mE?t=2220)] |
| UGround (OSU, 2024 H2, ASR: "youground") | Visual perception + pixel-level actions embodiment | §5 [[37:30](https://youtu.be/Xxz5uh0L1mE?t=2250)] |
| Claude Computer Use (2024-10) | Anthropic's adoption of pixel-level embodiment | §5 [[38:00](https://youtu.be/Xxz5uh0L1mE?t=2280)] |
| Claude Code | Cursor-driven Anthropic shift; exploded after Opus 4.5 | §5 [[39:00](https://youtu.be/Xxz5uh0L1mE?t=2340)] |
| OpenAI Operator / ChatGPT Agents (2025) | Adopt UGround-style embodiment | §5 [[38:30](https://youtu.be/Xxz5uh0L1mE?t=2310)] |
| OpenClaw (2025-11) | Watershed Agent moment; peaked ~2026-02 | §5 [[40:00](https://youtu.be/Xxz5uh0L1mE?t=2400)] |
| AgentBench (OSU + 唐杰 group joint) | "One of the earliest agent benchmarks" | §13 [[1:51:25](https://youtu.be/Xxz5uh0L1mE?t=6685)] |
| UI-TARS series (ByteDance, ASR: "UiTorch") | China computer-use agent line | §13 [[1:50:33](https://youtu.be/Xxz5uh0L1mE?t=6633)] |
| AutoGLM series (Zhipu) | Early Chinese computer-use agent | §13 [[1:51:01](https://youtu.be/Xxz5uh0L1mE?t=6661)] |
| *A Brief History of Intelligence* (2023) | Su Yu's #1 recommended book | §15 [[2:10:17](https://youtu.be/Xxz5uh0L1mE?t=7817)] |
| *Mice in the Imaginarium* (chapter) | Directly inspired Su Yu's paper *LRMs in the Imaginarium* | §15 [[2:11:08](https://youtu.be/Xxz5uh0L1mE?t=7868)] |
| *A Thousand Brains* (Jeff Hawkins) | Cortical-column theory; central to §11 | §11 [[1:25:00](https://youtu.be/Xxz5uh0L1mE?t=5100)] |
| *The Symbolic Species* (Terence Deacon) | Symbol-brain co-evolution | §11 [[1:32:00](https://youtu.be/Xxz5uh0L1mE?t=5520)] |
| McCulloch-Pitts neuron (1940s) | Su Yu's "personal canon" #1 | §15 [[2:12:08](https://youtu.be/Xxz5uh0L1mE?t=7928)] |
| AlexNet (2012) | "Renaissance of neural networks" | §15 [[2:13:00](https://youtu.be/Xxz5uh0L1mE?t=7980)] |
| word2vec (2013) | NN's entry into NLP | §15 [[2:13:20](https://youtu.be/Xxz5uh0L1mE?t=8000)] |
| BERT (2018) | "First foundation model for language at scale" | §15 [[2:13:50](https://youtu.be/Xxz5uh0L1mE?t=8030)] |
| Transformer (2017) | "Not from nowhere — components from earlier work" | §15 [[2:14:18](https://youtu.be/Xxz5uh0L1mE?t=8058)] |
| Attention (two 2014 papers — seq2seq + NYU MT) | Pre-Transformer attention | §15 [[2:14:00](https://youtu.be/Xxz5uh0L1mE?t=8040)] |
| ResNet / Highway Networks | Residual / shortcut connections | §15 [[2:14:30](https://youtu.be/Xxz5uh0L1mE?t=8070)] |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| Logical Agent era ~1950s-1990s | [Guest] | [[06:30](https://youtu.be/Xxz5uh0L1mE?t=390)] |
| AI Winter triggered by expert systems: 80s-90s | [Guest] | [[07:26](https://youtu.be/Xxz5uh0L1mE?t=446)] |
| AIMA 1st ed: ~1995 | [Guest] | [[12:02](https://youtu.be/Xxz5uh0L1mE?t=722)] |
| Backprop popularized: ~1985 | [Guest] | [[13:03](https://youtu.be/Xxz5uh0L1mE?t=783)] |
| Neural Agent scale: 数十million-100M params (≈1亿) | [Guest] | [[14:11](https://youtu.be/Xxz5uh0L1mE?t=851)] |
| Atari training: 几百万盘 plays per game | [Guest] | [[17:19](https://youtu.be/Xxz5uh0L1mE?t=1039)] |
| Symbolic expression: ~100,000 years | [Guest] | [[26:39](https://youtu.be/Xxz5uh0L1mE?t=1599)] |
| Written language: ~5,000-6,000 years | [Guest] | [[26:54](https://youtu.be/Xxz5uh0L1mE?t=1614)] |
| AutoGPT GitHub stars: 100k at peak (2023), ~180k now | [Guest] | [[34:02](https://youtu.be/Xxz5uh0L1mE?t=2042)] |
| Toolformer release: 2023-02 | [Guest] | §5 |
| WebArena release: 2023-07 | [Guest] | §5 |
| OSWorld release: 2024-03/04 | [Guest] | §5 |
| Claude Computer Use release: 2024-10 | [Guest] | §5 |
| OpenClaw release: 2025-11; peaked ~2026-02 | [Guest] | §5 |
| Neocognition seed: $40M (4000万美金) | [Guest] | [[1:07:08](https://youtu.be/Xxz5uh0L1mE?t=4028)] |
| Neocognition founded 2025-07/08 → seed closed 2026-03 (~6 months) | [Guest] | §10 |
| General agent success rate (Claude Code/OpenClaw/Project Computer): ~60-70% | [Guest] | [[1:02:54](https://youtu.be/Xxz5uh0L1mE?t=3774)] |
| OpenAI + Anthropic ≈ 30-50% of total market fundraising | [Guest] | §10 |
| Neocortex: ~70% of human brain | [Guest] | §11 |
| Cortical columns: ~150,000 | [Guest] | §11 |
| Mammalian evolution: ~200M+ years | [Guest] | §11 |
| Project Prometheus (Bezos): ~$6-7B raised | [Guest] | [[1:49:48](https://youtu.be/Xxz5uh0L1mE?t=6588)] |
| Sai Ning episode: 7 hours; listened ~2x | [Guest] | §15 |

## Open questions / gaps

- **OpenClaw Moment is defined by analogy but not on its own terms** at the time of first mention (§1). Resolved with structural analogue in §8 (interaction-form change + YOLO open permissions + open-source-enables-YOLO), but the underlying technical content is consistently downplayed as "nothing is new here" — the wiki reader should keep skepticism that interaction-form-only narratives have been overclaimed before.
- **"GPT was in large part inspired by ELMo / Zettlemoyer's earlier work"** asserted without specific citation chain (§3).
- **"Hassabis's personal love of games drove Deep RL's training environment"** is Su Yu's guess — he flags this himself: "我没有去真的做过那种 agent，所以我的答案可能不一定非常的完备" (§3).
- **"LLM training compresses into a world model"** (anti-stochastic-parrot view) asserted; the mechanism by which compression *yields* a world model isn't argued (§3).
- **Bio-evolution-to-AI-timeline analogy**: evocative; the causal claim ("language is the inflection variable in both") is asserted, not demonstrated (§3).
- **§5 historical timeline checks**: (a) Su Yu's reference to the "AI Engineer" precursor → Lovable lineage may conflate it with the GPT Engineer / Anton Osika lineage. (b) WebArena vs Mind2Web timing: Su Yu says ~1 month gap, but Mind2Web shipped early 2023 and WebArena was 2023-07 (~6 months) — likely a verbal slip.
- **Reliability / speed / cost** are stated as a triple problem (§12) but no ordering is given for which will break first.
- **2026 "different bets" beyond world-model continual learning** are gestured at but not enumerated (§12).
- **Job displacement framing in §9/§14 assumes** the bottleneck is barrier-to-entry and that "democratizing access to frontier agents" mitigates the impact; whether this actually balances displacement vs. new-job creation is an empirical claim not defended.
- **"魂不灵" rendered as transcribed** in §14; Su Yu uses it as a self-coined positive label, not the standard Mandarin sense of "dull-witted."

## Verification log

- **Sectioning**: chapters (15 author-supplied YouTube chapters); chapter #1 was *<Untitled Chapter 1>* and was renamed to "开场白与本期定位" (opening + episode framing) in the section JSON before chunking. All other 14 titles preserved verbatim from `info.json.chapters`.
- **Transcript source**: faster-whisper large-v3 (CPU int8, local M4) — produced by `docs/videos/transcripts/Xxz5uh0L1mE.txt` from the project's batch driver. YouTube provided no subtitles (manual or auto) for this video, so the standard yt-dlp path was unavailable.
- **Speaker name corrections**: guest's name was consistently rendered "苏玉" in ASR but the YouTube `info.json` title carries "苏煜" (Su Yu) — corrected throughout. Host name corrected from "小俊"/"小骏" to "张小珺" (Zhang Xiaojun) per channel metadata.
- **Sections covered**: 15/15 ✅
- **Notable quotes traced verbatim**: 8/8 ✅ (each anchored by a distinctive 6-15-char Chinese substring in the local transcript)
- **Numbers traced**: 24/24 ✅
- **Sectioning method used**: chapters
- **Removed during verification**: none

## See Also

- `docs/videos/2026-05-11-yao-shunyu-training-claude-gemini.md` — Zhang Xiaojun's episode with Yao Shunyu (the ReAct author Su Yu name-checks here); same host, same channel, complementary perspective from inside Anthropic / GDM rather than from academia → startup.
- `skills/research/youtube-wiki/SKILL.md` — the pipeline that produced this entry.
