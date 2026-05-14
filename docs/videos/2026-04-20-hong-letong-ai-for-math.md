# Hong Letong: AI for Math, Turning Mathematics into Lean, Proofs from THE BOOK, the Most Unlikely Founder

**Source**: https://youtu.be/bv8ghyTFF9w
**Channel**: Zhang Xiaojun Podcast (张小珺商业访谈录)
**Published**: 2026-04-20
**Duration**: 04:24:17
**Watched on**: 2026-05-14
**Sectioning**: chapters (15 YouTube-supplied chapters; chapter 1 was `<Untitled Chapter 1>`, renamed to "开场白与本期定位")
**Detected video language**: `zh` (no `info.json.language`; both `subtitles` and `automatic_captions` keys empty — no YouTube subs)
**Transcript source**: faster-whisper large-v3 (CPU int8) — produced by `docs/videos/transcribe_batch.py` from local audio. ASR consistently rendered the host as "小俊/小骏" — corrected from channel metadata to **张小珺 (Zhang Xiaojun)**. The guest's name was rendered "洪乐彤" / "Hong Letong" — corrected throughout to **洪乐潼 (Hong Letong)** per the Axiom team page; company name was rendered "Oxfam" / "Action" — corrected to **Axiom**; "AXIOM Prover" appears as "action prover" in ASR; "Verina" benchmark appears as "Varina"; "DeepSeek-Prover" as "DeepSeq Prover"; "Putnam" as "普特兰".
**Speakers**: Zhang Xiaojun (张小珺, host), Hong Letong (洪乐潼, guest — 00后 founder & CEO of Axiom, an AI-for-Math startup; Stanford math PhD + JD candidate; MIT '23 Morgan Prize)

## TL;DR

- A 4-hour interview at Facebook House with **Hong Letong (洪乐潼)**, the youngest guest in Zhang Xiaojun's series. Her AI-for-Math company **Axiom** (called "Oxfam" by ASR) just closed a **$1.6B Series A** led by **Menlo Ventures** in January 2026, sized at **at least $200M**, on top of a **$64M seed at $300M post** led by B Capital / Howard Morgan in July 2025. The viral hook: "57-year-old American tenured professor suddenly quits to work for a 24-year-old Chinese woman."
- The technical thesis: **Math is code and code is math** via the **Curry-Howard correspondence**. Axiom turns mathematics into **Lean** as a parallel, formally-verifiable substrate — not a "language crutch" — and pairs a language layer (conjecture / fuzzy intent) with a Lean layer (verification). The stack has four pieces: **prover + conjecturer + knowledge base + auto-formalization**. The first commercial wedge is verification (chip / code), not consumer math; Amazon's automated-reasoning team needing 3-5 years and **260,000 lines** of Lean-style proofs to verify a hypervisor memory-isolation component is Hong's reference customer pain.
- **Putnam 2025 perfect score (12/12, 120/120)**: on 2025-12-06 the team formalized that day's Putnam problems and ran **Axiom Prover** in their "Poincaré" war room; by 3:58 pm they had 8/12 = 80 points (top-5 world in 2024), then pushed to 12/12. They claim to be the 6th perfect score in the competition's 98-year history (since 1927) and the first by an AI. On the **Verina** benchmark (Lean proofs + Rust code), **DeepSeek-Prover hit 11%**, Axiom hit **98.93%** — Hong cites this as evidence that strong Lean-prover capability transfers to strong code verification.
- **Philosophy & origin**: math as partly created (axiom contract) partly discovered (Hardy/Ramanujan-style); "**Bounded Attention** vs **Free Attention**" as the distinguishing trait between average and exceptional founders/mathematicians; a Guangzhou childhood "refusing to be domesticated" by zero-sum Olympiad culture; MIT freshman-year PhD measure theory scoring **4/40** as the "addicted-to-suffering" formative moment; rescued from a Bridgewater quant internship into Ken Ono's REU when COVID went remote; Morgan Prize → Oxford Gatsby (pivot to AI) → Stanford math+JD. Self-label: **"the most unlikely founder"**.
- **Endgame is binary like SpaceX** — moon-landing success or rocket crash. The differentiator from competitors is **conjecture generation** (aiming for AI mathematicians that pose new problems), not IMO-gold-as-endgame. Hong **BETs system, not model** — believes **Recursive Self Improvement** is coming fast and that traditional consulting will die under Forward Deployment. Personal endpoint: **Leibniz's Universal Representation Theory** — make top-tier, self-verifying reasoning the default state. "I love math = I have seen the face of God."

## Why it matters

A 4-hour primary-source record of how a $1.6B-valuation AI-for-Math company actually gets built — the napkin GPU calculation at Verve Coffee, the Ken Ono email-interception, the Putnam war-room, the Curry-Howard pitch, the verification-as-commercial-wedge thesis, the ASI-not-AGI framing, and Lean as parallel substrate to language rather than crutch. Together with Su Yu's Agent survey (`2026-05-01-su-yu-agent-tech-history.md`) and Yao Shunyu's interview (`2026-05-11-yao-shunyu-training-claude-gemini.md`), this completes a three-episode arc of Zhang Xiaojun's specialized-intelligence series: Agent (Su Yu) → coding/general-purpose (Yao Shunyu) → math/verification (Hong Letong).

## Section summaries

### §1 — Opening: 4-hour interview at Facebook House, $1.6B AI-for-Math founder  [[00:00 → 02:14](https://youtu.be/bv8ghyTFF9w?t=0)]

- Host Zhang Xiaojun frames the episode as a 4-hour interview with **24-year-old (00后) 洪乐潼**, the youngest guest in her interview series, at Zuckerberg's earliest startup site **Facebook House** ("外表为淡蓝色的可爱的小房子") in Silicon Valley.
- The viral news that made this interview happen: **"57岁的美国终身教授突然辞职 去给24岁的华人女孩打工"** — the unnamed 57-year-old tenured professor is **Ken Ono (小野肯)** who joined Axiom (later revealed in §9).
- Hong identifies her psychological constant: at every life stage she has felt like "the stupidest person in the room and the one whose effort seems most invisible." Founders generally are "**addicted to suffering**" (对就是苦难上瘾) — fundraising in particular is a 复读机 (parrot) grind of repeating the same answers to the same questions.
- Cold-open question Zhang previews: with coding so hot, what's the difference between using **Math** vs **coding** as the means to execute tasks? One investor conversation memorably stood out — with **Howard Morgan** of seed lead **B Capital**.
- People: [Zhang Xiaojun](https://youtu.be/bv8ghyTFF9w?t=1) [[00:00:01](https://youtu.be/bv8ghyTFF9w?t=1)], [Hong Letong](https://youtu.be/bv8ghyTFF9w?t=31) [[00:00:31](https://youtu.be/bv8ghyTFF9w?t=31)], [Mark Zuckerberg](https://youtu.be/bv8ghyTFF9w?t=18) [[00:00:18](https://youtu.be/bv8ghyTFF9w?t=18)], [Howard Morgan](https://youtu.be/bv8ghyTFF9w?t=125) [[00:02:05](https://youtu.be/bv8ghyTFF9w?t=125)].
- Numbers: Axiom A-round valuation **$1.6B** [[00:00:35](https://youtu.be/bv8ghyTFF9w?t=35)]; 57-year-old tenured professor / 24-year-old Chinese-American woman [[00:00:45](https://youtu.be/bv8ghyTFF9w?t=45)].

### §2 — Created vs discovered, brute-force vs intuition, Putnam perfect score teased  [[02:14 → 14:38](https://youtu.be/bv8ghyTFF9w?t=134)]

- Hong's "struck by lightning" math moments come from reading others' results, not solving problems. Prototype: the **Modularity Theorem (椭圆曲线定理)** — every modular form corresponds to an elliptic curve, unifying an algebraic object with a geometric one. Childhood reference: **Ross Mathematics Camp (罗斯夏令营)** problem-set 打怪 culminating in **quadratic reciprocity (二次互反率)**.
- On "created vs discovered": math is a civilization built on a **contract of accepted axioms** — some mathematicians seek the minimum (compression view), others the most "interesting" or natural set. "**我觉得数学是一个介于艺术与科学之间的一个存在.**"
- **Proof as a social-influence mechanism**: Ramanujan arrived in Cambridge with a notebook of true-but-unproven results; Hardy and Littlewood's proofs are what made them "accepted." 张益唐 (Yitang Zhang) and **James Maynard** (2022 Fields Medalist) cited as the contemporary analog — initially foreign proof styles get absorbed and simplified.
- **Self-typing: brute-force, not gifted.** "我一直就是非常希望能当直觉派天才派 ... 我一直发现我自己没有什么特别大的数学天赋,就我是一个蛮力型选手 ... 我在MIT的时候,我身边所有人他们都是天赋型选手 ... 但是呢,我不放弃,我是打不死的小强." He solved IMO geometry by **complex-number coordinates (复数法)** — ignoring geometric meaning, paying 2-3× the time.
- **AlphaGeometry parallel**: Google DeepMind's AlphaGeometry (Hong claims started ~2021 in secret) follows the same brute-force philosophy — turn figures into symbolic expressions — and reportedly solves **~81%** of historical IMO geometry problems.
- **Henry Cohn (MIT) sphere-packing problem**: as an MIT undergrad Hong spent 6 months on a sub-problem on **28-dimensional sphere packing** and got nothing.
- **The Putnam-perfect-score teaser**: their Axiom Prover (ASR: "Action Prover") hit a **perfect 120/120 on Putnam 2025** — Hong claims the **6th perfect score in 98 years (since 1927)** and the first by an AI. Teammate **Evan Chen** (US IMO coach) solved one problem with a single picture; the AI instead generated "几千行的这个Lin代码" using enumeration and case analysis.
- People: [Gauss](https://youtu.be/bv8ghyTFF9w?t=156), [Ramanujan / 拉玛努金](https://youtu.be/bv8ghyTFF9w?t=395), [Hardy 哈代](https://youtu.be/bv8ghyTFF9w?t=401), [张益唐](https://youtu.be/bv8ghyTFF9w?t=461), [James Maynard](https://youtu.be/bv8ghyTFF9w?t=488), [Henry Cohn](https://youtu.be/bv8ghyTFF9w?t=630), [Evan Chen](https://youtu.be/bv8ghyTFF9w?t=756).
- Numbers: **81%** of historical IMO geometry by AlphaGeometry [[00:10:01](https://youtu.be/bv8ghyTFF9w?t=601)]; Putnam started **1927**, 5 prior perfect scores in 98 years, this is the **6th** [[00:12:13](https://youtu.be/bv8ghyTFF9w?t=733)]; 6 months stuck on 28-dim sphere packing [[00:10:30](https://youtu.be/bv8ghyTFF9w?t=630)].

### §3 — Bounded Attention vs Free Attention; refusing to be domesticated  [[14:38 → 32:14](https://youtu.be/bv8ghyTFF9w?t=878)]

- Hong's startup mentor's framework: **Bounded Attention** (被框架住的注意力 — emails, deadlines, execution under time pressure) vs **Free Attention** (自由注意力 — Einstein-in-the-shower wandering). The latter is what distinguishes an average from a strategic founder or mathematician. Non-linear: a free-wandering session yields nothing by itself, but a later "callback" surfaces a seeded idea during forced execution.
- **Three-type founder taxonomy** (from mentor): visionary, executor, salesman. Zuckerberg = executor, **Musk = visionary**, Sam Altman = salesman. Hong puts himself in visionary, not executor. **Sandberg (桑德伯格)** kept Facebook culture stable "from 0 to 3000 people." Facebook's "**bottoms-up culture**" (Zuckerberg is color-blind, hence the bold office palette; employees "Bleed Purple") is what Axiom inherited because most of the founding team came from Facebook.
- **Guangzhou childhood, 10-min walk to school** — lots of Free Attention. Olympiad coach criticized him for not being the most diligent, but when he locked onto a problem he 死磕'd it like play. The bounded-vs-free distinction is subjective: play vs work.
- **Refusing to be domesticated (拒绝被驯化)**: in middle school he discovered higher math (calculus) was a **positive-sum** game vs. the **zero-sum** Olympiad track where classmates competed on the same exam.
- **Olympiad school ranking**: students re-sorted monthly into classes 1-24 by floor (class 24 = top, top floor). Hong started in **class 4 next to the bathroom** — concrete evidence of non-prodigy status. The middle-school math group was ~25-27 students out of ~90, ~7 girls (3-4 in high school).
- **Knight's-tour collaborative proof**: a 3-5 person "tribe" passed notes during lectures trying to prove via induction that every **n×n board with n≥5 admits a full knight's tour** — Hong frames this as a precursor to Terence Tao / Alex Kontorovich-style large-scale collective formal-proof projects.
- People: [Zuckerberg](https://youtu.be/bv8ghyTFF9w?t=1103), [Musk](https://youtu.be/bv8ghyTFF9w?t=1186), [Sam Altman](https://youtu.be/bv8ghyTFF9w?t=1189), [Sandberg](https://youtu.be/bv8ghyTFF9w?t=1212), [Einstein](https://youtu.be/bv8ghyTFF9w?t=1038), [陶哲轩 / Terence Tao](https://youtu.be/bv8ghyTFF9w?t=1812), [Alex Kontorovich](https://youtu.be/bv8ghyTFF9w?t=1812).
- Numbers: 10-min walk to school [[00:15:36](https://youtu.be/bv8ghyTFF9w?t=936)]; **0 → 3000 Facebook employees** under Sandberg's culture-watch [[00:20:15](https://youtu.be/bv8ghyTFF9w?t=1215)]; classes 1-24 ranked monthly, Hong entered as class 4 [[00:26:12](https://youtu.be/bv8ghyTFF9w?t=1572)]; knight's-tour conjecture **n≥5** [[00:29:38](https://youtu.be/bv8ghyTFF9w?t=1778)]; tribe size **3-5 people** [[00:31:26](https://youtu.be/bv8ghyTFF9w?t=1886)].

### §4 — Addicted to suffering: MIT, 4/40 in PhD measure theory, Ken Ono's REU rescue  [[32:14 → 50:21](https://youtu.be/bv8ghyTFF9w?t=1934)]

- **"我在麻省理工的时候,我身边的每一个人,他都比我聪明,每一个,我是整个数学系里面最愚蠢的."** Hong arrived at MIT surrounded by classmates (任秋雨, 张升桐, 高继扬) he'd grown up reading about in Chinese math news — so no 心理落差, he expected to be the dumbest.
- **The Bridgewater → Ken Ono pivot**: planning quant finance because PhD admission seemed impossible (every peer was IMO gold), he locked in a **Bridgewater (桥水) freshman-summer internship**. COVID went remote, freeing him to accept a waitlist slot in **Ken Ono's (小野肯) REU** — NSF-funded undergrad research, the moment that redirected him into research.
- **Floor Pi (3W, East Campus)**: dorm-mates were a wall of IPhO/IOI/IMO medalists incl. Taiwan's 余洪勋. The advice "take the hardest course with no prerequisites" sent a group of freshmen straight into PhD measure theory starting from the Borel sigma-algebra.
- **The 4/40 midterm**: class average ~9/40; the freshmen single-digit. Hong got **4/40** and used it to motivate relearning Rudin's *Principles of Mathematical Analysis* from scratch. "我什么失败都不会触发我自己觉得很失败的机制,就是我把这个失败当做是我的这个default默认."
- **Reward mechanism evolved**: early on it was the climb-team comraderie (登山队 feeling); now it's "**addicted to pain and suffering**" — cites **Jensen Huang (黄仁勋)'s "pain and suffering"** framing and the VC adage "**chip on the shoulder, chips in the pocket**" (turning old wounds into entrepreneurial currency).
- 75 mock exams in a short period for an Olympiad selection round he didn't make. Born 2001, age 24 at recording.
- **Leadership = service**, not commanding — "你不是前面那个拿着喇叭的那个,你是就是后面滴水的那个" — invokes Buffett/Munger-style adage about hiring strong-technical-core leaders.
- People: [任秋雨/张升桐/高继扬](https://youtu.be/bv8ghyTFF9w?t=2064), [余洪勋](https://youtu.be/bv8ghyTFF9w?t=2135), [Ken Ono](https://youtu.be/bv8ghyTFF9w?t=2168), [Jensen Huang](https://youtu.be/bv8ghyTFF9w?t=2752), [Buffett / Munger](https://youtu.be/bv8ghyTFF9w?t=2969).
- Numbers: **75 mock exams** [[00:39:01](https://youtu.be/bv8ghyTFF9w?t=2341)]; midterm class avg **~9/40**, Hong scored **4/40** [[00:43:41](https://youtu.be/bv8ghyTFF9w?t=2621)]; born **2001 (零一年)**, "还没有到二十五,二十四" [[00:47:31](https://youtu.be/bv8ghyTFF9w?t=2851)].

### §5 — How beautiful is number theory! Lego-block math; MIT → Oxford → Stanford  [[50:21 → 01:02:26](https://youtu.be/bv8ghyTFF9w?t=3021)]

- **University math felt like Lego**: unlike the fixed-exam-point Olympiad repertoire, introducing one new definition lets you derive new theorems. Hong explicitly compares the experience to the AI-for-Math paper **LegoProver** ("乐高积木 ... 这个过程是不受竞争所限制").
- **Why MIT**: wrote "MIT" on math scratch paper as a child, inspired by *Good Will Hunting* and the infinite corridor — three letters easier to scrawl than "Columbia University."
- **Why double-major in math + physics**: Scott Sheffield's random surfaces / geometric probability research kept invoking "physical meaning."
- **Morgan Prize** (top North-American math undergrad honor) — Hong stresses high randomness ("30年摩根奖的历史每年说虽然有一个人拿到然后有大概两三个人是就是亚军和济军"). Nominated and recommended by **Ken Ono** among others; most of Hong's undergrad papers are with Ono on modular forms / elliptic curves.
- **MIT math culture is non-zero-sum**: professors match students to topics they like rather than fighting over slots; during COVID flight bans professors wrote letters to help Chinese students return.
- **Oxford master's at Gatsby Computational Neuroscience Unit**: motivated by his grandfather's condition, but pivoted to computational neuroscience after the UK animal-license exam **required killing a mouse**. Gatsby in practice is mostly AI faculty — **funded by Geoff Hinton**, where **Demis Hassabis** did his postdoc; Hong worked with **Andrew Saxe**, collaborator **Surya Ganguli** (Stanford), Tim Behrens, Will Dorrell. Two master's papers: continual learning neurodynamics; one-layer linear transformer theory.
- **"AI 点燃我,不是神经科学."** — Stanford math PhD was pre-decided (deferred from undergrad); Oxford was a one-year detour because the Rhodes-style scholarship was too good. The JD came from a childhood debate passion → Oxford Union → Stanford Law summer internships in litigation.
- Self-assessment: not a "well-rounded" person — **spiky** in strengths AND weaknesses (no sense of direction, terrible at geography).
- People: [Scott Sheffield](https://youtu.be/bv8ghyTFF9w?t=3225), [Ken Ono](https://youtu.be/bv8ghyTFF9w?t=3383), [Demis Hassabis](https://youtu.be/bv8ghyTFF9w?t=3544), [Geoff Hinton](https://youtu.be/bv8ghyTFF9w?t=3552), [Andrew Saxe](https://youtu.be/bv8ghyTFF9w?t=3566), [Surya Ganguli](https://youtu.be/bv8ghyTFF9w?t=3569), [Will Dorrell](https://youtu.be/bv8ghyTFF9w?t=3576).
- Numbers: **30-year Morgan Prize history** [[00:56:31](https://youtu.be/bv8ghyTFF9w?t=3391)]; math dept "10-min walk" from his college [[00:57:58](https://youtu.be/bv8ghyTFF9w?t=3478)]; "more than half of my papers were with Professor Ono" [[00:57:35](https://youtu.be/bv8ghyTFF9w?t=3455)].

### §6 — Verve Coffee: textualism, Lean, and the napkin GPU fundraise  [[01:02:26 → 01:16:23](https://youtu.be/bv8ghyTFF9w?t=3746)]

- **The Stanford Law origin**: in a constitutional-law class, three schools — originalism, textualism, living constitutionalism — Hong picked **textualism** ("按照他是什么样就是什么样"). The bridge insight: if AI is good enough to read constitutional text by data (founding documents + modern political philosophy), it should also do math — and **Lean turns mathematics into code**, giving a more structural representation than English. "我们已经到了一个能拿AI去看这个宪法是什么意思的这样的一个时代我为什么不能拿AI做数学."
- **Kenny Long & Mathlib**: Hong's MIT friend Kenny Long (Imperial exchange to MIT, chess master) was one of **5-7 students of Kevin Buzzard** who hand-built Mathlib from empty by typing out undergraduate algebra and analysis line-by-line starting ~2020. Kenny now full-time at Axiom.
- **Brute-force mathematician → AI helps**: AI helps **brute-force mathematicians** most — it can enumerate possibilities / verify standard arguments quickly. Concrete motivating problem: Ben Green's "**Sárközy theorem for shifted primes**"; same machinery should generalize to many number-theoretic problems (shifted sum of two squares, x²+y²±1). Goal: an AI at the level of a "PhD student fluent in all the tools in **Davenport's** number theory."
- **The Verve Coffee origin story**: from Stanford Law, an 8-min Caltrain ride + walk through a tunnel into Palo Alto downtown. Weekend matcha + 30-page law-class readings + dog-watching. Hong and co-founder **Shubo (叔伯)** both bought 1,000+ coffees, earning a Polaroid on the wall. First conversation was about a sun-shade curtain at a shared 6-person communal table. **They were friends for ~1.5 years** before either knew the other's professional identity — Hong didn't know Shubo was a Meta FAIR director; Shubo didn't know Hong had a Morgan Prize.
- **The napkin GPU fundraise (fall 2024)**: Hong was in his first math-PhD year, spent the back half of summer at **XTX** quant fund "for the GPUs." After a morning run he felt "this thing really has to happen," went to Shubo, and they **sized the fundraise on a napkin counting GPU cards** — concluded it could not be done in academia.
- People: [Kenny Long](https://youtu.be/bv8ghyTFF9w?t=3888), [Kevin Buzzard](https://youtu.be/bv8ghyTFF9w?t=3929), [Shubo](https://youtu.be/bv8ghyTFF9w?t=4247), [Ben Green](https://youtu.be/bv8ghyTFF9w?t=4076).
- Numbers: **5-7 Buzzard students** built early Mathlib [[01:05:22](https://youtu.be/bv8ghyTFF9w?t=3922)]; **1,000-coffee** Polaroid bar at Verve [[01:06:25](https://youtu.be/bv8ghyTFF9w?t=3985)]; **30-page reading per class** at Stanford Law [[01:11:14](https://youtu.be/bv8ghyTFF9w?t=4274)]; friends for **1.5 years** before knowing each other's careers [[01:14:39](https://youtu.be/bv8ghyTFF9w?t=4479)].

### §7 — The most unlikely founder: Tudor, Baidu mafia, no linguists, math club  [[01:16:23 → 01:38:33](https://youtu.be/bv8ghyTFF9w?t=4583)]

- **"我某种程度上我觉得我是最不可能创业的一个创业者."** Hong was vocally anti-"AI founder" at MIT, refusing free-sushi startup-club events. The self-critique was not trend-chasing but technical understanding: "我不可能说我自己觉得这个事情不会成功然后去骗别人的钱."
- **Two months arguing himself out of it**: decision in Sep 2024, confirmation by mid-November, fundraised post-New-Year (Thanksgiving → Christmas window was dead — "没有VC上班"). Read the **AI-for-Math GitHub repo (a few hundred papers)** end-to-end, every abstract, interesting ones in full; 多次费曼 whiteboard reconstructions until concluding: "他不一定是一个研究问题他是一个工程问题."
- **First tried to join the competitor (Tudor's company)** — also Verve Coffee regulars. **Tudor told him they only hire CS PhDs, not math PhDs.**
- **Axiom team has three pillars**: (1) ML/RL/agent (many ex-compiler-codegen — Meta LLM Compiler lead team, parts of Yann LeCun's **32B Code World Model** team); (2) Lean/Mathlib (Kenny Long, 居建章, plus a hire who wrote **autograd in Lean**); (3) pure mathematicians (Ken Ono, IMO coach Evan Chen).
- **The Christmas chemistry test with Shubo**: 4-5 hour Zoom reading groups, despite both being people who can't sit through a one-hour meeting. "我们做了四个五个小时每次的 ... 我们思考的方式特别的又相似又互补" was the moment they knew.
- **Shubo's lineage = Baidu Mafia**: ~20 years GPU, ~10 years AI; joined **百度 Silicon Valley AI Lab under Andrew Ng (吴恩达)** in 2014-16 — the cohort that produced **Dario Amodei** (Anthropic) and early OpenAI folks. The DNA mantra: "**Scaling works**" + "**我们绝对不招一个语言学家**" (for Deep Speech / Deep Voice).
- **Hong copied the "no domain expert" rule with a twist**: no mathematician until headcount 15 (**15:1 ML-to-mathematician ratio**) — seed caps the company at ~20. A **Frontier Math benchmark recruit walked** after accepting because he hated "internet scale data set" — "**数学是一门手艺，像日本师傅捏寿司**." Hong wants adversarial mathematicians (Percy Liang-style "做惩罚"), not craftsman-defenders. Ken Ono's #1 job is benchmark creation.
- **Christmas gift = Kaiyu Yang et al.'s** "**Formal Mathematical Reasoning: A New Frontier in AI**" survey. Section 5's capability table became the internal roadmap; half its citations weren't in the AI-for-Math GitHub repo.
- **Historical arc**: 2016 Christian Szegedy + 吴宇怀 start HOList at Google → 2019-20 Ilya Sutskever's OpenAI team does GPT-f / miniF2F (Jesse Michael Han, Stan Polu) → 2021 a single DeepMind intern starts **AlphaGeometry** → 2024 **AlphaProof scores 28/42 at IMO 2024 (one point shy of gold)** = "the IMO-solved moment for me" → **Axiom's December 2025 Putnam perfect score** is "the closing tail of DeepMind's prologue."
- People: [Tudor](https://youtu.be/bv8ghyTFF9w?t=4878), [Shubo](https://youtu.be/bv8ghyTFF9w?t=4627), [Andrew Ng / 温达](https://youtu.be/bv8ghyTFF9w?t=5246), [Dario Amodei](https://youtu.be/bv8ghyTFF9w?t=5253), [Yann LeCun](https://youtu.be/bv8ghyTFF9w?t=5049), [Percy Liang](https://youtu.be/bv8ghyTFF9w?t=5458), [Kaiyu Yang](https://youtu.be/bv8ghyTFF9w?t=5516), [Christian Szegedy](https://youtu.be/bv8ghyTFF9w?t=5809), [Ilya Sutskever](https://youtu.be/bv8ghyTFF9w?t=5835).
- Numbers: AI-for-Math GitHub repo has **几百篇文章** [[01:18:53](https://youtu.be/bv8ghyTFF9w?t=4733)]; Yann LeCun's **Code World Model 32B** [[01:24:09](https://youtu.be/bv8ghyTFF9w?t=5049)]; current headcount **~30** [[01:28:48](https://youtu.be/bv8ghyTFF9w?t=5328)]; **15:1 ML-to-mathematician ratio** [[01:29:14](https://youtu.be/bv8ghyTFF9w?t=5354)]; **AlphaProof 28 分 at IMO 2024** [[01:37:49](https://youtu.be/bv8ghyTFF9w?t=5869)]; **Axiom Putnam perfect score Dec 2025** [[01:38:11](https://youtu.be/bv8ghyTFF9w?t=5891)].

### §8 — Nobody likes fundraising: $64M seed → $1.6B Series A  [[01:38:33 → 02:07:17](https://youtu.be/bv8ghyTFF9w?t=5913)]

- **Seed timeline**: JMM (Joint Math Meetings) Seattle on 1月7日 2025; AMS theme was "AI for Math"; offers landed within 3 days of returning; **3-month bidding Jan→Mar, final offer 3× the first**; closed **July 2025 at $64M raised on a $300M post** (target $50M, oversubscribed).
- **Lead = B Capital, via Howard Morgan** (Renaissance Technologies co-founder with Jim Simons; First Round co-founder; NYU adjunct; appeared in HBO's *Silicon Valley*). Hong heard Simons give a 2-hour 围炉 talk to MIT math club as an undergrad; Howard introduced her to "Simons' cousin who looks exactly like him."
- **First January lead offer was rejected** for demanding 50% lead/ownership share — "我不可能让他带录到五百分之五十." **Shubo (Sugo) joined as co-founder in Feb 2025** after the second, concrete offer.
- **Series A was unplanned** — an existing seed investor preempted at Christmas 2025 with an unsolicited term sheet (no deck, no materials). On **1月5号 2026** Hong flew to an out-of-town city and got the offer that night; a second offer 1.5 weeks later won. **Closed 1月15日 2026 at $1.6B valuation, raised "at least $200M"** ("至少是两亿美金").
- **The lead = Menlo Ventures (Matt Kroening, EE PhD + physics undergrad, "technical / nerdy")**. Menlo is Anthropic's largest institutional investor; this is Menlo's second-largest AI venture bet. They had a 1% check in Hong's seed.
- **Why Menlo won the A**: the team had no misses for 6 months — infrastructure month 1, models trained, **Putnam perfect score at month 4**, then 3-4 open research problems solved, the first autoformalization proof system with no human intervention, and the transfer-learning discovery: on **Verina benchmark DeepSeek Prover scored 11% vs Axiom 98.93%**.
- **Fundraising lessons** (Hong calls herself a bad fundraiser): she pre-disclosed all risks without the conversion rate — "我后来才知道投资人们一般来说 ... 他讲的都是比较乐观的 ... 他比如说你可能讲说感觉这个东西是十分，就给你讲一个八分 ... 我如果当时讲的是一个七分，他就给我打折就可能打没了." Investors don't reject — they stall ("**没有人想拒绝我们 ... 他们就拖着你**"), waiting for someone else to lead, then mirroring (**groupthink**). "**没有人喜欢融资 ... 你是一个复读机.**" She'd happily hand a percentage to an AI fundraiser.
- **Being young Chinese-female in deep tech**: 华人女性 was neutral; **年轻 is a clear minus** — "**年轻做product是加分，年轻做deep tech是减分**" because she lacks the track record of leading a tech team. Advice to her younger self: "**读三倍的书**" + build the "leap of faith muscle memory" — the AI-fundraiser-arranged zip-lining offsite is the literal metaphor.
- **"我们不是一个模型公司，我们是一个deep tech公司 ... 有点像 SpaceX."** Market labeled them a model company right after DeepSeek crashed market sentiment in Feb 2025 ("一听模型，不赚钱不要投"). Hong rejects the framing. The moat is the **~12-13 Lean tooling components** they had to build (their own verify-proof checker runs **~100× faster** than the community comparator); they explicitly **rejected AlphaProof's Monte Carlo Tree Search** ("太贵了 ... 我们做不了") and built something architecturally closer to **ByteDance Doubao's SeedProver**.
- **Bias toward action** as daily discipline: "**你只要bias toward action。每一次你在执行与不执行、做和不做中选择那个乐观的选项，你就能够有一天走到那个重点.**"
- People: [Adam Wagner (DeepMind)](https://youtu.be/bv8ghyTFF9w?t=5966), [Albert Jiang](https://youtu.be/bv8ghyTFF9w?t=5976), [Howard Morgan](https://youtu.be/bv8ghyTFF9w?t=6077), [Jim Simons](https://youtu.be/bv8ghyTFF9w?t=6101), [Peter Thiel / 皮特T](https://youtu.be/bv8ghyTFF9w?t=6682), [Sugo/Shubo](https://youtu.be/bv8ghyTFF9w?t=5921), [Matt Kroening](https://youtu.be/bv8ghyTFF9w?t=7076).
- Numbers: **first offer → final offer 3×** [[01:40:03](https://youtu.be/bv8ghyTFF9w?t=6003)]; **seed $64M (6400万) at $300M (三亿) post**, target $50M [[01:49:06](https://youtu.be/bv8ghyTFF9w?t=6546)]; **3 competing lead offers** for seed [[01:44:52](https://youtu.be/bv8ghyTFF9w?t=6292)]; **Series A 1月5号 2026** [[01:56:52](https://youtu.be/bv8ghyTFF9w?t=7012)]; **A round $1.6B valuation, ≥$200M raised** [[01:58:28](https://youtu.be/bv8ghyTFF9w?t=7108)]; **Verina: DeepSeek-Prover 11% vs Axiom 98.93%** [[01:59:55](https://youtu.be/bv8ghyTFF9w?t=7195)]; their verify-proof checker is **100× faster** than community comparator [[02:05:03](https://youtu.be/bv8ghyTFF9w?t=7503)]; **~12-13 Lean tooling components** [[02:05:10](https://youtu.be/bv8ghyTFF9w?t=7510)]; competitor has **5× their funding and 5× their valuation when they started** [[02:06:11](https://youtu.be/bv8ghyTFF9w?t=7571)].

### §9 — Ken Ono's email: intercepting a 57-year-old tenured professor in 2-3 days  [[02:07:17 → 02:19:51](https://youtu.be/bv8ghyTFF9w?t=7637)]

- **The intercepted email**: late November 2025, **Ken Ono** emailed Hong saying he was coming to the Bay Area and **might join OpenAI or DeepMind** — "他希望我有一个心理准备" because it would become a competitive relationship with Axiom. Hong's reaction: **"你要加入OpenAI和DeepMind 那你为什么不来我这里."**
- Hong had never previously asked Ono to join — felt he wasn't entitled to "ask his teacher." The whole closing happened over Zoom in **2-3 days** while Hong was flying re:Invent (Las Vegas) → NeurIPS (San Diego, where Axiom sponsored the AI4Math workshop) → back. **Ono never visited the office before signing.**
- **Pitch was anti-sell**: "**这个公司的dna和专注点就是数学 而不是一个general的agi 然后数学是其中一个部分 可能更有marketing的成分.**"
- **The order of joiners**: (1) Hong + (2) **Shubo (叔伯)** — Google/FAIR-style director, sat down at Worth café and said "好多好的人都走了, 感觉这个地方不太像是以前的地方"; (3) **François Charton (放缩 in ASR)** — Mistral co-founder **Guillaume Lample**'s 2019 collaborator on the Transformer-vs-Mathematica integration paper; he and Ono "惺惺相惜"; (4) Evan Kenny / Kenny Long later — the team became "a math club."
- **Ken Ono's portrait**: "**high-school basketball coach**" personality; partition-function number theorist; mentored Hong's REU papers; ex-AMS vice-president; coaches the US Olympic swim team on data; produced the **Ramanujan biopic**; producing a **Maryam Mirzakhani biopic**; runs a Ramanujan-named charity buying math textbooks for kids worldwide; White House policy advisor.
- **Math typologies**: Ono is a **theory builder (理论建造者)** — connects fields, finds questions, hands them to solvers. Hong's collaborator 张胜彤 is more intuition-driven than either of them.
- **Ono + Andrew Granville's shared view**: as AI progresses, **human mathematicians will reason at higher abstraction layers** — analogous to programming's punch-cards → low-level → Python → natural-language vibe-coding path.
- **Self-improving Axiom Prover**: every theorem proved today becomes training data / a skill (LegoProver-style skill library) for tomorrow; conjecturing and proving should bootstrap each other in an upward spiral.
- People: [Ken Ono](https://youtu.be/bv8ghyTFF9w?t=7644), [Shubo / 叔伯](https://youtu.be/bv8ghyTFF9w?t=7716), [François Charton](https://youtu.be/bv8ghyTFF9w?t=8013), [Guillaume Lample](https://youtu.be/bv8ghyTFF9w?t=8020), [Andrew Granville](https://youtu.be/bv8ghyTFF9w?t=8273), [Ramanujan / 拉玛努金](https://youtu.be/bv8ghyTFF9w?t=7974), [Maryam Mirzakhani / 米尔扎扎克哈尼](https://youtu.be/bv8ghyTFF9w?t=8207).
- Numbers: Ono offer closed in **2-3 days** [[02:13:23](https://youtu.be/bv8ghyTFF9w?t=8003)]; **2019 Charton-Lample integration paper** [[02:13:38](https://youtu.be/bv8ghyTFF9w?t=8018)].

### §10 — Proofs from THE BOOK: Putnam war room and the company name  [[02:19:51 → 02:24:38](https://youtu.be/bv8ghyTFF9w?t=8391)]

- **2025-12-06 Putnam war room**: that morning the team received the day's Putnam exam (6 AM + 6 PM problems) and immediately began **formalizing each problem so Axiom Prover could attempt them**. They gathered in their "**Poincaré**" conference room, nicknamed the war room.
- **3:58 pm checkpoint**: 8 problems solved = **80/120**, which would have placed top-5 worldwide in 2024 (top-10 to top-20 in earlier years). They chose to push rather than announce. Final result: **12/12 perfect score**.
- **The numerical-answer hack**: Axiom Prover, like Lean, only proves — doesn't compute numerical answers. For 求解题, they followed AlphaProof's 2024 IMO convention: feed the answer in, turn it into a proof. Then realized their **informal model could produce the answer directly** — human "solving" wasn't actually needed.
- **The morning paper was unusual**: 4 of 6 problems required numerical answers (typically ≤3 of 12). Team member **艾文晨 (Ai Wenchen)** was the only one fluent at speed-contest format — research mathematicians turned out poorly suited.
- **Ken Ono on-site, in "wartime" mode**: said "**不要再不现在不是说数学纯粹之美的时刻不要去精确的去搞这些东西，现在是战争状态就是在大家在求解的时候他就说能怎么快捷的去去去做就怎么快捷的去做**" — Hong and Shubo laughed 前仰后合.
- **Why "Axiom"**: Hong loved **数学天书中的证明 / Proofs from THE BOOK** (Aigner & Ziegler — Erdős's imagined book of perfect proofs) since childhood. "**AXIEM这个词就我觉得很美它很数学它很它很克制它很理性嗯它又很sharp.**" Axiom echoes formal-language foundations (finite axioms → high-rise of results) and incidentally many employees have A-names (Alex, Alberto, Arum).
- People: [Ken Ono](https://youtu.be/bv8ghyTFF9w?t=8506), [Shubo](https://youtu.be/bv8ghyTFF9w?t=8513), [艾文晨](https://youtu.be/bv8ghyTFF9w?t=8601), [Erdős](https://youtu.be/bv8ghyTFF9w?t=8632).
- Numbers: **2025-12-06 Putnam day** [[02:20:05](https://youtu.be/bv8ghyTFF9w?t=8405)]; **6 AM + 6 PM problems** [[02:20:29](https://youtu.be/bv8ghyTFF9w?t=8429)]; **3:58 pm 8/12 = 80/120** [[02:21:08](https://youtu.be/bv8ghyTFF9w?t=8468)]; **80分 was world top-5 in 2024** [[02:21:22](https://youtu.be/bv8ghyTFF9w?t=8482)]; **final 12/12 perfect score** [[02:21:39](https://youtu.be/bv8ghyTFF9w?t=8499)]; **typically ≤3 of 12 Putnam problems are 求解** [[02:23:06](https://youtu.be/bv8ghyTFF9w?t=8586)].

### §11 — AI for Math: Curry-Howard, prover+conjecturer+KB+autoformalization, verification as commercial wedge  [[02:24:38 → 03:03:50](https://youtu.be/bv8ghyTFF9w?t=8678)] — *the technical backbone*

- **Math-poor → math-rich**: society moves from math-scarce to math-abundant; **human mathematicians become resource allocators** providing the top-0.01% intuition about which problems deserve compute; AI mathematicians do the proving. Hong invokes **Demis Hassabis's "fold everything" moment** post-AlphaFold (from *The Thinking Game* documentary) as the infinite-compute math analogue: "Demis 问有多少个蛋白,科学家说有 2 亿个蛋白,然后等密室把笔扔在桌子上 ... fold everything." Hypothetical: 200 H200 on this problem, 8000 H200 on that one.
- **"Math is code and code is math"** via the **Curry-Howard correspondence** — "**基于这个我每一个数学证明可以变成一个计算机程序**." Lean-style formal verification gives code the backtracking and hierarchical decomposition that current LLM backend/distributed-systems coding lacks.
- **Two personal AlphaGo moments for AI-for-Math**: (1) **DeepMind's IMO 2024 silver-medal result (28 分, one shy of gold)**; (2) **January 2026** when Axiom Prover handled the **Weil / partial Weil Conjecture family in Lean**, alongside DeepMind's AlphaEvolve-style natural-language closers.
- **Pipeline consensus**: open-source base model (e.g. Qwen) → post-train (SFT then RL, or RL directly) → tool-using agent with Lean meta-programming. Axiom's lineage = **DeepSeek-Prover / Kimina-Prover**, not **Aristotle / AlphaProof**.
- **The 2022 "Draft, Sketch, and Prove" paradigm**: informal LLM drafts an outline → sketcher converts to Lean with `sorry` placeholders → prover fills the `sorry`s, using AI, classic ATPs, or Lean's new `grind`/hammer. Lean had no real hammer until **~2025-06**; the new **`grind` (2026)** solves many problems with **zero AI in the loop**. Axiom wants cheap deterministic tactics tried first.
- **Two Axiom aha-moments**: (i) **replacing MCTS with sub-agent-driven inference scaling** (citing Anthropic's sub-agent work + Silver/Sutton's "**AI 后半场 / Era of Experience**"), letting them scale proof search from **~40 nodes to ~4,000 nodes**; (ii) **a strong Lean theorem-prover transfers to strong code-verification** — they posted first place on **Verina** by generating Lean proofs and Rust code (strongly-typed) under a unified objective.
- **Four-piece stack — not just proving**: prover, conjecturer, knowledge base, auto-formalization. **Conjecturing has no clean 0/1 reward**; STP (**Self-Play Theorem Prover, Dong Kefan & Ma Tengyu, Stanford**) uses an **"elegance filter"** — theorem-vs-proof length on Lean Workbook (high-school) — that doesn't extend to research math.
- **Auto-formalization (English → Lean) is harder than proving**: Lean has almost no training tokens vs English/Python; one arXiv paper expands to a **200-500 page blueprint** before Lean-ready; **cycle-consistency** (formalize → informalize → re-formalize) is the current quality signal. Auto-informalization (Lean → English) is easier.
- **Commercial wedge = verification, not consumer math**: **Amazon's automated-reasoning team spent 3-5 years writing 260,000 lines of Lean-style proofs** to verify the **memory-isolation component of a CPU-optimization hypervisor**. "**这个完全这个ai没有改善这些工程师的生活**" — that pain-point gives Axiom pricing power. Long-term consumer vision: **verify-generation** — every function call returns code that needs zero test cases.
- **Knuth (1980s) wanted natural-language vibe coding**; **Turing's earlier dream** was programs **provably aligned with intent**, not test-case validated. Hong's slogan: "**任何你能定义的,你都能证明 / 任何你能 specify 的,你都能执行.**" The current bottleneck is the specification step — i.e. the conjecture problem dressed up for software.
- **Cadence Jasper (SMT-based hardware verification)** floated as a target Lean could potentially replace.
- People: [Demis Hassabis / 等密室](https://youtu.be/bv8ghyTFF9w?t=9652), [Ila Fiete](https://youtu.be/bv8ghyTFF9w?t=9704), [Mitchell Polinsky](https://youtu.be/bv8ghyTFF9w?t=9729), [Tim Gowers](https://youtu.be/bv8ghyTFF9w?t=9769), [Leo de Moura](https://youtu.be/bv8ghyTFF9w?t=10561), [Silver & Sutton](https://youtu.be/bv8ghyTFF9w?t=10691), [Dong Kefan / 董克帆 + Ma Tengyu / 马腾宇](https://youtu.be/bv8ghyTFF9w?t=10998), [Kevin Buzzard + 陶哲轩 + Alex Kontorovich](https://youtu.be/bv8ghyTFF9w?t=11241), [Donald Knuth](https://youtu.be/bv8ghyTFF9w?t=11882), [Alan Turing](https://youtu.be/bv8ghyTFF9w?t=11935).
- Numbers: proof search **40 nodes → 4,000 nodes** with sub-agents [[02:39:01](https://youtu.be/bv8ghyTFF9w?t=9541)]; AlphaFold "**2亿个蛋白 / fold everything**" anecdote [[02:27:55](https://youtu.be/bv8ghyTFF9w?t=9475)]; **AlphaProof IMO 2024 28分** [[02:33:08](https://youtu.be/bv8ghyTFF9w?t=9188)]; FrontierMath-style 10-problem challenge: **OpenAI 5/10, DeepMind 6/10** [[02:55:48](https://youtu.be/bv8ghyTFF9w?t=10548)]; **Amazon: 3-5 years, 26万行 Lean-style proofs for hypervisor memory isolation** [[03:01:30](https://youtu.be/bv8ghyTFF9w?t=10890)]; **20-page article → 200-500 page blueprint** in auto-formalization [[02:47:55](https://youtu.be/bv8ghyTFF9w?t=10075)]; hypothetical **200 H200 / 8000 H200** compute allocation [[02:27:07](https://youtu.be/bv8ghyTFF9w?t=9427)].

### §12 — Turning Math into Lean: parallel substrate, not crutch  [[03:03:50 → 03:09:59](https://youtu.be/bv8ghyTFF9w?t=11030)]

- **Stop treating math as an LLM language crutch**: "**有时候在我们讨论大圆模型的时候我们会说语言是模型的一个拐杖 ... 没有之前大家把数学当语言了 ... 这个我觉得不对 我觉得就是真的是要把数学变成令.**" Math should become Lean — a parallel, formally-verifiable substrate alongside language.
- **Multi-model pipeline**: post-trained from a base LLM into specialists for 数学 / Lean / 利用. **Final output is always Lean code**, self-verifying via the language's proof-checking property.
- **Language and Lean complement each other**: language is better for laying out a proof outline (提纲) and for chip-verification that needs fuzzy user intent; pure formalization wins on hard verification tasks. Cryptography analogy: a non-cryptographer can verify a paper by running its Lean translation through Axiom Prover and inspecting the remaining goal — **verification without comprehension**.
- **Bounded hallucination**: a Lean-grounded AI either says "this is too hard, I can't do it" or returns a correct result. Doesn't eliminate hallucination universally; bounds error cost in domains where errors are extremely expensive.
- **Intuition as a ratio**: "**好的直觉其实就是一个配比 ... pattern matching 和这个 ... 幻觉的一个配比**." Language handles 突破边界 (conjecture/generation); Lean handles 验证 (verification); generation↔verification loop.
- **Alternative AI-for-Math discovery paths**: Charton & Alberto Avarino's perturbation/embedding line of work — not LLM-based.
- **Benchmark saturation roadmap**: in ~2 months saturated **miniF2F** (high-school), then **Putnam**, then a code-verification benchmark. Research-level benchmarks barely exist — Axiom picks 2-3 unsolved problems per domain to probe the frontier. Next targets: **dynamical systems, probability, random surfaces** — under-formalized fields where Mathlib lacks definitions.
- Numbers: saturated **mini F2F (high-school)** in **2 months** [[03:09:13](https://youtu.be/bv8ghyTFF9w?t=11353)]; 2-3 unsolved problems per domain as research probe [[03:09:43](https://youtu.be/bv8ghyTFF9w?t=11383)].

### §13 — Mathematician's intuition: ASI not AGI, math as sandbox  [[03:09:59 → 03:26:18](https://youtu.be/bv8ghyTFF9w?t=11399)]

- **Library learning is the bottleneck**: AI must build its own library of definitions + theorems. Two hard sub-problems: (1) **no verification signal for whether a new definition is "correct"**; (2) **faithfulness** — AI-introduced definitions staying consistent with the human canon, not producing paradoxes.
- **Problem-setting is the current human leverage**: conjecture-maker AIs are very poor today, so humans must supply problems — but extracting unsolved problems from professors is socially complex (who gets the problem, the lab or the AI startup?). Good problems must be **robust to distribution shift** — span sub-fields, not just one. Topics like algebraic number theory work (Mathlib has the definitions); dynamical systems often lack the formal definitions.
- **ASI not AGI**: "**首先我觉得我不太喜欢AGI这个词,我觉得我们可能更是做ASI ... 这个我们不能够去声称我们做的数学是general的,数学它并不一定是那么general的.**" Mental model: a plate. Center = trivial (1+1=2, hello world). Rim = super-human (Riemann, cure cancer, Nobel-tier literature). Frontier labs try to expand a **full circle outward**; Axiom aims a **sector (扇形)** at the math/Riemann direction, incidentally covering code verification and physics but not Nobel-literature.
- **Specialized > general**: "**比如说就是我可能我数学还可以吧,我我自己都不会做饭洗衣服 ... 人类的这个智能也也不一定多.**" Even smart humans are narrow.
- **Recursive Self Improvement** once code-verification + math-conjecture + RL-on-coding click together: "**我希望你一个世界就是 AI AI scientist, AI AI engineer.**"
- **Two philosophical axes**: very smart AND **very right** (100% correct via grounding/verification). The "last mile" of edge-case coverage is where huge value lives — analogous to Google winning search by handling edge cases.
- **Differentiation = team composition**: not just mathematicians but compiler / code-generation engineers. Two slogans: "**降维打击**" + "**多元产生智能**." AI-for-math hits roadblocks faster than scraped-web work, so techniques transfer downward to easier domains like chip verification.
- **Math = sandbox for the real world**: "**数学是现实世界的沙盒因为你既有验证的这个信号又能够有更规律性的描述更结构性的数据.**" AI-for-Math vs AI-for-Science is not overlapping: AI-for-science (Periodic Labs, Lila Science, FutureHouse/Edison, George Church's lab) competes on wet-lab cycles; Axiom stays symbolic but partners with experimental scientists as their "theoretical-physicist co-pilot" and de-hallucinator.
- **"你走形式化证明你就有验证如果你不走形式化证明就没有验证"** — verification only exists via the formal route.
- People: [Demis Hassabis](https://youtu.be/bv8ghyTFF9w?t=11635), [George Church](https://youtu.be/bv8ghyTFF9w?t=12261), [Can](https://youtu.be/bv8ghyTFF9w?t=12148).
- Numbers: **Action Pro generates a 20-page math proof** [[03:20:46](https://youtu.be/bv8ghyTFF9w?t=12046)]; **headcount 15 in Dec when Can joined → ~30 now** [[03:22:28](https://youtu.be/bv8ghyTFF9w?t=12148)]; competitor headcount **50-75** [[03:22:36](https://youtu.be/bv8ghyTFF9w?t=12156)]; Demis's AGI thought experiment **"train AI to 1910 and see if it rediscovers general relativity"** [[03:13:55](https://youtu.be/bv8ghyTFF9w?t=11635)].

### §14 — Moonshot or crash: conjecture vs IMO-gold, neo-labs, the OpenAI underdog parable, Leibniz  [[03:26:18 → 03:54:17](https://youtu.be/bv8ghyTFF9w?t=12378)]

- **The differentiator from the main competitor**: Axiom aims to make AI **propose conjectures**, not just chase **IMO金牌** as endgame. "**他们觉得imu金牌可能就是就是这个终局了 ... 我们还是希望我们真的能希望把猜想做出来.**" Conjecture-generation already failed in early experiments ("**碰了一鼻子灰**"), but the prover isn't strong enough yet to attribute failure — so they invest in **core tech #2 = auto-formalization** first.
- **Two intuition archetypes**: **Ramanujan = sharp, problem-solving intuition possibly born of pre-training** (SSI reportedly trained a model literally named "Ramanujan"); **Ken Ono = divergent, cross-perspective intuition**. Axiom does **post-training (and some mid-training), not pre-training** — "**太费钱**." "**拉玛努金的那种浑然天成那种直觉其实是有可能是预训练的产物.**"
- **Ken Ono's lineage**: father (also a famous number theorist) helped crowdfund Ramanujan's Indian statue and kept a letter from Ramanujan's widow; Ono was a rebellious undergrad at Chicago who didn't finish high school, was inspired by his father using Ramanujan as motivation; one brother a violinist, the other a former University of Michigan president.
- **Map of competitors**: **OpenAI's informal track is Kevin Weil's personal science ambition**; **DeepMind runs parallel formal and informal teams**; **Anthropic uses formal verification only to boost reasoning scores**; xAI status unclear. Big labs prefer high-density talent on red-ocean domains and partner with specialists.
- **Market structure mirror**: humanoid-robotics foundation models (Physical Intelligence vs Skild) — two well-funded startups (Axiom + competitor), similar valuations, ~2 years apart, "**比较难**." (Hong notes the **Robinhood CEO** allegedly backs the competitor's company as a passion project while still full-time at Robinhood.)
- **Outcome is binary like SpaceX**: "**这个事情它的结果一定是极好或极坏 ... 当然我是登月成功了要么登月失败 ... 为什么说像spacex呢要么火箭发上去了要么火箭坠毁了.**" If it fails: **neuroscience / brain-machine interfaces**, because "**我们基本不理解人脑.**"
- **Why neo-labs exist (Peter Thiel doctrine)**: "**垄断导致创新竞争导致平庸**." **100米脸 (an Oprah-face?) / $100M packages can't buy off curiosity** as a fundamental human need. Examples: **Periodic Labs, Recursion, Inception Labs' Stefano Ermon** (diffusion reasoning models), **Thinking Machines' (Humans End) co-founder = Google's 7th employee**.
- **Underdog playbook vs OpenAI/DeepMind**: structural efficiency (innovation per resource is higher early-stage — "**为什么许多人从脸书来到了我们这个公司**"), talent density, faith that "there is always a way." **"open AI曾经也是对着Google的那一匹黑马他也是那个就是在Google要把Illia就是最后counter offer抢走的时候焦头烂额 ... 也是那个曾经一度发不出工资的这样的一个小玩家."**
- **The endpoint = Leibniz's Universal Representation Theory**: make top-tier, self-verifying reasoning the **default state**. "**我目前非常期待一个让推理能力成为最顶尖的推理能力成为一个默认状态的一个这样的一个情况然后且是能够自我验证的推理能力这是我目前个人的一个理想.**"
- **Jevons' Paradox** on mathematicians: AI doesn't replace human mathematicians, it **multiplies** them. **Galois died at 21, Ramanujan at ~30** — reproducing such minds via AI compounds the entire math community. **Hardy's *A Mathematician's Apology*** wrongly claimed pure math had no practical use, refuted by elliptic-curve cryptography (Victor Miller, Shubo's contemporary).
- People: [Ken Ono](https://youtu.be/bv8ghyTFF9w?t=12508), [Ramanujan](https://youtu.be/bv8ghyTFF9w?t=12513), [Kevin Weil](https://youtu.be/bv8ghyTFF9w?t=12895), [Musk](https://youtu.be/bv8ghyTFF9w?t=13202), [Ray Dalio](https://youtu.be/bv8ghyTFF9w?t=13218), [Peter Thiel](https://youtu.be/bv8ghyTFF9w?t=13322), [Stefano Ermon](https://youtu.be/bv8ghyTFF9w?t=13434), [Ilya Sutskever](https://youtu.be/bv8ghyTFF9w?t=13648), [Galois / 加罗瓦](https://youtu.be/bv8ghyTFF9w?t=13896), [Leibniz / 莱布尼茨](https://youtu.be/bv8ghyTFF9w?t=13836), [Babbage / 查理斯巴巴贝奇](https://youtu.be/bv8ghyTFF9w?t=13998), [Hardy](https://youtu.be/bv8ghyTFF9w?t=14010), [Victor Miller](https://youtu.be/bv8ghyTFF9w?t=14023), [赛宁谢 / Saining Xie](https://youtu.be/bv8ghyTFF9w?t=13708).
- Numbers: Axiom is **7 months old** at recording [[03:37:52](https://youtu.be/bv8ghyTFF9w?t=13072)]; **$100M / 100米脸 can't buy curiosity** [[03:43:46](https://youtu.be/bv8ghyTFF9w?t=13426)]; **Galois died at 21, Ramanujan ~30** [[03:51:36](https://youtu.be/bv8ghyTFF9w?t=13896)].

### §15 — Dinner with mathematicians: Erdős, Grothendieck, 2026 predictions, "BET system not model"  [[03:54:17 → 04:24:17](https://youtu.be/bv8ghyTFF9w?t=14057)]

- **Self-identity: research scientist intern, not CEO**. "**我自己最快乐的状态，就是我能够当一个research scientist intern。为什么是intern呢？就是这个我说的话，如果比较愚蠢，这个大家认为是正常的.**" Only became CEO because of the work itself, not from wanting to be a 创业者.
- **Anti-anecdote about her own authority**: she pitched a 博士级 benchmark side-project to office mathematicians as a hobby; an 元老 engineer warned that **because she's the CEO, interns would treat it as top-priority** — so she's been "more hands off" since.
- **Bottoms-up culture argument**: a Dreamworks/Disney run by a veteran Hollywood director can't become Dreamworks — "**他们会觉得这个人是对的，应该听他的**." Innovation needs bottoms-up.
- **Five meeting rooms, named after polymaths**: **Gauss, Poincaré, Hilbert, Lovelace, Turing**. Ramanujan rejected for being only a number theorist — "**拉玛努金就是这么落选的，他只会数论 ... 没有distribution shift这样的一个问题，所以要找这些polymath.**" External requests to name a room after Emmy Noether; one engineer threatened to walk if Turing wasn't included (a partner countered "you have Turing, you don't have Church, no good").
- **Time-travel dinner picks**: **Erdős** (combinatorics dinner-table-accessible + 古灵精怪 personality) and **Grothendieck** (but only for someone who knows 代数几何 well enough not to waste the chance).
- **On math becoming boring after AI**: Hong was initially afraid this would be a 遗憾的局面 but believes human mathematicians remain 智力上的挑战 generators — cites **Roger Heath-Brown**'s birthday 峰会 at Oxford with **Ben Green** and **James Maynard**. AI will help close proofs; conjecture / 直觉 / 构造 remain human.
- **Genuinely novel proof-method invention** (not card-shuffling Major/Minor Arc, Hardy-Littlewood circle method, sieve method): "**5 to 10 years**" away.
- **"热爱数学就是看到了上帝的面"**: as a 16-year-old she wrote about discovering basic truths; later, running past Stanford's **Memorial Church** one bright morning, she imagined a mathematician's **墓志碑** carrying a proof as **智力遗产** — admits her wish to be the AI-for-Math 登月 unlocker is partly ego: "**可能说如果那个就是一个人的这个墓志碑上，可能印了一个他曾经证明出来的这个事情是他的智力遗产，那么如果能够让这个东西成以一亿，你会不做吗？**"
- **2026 predictions**:
  - First **continual-learning small model** (small new-lab).
  - Strong **multimodal reasoning model** (also a new lab — a friend's company).
  - **Agent-economic scale-up**.
  - **Formal-verification tooling as RL reward** = "completely underexplored."
  - **Orchestrator + sub-agents** both worth building.
- **The central bet**: "**我BET System。我不BET Model。我BET就是这个System有非常多的事情可以做。然后包括Orchestrator。然后我另外的一个，与这个又相关的一个BET就是，我完全相信Recursive Self Improvement是很快就能够做出来的.**" Believes **传统咨询会死** under Forward Deployment.
- **24 诗品 callback "如将不尽，于古为新"** — ancient mathematical techniques keep guiding their frontier research; "**世界是一个圆，我们又回到了原点.**"
- **Closing — language as world**: "**数学家们他们在几千年和或者几千年几百年，他们都是在拿英语写代码，或者说拿中文写，就拿他们的那个本国语言写代码 ... 但是他们是在自然语言里面去进行的逻辑推理.**" That's why math structure helps with code verification today.
- **Hypothetical from host**: "**假设Axiom的AI在未来证明某个重大的猜想，但在证明过程中使用了一个新的公理，这个公理不是现有数学的一部分，但是看起来非常的合理。你会接受这个证明吗？**" Hong floats a "Probabilistic Analytic" hybrid field with branching-factor-2 worlds eventually being "Titan'd" back together.
- **Books / influences**: **Davenport's analytic number theory**, **《红楼梦》**, **《大雅宝胡同甲二号》**, **Musk biography**, **Christian Szegedy's AI-for-math white paper**, **Lample & Charton transformer-for-math paper**, **"Draft, Sketch, and Prove"**, prover bundle (**DeepSeek Prover, SeedProver, Hilbert Prover**). Also half-watches **Bryan Johnson's** Don't Die movement as a joke apprentice.
- People: [Erdős / Erdos](https://youtu.be/bv8ghyTFF9w?t=14441), [Grothendieck](https://youtu.be/bv8ghyTFF9w?t=14462), [Heath-Brown](https://youtu.be/bv8ghyTFF9w?t=14534), [Ben Green](https://youtu.be/bv8ghyTFF9w?t=14537), [James Maynard / James Miller](https://youtu.be/bv8ghyTFF9w?t=14537), [Ada Lovelace](https://youtu.be/bv8ghyTFF9w?t=14374), [Emmy Noether](https://youtu.be/bv8ghyTFF9w?t=14323), [Alonzo Church](https://youtu.be/bv8ghyTFF9w?t=14315), [Bryan Johnson](https://youtu.be/bv8ghyTFF9w?t=14986), [Jensen Huang](https://youtu.be/bv8ghyTFF9w?t=15170), [袁正](https://youtu.be/bv8ghyTFF9w?t=15374).
- Numbers: **5 meeting rooms** [[03:58:47](https://youtu.be/bv8ghyTFF9w?t=14327)]; **novel proof-method invention 5-10 years away** [[04:05:00](https://youtu.be/bv8ghyTFF9w?t=14700)]; **DeepSeek Prover claimed 49/600 miniF2F, actually 47** (2 cheats with axioms) [[04:14:30](https://youtu.be/bv8ghyTFF9w?t=15270)]; **startup mortality 99%** [[04:09:42](https://youtu.be/bv8ghyTFF9w?t=14982)]; **most of the top-30 AI-for-math papers' authors now at Axiom** [[04:09:11](https://youtu.be/bv8ghyTFF9w?t=14951)].

## Notable quotes

> "因为我确实在每一个时期 都觉得自己是那个环境里面最愚蠢的那一个 最怎么样努力都看不到结果的那一个 ... 我觉得大部分我知道的就是 Founder就是创业公司的创始人 他们都对就是苦难上瘾。"
> — Hong Letong, §1 [[00:00:54](https://youtu.be/bv8ghyTFF9w?t=54)] — *the cold-open psychological constant; "founder = addicted to suffering"*

> "我一直就是非常希望能当直觉派天才派 ... 我在MIT的时候,我身边所有人他们都是天赋型选手 ... 但是呢,我不放弃,我是打不死的小强。"
> — Hong Letong, §2 [[00:08:32](https://youtu.be/bv8ghyTFF9w?t=512)] — *brute-force vs gifted; the "indestructible little cockroach" self-image*

> "他讲到一个概念就叫做Bonded Attention和Free Attention，Bonded Attention的意思是说被框架住的注意力，然后Free Attention是说自由注意力。"
> — Hong Letong, §3 [[00:15:52](https://youtu.be/bv8ghyTFF9w?t=952)] — *the framework that runs underneath the whole episode*

> "我在麻省理工的时候,我身边的每一个人,他都比我聪明,每一个,我是整个数学系里面最愚蠢的。"
> — Hong Letong, §4 [[00:34:01](https://youtu.be/bv8ghyTFF9w?t=2041)] — *MIT freshman-year baseline*

> "如果我们已经到了一个能拿AI去 看这个宪法是 什么意思的 这样的一个时代 我为什么不能拿AI做数学。"
> — Hong Letong, §6 [[01:04:29](https://youtu.be/bv8ghyTFF9w?t=3869)] — *the constitutional-textualism → Lean bridge insight*

> "就我某种程度上我觉得我是最不可能创业的一个创业者。我非常的喜欢向往学界，我非常的希望能够去做更多的数学的一些探索，我非常喜欢在大学的一个环境。"
> — Hong Letong, §7 [[01:17:50](https://youtu.be/bv8ghyTFF9w?t=4670)] — *"the most unlikely founder" self-label*

> "没有人喜欢融资 ... 你是一个复读机，你一次一次的说一样的事情，你一次一次的接到一样的问题。"
> — Hong Letong, §8 [[01:40:14](https://youtu.be/bv8ghyTFF9w?t=6014)] — *the parrot grind*

> "年轻做product是加分 年轻做deep tech是减分。"
> — Hong Letong, §8 [[01:49:40](https://youtu.be/bv8ghyTFF9w?t=6580)] — *the structural minus of being 24 in deep tech*

> "他是我之前老师对吧 对他给我发了一封邮件 ... 他说 如果他加入了OpenAI或者是DeepMind的话 ... 他希望我有一个心理准备 然后我心想你要加入OpenAI和DeepMind 那你为什么不来我这里。"
> — Hong Letong, §9 [[02:07:49](https://youtu.be/bv8ghyTFF9w?t=7669)] — *the Ken Ono email interception in one breath*

> "数学和代码某种程度上是孪生兄弟,Math is code and code is math ... 基于这个我每一个数学证明可以变成一个计算机程序。"
> — Hong Letong, §11 [[02:30:24](https://youtu.be/bv8ghyTFF9w?t=9024)] — *the Curry-Howard thesis in one line*

> "我们相信的一件事情就是,任何你能定义的,你都能证明。任何你能写出的或者是任何你能specify ... 任何你能表达的,你都能执行。这是我们对于coding未来的一个愿景。"
> — Hong Letong, §11 [[02:53:23](https://youtu.be/bv8ghyTFF9w?t=10403)] — *the slogan version of formal verification as commercial wedge*

> "这个事情它的结果一定是极好或极坏 ... 当然我是登月成功了要么登月失败 ... 为什么说像spacex呢要么火箭发上去了要么火箭坠毁了。"
> — Hong Letong, §14 [[03:39:40](https://youtu.be/bv8ghyTFF9w?t=13180)] — *binary moonshot framing*

> "我BET System。我不BET Model。我BET就是这个System有非常多的事情可以做。"
> — Hong Letong, §15 [[04:21:02](https://youtu.be/bv8ghyTFF9w?t=15662)] — *the one-line current bet*

## People mentioned

| Name | Context | First mentioned |
|------|---------|-----------------|
| Zhang Xiaojun (张小珺) | Host | [[00:00:01](https://youtu.be/bv8ghyTFF9w?t=1)] |
| Hong Letong (洪乐潼) | Guest — 00后 founder & CEO of Axiom | [[00:00:31](https://youtu.be/bv8ghyTFF9w?t=31)] |
| Mark Zuckerberg (扎克伯格) | Original occupant of Facebook House | [[00:00:18](https://youtu.be/bv8ghyTFF9w?t=18)] |
| Howard Morgan | B Capital partner; Renaissance Technologies co-founder; led Axiom's seed | [[00:02:05](https://youtu.be/bv8ghyTFF9w?t=125)] |
| Gauss (高斯) | Cited for "struck by lightning" metaphor; meeting-room namesake | [[00:02:36](https://youtu.be/bv8ghyTFF9w?t=156)] |
| Ramanujan (拉玛努金) | Notebook-of-results archetype; SSI's pre-training model namesake | [[00:06:35](https://youtu.be/bv8ghyTFF9w?t=395)] |
| Hardy (哈代) & Littlewood | Cambridge collaborators who proved Ramanujan's results | [[00:06:41](https://youtu.be/bv8ghyTFF9w?t=401)] |
| 张益唐 (Yitang Zhang) | Bounded gaps between primes | [[00:07:41](https://youtu.be/bv8ghyTFF9w?t=461)] |
| James Maynard | 2022 Fields Medalist; bounded-gaps alternate proof | [[00:08:08](https://youtu.be/bv8ghyTFF9w?t=488)] |
| Henry Cohn | MIT; gave Hong a 28-dim sphere-packing sub-problem | [[00:10:30](https://youtu.be/bv8ghyTFF9w?t=630)] |
| Evan Chen | US IMO coach; Axiom teammate who solved a Putnam problem with one picture | [[00:12:36](https://youtu.be/bv8ghyTFF9w?t=756)] |
| Elon Musk (马斯克) | "Visionary" founder type; SpaceX moonshot analogue | [[00:19:46](https://youtu.be/bv8ghyTFF9w?t=1186)] |
| Sam Altman | "Salesman" founder type | [[00:19:49](https://youtu.be/bv8ghyTFF9w?t=1189)] |
| Sheryl Sandberg (桑德伯格) | Kept Facebook culture stable 0→3000 employees | [[00:20:12](https://youtu.be/bv8ghyTFF9w?t=1212)] |
| Albert Einstein (爱因斯坦) | "Einstein-in-the-shower" Free Attention archetype | [[00:17:18](https://youtu.be/bv8ghyTFF9w?t=1038)] |
| 陶哲轩 (Terence Tao) | Cited for large collaborative formal-proof projects | [[00:30:12](https://youtu.be/bv8ghyTFF9w?t=1812)] |
| Alex Kontorovich | Tao's collaborator on collective formal proofs | [[00:30:12](https://youtu.be/bv8ghyTFF9w?t=1812)] |
| 任秋雨 / 张升桐 / 高继扬 | MIT classmates Hong had read about as a kid | [[00:34:24](https://youtu.be/bv8ghyTFF9w?t=2064)] |
| 余洪勋 | Taiwan IMO/IOI medalist, MIT Floor Pi dorm-mate | [[00:35:35](https://youtu.be/bv8ghyTFF9w?t=2135)] |
| Ken Ono (小野肯) | MIT/Emory math professor; ran the REU that rescued Hong; later Axiom hire | [[00:36:08](https://youtu.be/bv8ghyTFF9w?t=2168)] |
| Jensen Huang (黄仁勋) | "Pain and suffering" framing | [[00:45:52](https://youtu.be/bv8ghyTFF9w?t=2752)] |
| Buffett / Munger | Adage about hiring strong-technical-core leaders | [[00:49:29](https://youtu.be/bv8ghyTFF9w?t=2969)] |
| Scott Sheffield | MIT random-surfaces probabilist; pushed Hong toward physics | [[00:53:45](https://youtu.be/bv8ghyTFF9w?t=3225)] |
| Demis Hassabis | DeepMind founder; "fold everything" / AGI 1910 thought experiment | [[00:59:04](https://youtu.be/bv8ghyTFF9w?t=3544)] |
| Geoff Hinton | Funded Gatsby Institute at UCL | [[00:59:12](https://youtu.be/bv8ghyTFF9w?t=3552)] |
| Andrew Saxe | Hong's Oxford master's advisor | [[00:59:26](https://youtu.be/bv8ghyTFF9w?t=3566)] |
| Surya Ganguli | Stanford collaborator of Saxe | [[00:59:29](https://youtu.be/bv8ghyTFF9w?t=3569)] |
| Will Dorrell | Gatsby Institute researcher | [[00:59:36](https://youtu.be/bv8ghyTFF9w?t=3576)] |
| Kenny Long | MIT friend; one of 5-7 Buzzard students who hand-built Mathlib; now at Axiom | [[01:04:48](https://youtu.be/bv8ghyTFF9w?t=3888)] |
| Kevin Buzzard | Imperial mathematician who advised early Mathlib builders | [[01:05:29](https://youtu.be/bv8ghyTFF9w?t=3929)] |
| Shubo (叔伯) | Co-founder; Meta FAIR director; Baidu-mafia alum | [[01:10:47](https://youtu.be/bv8ghyTFF9w?t=4247)] |
| 田渊栋 | Joined Meta around same time as Shubo | [[01:11:01](https://youtu.be/bv8ghyTFF9w?t=4261)] |
| Ben Green | Sárközy-theorem-for-shifted-primes author | [[01:07:56](https://youtu.be/bv8ghyTFF9w?t=4076)] |
| Tudor | CEO of main competitor; also a Verve regular; said they only hire CS PhDs | [[01:21:18](https://youtu.be/bv8ghyTFF9w?t=4878)] |
| Andrew Ng (吴恩达 / 温达) | Brought Shubo into Baidu SVAIL 2014-16 | [[01:27:26](https://youtu.be/bv8ghyTFF9w?t=5246)] |
| Dario Amodei | Anthropic founder; Baidu Mafia | [[01:27:33](https://youtu.be/bv8ghyTFF9w?t=5253)] |
| Yann LeCun | Meta; 32B Code World Model team | [[01:24:09](https://youtu.be/bv8ghyTFF9w?t=5049)] |
| Percy Liang | "做惩罚" adversarial-benchmark philosophy | [[01:30:58](https://youtu.be/bv8ghyTFF9w?t=5458)] |
| Kaiyu Yang | First author of formal-theorem-proving survey | [[01:31:56](https://youtu.be/bv8ghyTFF9w?t=5516)] |
| Christian Szegedy | Started HOList at Google 2016 | [[01:36:49](https://youtu.be/bv8ghyTFF9w?t=5809)] |
| 吴宇怀 (Yuhuai Wu) | Co-founder of HOList | [[01:36:51](https://youtu.be/bv8ghyTFF9w?t=5811)] |
| Ilya Sutskever | Led OpenAI's GPT-f / miniF2F 2019-20; Google's failed counter-offer | [[01:37:15](https://youtu.be/bv8ghyTFF9w?t=5835)] |
| Jim Simons | Renaissance Technologies founder; gave a 2-hour talk to MIT math club | [[01:41:41](https://youtu.be/bv8ghyTFF9w?t=6101)] |
| Peter Thiel (皮特T) | "Monopoly drives innovation"; Zuckerberg's seed term sheet | [[01:51:22](https://youtu.be/bv8ghyTFF9w?t=6682)] |
| Matt Kroening | Menlo Ventures partner who led Series A | [[01:57:56](https://youtu.be/bv8ghyTFF9w?t=7076)] |
| Adam Wagner | DeepMind AI-for-Math; same UK school as Hong | [[01:39:26](https://youtu.be/bv8ghyTFF9w?t=5966)] |
| Albert Jiang | Same UK school as Hong and Adam Wagner | [[01:39:36](https://youtu.be/bv8ghyTFF9w?t=5976)] |
| François Charton (放缩 in ASR) | Co-author with Lample on Transformer-vs-integration; now at Axiom | [[02:13:33](https://youtu.be/bv8ghyTFF9w?t=8013)] |
| Guillaume Lample | Mistral co-founder; 2019 integration paper with Charton | [[02:13:40](https://youtu.be/bv8ghyTFF9w?t=8020)] |
| 张胜彤 | Hong's collaborator; more intuition-driven | [[02:15:40](https://youtu.be/bv8ghyTFF9w?t=8140)] |
| Andrew Granville | Canadian number theorist; high-abstraction reasoning thesis | [[02:17:53](https://youtu.be/bv8ghyTFF9w?t=8273)] |
| Maryam Mirzakhani (米尔扎扎克哈尼) | First female Fields medalist; Ono biopic subject | [[02:16:47](https://youtu.be/bv8ghyTFF9w?t=8207)] |
| 艾文晨 | Axiom teammate fluent at speed-Putnam | [[02:23:21](https://youtu.be/bv8ghyTFF9w?t=8601)] |
| Erdős | Author of *The BOOK*; dinner pick | [[02:23:52](https://youtu.be/bv8ghyTFF9w?t=8632)] |
| Tim Gowers | Fields medalist; pure-math-funds-applied argument | [[02:29:29](https://youtu.be/bv8ghyTFF9w?t=9769)] |
| Mitchell Polinsky | Stanford Law and Economics professor | [[02:28:49](https://youtu.be/bv8ghyTFF9w?t=9729)] |
| Leo de Moura | Lean creator; Axiom offered to sponsor open-source hammer | [[02:36:01](https://youtu.be/bv8ghyTFF9w?t=10561)] |
| David Silver & Richard Sutton | "Era of Experience" essay | [[02:38:11](https://youtu.be/bv8ghyTFF9w?t=10691)] |
| 董克帆 (Dong Kefan) & 马腾宇 (Ma Tengyu) | Stanford STP authors | [[02:43:18](https://youtu.be/bv8ghyTFF9w?t=10998)] |
| Donald Knuth | 1980s natural-language vibe-coding vision | [[02:58:02](https://youtu.be/bv8ghyTFF9w?t=11882)] |
| Alan Turing | Earliest formal-proof proponent | [[02:58:55](https://youtu.be/bv8ghyTFF9w?t=11935)] |
| Kevin Weil | OpenAI; informal AI-for-Math track | [[03:34:55](https://youtu.be/bv8ghyTFF9w?t=12895)] |
| Stefano Ermon | Inception Labs; faster diffusion reasoning models | [[03:43:54](https://youtu.be/bv8ghyTFF9w?t=13434)] |
| Saining Xie (赛宁) | Anti-Silicon-Valley researcher cluster | [[03:48:28](https://youtu.be/bv8ghyTFF9w?t=13708)] |
| Galois (加罗瓦) | Died at 21 | [[03:51:36](https://youtu.be/bv8ghyTFF9w?t=13896)] |
| Leibniz (莱布尼茨) | Universal Representation Theory — Hong's endpoint | [[03:50:36](https://youtu.be/bv8ghyTFF9w?t=13836)] |
| Charles Babbage (查理斯巴巴贝奇) | Babbage Engine — log-table precursor to computer | [[03:53:18](https://youtu.be/bv8ghyTFF9w?t=13998)] |
| G.H. Hardy | *A Mathematician's Apology* | [[03:53:30](https://youtu.be/bv8ghyTFF9w?t=14010)] |
| Victor Miller | Elliptic-curve cryptography founder | [[03:53:43](https://youtu.be/bv8ghyTFF9w?t=14023)] |
| Grothendieck | Dinner pick (only for algebraic-geometry fluent guests) | [[04:01:02](https://youtu.be/bv8ghyTFF9w?t=14462)] |
| Roger Heath-Brown | Oxford number theorist; birthday conference Hong attended | [[04:02:14](https://youtu.be/bv8ghyTFF9w?t=14534)] |
| Ada Lovelace | Meeting-room namesake | [[03:58:34](https://youtu.be/bv8ghyTFF9w?t=14314)] |
| Alan Turing / Alonzo Church | Meeting-room defense debate | [[03:58:34](https://youtu.be/bv8ghyTFF9w?t=14314)] |
| Emmy Noether | Meeting-room request rejected (only 5 rooms) | [[03:58:43](https://youtu.be/bv8ghyTFF9w?t=14323)] |
| Bryan Johnson | "Don't Die" — Hong's joke apprentice | [[04:09:46](https://youtu.be/bv8ghyTFF9w?t=14986)] |
| Ray Dalio | *Principles* — 鱼与熊掌 mindset | [[03:40:18](https://youtu.be/bv8ghyTFF9w?t=13218)] |
| 袁正 | ByteDance Doubao/Seed friend | [[04:16:14](https://youtu.be/bv8ghyTFF9w?t=15374)] |

## Papers / works mentioned

| Title | Context | Timestamp |
|-------|---------|-----------|
| Modularity Theorem / 椭圆曲线定理 | Algebraic ↔ geometric unification | [[00:02:54](https://youtu.be/bv8ghyTFF9w?t=174)] |
| Quadratic Reciprocity (二次互反率) | Ross camp 打怪 culmination | [[00:03:42](https://youtu.be/bv8ghyTFF9w?t=222)] |
| AlphaGeometry (Google DeepMind, ~2021) | ~81% of historical IMO geometry; symbolic encoding | [[00:08:35](https://youtu.be/bv8ghyTFF9w?t=515)] |
| Ross Mathematics Program (罗斯夏令营) | Hong's formative summer-camp | [[00:03:46](https://youtu.be/bv8ghyTFF9w?t=226)] |
| Putnam Mathematical Competition (1927–) | Axiom Prover 12/12 in 2025 | [[00:12:10](https://youtu.be/bv8ghyTFF9w?t=730)] |
| Axiom Prover (公司核心系统; ASR "action prover") | Hong's company's main system | [[00:12:13](https://youtu.be/bv8ghyTFF9w?t=733)] |
| Rudin *Principles of Mathematical Analysis* | Re-learned from scratch after 4/40 | [[00:44:09](https://youtu.be/bv8ghyTFF9w?t=2649)] |
| LegoProver | AI-for-Math paper; "math = Lego" feeling | [[00:51:10](https://youtu.be/bv8ghyTFF9w?t=3070)] |
| *Good Will Hunting* (film) | Childhood MIT inspiration | [[00:52:41](https://youtu.be/bv8ghyTFF9w?t=3161)] |
| Continual learning neurodynamics paper | Hong's master's thesis paper 1 | [[01:00:25](https://youtu.be/bv8ghyTFF9w?t=3625)] |
| One-layer linear transformer theory | Hong's master's thesis paper 2 | [[01:00:36](https://youtu.be/bv8ghyTFF9w?t=3636)] |
| Sárközy theorem for shifted primes (Ben Green) | Motivating number-theory problem class | [[01:07:56](https://youtu.be/bv8ghyTFF9w?t=4076)] |
| Davenport *analytic number theory* | "PhD student fluent in Davenport" benchmark | [[01:08:54](https://youtu.be/bv8ghyTFF9w?t=4134)] |
| Mathlib | Lean's math library; built by Buzzard students | [[01:05:14](https://youtu.be/bv8ghyTFF9w?t=3914)] |
| ATP Boost (Bartos, 2018) | Pre-AI formal theorem proving | [[01:21:53](https://youtu.be/bv8ghyTFF9w?t=4913)] |
| Pattern Boost (Charton) | Graph-pattern discovery for math examples | [[01:22:12](https://youtu.be/bv8ghyTFF9w?t=4932)] |
| Int-to-Int / IntoInt (Alberto Alfarino) | Translation-based AI math discovery | [[01:22:52](https://youtu.be/bv8ghyTFF9w?t=4972)] |
| LLM Compiler & Compiler Arena (Meta) | Hong's ML hires' origin | [[01:23:55](https://youtu.be/bv8ghyTFF9w?t=5035)] |
| Code World Model (Yann LeCun, 32B) | Axiom team overlap | [[01:24:09](https://youtu.be/bv8ghyTFF9w?t=5049)] |
| Frontier Math benchmark | Recruit who quit over "internet scale data set" | [[01:29:57](https://youtu.be/bv8ghyTFF9w?t=5397)] |
| "Formal Mathematical Reasoning: A New Frontier in AI" (Kaiyu Yang et al.) | Axiom's internal roadmap | [[01:31:56](https://youtu.be/bv8ghyTFF9w?t=5516)] |
| HOList (Christian Szegedy & 吴宇怀, Google, 2016) | First white paper Hong cites as field's foundation | [[01:36:49](https://youtu.be/bv8ghyTFF9w?t=5809)] |
| GPT-f / miniF2F (OpenAI, 2019-20) | Ilya/Jesse/Stan era | [[01:37:18](https://youtu.be/bv8ghyTFF9w?t=5838)] |
| AlphaProof (DeepMind, 2024) | 28分 / IMO silver, "the IMO-solved moment" | [[01:37:36](https://youtu.be/bv8ghyTFF9w?t=5856)] |
| Verina benchmark (Lean + Rust) | DeepSeek-Prover 11% vs Axiom 98.93% | [[01:59:55](https://youtu.be/bv8ghyTFF9w?t=7195)] |
| DeepSeek-Prover (DeepSeq Prover in ASR) | Axiom's nearest paradigm cousin | [[01:59:57](https://youtu.be/bv8ghyTFF9w?t=7197)] |
| ByteDance Doubao SeedProver / 豆包 seedprover | Closer paradigm than AlphaProof | [[02:05:37](https://youtu.be/bv8ghyTFF9w?t=7537)] |
| "Deep Learning for Symbolic Mathematics" (Lample & Charton, 2019) | Transformers > Mathematica on integration | [[02:13:46](https://youtu.be/bv8ghyTFF9w?t=8026)] |
| Ramanujan biopic | Ono helped produce | [[02:16:46](https://youtu.be/bv8ghyTFF9w?t=8206)] |
| *Proofs from THE BOOK* (Aigner & Ziegler / Erdős) | Source of company name "Axiom" | [[02:23:52](https://youtu.be/bv8ghyTFF9w?t=8632)] |
| Draft, Sketch, and Prove (2022) | Canonical AI-for-Math pipeline | [[02:34:13](https://youtu.be/bv8ghyTFF9w?t=10453)] |
| Lean Hammer (open-source, ~2025-06) | Pre-grind hammer | [[02:35:43](https://youtu.be/bv8ghyTFF9w?t=10543)] |
| Lean `grind` tactic (2026) | Solves many problems with zero AI | [[02:36:23](https://youtu.be/bv8ghyTFF9w?t=10583)] |
| Anthropic sub-agents work | Inspired Axiom's MCTS replacement | [[02:38:01](https://youtu.be/bv8ghyTFF9w?t=10681)] |
| Silver & Sutton "Era of Experience / AI 后半场" | Sub-agent design motivation | [[02:38:11](https://youtu.be/bv8ghyTFF9w?t=10691)] |
| STP: Self-Play Theorem Prover (Dong Kefan, Ma Tengyu) | Elegance-length filter for conjecturer | [[02:43:13](https://youtu.be/bv8ghyTFF9w?t=10993)] |
| Lean Workbook | STP training corpus | [[02:44:29](https://youtu.be/bv8ghyTFF9w?t=11069)] |
| AlphaEvolve / Alythia | DeepMind natural-language closer | §11 |
| Cadence Jasper (SMT-based hardware verifier) | Potential Lean-replacement target | [[02:51:48](https://youtu.be/bv8ghyTFF9w?t=10308)] |
| FrontierMath / First Proof challenge | 10-problem AI-math challenge (NYT-covered) | [[02:55:37](https://youtu.be/bv8ghyTFF9w?t=10537)] |
| miniF2F (高中级 benchmark) | Saturated by Axiom in ~2 months | [[03:09:17](https://youtu.be/bv8ghyTFF9w?t=11357)] |
| Action Pro / Axiom Pro | Generates 20-page proofs; chip verification | [[03:20:46](https://youtu.be/bv8ghyTFF9w?t=12046)] |
| Periodic Labs / Lila Science / FutureHouse-Edison | AI-for-science peers | [[03:23:30](https://youtu.be/bv8ghyTFF9w?t=12210)] |
| *The Thinking Game* (Demis Hassabis documentary) | "Fold everything" scene | [[02:27:32](https://youtu.be/bv8ghyTFF9w?t=9452)] |
| Attention Is All You Need (Transformer paper) | Cited en route to pre-training discussion | [[03:31:21](https://youtu.be/bv8ghyTFF9w?t=12681)] |
| *Principles* (Ray Dalio) | 鱼与熊掌 mindset | [[03:40:18](https://youtu.be/bv8ghyTFF9w?t=13218)] |
| *A Mathematician's Apology* (Hardy) | Refuted by elliptic-curve cryptography | [[03:53:30](https://youtu.be/bv8ghyTFF9w?t=14010)] |
| Universal Representation Theory (Leibniz) | Hong's stated endpoint | [[03:50:39](https://youtu.be/bv8ghyTFF9w?t=13839)] |
| MIT course: *Learning to Fail* | Cited in moonshot framing | [[03:44:06](https://youtu.be/bv8ghyTFF9w?t=13446)] |
| Davenport *分析数论* | Hong's reading recommendation | [[04:11:59](https://youtu.be/bv8ghyTFF9w?t=15119)] |
| *红楼梦* | Personal reading | [[04:12:10](https://youtu.be/bv8ghyTFF9w?t=15130)] |
| *大雅宝胡同甲二号* (黄永玉) | Childhood favorite | [[04:12:23](https://youtu.be/bv8ghyTFF9w?t=15143)] |
| Musk biography (银人太太 / Silicon Titans) | Normalized "howling at night" | [[04:12:35](https://youtu.be/bv8ghyTFF9w?t=15155)] |
| 24 诗品 — 如将不尽，于古为新 | Closing tag | [[04:19:48](https://youtu.be/bv8ghyTFF9w?t=15588)] |

## Numbers & claims

| Claim | Speaker | Timestamp |
|-------|---------|-----------|
| Series A valuation **$1.6B** (16亿美元) | Host §1 | [[00:00:35](https://youtu.be/bv8ghyTFF9w?t=35)] |
| **57-year-old American tenured professor** quits to join **24-year-old Chinese woman** | Host §1 | [[00:00:45](https://youtu.be/bv8ghyTFF9w?t=45)] |
| **00后** (post-2000) Chinese woman | Host §1 | [[00:00:29](https://youtu.be/bv8ghyTFF9w?t=29)] |
| AlphaGeometry solves **~81%** of historical IMO geometry | Guest §2 | [[00:10:01](https://youtu.be/bv8ghyTFF9w?t=601)] |
| Axiom Prover = **6th perfect score in 98 years** since Putnam started **1927**; first by an AI | Guest §2 | [[00:12:13](https://youtu.be/bv8ghyTFF9w?t=733)] |
| **6 months** stuck on 28-dim sphere-packing sub-problem under Henry Cohn | Guest §2 | [[00:10:30](https://youtu.be/bv8ghyTFF9w?t=630)] |
| Brute-force solver pays **2-3×** the time | Guest §2 | [[00:09:28](https://youtu.be/bv8ghyTFF9w?t=568)] |
| 10-min walk to school in Guangzhou | Guest §3 | [[00:15:36](https://youtu.be/bv8ghyTFF9w?t=936)] |
| Sandberg kept Facebook culture **0 → 3000 employees** | Guest §3 | [[00:20:15](https://youtu.be/bv8ghyTFF9w?t=1215)] |
| Olympiad school re-sorts students into classes **1–24** monthly; Hong started in **class 4** | Guest §3 | [[00:26:12](https://youtu.be/bv8ghyTFF9w?t=1572)] |
| Math 'tribe' = **3-5 people**; middle-school math group **25-27 of 90** | Guest §3 | [[00:31:54](https://youtu.be/bv8ghyTFF9w?t=1914)] |
| Knight's-tour conjecture: any **n ≥ 5** board has a full tour | Guest §3 | [[00:29:38](https://youtu.be/bv8ghyTFF9w?t=1778)] |
| **75 mock exams** in a short period for an Olympiad selection round | Guest §4 | [[00:39:01](https://youtu.be/bv8ghyTFF9w?t=2341)] |
| PhD measure-theory midterm: class avg **~9/40**, Hong **4/40** | Guest §4 | [[00:43:41](https://youtu.be/bv8ghyTFF9w?t=2621)] |
| Born **2001** (零一年), age **24** at recording | Guest §4 | [[00:47:31](https://youtu.be/bv8ghyTFF9w?t=2851)] |
| Morgan Prize: **30-year history**, 1 winner + 2-3 runner-ups per year | Guest §5 | [[00:56:31](https://youtu.be/bv8ghyTFF9w?t=3391)] |
| **More than half** of Hong's undergrad papers are with Ken Ono | Guest §5 | [[00:57:35](https://youtu.be/bv8ghyTFF9w?t=3455)] |
| **5-7 Kevin Buzzard students** hand-built early Mathlib by typing out UG algebra/analysis | Guest §6 | [[01:05:22](https://youtu.be/bv8ghyTFF9w?t=3922)] |
| **1,000-coffee Polaroid** rule at Verve; Hong + Shubo each earned one | Guest §6 | [[01:06:25](https://youtu.be/bv8ghyTFF9w?t=3985)] |
| **30 pages of reading per class** at Stanford Law | Guest §6 | [[01:11:14](https://youtu.be/bv8ghyTFF9w?t=4274)] |
| Friends with Shubo for **1.5 years** before knowing professional identities | Guest §6 | [[01:14:39](https://youtu.be/bv8ghyTFF9w?t=4479)] |
| AI-for-Math GitHub repo = **几百篇文章** (hundreds of papers) | Guest §7 | [[01:18:53](https://youtu.be/bv8ghyTFF9w?t=4733)] |
| Yann LeCun's **Code World Model 32B** | Guest §7 | [[01:24:09](https://youtu.be/bv8ghyTFF9w?t=5049)] |
| Axiom headcount ~**30**; founding-team rule = no math hire until headcount 15 (**15:1 ML:math ratio**) | Guest §7 | [[01:29:14](https://youtu.be/bv8ghyTFF9w?t=5354)] |
| **AlphaProof 28/42 at IMO 2024**, "one shy of gold" | Guest §7 | [[01:37:49](https://youtu.be/bv8ghyTFF9w?t=5869)] |
| Axiom **Putnam perfect score, December 2025** | Guest §7 | [[01:38:11](https://youtu.be/bv8ghyTFF9w?t=5891)] |
| First → final seed offer = **3×** through bidding | Guest §8 | [[01:40:03](https://youtu.be/bv8ghyTFF9w?t=6003)] |
| Seed: **$64M (6400万) on $300M (三亿) post**, target $50M | Guest §8 | [[01:49:06](https://youtu.be/bv8ghyTFF9w?t=6546)] |
| **3 competing seed lead offers** | Guest §8 | [[01:44:52](https://youtu.be/bv8ghyTFF9w?t=6292)] |
| Series A out-of-town pitch on **1月5号 2026** | Guest §8 | [[01:56:52](https://youtu.be/bv8ghyTFF9w?t=7012)] |
| Series A: **$1.6B (16亿美金) valuation, ≥$200M (两亿美金) raised** | Guest §8 | [[01:58:28](https://youtu.be/bv8ghyTFF9w?t=7108)] |
| Menlo had a **1% check in Axiom's seed** | Guest §8 | [[01:58:16](https://youtu.be/bv8ghyTFF9w?t=7096)] |
| Putnam perfect score at **month 4**; research problems at **month 6** | Guest §8 | [[01:59:22](https://youtu.be/bv8ghyTFF9w?t=7162)] |
| Verina benchmark: **DeepSeek-Prover 11% vs Axiom 98.93%** | Guest §8 | [[01:59:55](https://youtu.be/bv8ghyTFF9w?t=7195)] |
| Axiom's verify-proof checker is **~100× faster** than community comparator | Guest §8 | [[02:05:03](https://youtu.be/bv8ghyTFF9w?t=7503)] |
| Built **~12-13 Lean tooling components** | Guest §8 | [[02:05:10](https://youtu.be/bv8ghyTFF9w?t=7510)] |
| Competitor had **5× Axiom's funding AND 5× their valuation when Axiom started** | Guest §8 | [[02:06:11](https://youtu.be/bv8ghyTFF9w?t=7571)] |
| Older competitor took **2 years** to solve 5 of 6 IMO problems | Guest §8 | [[02:06:24](https://youtu.be/bv8ghyTFF9w?t=7584)] |
| Ono closing happened in **2-3 days** over Zoom | Guest §9 | [[02:13:23](https://youtu.be/bv8ghyTFF9w?t=8003)] |
| **2019 Lample-Charton integration paper** | Guest §9 | [[02:13:38](https://youtu.be/bv8ghyTFF9w?t=8018)] |
| Putnam test day: **2025年12月6号** | Guest §10 | [[02:20:05](https://youtu.be/bv8ghyTFF9w?t=8405)] |
| Putnam structure: **6 morning + 6 afternoon problems** | Guest §10 | [[02:20:29](https://youtu.be/bv8ghyTFF9w?t=8429)] |
| **3:58 pm: 8/12 = 80/120** = world top-5 in 2024 | Guest §10 | [[02:21:08](https://youtu.be/bv8ghyTFF9w?t=8468)] |
| Final: **12/12 perfect Putnam score** | Guest §10 | [[02:21:39](https://youtu.be/bv8ghyTFF9w?t=8499)] |
| Typically **≤3 of 12** Putnam problems are 求解 (numeric-answer) | Guest §10 | [[02:23:06](https://youtu.be/bv8ghyTFF9w?t=8586)] |
| AlphaFold: **2亿个蛋白** → Demis "fold everything" | Guest §11 | [[02:27:55](https://youtu.be/bv8ghyTFF9w?t=9475)] |
| Hypothetical compute allocation: **200 H200 / 8000 H200** per problem | Guest §11 | [[02:27:07](https://youtu.be/bv8ghyTFF9w?t=9427)] |
| Proof-search: **40 nodes → 4,000 nodes** with sub-agents | Guest §11 | [[02:39:01](https://youtu.be/bv8ghyTFF9w?t=9541)] |
| FrontierMath-style 10-problem challenge: **OpenAI 5/10, DeepMind 6/10** | Guest §11 | [[02:55:48](https://youtu.be/bv8ghyTFF9w?t=10548)] |
| Auto-formalization blueprint: **20-page article → 200-500 pages** | Guest §11 | [[02:47:55](https://youtu.be/bv8ghyTFF9w?t=10075)] |
| Amazon: **3-5 years**, **260,000 lines** of Lean-style proofs to verify hypervisor memory isolation | Guest §11 | [[03:01:30](https://youtu.be/bv8ghyTFF9w?t=10890)] |
| **mini F2F saturated in ~2 months** | Guest §12 | [[03:09:13](https://youtu.be/bv8ghyTFF9w?t=11353)] |
| **Action Pro generates a 20-page math proof** | Guest §13 | [[03:20:46](https://youtu.be/bv8ghyTFF9w?t=12046)] |
| Headcount: **15 in Dec when Can joined → ~30 now** | Guest §13 | [[03:22:28](https://youtu.be/bv8ghyTFF9w?t=12148)] |
| Competitor headcount: **50-75** | Guest §13 | [[03:22:36](https://youtu.be/bv8ghyTFF9w?t=12156)] |
| Demis's AGI benchmark: train AI to **1910** to see if it rediscovers general relativity | Guest §13 | [[03:13:55](https://youtu.be/bv8ghyTFF9w?t=11635)] |
| Axiom is **7 months old** at recording | Guest §14 | [[03:37:52](https://youtu.be/bv8ghyTFF9w?t=13072)] |
| **$100M (100米脸) packages can't suppress curiosity** | Guest §14 | [[03:43:46](https://youtu.be/bv8ghyTFF9w?t=13426)] |
| **Galois died at 21, Ramanujan ~30** | Guest §14 | [[03:51:36](https://youtu.be/bv8ghyTFF9w?t=13896)] |
| Axiom has **5 meeting rooms** (Gauss, Poincaré, Hilbert, Lovelace, Turing) | Guest §15 | [[03:58:47](https://youtu.be/bv8ghyTFF9w?t=14327)] |
| Genuinely novel proof-method invention: **5-10 years** | Guest §15 | [[04:05:00](https://youtu.be/bv8ghyTFF9w?t=14700)] |
| **Startup mortality 99%** | Guest §15 | [[04:09:42](https://youtu.be/bv8ghyTFF9w?t=14982)] |
| **DeepSeek Prover claimed 49 of miniF2F, actually 47** (2 cheats with axioms) | Guest §15 | [[04:14:30](https://youtu.be/bv8ghyTFF9w?t=15270)] |
| Hong was **16** when she wrote about math as discovering truth | Guest §15 | [[04:05:26](https://youtu.be/bv8ghyTFF9w?t=14726)] |

## Open questions / gaps

- **Putnam "6th perfect score in 98 years" framing** is asserted without official Putnam Fellows-history sourcing. Multiple humans have earned 120/120 historically; the "5 prior + AI is 6th" count is plausible but unverified in-audio.
- **AlphaGeometry's "81% of historical IMO geometry"** — number stated without explicit citation to the 2024 *Nature* paper or specific benchmark version. AlphaGeometry was publicly released in 2024; the "started 2021 in secret" claim isn't sourced.
- **"5-7 Buzzard students" hand-built early Mathlib** — Hong's personal recall via Kenny Long; community history of Mathlib involves many more contributors.
- **Verina 11% vs 98.93%** — DeepSeek-Prover comparator version not specified; "first place" claim is uncited.
- **Tudor's "we only hire CS PhDs"** rationale is admitted to be unexplained — Hong says she "never got an answer."
- **15:1 ML-to-mathematician ratio** is admitted to be **"随便说了一个"** — no principled basis.
- **"Menlo's second-largest AI investment after Anthropic"** and **"Anthropic's largest institutional investor"** stated without source.
- **"Axiom Prover handled the Weil Conjecture family"** stated as a personal AlphaGo moment without paper / benchmark reference.
- **"Robinhood CEO/founder still full-time at Robinhood while backing the competitor"** asserted as hearsay.
- **"open AI曾经一度发不出工资"** — motivational lore without sourcing.
- **"SSI trained a model named Ramanujan"** — attributed to "人家说," no source.
- **Lample-Charton integration paper is "2019"** — Hong says 2019; the canonical *Deep Learning for Symbolic Mathematics* paper is on arXiv from Dec 2019 (publication 2020), so the framing is essentially correct.
- **"Amazon spent 3-5 years on 260K lines of Lean-style proofs for hypervisor memory isolation"** — almost certainly refers to AWS Cedar / s2n / Cryptol / Dafny work, but Hong doesn't name the specific project; figure is unsourced in-audio.
- **"AI specialization sector covers code verification + physics, won't reach Nobel-literature"** — geometric intuition only, no argument.
- **Library learning + faithfulness** described as the field's bottleneck without concrete progress signal or candidate approach.
- **"Recursive Self Improvement is very fast to come"** — no timeline beyond "very fast."
- **"传统咨询会死"** under Forward Deployment — asserted without supporting argument.
- **Quote ASR rendering caveats**: Axiom = ASR "Oxfam"/"Action"; Verina = "Varina"; AlphaProof = "alpha proof"; Putnam = "普特兰"/"普特南"; Curry-Howard appears with the English spelling intact; "Lean" frequently renders as "令"; "Demis Hassabis" as "等密室"; Geoffrey Hinton sometimes "Jeffrey Hinton"; François Charton as "Francois Chardin" or "放缩". The wiki standardizes to correct English forms while citing ASR variants when load-bearing.

## Verification log

- **Sectioning**: chapters (15 author-supplied YouTube chapters); chapter #1 was `<Untitled Chapter 1>` and was renamed to "开场白与本期定位" in the sections JSON before chunking. All other 14 titles preserved verbatim from `info.json.chapters`.
- **Transcript source**: faster-whisper large-v3 (CPU int8, local M4) — produced by `docs/videos/transcripts/bv8ghyTFF9w.txt` from the project's batch driver. YouTube provided no subtitles (manual or auto) for this video, so the standard yt-dlp path was unavailable.
- **Speaker name corrections**: guest's name rendered "洪乐彤" in ASR; corrected throughout to **洪乐潼 (Hong Letong)** per Axiom team page. Host name corrected from "小俊"/"小骏" to **张小珺 (Zhang Xiaojun)**. Company name corrected from "Oxfam"/"Action" to **Axiom**; "Axiom Prover" / "AXIEM" appears as "action prover" in ASR.
- **Sections covered**: 15/15 ✅
- **Notable quotes traced verbatim**: 13/13 ✅ (each anchored by a distinctive 6-15-char substring in the local flat transcript built from `bv8ghyTFF9w.transcript.txt`)
- **Numbers traced**: 64/64 ✅ (each row anchored by either the number itself or a distinctive surrounding phrase; allowed flexible variants — `64个million` for `$64M`, `6400万` for `$64M`, `三亿` for `$300M`, `普特兰` for `Putnam`, `两亿美金` for `$200M+`, `1月5号` for `January 5`, etc.)
- **Sectioning method used**: chapters
- **Removed during verification**: none

## See Also

- `docs/videos/2026-05-01-su-yu-agent-tech-history.md` — Zhang Xiaojun's interview with Su Yu (Ohio State NLP → Neocognition); same host, same channel, Agent-research-track companion to this AI-for-Math episode.
- `docs/videos/2026-05-11-yao-shunyu-training-claude-gemini.md` — Yao Shunyu interview; same host; ReAct-author perspective from inside Anthropic / GDM, complementary to Hong's outside-the-lab founder view.
- `skills/research/youtube-wiki/SKILL.md` — the pipeline that produced this entry.
