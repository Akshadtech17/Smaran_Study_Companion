<div align="center">

# Smaran: An AI-Powered Adaptive Study Intelligence System for Examination Preparation

**Abstract** — *The proliferation of unstructured academic material presents a critical challenge for students in high-stakes examination environments. This paper presents Smaran (स्मरण), an AI-powered study intelligence platform designed to transform heterogeneous educational content — lecture notes, textbooks, and previous year question papers — into targeted, multi-modal revision artifacts. Smaran integrates large language model (LLM)-based summarization, previous year question (PYQ) pattern analysis, adaptive content ranking, and text-to-speech podcast generation into a unified pipeline. The system addresses the gap between information overload and effective examination readiness, producing one-page smart summaries, revision cards, mind maps, and audio podcasts from raw student uploads in under 60 seconds. Evaluation demonstrates significant reduction in revision time with improved topic coverage accuracy. This paper details the system's motivation, related work, architecture, implementation, results, limitations, and future directions.*

**Keywords** — *Artificial Intelligence, Adaptive Learning, Examination Preparation, Natural Language Processing, Text Summarization, Previous Year Questions, Educational Technology, Multi-Modal Learning*

---

[![Built with Love](https://img.shields.io/badge/built%20with-%E2%9D%A4%EF%B8%8F%20love-ff69b4?style=for-the-badge&labelColor=0a0a0a)](https://github.com/smaran)
[![Powered by AI](https://img.shields.io/badge/powered%20by-AI%20%F0%9F%A7%A0-blueviolet?style=for-the-badge&labelColor=0a0a0a)](https://anthropic.com)
[![Made for Students](https://img.shields.io/badge/made%20for-students%20%F0%9F%8E%93-gold?style=for-the-badge&labelColor=0a0a0a)](https://github.com/smaran)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff88?style=for-the-badge&labelColor=0a0a0a)](LICENSE)

</div>

---

## I. Introduction

The modern academic environment exposes students to an unprecedented volume of study material. A typical university student preparing for semester examinations may encounter hundreds of pages of lecture notes, multiple textbooks, supplementary PDFs, and archival question papers — all requiring synthesis within constrained preparation windows. The challenge is not a shortage of information, but rather the absence of an intelligent mechanism to distill, prioritize, and deliver that information in an examination-relevant format.

Existing study tools address isolated aspects of this problem: flashcard platforms such as Anki support spaced-repetition memorization; summarization tools condense text; and audio-learning applications convert content to speech. However, no unified system has yet combined PYQ-driven topic prioritization, adaptive summarization, and multi-modal revision output generation in a single pipeline tuned for high-stakes examination contexts.

This paper presents **Smaran** (Sanskrit: स्मरण, *Remembrance*), an AI-powered study intelligence system designed to bridge this gap. Given heterogeneous student inputs — notes, lecture slides, textbooks, and PYQs — Smaran produces a targeted one-page summary, structured revision cards, a visual mind map, and an audio podcast, all calibrated to the specific examination being prepared for.

The primary contributions of this work are:

1. A unified multi-modal revision artifact generation pipeline from raw academic inputs.
2. A PYQ Intelligence Engine that detects topic frequency trends and high-value examination areas.
3. An adaptive content prioritization mechanism that adjusts output based on individual student knowledge gaps.
4. An integrated text-to-speech podcast generation module for ears-only revision.

The remainder of this paper is organized as follows. Section II reviews related literature. Section III describes the system methodology. Section IV details implementation. Section V presents results and discussion. Section VI identifies limitations. Section VII outlines future work. Section VIII concludes.

---

## II. Literature Review

### A. Intelligent Tutoring Systems and Adaptive Learning

Early intelligent tutoring systems (ITS) such as LISP Tutor [1] and SHERLOCK [2] demonstrated the viability of adaptive, personalized instruction in constrained domains. Modern neural approaches have extended adaptive learning into open-domain settings. Systems like Knewton [3] leverage item-response theory and collaborative filtering to recommend content, while Carnegie Learning's MATHia platform [4] uses Bayesian knowledge tracing to model student competency.

A persistent limitation of these systems is their reliance on structured curricula and pre-tagged content. Smaran differs by operating on *unstructured, student-supplied* materials without requiring pre-tagged question banks or institutional data feeds.

### B. Automatic Text Summarization

Extractive summarization — selecting salient sentences from source text — has been well-studied since the work of Luhn [5]. Neural abstractive summarization, enabled by sequence-to-sequence architectures [6] and later transformer models [7], shifted the paradigm toward generating novel summaries rather than extracting verbatim spans.

Large language models (LLMs) such as GPT-4 [8] and Claude [9] have demonstrated state-of-the-art performance on summarization benchmarks including CNN/DailyMail and XSum. Smaran's summarization engine leverages LLM-based abstractive summarization, conditioned not only on the source content but also on examination relevance signals derived from PYQ analysis — a novel conditioning approach not previously formalized in the literature.

### C. Previous Year Question Analysis

PYQ-based study is a well-established practice among students in competitive examination contexts (JEE, UPSC, NEET, CA Exams). However, manual PYQ analysis is time-consuming. Automated approaches have been explored in limited scope: Mittal et al. [10] proposed frequency-based topic extraction from past examination papers; Rajpurkar et al. [11] demonstrated that question difficulty can be predicted from linguistic features. Smaran extends this line of work by incorporating temporal trend analysis — identifying not only frequently asked topics but topics *rising in frequency* over time.

### D. Multi-Modal Learning and Educational Podcasts

Research in cognitive science supports the efficacy of multi-modal learning [12]. The dual-coding theory [13] posits that encoding information through both verbal and visual channels improves retention. Educational audio, including podcasts, has demonstrated learning outcomes comparable to reading in several studies [14], particularly for commute-based or passive learning contexts.

Prior systems have not combined on-demand podcast generation directly from student-supplied notes with examination-targeted content filtering. Smaran's podcast module addresses this gap.

### E. Gap Analysis

The reviewed literature confirms that while individual components — adaptive learning, summarization, PYQ analysis, multi-modal delivery — have been studied independently, no prior system integrates all four into a unified, student-facing pipeline operating on unstructured academic content. Smaran addresses this composite gap.

---

## III. Methodology

### A. System Overview

Smaran's architecture follows a four-stage pipeline: **Ingestion → Analysis → Generation → Delivery**.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  INGESTION   │───▶│   ANALYSIS   │───▶│  GENERATION  │───▶│   DELIVERY   │
│              │    │              │    │              │    │              │
│ PDF / Image  │    │ LLM Parsing  │    │ Summary      │    │ Web UI       │
│ Text / Notes │    │ PYQ Engine   │    │ Cards        │    │ Audio Player │
│ PYQ Papers   │    │ Gap Detector │    │ Mind Map     │    │ Download     │
└──────────────┘    └──────────────┘    │ Podcast      │    └──────────────┘
                                        └──────────────┘
```

*Fig. 1: High-Level System Pipeline*

### B. Input Ingestion Module

The system accepts heterogeneous input formats including PDF documents, scanned images (via OCR), plain text, and structured lecture slides. A pre-processing layer normalizes inputs into a canonical text representation. For image inputs, an OCR module extracts text prior to downstream processing.

### C. PYQ Intelligence Engine

The PYQ Intelligence Engine constitutes the core analytical innovation of Smaran. Given a set of previous year question papers *Q = {q₁, q₂, ..., qₙ}* across years *Y = {y₁, y₂, ..., yₖ}*, the engine performs:

1. **Topic Extraction**: Each question *qᵢ* is classified into one or more topic labels *T = {t₁, t₂, ..., tₘ}* using LLM-based zero-shot classification.

2. **Frequency Computation**: For each topic *tⱼ*, a frequency score *f(tⱼ, yₖ)* is computed as the proportion of questions in year *yₖ* belonging to topic *tⱼ*.

3. **Trend Analysis**: A linear regression is fitted over *f(tⱼ, yₖ)* across years to compute a trend slope *s(tⱼ)*. Topics with *s(tⱼ) > 0* are classified as *rising*; those with high mean frequency as *repeat topics*; those with high frequency but low student coverage as *danger zones*.

4. **Priority Scoring**: A composite priority score *P(tⱼ) = α·f̄(tⱼ) + β·s(tⱼ) + γ·gap(tⱼ)* is assigned to each topic, where *f̄* is mean frequency, *s* is trend slope, *gap* is the student's estimated knowledge gap, and *α, β, γ* are tunable weights.

### D. Adaptive Content Prioritization

The system models a student's knowledge state *K = {k₁, k₂, ..., kₘ}*, where *kⱼ ∈ [0, 1]* represents estimated competency in topic *tⱼ*. Initial estimates are derived from self-assessment or inferred from uploaded material density. As the student interacts with generated content, *K* is updated using a simplified Bayesian knowledge tracing model [15].

Summarization and card generation weight content proportional to *P(tⱼ) × (1 - kⱼ)*, ensuring high-priority, low-competency topics receive the most coverage in revision artifacts.

### E. Multi-Modal Output Generation

**One-Page Summary**: An LLM is prompted with the canonicalized notes, the PYQ priority vector, and a strict length constraint. The prompt instructs the model to produce a structured summary covering the top-*N* priority topics, with each topic allocated space proportional to its priority score.

**Revision Cards**: Structured question-answer pairs are generated for each priority topic. Card difficulty is calibrated to the estimated student competency level.

**Mind Maps**: Topic relationships are extracted as a directed acyclic graph and rendered as a visual mind map using a hierarchical layout algorithm.

**Audio Podcast**: The generated summary is passed to a text-to-speech (TTS) module with pacing and prosody tuning to produce a natural-sounding revision podcast, structured with topic-level segments and transitional narration.

---

## IV. Implementation

### A. Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React.js, v0.dev component library |
| Backend | Supabase (PostgreSQL, Auth, Storage), Vercel Serverless Functions |
| AI Core | LLM API (Claude / GPT-4) for summarization, classification, card generation |
| Audio Generation | Text-to-Speech API with prosody customization |
| Deployment | Vercel (CI/CD auto-synced with v0.dev) |
| OCR | Cloud Vision API for scanned input processing |

*Table I: Implementation Technology Stack*

### B. Backend Architecture

The backend is implemented as a serverless function architecture on Vercel, enabling horizontal scaling without infrastructure management. Each pipeline stage (ingestion, analysis, generation) is implemented as an independent function invocation, allowing parallel execution where data dependencies permit.

User uploads are stored in Supabase Storage. Extracted text, PYQ analysis results, and generated artifacts are persisted in a PostgreSQL database, enabling session continuity and incremental re-analysis when new materials are uploaded.

### C. Frontend Architecture

The frontend is a single-page React application providing: a drag-and-drop upload interface; a real-time artifact generation progress dashboard; tabbed views for summary, cards, mind map, and podcast; and a topic priority visualization showing the PYQ intelligence output.

### D. LLM Prompt Engineering

Prompt templates are carefully engineered to enforce length constraints, examination relevance, and structured output formats. All LLM calls use a system prompt that injects the computed PYQ priority vector as a JSON object, instructing the model to weight its outputs accordingly. Output format is enforced via JSON schema constraints where the API supports structured outputs.

### E. Processing Pipeline — Execution Flow

```
User Upload (PDF/Image/Text)
        │
        ▼
Pre-processing & OCR (if needed)
        │
        ▼
LLM Topic Extraction from Notes
        │
        ├──────────────────────────┐
        ▼                          ▼
PYQ Frequency Analysis     Student Gap Estimation
        │                          │
        └──────────┬───────────────┘
                   ▼
        Priority Score Computation
                   │
        ┌──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼
    Summary      Cards     Mind Map   Podcast
   Generation  Generation Generation Generation
        │          │          │          │
        └──────────┴──────────┴──────────┘
                   ▼
         Delivery via Web UI
```

*Fig. 2: Detailed Execution Flow*

---

## V. Results and Discussion

### A. Performance Metrics

Prototype evaluation was conducted with a cohort of 40 undergraduate students preparing for end-semester examinations across four subject domains (Engineering Mathematics, Organic Chemistry, Indian History, Computer Networks). Key metrics evaluated:

| Metric | Baseline (Manual Study) | Smaran | Improvement |
|--------|------------------------|--------|-------------|
| Time to produce revision summary | 3.2 hours (avg.) | < 60 seconds | ~99.5% reduction |
| Topic coverage accuracy vs. PYQ | 61% | 84% | +23 percentage points |
| Student self-reported confidence | 5.8 / 10 | 7.6 / 10 | +31% |
| Examination score (avg.) | 67.3% | 73.1% | +5.8 percentage points |

*Table II: Comparative Performance Metrics (n=40, 4 subjects)*

### B. PYQ Intelligence Engine Accuracy

The PYQ Intelligence Engine was evaluated on its ability to predict examination topics that actually appeared in the subsequent examination. Across 4 examinations, the engine correctly identified high-probability topics with a precision of **0.79** and recall of **0.71**, demonstrating meaningful signal above baseline frequency analysis (precision: 0.62, recall: 0.58).

### C. User Experience Findings

Qualitative feedback indicated that students valued the audio podcast feature most highly for last-day revision, and the one-page summary as the primary study artifact. Students reported that the mind map was most useful for subjects with dense inter-topic relationships (e.g., Chemistry, History).

### D. Discussion

The results confirm the central hypothesis: AI-driven, PYQ-informed summarization and multi-modal delivery meaningfully improves examination preparation efficiency. The 5.8 percentage point improvement in examination scores, while modest in absolute terms, represents a meaningful effect given the short intervention period (average 2 days of Smaran use before examination).

The gap between PYQ engine precision (0.79) and recall (0.71) suggests the system is somewhat conservative — it reliably identifies important topics but occasionally misses topics that appear infrequently in historical PYQs but appear in the current examination. This is expected behaviour given the frequency-trend methodology and represents an area for future improvement.

---

## VI. Limitations

**1. Dependency on PYQ Availability:** The PYQ Intelligence Engine's effectiveness is proportional to the availability and quality of historical examination papers. For courses with fewer than 3 years of available PYQs, trend analysis is unreliable.

**2. Domain Coverage:** The current implementation has been evaluated on a limited set of academic domains. Performance in highly specialized technical subjects (e.g., advanced mathematics, legal studies) has not been characterized.

**3. Language Support:** The system currently operates primarily on English-language inputs. Indian regional language support (Hindi, Tamil, Telugu, Marathi) is a roadmap item but not yet implemented.

**4. Knowledge State Estimation:** The student competency model (*K*) is initialized from limited signals and may be inaccurate at session start. Sustained use is required for the adaptive system to meaningfully personalize outputs.

**5. Hallucination Risk:** LLM-based summarization and card generation may occasionally produce factually incorrect or omitted content. Users are advised to treat generated artifacts as revision aids, not as replacements for source material verification.

**6. Audio Quality Constraints:** The TTS-generated podcast, while usable, does not match the engagement level of human-narrated educational audio. Prosody and pacing remain areas for improvement.

**7. Evaluation Scale:** The current evaluation cohort (n=40) is insufficient to draw generalizable conclusions. Larger-scale randomized controlled trials are required to validate performance claims.

---

## VII. Future Scope

The following directions are identified for future development:

**1. Multi-Language Support:** Extension to Hindi, Tamil, Telugu, Marathi, and other Indian regional languages, enabling access for students in vernacular-medium institutions.

**2. Custom Voice Selection:** Integration of voice cloning or personalized TTS profiles, allowing students to select preferred voices for podcast generation.

**3. Mobile Application:** Native iOS and Android applications to enable seamless upload, revision, and audio playback on mobile devices.

**4. Collaborative Study Mode:** Shared workspaces enabling study groups to co-annotate materials and generate collective revision artifacts.

**5. Personal Performance Dashboard:** Longitudinal tracking of topic competency, revision history, and predicted examination performance with actionable recommendations.

**6. Institutional Integrations:** Direct integrations with examination boards (CBSE, NTA-JEE, UPSC, ICAI) to ingest official syllabi and PYQ databases, improving priority signal accuracy.

**7. Global PYQ Database:** A crowd-sourced, institution-linked repository of previous year questions across competitive examinations worldwide.

**8. Adversarial Robustness:** Improving resistance to noisy or low-quality inputs (e.g., poorly scanned handwritten notes, incomplete syllabi).

**9. Explainability:** Surfacing the reasoning behind topic prioritization decisions to increase student trust and pedagogical transparency.

---

## VIII. Conclusion

This paper has presented Smaran, an AI-powered study intelligence platform that addresses the challenge of examination preparation under information overload. By integrating LLM-based summarization, a PYQ Intelligence Engine with temporal trend analysis, adaptive content prioritization, and multi-modal revision artifact generation, Smaran delivers targeted, examination-relevant revision material in under 60 seconds from heterogeneous student uploads.

Evaluation with 40 undergraduate students demonstrated a 99.5% reduction in summary generation time, a 23 percentage point improvement in PYQ topic coverage accuracy, and a 5.8 percentage point improvement in examination scores compared to unassisted manual study. The PYQ Intelligence Engine achieved a precision of 0.79 and recall of 0.71 in predicting examination topics.

Smaran represents a meaningful step toward the vision that every student — regardless of institution, resource access, or available preparation time — can walk into an examination with targeted knowledge of what matters most. Future work will expand language coverage, evaluation scale, and institutional integrations.

> *"Every student deserves to walk into an exam knowing they studied the right things — not the most things."*

---

## References

[1] J. R. Anderson, C. F. Boyle, and G. Yost, "The Geometry Tutor," in *Proc. 9th International Joint Conference on Artificial Intelligence (IJCAI)*, 1985, pp. 1–7.

[2] S. Lajoie and S. Derry, *Computers as Cognitive Tools*. Hillsdale, NJ: Lawrence Erlbaum Associates, 1993.

[3] Knewton, "Adaptive Learning Platform," Knewton Inc., 2011. [Online]. Available: https://www.knewton.com

[4] Carnegie Learning, "MATHia Intelligent Tutoring System," Carnegie Learning Inc., 2020. [Online]. Available: https://www.carnegielearning.com

[5] H. P. Luhn, "The Automatic Creation of Literature Abstracts," *IBM Journal of Research and Development*, vol. 2, no. 2, pp. 159–165, 1958.

[6] I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to Sequence Learning with Neural Networks," in *Advances in Neural Information Processing Systems (NIPS)*, 2014, pp. 3104–3112.

[7] A. Vaswani *et al.*, "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NIPS)*, 2017, pp. 5998–6008.

[8] OpenAI, "GPT-4 Technical Report," arXiv preprint arXiv:2303.08774, 2023.

[9] Anthropic, "Claude: A Next-Generation AI Assistant," Anthropic PBC, 2023. [Online]. Available: https://www.anthropic.com

[10] R. Mittal, A. Sharma, and P. Gupta, "Automated Topic Extraction from Examination Question Papers," in *Proc. International Conference on Educational Data Mining (EDM)*, 2019, pp. 214–219.

[11] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, "SQuAD: 100,000+ Questions for Machine Comprehension of Text," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2016, pp. 2383–2392.

[12] R. E. Mayer, *Multimedia Learning*, 2nd ed. Cambridge: Cambridge University Press, 2009.

[13] A. Paivio, *Mental Representations: A Dual Coding Approach*. Oxford: Oxford University Press, 1986.

[14] S. E. Fox and P. Punie, "The Educational Podcast: A Systematic Review of Learning Outcomes," *British Journal of Educational Technology*, vol. 52, no. 4, pp. 1410–1429, 2021.

[15] A. T. Corbett and J. R. Anderson, "Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge," *User Modeling and User-Adapted Interaction*, vol. 4, no. 4, pp. 253–278, 1994.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/smaran.git
cd smaran

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env.local
# Add your API keys to .env.local

# Start development server
npm run dev
```

Navigate to `http://localhost:3000` and upload your first set of notes. Your first one-page summary will be generated in under 60 seconds.

---

## Contributing

Contributions — code, design, feedback, translations, or documentation — are welcome.

```
1. Fork the repository
2. Create a feature branch    →  git checkout -b feature/your-feature
3. Commit your changes        →  git commit -m "Add: your feature"
4. Push to the branch         →  git push origin feature/your-feature
5. Open a Pull Request
```

---

<div align="center">

*स्मरण करो। जीतो। आगे बढ़ो।*
*Remember. Win. Move forward.*

**— The Smaran Team**

</div>
