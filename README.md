# CosyVoice_DPO_NOTES

🎙️ **CosyVoice_DPO_NOTES: Supercharge Your TTS with Cutting-Edge DPO Fine-Tuning!** 🎙️

Welcome to **CosyVoice_DPO_NOTES**, the go-to resource for TTS practitioners looking to push the boundaries of speech synthesis with Direct Preference Optimization (DPO)! Built on the powerful CosyVoice framework by FunAudioLLM, this repository is your treasure trove of practical insights, code snippets, and detailed notes on fine-tuning CosyVoice models using DPO to achieve unparalleled speaker similarity, pronunciation accuracy, and naturalness in multilingual and zero-shot scenarios. 🌍🎵

### Why This Repo Rocks for TTS Enthusiasts
- 🚀 **DPO-Powered Fine-Tuning**: Dive into hands-on examples and step-by-step guides on using DPO to optimize CosyVoice’s LLM for speech token generation, leveraging metrics like Word Error Rate (WER) and Speaker Similarity (SS) to craft human-like voices.
- 🎤 **Multilingual Mastery**: Explore techniques for zero-shot voice cloning and cross-lingual synthesis across languages like Chinese, English, Japanese, Korean, and 18 Chinese dialects, with tips to tackle tricky prosody and timbre challenges.
- ⚡ **Low-Latency Streaming**: Learn how to harness CosyVoice 2’s bidirectional streaming capabilities for ultra-low 150ms latency, perfect for real-time applications like virtual assistants or audiobooks.
- 🛠️ **Practical Hacks & Workarounds**: Get insider tips from a seasoned CosyVoice user on avoiding common pitfalls, like overfitting to speaker prompts or handling out-of-domain inputs, with real-world solutions tested on datasets like Seed-TTS.
- 📝 **Comprehensive Notes**: From model architecture tweaks to dataset scaling (up to 1M hours!), this repo distills the latest CosyVoice 2 and 3 papers into actionable advice, saving you hours of research.

### What’s Inside
- **DPO Implementation Guides**: Code and configs for integrating DPO with CosyVoice’s text-speech LLM, including how to use differentiable ASR rewards for better generalization.
- **Prompt Engineering Tips**: Best practices for using speaker prompts like `"Speaker A<|endofprompt|>"` in multi-speaker SFT, with workarounds for reducing errors.
- **Performance Benchmarks**: Notes on achieving human-parity MOS scores (5.53 vs. 5.52 for commercial models) and slashing pronunciation errors by 30-50%.
- **Dataset & Model Scaling**: Strategies for handling large-scale multilingual datasets and scaling models from 0.5B to 1.5B parameters.
- **Community-Driven Insights**: Contributions from TTS practitioners, addressing real issues like slow inference or complex voice handling (e.g., Speaker E in Seed-TTS).

### Who’s This For?
- TTS researchers and developers eager to fine-tune state-of-the-art models like CosyVoice 2 and 3.
- Voice cloning enthusiasts aiming for zero-shot or cross-lingual synthesis with high fidelity.
- Real-time TTS app builders needing low-latency, high-quality solutions.
- Anyone curious about DPO’s magic in making synthetic voices sound *scarily* human!

### Get Started
Clone this repo, dive into the code, and start experimenting with CosyVoice’s pretrained models (`CosyVoice2-0.5B`, `CosyVoice-300M-SFT`, etc.) available on Hugging Face or ModelScope. Join the conversation on GitHub Issues to share your tweaks or ask for help. Whether you’re crafting a multilingual chatbot or a next-gen audiobook narrator, **CosyVoice_DPO_NOTES** is your shortcut to TTS excellence! 🚀

🔗 **Repo**: https://github.com/ScottishFold007/Cosyvoice_DPO_NOTES  
📚 **Inspired by**: CosyVoice 2 (arXiv:2412.10117) & CosyVoice 3 (arXiv:2505.16736)  
🎧 **Demos**: Check out CosyVoice’s demos at https://funaudiollm.github.io/cosyvoice2  

Let’s make synthetic voices sound more human than ever! 🎙️

- **行动号召**：鼓励用户克隆仓库、参与社区，并提供相关链接和资源。

如果需要更简短或更技术化的版本，请告诉我！
