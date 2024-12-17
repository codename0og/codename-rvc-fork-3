# <p align="center">` Codename-RVC-Fork 🍇 v3.0 ` </p>
## <p align="center">Based on Applio</p>

<p align="center"> ㅤㅤ👇 Applio's official links below 👇ㅤㅤ </p>

</p>
<p align="center">
  <a href="https://applio.org" target="_blank">🌐 Website</a>
  •
  <a href="https://docs.applio.org" target="_blank">📚 Documentation</a>
  •
  <a href="https://discord.gg/urxFjYmYYh" target="_blank">☎️ Discord</a>
</p>



## A lil bit more the project:

### This fork is pretty much my personal take on Applio. ✨
`Goal of this project is to have a much more flexible base than mainline rvc.`
<br/>
ㅤ
<br/>
**Features that are, at the time of writing, already added in:**
- Configurable learning-rate warmup <br/> ( Provides an ability to give your training a lil warmup, potentially yielding better results. )

- Configurable moving average loss for Generator and Discriminator <br/> ( Helps with better judgement on which epoch to choose. )

- Features a different optimizer: Ranger2020 <br/> ( More advanced than stock AdamW. )
  
- Support for following vocoders: HiFi-gan, MRF-HiFi-gan, Refine-GAN
<br/>``⚠️ 1: HiFi-gan is the stock rvc/applio vocoder, hence it's what you use for og pretrains and customs ( for now ). ``
<br/>``⚠️ 2: Both MRF-HiFi-GAN and Refine-Gan are experimental and shouldn't be used just yet. ``
<br/>``⚠️ 3: MRF-HiFi-GAN and Refine-GAN require new pretrained models. They can't be used with original rvc's G/D pretrains. ``

ㅤ
✨ to-do list ✨
> - More / different configurable optimizers.
> - Adjustable hop length for RMVPE.
> - Custom initial learning rate per Generator and Discriminator.
> - Custom gradient norm value  ( from the ui level )
> - Ability to delay / headstart the Generator or Discriminator.
> - Avg for other metrics. ( fm, mel and kl ) 
> - and more...

### ❗ For contact, please use AI HUB by Weights discord server ❗


## Getting Started:

### 1. Installation

Run the installation script based on your operating system:

- **Windows:** Double-click `run-install.bat`.
- **Linux/macOS:** Execute `run-install.sh`.

### 2. Running Applio

Start Applio using:

- **Windows:** Double-click `run-fork.bat`.
- **Linux/macOS:** Run `run-fork.sh`.

This launches the Gradio interface in your default browser.

### 3. Optional: TensorBoard Monitoring

To monitor training or visualize data:

- **Windows:** Run `run-tensorboard.bat`.
- **Linux/macOS:** Run `run-tensorboard.sh`.

For more detailed instructions, visit the [documentation](https://docs.applio.org).

## Disclaimer
``The creators of the original Applio repository, Applio's contributors, and the maintainer of this fork (Codename;0), built upon Applio, are not responsible for any legal issues, damages, or consequences arising from the use of this repository or the content generated from it. By using this fork, you acknowledge that:``

- The use of this fork is at your own risk.
- This repository is intended solely for educational, and experimental purposes.
- Any misuse, including but not limited to illegal activities or violation of third-party rights, <br/> is not the responsibility of the original creators, contributors, or this fork’s maintainer.
- You willingly agree to comply with this repository's [Terms of Use](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md)
