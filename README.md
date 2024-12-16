# <p align="center">` Codename-RVC-Fork 🍇 v3.0 ` </p>
## <p align="center">Based on Applio:ㅤv3.2.8 </p>

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

ㅤ
⚠️ to-do list ⚠️
> - More / different configurable optimizers.
> - Adjustable hop length for RMVPE.
> - Custom initial learning rate per Generator and Discriminator.
> - Custom gradient norm value  ( from the ui level )
> - Ability to delay / headstart the Generator or Discriminator.
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
``The creators of original Applio repository, Applio's contributors, and the maintainer of this fork (Codename;0) which is built upon Applio are not responsible for any legal issues, damages, or consequences arising from the use of this repository or the generated content from it. By using this fork, you acknowledge that:``

- The use of this fork is at your own risk.
- This repository is intended solely for educational, and experimental purposes.
- Any misuse, including but not limited to illegal activities or violation of third-party rights, <br/> is not the responsibility of the original creators, contributors, or this fork’s maintainer.
