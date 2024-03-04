<!-- TODO:
- Need a logo
- Screenshot of the GUI
- Some of the pipeline unfinished
- Explain the GUI's output
- List attacks and defenses like Poison-Playground
- Explain how to add an attack or defense to the GUI
- Marom's gmail -->



<div style="text-align: center;">
  <h1>Poison Playground</h1>
</div>




<!-- PROJECT SHIELDS -->
<div align="left">
<img src="https://badgen.net/github/contributors/MaromSv/Poison-Playground">
<img src="https://badgen.net/github/stars/MaromSv/Poison-Playground?color=green">
<img src="https://badgen.net/github/forks/MaromSv/Poison-Playground">
<img src="https://badgen.net/github/watchers/MaromSv/Poison-Playground">
<img src="https://badgen.net/github/issues/MaromSv/Poison-Playground">
<img src="https://img.shields.io/github/commit-activity/m/MaromSv/Poison-Playground">
<img src="https://img.shields.io/github/languages/code-size/MaromSv/Poison-Playground">
</div>



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/MaromSv/Poison-Playground">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
</div>



<!-- TABLE OF CONTENTS -->
<div>
  <h2>Table of contents</h2>

  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#poison-playgrounds-pipeline">Poison Playground's pipeline</a></li>
    <li><a href="#attack--defense-sources">Attack & Defense sources</a></li>
    <li><a href="#adding-new-attacks--defenses">Adding new attacks and defenses</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</div>



<!-- ABOUT THE PROJECT -->
## About the project

<div align="center">
  <a href="https://github.com/MaromSv/Poison-Playground">
    <img src="images/teaser.png" alt="Teaser" width="80" height="80">
  </a>
</div>

Poison Playground is a benchmarking tool that can be used to compare how vertical and horizontal federated learning models handle different attacks and defenses. The base tool includes some poisoning attacks and defenses. <br />
Poison Playgroud uses the <a href="https://www.tensorflow.org/datasets/catalog/mnist">MNIST Digits</a> dataset using TensorFlow.

Poison Playground was created by researchers, for researchers. It is meant to help make comparing the effects of various attacks and defenses on horizontal and vertical federated learning much easier. It is intended to speed up research by providing a useful metric collection and comparison tool.



<!-- GETTING STARTED -->
## Getting started

To ensure you have all of the necessary packages, first run the [requirements.txt](/requirements.txt) file using the below command
  ```sh
  pip install -r requirements.txt
  ```
  Once the files are installed, you can simply run the [main.py](/Federated_Learning/main.py) file. This will open the GUI, which you can use to orchestrate your experiments.

## Poison Playground's pipeline

1. Run [main.py](/Federated_Learning/main.py)

2. In the GUI, fill in <span style="font-size: x-large;">(more info about how the user inputs parameters)</span>. Then press the "Run Simulation" button.

3. Once the "Run Simulation" button has been pressed, the described experiments will run. The first step in running an instance is to partition the data, either horizontally or vertically. This is done in the [dataPartitioning.py](/Federated_Learning//dataPartitioning.py) file.

4. Then, the instances will be run one by one. To run an instance, the parameters inputted for the instance are passed to the [simulationHorizontal.py](/Federated_Learning/simulationHorizontal.py) or [simulationVertical.py](/Federated_Learning/simulationVertical.py) simulation file, depending on which federated learning model was chosen.

5. Finally, once all of the instances are done, the GUI will display a new window with <span style="font-size: x-large;">(describe the metrics and graphs displayed)</span>.



<!-- ATTACKS AND DEFENSES -->
## Attack & Defense sources

| Attack/Defense | Name | Paper source |
|---------|----------|----------|
| Attack  | Label flipping | [Data Poisoning Attacks Against Federated Learning Systems](https://arxiv.org/abs/2007.08432) |
| Attack  | Model poisoning | [MPAF: Model Poisoning Attacks to Federated Learning based on Fake Clients](https://arxiv.org/abs/2203.08669) |
| Defense | Fools gold | [Mitigating Sybils in Federated Learning Poisoning](https://arxiv.org/abs/1808.04866) |
| Defense | Two norm | [Data Poisoning Attacks Against Federated Learning Systems](https://arxiv.org/abs/2203.08669) |



<!-- ADDING CUSTOM ATTACKS AND DEFENSES -->
## Adding new attacks & defenses

The below three steps describe how you can add your own attack or defense to Poison Playground:

1) **Create the attack's/defense's file:** The first step is to create your new file in either the [attacks](/Federated_Learning/attacks/) or [defenses](/Federated_Learning/defenses/) folder. Then you need to write your attack/defense code. It is recommended to make your attack/defense executable by calling only one function, and that it obtains all needed parameters from the simulation file.

2) **Add the attack/defense to both simulation's code:** To execute your attack/defense, you need to add it to the simulation files. Depending on how your code works, the location of execution will vary. Some examples can be seen with our implemented attacks and defenses.

3) **Add the attack/defense to the GUI:** <span style="font-size: x-large;">(Not sure)</span>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- CONTACT -->
## Contact

Yusef Ahmed - [LinkedIn](https://www.linkedin.com/in/yusefahmd/) - yusefahmed0403@gmail.com

Marom Sverdlov - [LinkedIn](https://www.linkedin.com/in/marom-sverdlov-251370252/) - GMAIL



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()
