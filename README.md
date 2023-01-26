[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache License 2.0][license-shield]][license-url]

# YoY Challenge 2022: Yum or Yuck Butterfly Classification 2022

<a name="readme-top">Main repository for a Kaggle machine learning competition, concerning the creation of a classification model for various butterfly spiecies. These include Black Swallowtail (Papilio polyxenes), Monarc (Danaus plexippus), Pipevine Swallowtail (Battus philenor), Spicebush Swallowtail (Papilio troilus), Eastern Tiger Swallowtail (Papilio glaucus) and Viceroy (Limenitis archippus).</a>

<!-- PROJECT LOGO -->
<br/>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#network-architecture">Network architecture</</li>
        <li><a href="#confusion-matrix-binary">Confusion matrix (binary)</</li>
        <li><a href="#confusion-matrix-complete">Confusion matrix (complete)</</li>
        <li><a href="#results">Results</</li>
        <li><a href="#complete-project-report-italian-language">Complete project report (italian language)</</li>
      </ul>
    </li>
    <li><a href="#technologies">Technologies</a>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This neural network-based computer vision project is aimed at classifying six different species of butterflies spread in various parts of the planet. The system was developet to participate in a Kaggle challenge, known as [Yum or Yuck Butterfly Mimics 2022](https://www.kaggle.com/competitions/yum-or-yuck-butterfly-mimics-2022). The ultimate goal of the latter is the development of reliable systems for automatic butterfly recognition in order to order to be able to conduct detailed studies on the distribution and populations of these insect species, some of which are considered endangered. The results obtained with some state-of-the-art convolutional systems are encouraging despite the limitations of the available hardware, achieving about 94%/95% accuracy on the test set.

### Network architecture

### Confusion matrix (binary)

### Confusion matrix (complete)

### Results

| **\#**          | **Sc./Ver.**          | **Freeze pre-trained**          | **Dense dim.**                          | **Regol. Bias**                               | | **Regol. Pesi**                                  | **Regol. Attiv.**      |  **Ottimizzatore** | **$\boldsymbol{\eta}$** | **\textbf{Precision}** | **\textbf{Recall}** | **\textbf{Accuratezza}** | **\textbf{F1-score}** |
|-----------------|-----------------------|---------------------------------|-----------------------------------------|-----------------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| 1               | B0                    | Sì                              | $\left[512, 128, 6\right]$              | $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$              |  $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$              | $\left[-, L_1: 10^{-6}, L_1: 10^{-6}\right]$                      | Adadelta                   | 1                       | 88.30\%                | 88.51\%             | 89.78\%                  | 88.11\%               |
| 2               | B0                    | No                              | $\left[512, 256, 6\right]$              | $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$              |  $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$              | $\left[-, L_1: 10^{-6}, L_1: 10^{-6}\right]$                         | Adadelta                   | 1                       | 91.18\%                | 91.28\%             | 91.25\%                  | 91.24\%               |
| 3               | B2                    | No                              | $\left[512, 6\right]$                   | $\left[L_1: 10^{-6}, -\right]$                            | $\left[L_1: 10^{-6}, -\right]$                            | $\left[-, L_1: 10^{-6}\right]$                                       | 3               | Adadelta                   | 1                       | 92.58\%                | 92.25\%             | 92.70\%                  | 92.40\%               |
| 4               | B1                    | No                              | $\left[512, 256, 6\right]$              | $\left[L_2: 10^{-5}, L_2: 10^{-5}, -\right]$              | $\left[L_2: 10^{-5}, L_2: 10^{-5}, -\right]$              | $\left[L_1: 10^{-6}, L_1: 10^{-6}, L_1: 10^{-6}\right]$              | Adam                       | $3 \cdot 10^{-7}$       | 86.74\%                | 86.55\%             | 86.55\%                  | 86.28\%               |
| 5               | B1                    | No                              | $\left[512, 6\right]$                   | $\left[ -, -\right]$                                      | $\left[L_1: 10^{-6}, -\right]$                            | $\left[-, -\right]$                                                  | Adadelta                   | 1                       | 92.10\%                | 91.23\%             | 91.23\%                  | 91.21\%               |
| 6               | B1                    | No                              | $\left[512, 512, 6\right]$              | $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$              |  $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$              | $\left[L_1: 10^{-6}, L_1: 10^{-6}, L_1: 10^{-6}\right]$              | Adadelta                   | 1                       | 94.88\%                | 93.97\%             | 94.15\%                  | 94.12\%               |
| **7**    | **B1**                       | **No**                          | **$\boldsymbol{\left[512, 256, 6\right]}$** | **$\boldsymbol{\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]}$** | $\boldsymbol{\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]}$ | $\boldsymbol{\left[L_1: 10^{-6}, L_1: 10^{-6}, L_1: 10^{-6}\right]}$ | \textbf{Adadelta}          | \textbf{1}              | \textbf{96.64\%}       | \textbf{96.49\%}    | \textbf{96.49\%}         | \textbf{96.48\%}      |
| 8               | B1                    | No                              | $\left[512, 512, 6\right]$              | $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$              | $\left[L_1: 10^{-5}, L_1: 10^{-6}, -\right]$              | $\left[L_1: 10^{-6}, L_1: 10^{-6}, L_1: 10^{-6}\right]$              | Adadelta, Adam             | 1, $3 \cdot 10^{-7}$    | 94.30\%                | 94.15\%             | 94.74\%                  | 94.68\%               |
| 9               | VGG19                 | No                              | $\left[512, 6\right]$                   | $\left[-, -\right]$                                       | $\left[-, -\right]$                                       | $\left[-, -\right]$                                                  | Adam                       | $3 \cdot 10^{-7}$       | 83.58\%                | 83.34\%             | 83.63\%                  | 82.27\%               |
| 10              | VGG19                 | No                              | $\left[512, 256, 6\right]$              | $\left[ L_1: 10^{-6},  L_1: 10^{-6}, -\right]$            | $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$              | $\left[L_1: 10^{-6}, L_1: 10^{-6}, -\right]$                               | Adam                       | $3 \cdot 10^{-7}$       | 90.69\%                | 90.97\%             | 90.64\%                  | 90.47\%               |
| 11              | VGG16                 | No                              | $\left[512, 256, 6\right]$              | $\left[L_1: 10^{-5},  L_1: 10^{-5}, -\right]$             | $\left[L_1: 10^{-5}, L_1: 10^{-5}, -\right]$              | $\left[L_1: 10^{-5}, L_1: 10^{-5}, L_1: 10^{-5}\right]$              | Adam                       | $3 \cdot 10^{-7}$       | 88.82\%                | 88.08\%             | 88.89\%                  | 87.89\%               |
| 12              | VGG16                 | No                              | $\left[512, 6\right]$                   | $\left[L_2: 10^{-5},-\right]$                             | $\left[L_2: 10^{-5}, -\right]$                            | $\left[-, L_1: 10^{-6}\right]$                | Adam                       | $3 \cdot 10^{-6}$       | 88.30\%                | 88.66\%             | 88.89\%                  | 88.26\%               |


### Complete project report (italian language)
![Alt text](/app/assets/readme-imgs/home-alg-options.png?raw=true)


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Technologies

#### Data & graph processing management
* [![NetworkX]][NetworkX-url]
* [![Pandas]][Pandas-url]
* [![Numpy]][Numpy-url]

#### Data visualization
* [![Plotly]][Plotly-url]

#### Server-side
* [![Dash]][Dash-url]
* [![Flask]][Flask-url]

#### Front-end
* [![Dash]][Dash-url]
* [![React.js]][React-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]


<!-- LICENSE -->
## License

Distributed under the Apache License 2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Link to the Kaggle competition](https://www.kaggle.com/competitions/yum-or-yuck-butterfly-mimics-2022)
* [Best-README template GitHub repository](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[product-screenshot]: images/screenshot.png
[project-logo]: app/assets/ecvt.png
[contributors-shield]: https://img.shields.io/github/contributors/Attornado/yum-or-yuck-butterfly-classification.svg?style=for-the-badge
[contributors-url]: https://github.com/Attornado/yum-or-yuck-butterfly-classification/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Attornado/yum-or-yuck-butterfly-classification.svg?style=for-the-badge
[forks-url]: https://github.com/Attornado/yum-or-yuck-butterfly-classification/network/members
[stars-shield]: https://img.shields.io/github/stars/Attornado/yum-or-yuck-butterfly-classification.svg?style=for-the-badge
[stars-url]: https://github.com/Attornado/yum-or-yuck-butterfly-classification/stargazers
[issues-shield]: https://img.shields.io/github/issues/Attornado/yum-or-yuck-butterfly-classification.svg?style=for-the-badge
[issues-url]: https://github.com/Attornado/yum-or-yuck-butterfly-classification/issues
[license-shield]: https://img.shields.io/github/license/Attornado/yum-or-yuck-butterfly-classification.svg?style=for-the-badge
[license-url]: https://github.com/Attornado/yum-or-yuck-butterfly-classification/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
[Solidity]: https://img.shields.io/badge/solidity-gray?style=for-the-badge&logo=solidity
[Solidity-url]: https://soliditylang.org
[Web3Py]: https://img.shields.io/badge/Web3.py-yellow?style=for-the-badge&logo=Web3.js&logoColor=black
[Web3Py-url]: https://pypi.org/project/web3/
[MongoDB]: https://img.shields.io/badge/MongoDB-darkgreen?style=for-the-badge&logo=mongodb&logoWidth=15
[MongoDB-url]: https://www.mongodb.com/
[Pandas]: https://img.shields.io/badge/Pandas-red?style=for-the-badge&logo=pandas&logoWidth=15
[Pandas-url]: https://pandas.pydata.org/
[Truffle]: https://svgshare.com/getbyhash/sha1-NX499/URB+khENlHOWdGS/+GJNw=
[Truffle-url]: https://trufflesuite.com/
[Numpy]: https://img.shields.io/badge/Numpy-yellow?style=for-the-badge&logo=numpy&logoColor=black
[Numpy-url]: https://numpy.org/
[Flask]: https://img.shields.io/badge/Flask-darkred?style=for-the-badge&logo=flask
[Flask-url]: https://flask.palletsprojects.com/en/2.2.x/
[Ganache]: https://svgshare.com/getbyhash/sha1-4Z5dD5/nHgiA9ULH6Jk1JgFiSBE=
[Ganache-url]: https://trufflesuite.com/ganache/
[Ganache-url]: https://flask.palletsprojects.com/en/2.2.x/
[IPFS]: https://img.shields.io/badge/IPFS-154c79?style=for-the-badge&logo=ipfs
[IPFS-url]: https://ipfs.tech/  
[Ethereum]: https://img.shields.io/badge/Ethereum-76b5c5?style=for-the-badge&logo=ethereum&logoColor=black
[Ethereum-url]: https://ethereum.org/en/
[Matplotlib]: https://svgshare.com/getbyhash/sha1-DUTrNq/OGl0noPQdTr2YgrvYhIw=
[Matplotlib-url]: https://matplotlib.org/
[Dash]: https://svgshare.com/getbyhash/sha1-rP+R9ynV+Lb+plNuV5j6jx9G10c=
[Dash-url]: https://dash.plotly.com/
[NetworkX]: https://svgshare.com/getbyhash/sha1-xg9rckqBiF6LDNPNh+JBiGiAr7s=
[NetworkX-url]: https://networkx.org/documentation/stable/index.html
[Plotly]: https://img.shields.io/badge/Plotly-100000?style=for-the-badge&logo=plotly&logoColor=white&labelColor=660169&color=660169
[Plotly-url]: https://plotly.com/
