---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Deep Dream for Transformers"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2023-08-01T23:30:23-07:00
lastmod: 2023-08-01T23:30:23-07:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

<head>
  <title>GPT2 Visualization</title>
  <style>
    body {
      font-family: Arial, sans-serif;
    }

    #modelContainer {
      margin: 20px;
    }

    #info {
      margin: 20px;
    }
  </style>
</head>
<body>
  <h1>GPT2 Model Structure</h1>

  <div id="modelContainer">
    <!-- Our D3.js visualization will be inserted here -->
  </div>

  <div id="info">
    <p id="layerIndex"></p>
    <p id="neuronIndex"></p>
    <p id="explanation"></p>
  </div>

  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
    // Fetch the JSON file
    fetch('/files/GPT2NeuronExplanations.json')
      .then(response => response.json())
      .then(data => {
        // Save the data for later use
        window.neuronExplanations = data;

        // Set up the SVG canvas and render the model
        const svg = d3.select("#modelContainer")
          .append("svg")
          .attr("width", 600)
          .attr("height", 500);

        renderModel(GPT2Model, svg);
      });

    // Model Creation
    const GPT2Model = {
      transformer: {
        wte: { type: 'Embedding' },
        h: {
          moduleList: {
            GPT2Blocks: []
          }
        }
      }
    };

    // Function to create neurons
    function createNeurons(numNeurons) {
      let neurons = [];
      for (let i = 0; i < numNeurons; i++) {
        neurons.push({ index: i });
      }
      return neurons;
    }

    // Function to create GPT2Blocks
    function createGPT2Blocks(numBlocks, numNeurons) {
      for (let i = 0; i < numBlocks; i++) {
        let block = {
          index: i,
          mlp: {
            c_fc: {
              type: 'Conv1D',
              neurons: createNeurons(numNeurons)
            }
          }
        };
        GPT2Model.transformer.h.moduleList.GPT2Blocks.push(block);
      }
    }

    createGPT2Blocks(6, 10);

    // Function to render the GPT2 model
    function renderModel(model, svg) {
      // For each layer in the model...
      model.transformer.h.moduleList.GPT2Blocks.forEach((block, i) => {
        // Create a group for the layer
        const layerGroup = svg.append("g")
          .attr("transform", `translate(0,${i * 100})`);

        // For each neuron in the layer...
        block.mlp.c_fc.neurons.forEach((neuron, j) => {
          // Create a circle for the neuron
          layerGroup.append("circle")
            .attr("cx", 50 + j * 50)
            .attr("cy", 50)
            .attr("r", 20)
            .on("click", function() {
              // Display the layer and neuron index when clicked
              d3.select("#layerIndex").text(`Layer: ${i}`);
              d3.select("#neuronIndex").text(`Neuron: ${j}`);

              // Look up the explanation for this neuron
              const explanation = window.neuronExplanations[i][j];
              // Display the explanation
              d3.select("#explanation").html(`Without Autoencoder: <br>${explanation[0]}<br><br>With Autoencoder:<br> ${explanation[1]}`);
            });
        });
      });
    }
  </script>
</body>
