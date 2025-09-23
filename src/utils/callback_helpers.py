# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional
import torch

import networkx as nx
import dwave_networkx as dnx
from dwave.system import DWaveSampler
from dwave.plugins.torch.models import DiscreteVariationalAutoencoder
from src.model_wrapper import get_dataset
from plotly import graph_objects as go
from torchvision.utils import save_image

from demo_configs import GENERATE_NEW_MODEL_DIAGRAM, GRAPH_COLORS, SHARPEN_OUTPUT, THEME_COLOR_SECONDARY
from src.utils.common import get_graph_mapping, greedy_get_subgraph

MODEL_PATH = Path("models")
JSON_FILE_DIR = "generated_json"
PROBLEM_DETAILS_PATH = f"{JSON_FILE_DIR}/problem_details.json"
IMAGE_GEN_FILE_PREFIX = "generated_epoch_"
IMAGE_RECON_FILE_PREFIX = "reconstructed_epoch_"
LOSS_PREFIX = "loss_"

def get_example_image(index: int = 0) -> torch.Tensor:
    """Gets the first image of the MNIST dataset.

    Args:
        index: Which MNIST image to get, defaults to the first.

    Returns:
        example_image: The first image of the MNIST dataset. This is always the same image.
    """
    dataset = get_dataset(image_size=32)

    example_image = dataset[index][0]

    save_image(example_image, "static/model_diagram/step_1_input.png")

    return example_image

def create_model_files(
    model: DiscreteVariationalAutoencoder,
    file_name: str,
    qpu: str,
    n_latents: int,
    n_epochs: int,
    loss_data: dict,
):
    """Creates model files, losses file, and parameters file.

    Args:
        model: The DVAE model.
        file_name: The directory name to save all the files to.
        qpu: The QPU associated with the model.
        n_latents: The number of latents.
        n_epochs: The number of epochs.
        loss_data: The loss data to save.
    """
    model.save(file_path=MODEL_PATH / file_name)

    with open(MODEL_PATH / file_name / "parameters.json", "w") as f:
        json.dump(
            {
                "n_latents": n_latents,
                "n_epochs": n_epochs,
                "prefactor": model.PREFACTOR,
                "qpu": qpu,
                "num_read": model.NUM_READS,
                "loss_function": model.LOSS_FUNCTION,
                "image_size": model.IMAGE_SIZE,
                "batch_size": model.BATCH_SIZE,
                "dateset_size": model.DATASET_SIZE,
                "random_seed": model.RANDOM_SEED,
            },
            f,
        )

    with open(MODEL_PATH / file_name / "losses.json", "w") as f:
        json.dump(loss_data, f)


def generate_model_diagram(model: DiscreteVariationalAutoencoder, example_image: torch.Tensor):
    """Generates images of each step in the model diagram.

    Args:
        model: The DVAE model.
        example_image: A tensor to show in the model diagram UI as an example.
    """
    # Update saved UI images
    step_2 = model._dvae.encoder.conv(example_image)
    save_image(step_2[0].unsqueeze(1), "static/model_diagram/step_2_encode.png", padding=1)

    latents = model._dvae.encoder(example_image)
    discretes = model._dvae.latent_to_discrete(latents, 1)
    with open("static/model_diagram/latent_notqpu.json", "w") as f:
        json.dump(discretes[0, 0].tolist(), f)

    step_4 = model._dvae.decoder.merge_batch_dim_and_replica_dim(
        model._dvae.decoder.make_2x2_images(
            model._dvae.decoder.increase_latent_dim(discretes)
        )
    )
    save_image(
        step_4[0].unsqueeze(1),
        "static/model_diagram/step_4_decode.png",
        normalize=True,
        scale_each=True,
        padding=1
    )

    step_5 = model._dvae.decoder(discretes)
    save_image(step_5[0], "static/model_diagram/step_5_output.png")


def execute_training(
    set_progress,
    model: DiscreteVariationalAutoencoder,
    n_epochs: int,
    qpu: str,
    n_latents: int,
    loss_data: Optional[list] = None,
    example_image: Optional[torch.Tensor] = None,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Orchestrates training or tuning of model.

    Args:
        model: The DVAE model.
        n_epochs: The number of epochs to run training for.
        qpu: The selected QPU.
        n_latents: The size of the latent space for the training.
        loss_data: Old loss data from previous training.
        example_image: A tensor to show in the model diagram UI as an example.

    Returns:
        fig_output: The generated image output.
        fig_reconstructed: The image comparing the reconstructed image to the original.
        fig_mse_loss: The graph showing the MSE Loss.
        fig_total_loss: The graph showing the total Loss (MMD + MSE).
    """
    if example_image is not None and GENERATE_NEW_MODEL_DIAGRAM:
        example_image = example_image.unsqueeze(0)

    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        print(f"Starting epoch {epoch + 1}/{n_epochs}")

        total = len(model._dataloader)
        for i, batch in enumerate(model._dataloader):
            set_progress((str(total * epoch + i), str(total * n_epochs)))
            mse_loss = model.step(batch, epoch)

            if example_image is not None and GENERATE_NEW_MODEL_DIAGRAM:
                generate_model_diagram(model, example_image)

        learning_rate_dvae = model._tpar["dvae_lr_schedule"][model._tpar["opt_step"]]
        learning_rate_grbm = model._tpar["grbm_lr_schedule"][model._tpar["opt_step"]]
        print(
            f"Epoch {epoch + 1}/{n_epochs} - MSE Loss: {mse_loss.item():.4f} - "
            f"Learning rate DVAE: {learning_rate_dvae:.3E} "
            f"Learning rate GRBM: {learning_rate_grbm:.3E} "
            f"Time: {(time.perf_counter() - start_time)/60:.2f} mins. "
        )
        with open(PROBLEM_DETAILS_PATH, "w") as f:
            json.dump(
                {
                    "QPU": qpu,
                    "Epoch": f"{epoch + 1}/{n_epochs}",
                    "Batch Size": model.BATCH_SIZE,
                    "Latents": n_latents,
                    "Learning rate DVAE": f"{learning_rate_dvae:.3E}",
                    "Learning rate GRBM": f"{learning_rate_grbm:.3E}",
                    "Mean Squared Error Loss": f"{mse_loss.item():.4f}",
                },
                f,
            )

        fig_output = model.generate_output(
            sharpen=SHARPEN_OUTPUT,
            save_to_file=f"{JSON_FILE_DIR}/{IMAGE_GEN_FILE_PREFIX}{epoch+1}.json",
        )
        fig_reconstructed = model.generate_reconstucted_samples(
            sharpen=SHARPEN_OUTPUT,
            save_to_file=f"{JSON_FILE_DIR}/{IMAGE_RECON_FILE_PREFIX}{epoch+1}.json",
        )
        fig_mse_loss, fig_dvae_loss = model.generate_loss_plot(
            save_to_file_mse=f"{JSON_FILE_DIR}/{LOSS_PREFIX}mse_{epoch+1}.json",
            save_to_file_total=f"{JSON_FILE_DIR}/{LOSS_PREFIX}total_{epoch+1}.json",
            old_loss_data=loss_data,
        )

    return fig_output, fig_reconstructed, fig_mse_loss, fig_dvae_loss


def get_edge_trace(
    G: nx.Graph, node_coords: dict[int, tuple], color: str, line_width: float
) -> go.Scatter:
    """Create a Plotly scatter trace of graph edges.

    Args:
        G (nx.Graph): The graph to plot.
        node_coords (dict): Dictionary mapping nodes to (x, y) coordinates.
        color (str): The color of the edges.
        line_width (float): The width of the edges.

    Returns:
        go.Scatter: A Plotly scatter trace of edges.
    """
    edge_x = []
    edge_y = []
    for start, end in G.edges():
        x0, y0 = node_coords[start]
        x1, y1 = node_coords[end]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=line_width, color=color), hoverinfo="none", mode="lines"
    )

    return edge_trace


def get_node_trace(
    G: nx.Graph,
    node_coords: dict[int, tuple],
    mapping: dict[int, int],
    file_name: str,
) -> go.Scatter:
    """Create a Plotly scatter trace of graph nodes.

    Args:
        G (nx.Graph): The graph to plot.
        pos (dict): Dictionary mapping nodes to (x, y) coordinates.
        color (str): The node color.

    Returns:
        go.Scatter: A Plotly scatter trace of nodes.
    """
    node_x = [node_coords[node][0] for node in G.nodes()]
    node_y = [node_coords[node][1] for node in G.nodes()]

    try:
        with open(file_name, "r") as f:
            latent = json.load(f)

        color_mapping = [GRAPH_COLORS[int(latent[i] > 0)] for i in mapping]

    except Exception:  # Expected when QPU or latents setting is updated
        print("Accurate latent color mapping not available for the requested graph nodes.")
        color_mapping = [GRAPH_COLORS[random.randint(0, 1)] for _ in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color=color_mapping,
            size=5,
        ),
    )

    return node_trace


def get_fig(G: nx.Graph, node_coords: dict[int, tuple], mapping: dict[int, int], file_name: str, show_edges: bool=True) -> go.Figure:
    """Generate a Plotly fig of a graph with highlighted subgraph.

    Args:
        G (nx.Graph): The complete graph.
        node_coords (dict): Dictionary mapping nodes to (x, y) coordinates.

    Returns:
        go.Figure: A Plotly figure showing a graph with highlighted subgraph.
    """
    random.seed(10)
    data = []

    if show_edges:
        edge_trace = get_edge_trace(G, node_coords, "#5e9ce8", 0.3)
        data.append(edge_trace)

    node_trace = get_node_trace(G, node_coords, mapping, file_name)
    data.append(node_trace)

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    return fig


def generate_model_fig(
    qpu: str, n_latents: int, random_seed: int
) -> tuple[go.Figure, go.Figure, list]:
    """Generates a figure of the machine learning model.

    Args:
        qpu: The selected qpu title.
        n_latents: The size of the latent space.
        random_seed: The random seed for node selection.

    Returns:
        fig_output: The generated image output.
        fig_reconstructed: The image comparing the reconstructed image to the original.
        fig_mse_loss: The graph showing the MSE Loss.
        fig_total_loss: The graph showing the total Loss (MMD + MSE).
    """
    qpu = DWaveSampler(solver=qpu)
    qpu_graph = qpu.to_networkx_graph()
    subgraph = greedy_get_subgraph(n_nodes=n_latents, random_seed=random_seed, graph=qpu_graph)
    _, mapping = get_graph_mapping(subgraph)

    latent_mapping = [mapping[node] for node in subgraph.nodes()]

    qpu_shape = qpu.properties["topology"]["shape"][0]
    qpu_topology = qpu.properties["topology"]["type"]

    if qpu_topology == "pegasus":
        node_coords = dnx.drawing.pegasus_layout(dnx.pegasus_graph(qpu_shape), crosses=True)
    elif qpu_topology == "zephyr":
        node_coords = dnx.drawing.zephyr_layout(dnx.zephyr_graph(qpu_shape))
    elif qpu_topology == "chimera":
        node_coords = dnx.drawing.chimera_layout(dnx.chimera_graph(qpu_shape))
    else:
        raise ValueError(f"Unknown QPU topology: {qpu_topology}")

    fig_qpu = get_fig(subgraph, node_coords, latent_mapping, "static/model_diagram/latent_qpu.json")
    fig_not_qpu = get_fig(subgraph, node_coords, latent_mapping, "static/model_diagram/latent_notqpu.json", False)

    return fig_qpu, fig_not_qpu, latent_mapping
