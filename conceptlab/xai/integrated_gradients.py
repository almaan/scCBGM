import torch


def _integrated_gradients_ix(model, concept, unknown, concept_index, steps=50):

    device = model.device

    # Create baseline and target for the specified concept index
    concept_baseline = concept.clone()
    concept_target = concept.clone()
    concept_baseline[:, concept_index] = 0
    concept_target[:, concept_index] = 1

    concept_baseline, concept, unknown, concept_target = (
        concept_baseline.to(device),
        concept.to(device),
        unknown.to(device),
        concept_target.to(device),
    )

    concept_baseline.unsqueeze(0)
    concept_target = concept_target.unsqueeze(0)

    # alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1)  # Shape: [steps, 1, 1]
    # interpolated_concepts = concept_baseline.unsqueeze(0) + alphas * (concept_target.unsqueeze(0) - concept_baseline.unsqueeze(0))

    alphas = torch.linspace(0, 1, steps, device=device)
    # Initialize the IG accumulator
    gradients = torch.zeros_like(interpolated_concepts)  # Shape: [steps, N, F]

    # Difference
    concept_delta = concept_target - concept_baseline

    # Compute gradients at each interpolation step
    for step in range(steps):

        interpolated_concept = concept_baseline + alphas[k] * concept_delta

        outputs = model.decode(interpolated_concept, unknown)  # Shape: [N, D]
        output = outputs["x_pred"]

        grad_mask = torch.ones_like(output)

        output.backward(grad_mask)  # Compute gradients

        print(interpolated_concept.grad.shape)
        gradients[step] = interpolated_concept.grad  # Save gradients

    # Average gradients across steps
    avg_gradients = gradients.mean(dim=0)  # Shape: [N, F]

    # Compute IG scores
    (concept_target - concept_baseline) * avg_gradients
    pre_ig_c = ig_c[:, concept_index].unsqueeze(1)

    pre_ig_c * delta

    return ig
