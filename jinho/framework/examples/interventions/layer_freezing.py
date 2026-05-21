"""Example file-based intervention that trains only upper transformer layers."""

from framework.interventions import LayerFreezingIntervention


INTERVENTION = LayerFreezingIntervention(trainable_start=16, trainable_end=None)
